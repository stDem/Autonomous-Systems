#!/usr/bin/env python3
# drive_autonomous_only.py  (Python 3.6 compatible)
#
# Controls:
#   A (BTN_SOUTH) : toggle STOP <-> AUTO
#   B (BTN_EAST)  : STOP immediately (emergency stop)
#
# Optional override:
#   Left stick X (ABS_X)  : manual steering override when moved strongly
#   Right stick (ABS_RZ)  : manual throttle override when moved strongly
#
# Requirements:
#   ./models/best_control_cnn.pth
#   ./models/control_norm.json
#
# Dataset preprocessing MUST match training (resize + mean/std normalization).

from __future__ import print_function
import time
import json
import math
from threading import Thread, Lock

import cv2
import numpy as np

import torch
import torch.nn as nn

from inputs import get_gamepad, UnpluggedError
from jetcam.csi_camera import CSICamera
from jetracer.nvidia_racecar import NvidiaRacecar


# -----------------------------
# Model (must match training exactly)
# -----------------------------
class Dave2Small(nn.Module):
    def __init__(self, dropout_p=0.6):
        super(Dave2Small, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),

            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.BatchNorm2d(36),
            nn.ReLU(inplace=True),

            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),

            nn.Conv2d(48, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 1 * 18, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),

            nn.Linear(100, 50),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),

            nn.Linear(50, 10),
            nn.ReLU(inplace=True),

            nn.Linear(10, 2),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# -----------------------------
# Safety / tuning parameters
# -----------------------------
STEERING_CLAMP = 0.35     # limit steering magnitude
THROTTLE_MIN = 0.00       # no reverse for first real test
THROTTLE_MAX = 0.16       # keep below your MAX_THROTTLE (0.2) at first

# smoothing (0..1). Higher = smoother, slower reaction.
STEER_SMOOTH = 0.35
THROTTLE_SMOOTH = 0.0

# manual override thresholds (set higher if you want “mostly pure auto”)
OVERRIDE_STEER_THRESH = 0.20
OVERRIDE_THROTTLE_THRESH = 0.08


# -----------------------------
# Simple mode
# -----------------------------
MODE_STOP = 0
MODE_AUTO = 1


# -----------------------------
# Shared state from gamepad thread
# -----------------------------
class SharedState(object):
    def __init__(self):
        self.lock = Lock()

        self.mode_toggle = False  # A pressed event
        self.stop_now = False     # B pressed event

        self.manual_steer = 0.0
        self.manual_throttle = 0.0

        self.running = True


def steering_transform(raw):
    """
    Same idea as your data_collection steering_transform:
    deadzone + square curve + scale.
    raw is expected in [-1,1].
    """
    x = raw
    deadzone = 0.05
    if abs(x) < deadzone:
        return 0.0

    if x > 0:
        x = (x - deadzone) / (1.0 - deadzone)
    else:
        x = (x + deadzone) / (1.0 - deadzone)

    x = math.copysign(x * x, x)
    x *= 0.7  # scale down
    if x > 1.0:
        x = 1.0
    if x < -1.0:
        x = -1.0
    return x


def gamepad_thread(shared):
    """
    Reads controller events.
    Mapping:
      ABS_X  -> steering
      ABS_RZ -> throttle forward only
      BTN_SOUTH -> A toggle auto
      BTN_EAST  -> B stop
    """
    print("[INFO] Gamepad thread started.")
    print("[INFO] A (BTN_SOUTH) toggle STOP<->AUTO, B (BTN_EAST) STOP immediately.")

    while True:
        with shared.lock:
            if not shared.running:
                break

        try:
            events = get_gamepad()
        except UnpluggedError:
            time.sleep(0.5)
            continue

        for e in events:
            if e.ev_type == "Key":
                if e.code == "BTN_SOUTH" and e.state == 1:
                    with shared.lock:
                        shared.mode_toggle = True
                elif e.code == "BTN_EAST" and e.state == 1:
                    with shared.lock:
                        shared.stop_now = True

            elif e.ev_type == "Absolute":
                # left stick X (0..255)
                if e.code == "ABS_X":
                    v = float(e.state)
                    norm = (v - 127.0) / 127.0
                    if norm > 1.0:
                        norm = 1.0
                    if norm < -1.0:
                        norm = -1.0
                    steer = steering_transform(norm)
                    with shared.lock:
                        shared.manual_steer = float(steer)

                # throttle on ABS_RZ (0..255), forward only
                elif e.code == "ABS_RZ":
                    v = float(e.state)
                    norm = (127.0 - v) / 127.0  # up -> +, center -> 0, down -> -
                    if norm < 0.0:
                        norm = 0.0
                    if norm > 1.0:
                        norm = 1.0
                    thr = norm * THROTTLE_MAX
                    with shared.lock:
                        shared.manual_throttle = float(thr)

    print("[INFO] Gamepad thread exit.")


def clamp(x, lo, hi):
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def ema(prev, new, smooth):
    # smooth in [0,1): 0 -> no smoothing, 0.9 -> very smooth
    return smooth * prev + (1.0 - smooth) * new


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- load norm ----
    with open("./models/control_norm.json", "r") as f:
        norm = json.load(f)
    W = int(norm["input_w"])
    H = int(norm["input_h"])
    img_mean = np.array(norm["img_mean"], dtype=np.float32)
    img_std = np.array(norm["img_std"], dtype=np.float32)

    steer_bias = float(norm.get("steer_bias", 0.0))
    # ---- load model ----
    model = Dave2Small(dropout_p=0.6).to(device)
    model.load_state_dict(torch.load("./models/best_control_cnn.pth", map_location=device))
    model.eval()
    print("[INFO] Loaded model + norm. Device:", device)

    # ---- init car ----
    car = NvidiaRacecar()
    car.steering_gain = 1.0
    car.steering_offset = 0.0
    car.throttle_gain = 1.0
    car.steering = 0.0
    car.throttle = 0.0

    # ---- init camera ----
    camera = CSICamera(width=224, height=224, capture_width=1280, capture_height=720, capture_fps=30)
    camera.running = True

    # ---- gamepad thread ----
    shared = SharedState()
    t = Thread(target=gamepad_thread, args=(shared,), daemon=True)
    t.start()

    mode = MODE_STOP
    print("[MODE] STOP (A to start AUTO, B to STOP)")

    # smoothed commands
    steer_s = 0.0
    thr_s = 0.0

    try:
        while True:
            frame_rgb = camera.value
            if frame_rgb is None:
                time.sleep(0.01)
                continue

            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # ---- read button events + manual ----
            with shared.lock:
                a = shared.mode_toggle
                b = shared.stop_now
                shared.mode_toggle = False
                shared.stop_now = False
                manual_steer = shared.manual_steer
                manual_thr = shared.manual_throttle

            if b:
                mode = MODE_STOP
                print("[MODE] STOP (B pressed)")

            if a:
                if mode == MODE_STOP:
                    mode = MODE_AUTO
                    print("[MODE] AUTO (A pressed)")
                else:
                    mode = MODE_STOP
                    print("[MODE] STOP (A pressed)")

            # ---- predict ----
            # preprocess exactly like training
            img = cv2.resize(frame_bgr, (W, H))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            img = (img - img_mean) / img_std
            img = np.transpose(img, (2, 0, 1))
            x = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                pred = model(x)[0].cpu().numpy()

            auto_steer = float(pred[0]) + steer_bias
            auto_thr = float(pred[1])

            # ---- choose command by mode ----
            if mode == MODE_STOP:
                steer_cmd = 0.0
                thr_cmd = 0.0
            else:
                steer_cmd = auto_steer
                thr_cmd = max(auto_thr, THROTTLE_MIN)

                # manual override (help recovery)
                if abs(manual_steer) > OVERRIDE_STEER_THRESH:
                    steer_cmd = -manual_steer
                if manual_thr > OVERRIDE_THROTTLE_THRESH:
                    thr_cmd = manual_thr

            # ---- clamp & smooth ----
            steer_cmd = clamp(steer_cmd, -STEERING_CLAMP, STEERING_CLAMP)
            thr_cmd = clamp(thr_cmd, THROTTLE_MIN, THROTTLE_MAX)

            steer_s = ema(steer_s, steer_cmd, STEER_SMOOTH)
            thr_s = ema(thr_s, thr_cmd, THROTTLE_SMOOTH)

            # ---- apply to car ----
            car.steering = -float(steer_s)
            car.throttle = float(thr_s)

            # ---- optional debug window ----
            applied_steer = -steer_s
            cv2.putText(frame_bgr,
                "steer={:+.3f} thr={:+.3f}".format(applied_steer, thr_s),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)


            cv2.imshow("Autonomous (B=STOP, A=toggle)", frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] q pressed, exiting.")
                break

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("[INFO] KeyboardInterrupt")

    finally:
        # stop everything safely
        with shared.lock:
            shared.running = False
        car.steering = 0.0
        car.throttle = 0.0
        camera.running = False
        cv2.destroyAllWindows()
        print("[INFO] Stopped safely.")


if __name__ == "__main__":
    main()
