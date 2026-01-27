#!/usr/bin/env python3
# drive_autonomous_yolo.py  (Python 3.6 compatible)

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
STEERING_CLAMP = 0.9
THROTTLE_MIN = 0.13
THROTTLE_MAX = 0.13

STEER_SMOOTH = 0.0
THROTTLE_SMOOTH = 0.5

OVERRIDE_STEER_THRESH = 0.20
OVERRIDE_THROTTLE_THRESH = 0.08


# -----------------------------
# Object detection settings (YOLOv5n)
# -----------------------------
OD_WEIGHTS = "./models/od_best.pt"     # your trained YOLO weights
OD_IMG_SIZE = 416                     # 320/416/640 (Nano: start 416 or 320)
DETECT_EVERY = 3                      # run detector every N frames (speed)

# Stability + cooldown (to trigger once per pass)
OD_CONF_DETECT = 0.25                 # show boxes
OD_CONF_TRIGGER = 0.60                # trigger actions only if very confident
OD_STABLE_FRAMES = 2                  # must be seen this many consecutive detector runs
OD_COOLDOWN_SEC = 8.0                 # prevent repeated triggers while passing sign

# Throttle behavior
SLOW_FACTOR_PERSON = 0.75             # child girl / woman / man
SLOW_FACTOR_ZONE30 = 0.80             # zone30
STOP_HOLD_SEC = 1.0                   # stop sign: full stop for 1 sec


# class list (must match your classes.txt order)
CLASSES = [
    "child girl",
    "woman",
    "man",
    "give way sign",
    "priority road sign",
    "stop sign",
    "turn right ahead sign",
    "warning sign",
    "zone 30 sign",
    "zone 30 stop sign",
]

# convenient ids
ID_CHILD_GIRL = 0
ID_WOMAN      = 1
ID_MAN        = 2
ID_STOP       = 5
ID_ZONE30     = 8
ID_ZONE30STOP = 9


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
    x = raw
    deadzone = 0.05
    if abs(x) < deadzone:
        return 0.0

    if x > 0:
        x = (x - deadzone) / (1.0 - deadzone)
    else:
        x = (x + deadzone) / (1.0 - deadzone)

    x = math.copysign(x * x, x)
    x *= 0.7
    if x > 1.0:
        x = 1.0
    if x < -1.0:
        x = -1.0
    return x


def gamepad_thread(shared):
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
                if e.code == "ABS_X":
                    v = float(e.state)
                    norm = (v - 127.0) / 127.0
                    norm = max(-1.0, min(1.0, norm))
                    steer = steering_transform(norm)
                    with shared.lock:
                        shared.manual_steer = float(steer)

                elif e.code == "ABS_RZ":
                    v = float(e.state)
                    norm = (127.0 - v) / 127.0
                    norm = max(0.0, min(1.0, norm))
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
    return smooth * prev + (1.0 - smooth) * new


def load_yolov5_model(device):
    """
    Loads YOLOv5 model. Best practice on Jetson:
    - clone yolov5 repo locally
    - use torch.hub with source='local'
    """
    try:
        # If you have yolov5 folder рядом (recommended):
        #   git clone https://github.com/ultralytics/yolov5
        # Then this works offline:
        model = torch.hub.load("yolov5", "custom", path=OD_WEIGHTS, source="local")
    except Exception as e:
        print("[ERROR] Failed to load YOLOv5 via local hub. Ensure yolov5 repo exists in current dir.")
        raise

    model.to(device)
    model.eval()
    # set inference size and conf threshold for display
    model.conf = OD_CONF_DETECT
    model.iou = 0.45
    return model


def draw_detections(frame_bgr, dets_xyxy, names):
    """
    dets_xyxy: numpy Nx6 = [x1,y1,x2,y2,conf,cls]
    """
    out = frame_bgr.copy()
    for d in dets_xyxy:
        x1, y1, x2, y2, conf, cls = d
        cls = int(cls)
        label = "{} {:.2f}".format(names[cls], float(conf))
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(out, label, (int(x1), int(y1) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return out


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

    # ---- load driving model ----
    model = Dave2Small(dropout_p=0.6).to(device)
    model.load_state_dict(torch.load("./models/best_control_cnn.pth", map_location=device))
    model.eval()
    print("[INFO] Loaded control model. Device:", device)

    # ---- load YOLO ----
    od_model = load_yolov5_model(device)
    print("[INFO] Loaded YOLOv5 detector:", OD_WEIGHTS)

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

    steer_s = 0.0
    thr_s = 0.0

    # OD state
    frame_i = 0
    last_dets = np.zeros((0, 6), dtype=np.float32)
    last_det_vis = None

    stable_count = {}      # class_id -> consecutive hits
    last_trigger = {}      # class_id -> time.time() last triggered
    zone30_active = False
    stop_until_time = 0.0

    try:
        while True:
            frame_rgb = camera.value
            if frame_rgb is None:
                time.sleep(0.01)
                continue

            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # ---- read gamepad ----
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

            # ---- driving prediction ----
            img = cv2.resize(frame_bgr, (W, H))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            img = (img - img_mean) / img_std
            img = np.transpose(img, (2, 0, 1))
            x = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                pred = model(x)[0].cpu().numpy()

            auto_steer = float(pred[0]) + steer_bias
            auto_thr = float(pred[1])

            # ---- run detector every N frames ----
            frame_i += 1
            if (frame_i % DETECT_EVERY) == 0:
                # YOLOv5 expects RGB
                od_in = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                results = od_model(od_in, size=OD_IMG_SIZE)
                # results.xyxy[0] tensor Nx6
                det = results.xyxy[0]
                if det is None or len(det) == 0:
                    last_dets = np.zeros((0, 6), dtype=np.float32)
                else:
                    last_dets = det.detach().cpu().numpy().astype(np.float32)

                last_det_vis = draw_detections(frame_bgr, last_dets, CLASSES)

                # update stability counters (per class)
                seen_classes = set()
                for d in last_dets:
                    conf = float(d[4])
                    cls = int(d[5])
                    if conf >= OD_CONF_TRIGGER:
                        seen_classes.add(cls)

                # increment seen, reset not seen
                for cls in range(len(CLASSES)):
                    if cls in seen_classes:
                        stable_count[cls] = stable_count.get(cls, 0) + 1
                    else:
                        stable_count[cls] = 0

            # ---- choose command by mode ----
            if mode == MODE_STOP:
                steer_cmd = 0.0
                thr_cmd = 0.0
            else:
                steer_cmd = auto_steer
                thr_cmd = max(auto_thr, THROTTLE_MIN)

                # manual override
                if abs(manual_steer) > OVERRIDE_STEER_THRESH:
                    steer_cmd = -manual_steer
                if manual_thr > OVERRIDE_THROTTLE_THRESH:
                    thr_cmd = manual_thr

            # ---- apply OD rules (only when AUTO) ----
            now = time.time()
            if mode != MODE_STOP:
                # stop logic overrides everything
                if now < stop_until_time:
                    thr_cmd = 0.0
                else:
                    # helper: check trigger allowed
                    def can_trigger(cls):
                        if stable_count.get(cls, 0) < OD_STABLE_FRAMES:
                            return False
                        last = last_trigger.get(cls, 0.0)
                        return (now - last) >= OD_COOLDOWN_SEC

                    # STOP sign
                    if can_trigger(ID_STOP):
                        last_trigger[ID_STOP] = now
                        stop_until_time = now + STOP_HOLD_SEC
                        # (optional) also clear zone
                        # zone30_active = False

                    # zone30 / zone30stop
                    if can_trigger(ID_ZONE30):
                        last_trigger[ID_ZONE30] = now
                        zone30_active = True

                    if can_trigger(ID_ZONE30STOP):
                        last_trigger[ID_ZONE30STOP] = now
                        zone30_active = False

                    # people slow-down (any of these)
                    people_seen = (
                        stable_count.get(ID_CHILD_GIRL, 0) >= OD_STABLE_FRAMES or
                        stable_count.get(ID_WOMAN, 0) >= OD_STABLE_FRAMES or
                        stable_count.get(ID_MAN, 0) >= OD_STABLE_FRAMES
                    )
                    if people_seen:
                        thr_cmd = thr_cmd * SLOW_FACTOR_PERSON

                    if zone30_active:
                        thr_cmd = thr_cmd * SLOW_FACTOR_ZONE30

            # ---- clamp & smooth ----
            steer_cmd = clamp(steer_cmd, -STEERING_CLAMP, STEERING_CLAMP)

            if mode == MODE_STOP:
                thr_cmd = 0.0
            else:
                thr_cmd = clamp(thr_cmd, THROTTLE_MIN, THROTTLE_MAX)

            steer_s = ema(steer_s, steer_cmd, STEER_SMOOTH)
            thr_s = ema(thr_s, thr_cmd, THROTTLE_SMOOTH)

            # ---- apply to car ----
            car.steering = -float(steer_s)
            car.throttle = float(thr_s)

            # ---- windows ----
            applied_steer = -steer_s
            cv2.putText(frame_bgr,
                "steer={:+.3f} thr={:+.3f} zone30={} stop={}".format(
                    applied_steer, thr_s, int(zone30_active), int(time.time() < stop_until_time)
                ),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            cv2.imshow("Autonomous (B=STOP, A=toggle)", frame_bgr)

            if last_det_vis is None:
                cv2.imshow("Detections", frame_bgr)
            else:
                cv2.imshow("Detections", last_det_vis)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] q pressed, exiting.")
                break

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("[INFO] KeyboardInterrupt")

    finally:
        with shared.lock:
            shared.running = False
        car.steering = 0.0
        car.throttle = 0.0
        camera.running = False
        cv2.destroyAllWindows()
        print("[INFO] Stopped safely.")


if __name__ == "__main__":
    main()
