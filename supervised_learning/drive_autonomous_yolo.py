#!/usr/bin/env python3
# drive_autonomous_yolo.py  (Python 3.6 compatible)

from __future__ import print_function
import time
import json
import math
from threading import Thread, Lock

import cv2
import numpy as np

import os, sys
import torch
import torch.nn as nn

from inputs import get_gamepad, UnpluggedError
from jetcam.csi_camera import CSICamera
from jetracer.nvidia_racecar import NvidiaRacecar

# ---- Torch 1.6 compatibility: add nn.SiLU if missing ----
if not hasattr(nn, "SiLU"):
    class SiLU(nn.Module):
        def __init__(self, inplace=False):
            super(SiLU, self).__init__()
            self.inplace = inplace

        def forward(self, x):
            return x * torch.sigmoid(x)

    nn.SiLU = SiLU  # torch.nn.SiLU

    # also patch the exact path torch.load() expects
    import torch.nn.modules.activation as activation
    activation.SiLU = SiLU


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
THROTTLE_MIN = 0.0
THROTTLE_MAX = 0.155

STEER_SMOOTH = 0.5
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
    HERE = os.path.dirname(os.path.abspath(__file__))
    YOLO_DIR = os.path.join(HERE, "yolov5")

    # IMPORTANT: use absolute weights path
    WEIGHTS = os.path.join(HERE, "models", "od_best.pt")  # or OD_WEIGHTS but absolute is safer

    # Put yolov5 repo FIRST
    if YOLO_DIR not in sys.path:
        sys.path.insert(0, YOLO_DIR)

    # If Python already loaded some other "models" or "utils", drop them
    # (this is the usual reason for "models.common" failing)
    for k in list(sys.modules.keys()):
        if k == "models" or k.startswith("models."):
            sys.modules.pop(k, None)
        if k == "utils" or k.startswith("utils."):
            sys.modules.pop(k, None)

    # Now import from YOLOv5 repo
    from models.common import DetectMultiBackend
    from utils.general import check_img_size
    from utils.torch_utils import select_device

    dev = select_device('0' if device == "cuda" else 'cpu')

    model = DetectMultiBackend(
        WEIGHTS,
        device=dev,
        dnn=False,
        data=None,
        fp16=(device == "cuda")
    )
    stride = int(model.stride)
    imgsz = check_img_size(OD_IMG_SIZE, s=stride)

    model.eval()
    return model, imgsz, stride, YOLO_DIR




def _wrap_text(text, max_px, font, font_scale, thickness):
    """Wrap text into multiple lines so each line fits max_px in width."""
    words = text.split(" ")
    lines = []
    cur = ""
    for w in words:
        trial = (cur + " " + w).strip()
        (tw, th), _ = cv2.getTextSize(trial, font, font_scale, thickness)
        if tw <= max_px or cur == "":
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def draw_detections(frame_bgr, dets_xyxy, names):
    out = frame_bgr.copy()

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1

    h, w = out.shape[:2]

    for d in dets_xyxy:
        x1, y1, x2, y2, conf, cls = d
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cls = int(cls)

        # Box
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Full label text
        label = "{} {:.2f}".format(names[cls], float(conf))

        # Decide where to place label: below if near top
        pad = 4
        max_label_width = max(60, min(w - x1 - 2, 240))  # wrap if too long

        lines = _wrap_text(label, max_label_width, font, font_scale, thickness)

        # Compute label block size
        line_h = cv2.getTextSize("Ag", font, font_scale, thickness)[0][1] + 6
        block_h = line_h * len(lines) + pad
        block_w = 0
        for line in lines:
            (tw, th), _ = cv2.getTextSize(line, font, font_scale, thickness)
            block_w = max(block_w, tw)
        block_w += 2 * pad

        # Default: above the box
        y_top = y1 - block_h - 2
        if y_top < 0:
            # not enough space -> draw below
            y_top = y2 + 2
            if y_top + block_h > h:
                # clamp inside frame
                y_top = max(0, h - block_h - 2)

        x_left = x1
        if x_left + block_w > w:
            x_left = max(0, w - block_w - 2)

        # Filled background
        cv2.rectangle(out,
                      (x_left, y_top),
                      (x_left + block_w, y_top + block_h),
                      (0, 255, 0), -1)

        # Draw each line
        y_text = y_top + pad + line_h - 8
        for line in lines:
            cv2.putText(out, line,
                        (x_left + pad, y_text),
                        font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
            y_text += line_h

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
    od_model, od_imgsz, od_stride, YOLO_DIR = load_yolov5_model(device)

    # make sure YOLO_DIR is still first
    if YOLO_DIR not in sys.path:
        sys.path.insert(0, YOLO_DIR)

    from utils.general import non_max_suppression, scale_coords
    from utils.augmentations import letterbox


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

            frame_bgr = frame_rgb

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
                # 1) Letterbox resize to od_imgsz (keeps aspect ratio)
                im0 = frame_bgr  # BGR original
                lb = letterbox(im0, new_shape=od_imgsz, stride=od_stride, auto=True)[0]

                # 2) BGR->RGB, HWC->CHW, to tensor
                img_od = lb[:, :, ::-1].transpose(2, 0, 1)
                img_od = np.ascontiguousarray(img_od)

                img_od = torch.from_numpy(img_od).to(od_model.device)
                img_od = img_od.half() if getattr(od_model, "fp16", False) else img_od.float()
                img_od /= 255.0
                if img_od.ndimension() == 3:
                    img_od = img_od.unsqueeze(0)

                # 3) Inference
                with torch.no_grad():
                    pred_od = od_model(img_od)

                # 4) NMS
                pred_od = non_max_suppression(
                    pred_od,
                    conf_thres=OD_CONF_DETECT,  # show boxes threshold
                    iou_thres=0.45,             # good default
                    max_det=20
                )

                det = pred_od[0]  # detections for first image
                if det is None or len(det) == 0:
                    last_dets = np.zeros((0, 6), dtype=np.float32)
                else:
                    # 5) Scale boxes back to original image size
                    det[:, :4] = scale_coords(img_od.shape[2:], det[:, :4], im0.shape).round()
                    last_dets = det.detach().cpu().numpy().astype(np.float32)

                # Draw detections (expects Nx6: x1 y1 x2 y2 conf cls)
                last_det_vis = draw_detections(frame_bgr, last_dets, CLASSES)

                # update stability counters (per class)
                seen_classes = set()
                for d in last_dets:
                    conf = float(d[4])
                    cls = int(d[5])
                    if conf >= OD_CONF_TRIGGER:
                        seen_classes.add(cls)

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
