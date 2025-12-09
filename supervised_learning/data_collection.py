#!/usr/bin/env python3
"""
JetRacer Pro – Gamepad teleop + data collection with VISUAL cropping
and live camera preview.

- Teleoperate JetRacer with a gamepad.
- At startup: show one camera frame, let you draw ROI with the mouse.
- Save CROPPED images + steering + throttle + timestamp.
- One folder per session for easy cleanup.
- Shows a live preview window of the (cropped) camera while running.

Controls:
    Left stick horizontal (ABS_X)  -> steering
    Right stick vertical  (ABS_RY) -> throttle (forward only, for safety)
    A / Cross (BTN_SOUTH)          -> toggle recording
    B / Circle (BTN_EAST)          -> quit

Run (from your Mac with XQuartz):
    ssh -Y jetson@<JETSON_IP>
    cd ~/anst/supervised_learning
    python3 data_collection.py --data-root ./data --session-name run1
"""

import os
import csv
import time
import math
import argparse
from datetime import datetime
from threading import Thread, Lock
from inputs import get_gamepad, UnpluggedError

import cv2

from jetcam.csi_camera import CSICamera
from jetracer.nvidia_racecar import NvidiaRacecar


# ----------------------------
# CONFIG DEFAULTS
# ----------------------------

MAX_THROTTLE = 0.15              # safety cap
STEERING_LIMIT = 0.9
DEFAULT_SAVE_INTERVAL = 0.10     # seconds between saved frames (~10 FPS)
AXIS_MAX_ABS = 32767.0           # typical gamepad axis range


# ----------------------------
# TELEOP STATE
# ----------------------------

class TeleopState:
    def __init__(self):
        self.steering = 0.0
        self.throttle = 0.0
        self.recording = False
        self.running = True
        self._lock = Lock()

    def set_steering(self, value: float):
        with self._lock:
            self.steering = value

    def set_throttle(self, value: float):
        with self._lock:
            self.throttle = value

    def toggle_recording(self):
        with self._lock:
            self.recording = not self.recording
            print(f"[INFO] Recording = {self.recording}")

    def stop(self):
        with self._lock:
            self.running = False

    def snapshot(self):
        with self._lock:
            return self.steering, self.throttle, self.recording, self.running


# ----------------------------
# GAMEPAD HELPERS
# ----------------------------

def axis_to_unit(state: int) -> float:
    """Map raw gamepad axis value to [-1, 1]."""
    return max(-1.0, min(1.0, state / AXIS_MAX_ABS))


def steering_transform(raw: float) -> float:
    """
    Map raw axis [-1, 1] to steering [-1, 1]
    with deadzone + non-linear curve for fine control near center.
    """
    x = raw  # flip sign here if your left/right feels inverted

    deadzone = 0.05
    if abs(x) < deadzone:
        return 0.0

    if x > 0:
        x = (x - deadzone) / (1.0 - deadzone)
    else:
        x = (x + deadzone) / (1.0 - deadzone)

    x = math.copysign(x * x, x)
    return max(-1.0, min(1.0, x))


def throttle_transform(raw: float) -> float:
    """
    Map raw axis [-1, 1] to throttle [0, MAX_THROTTLE] (forward only).
    If you want reverse later, we can extend this to [-MAX, MAX].
    """
    # For many controllers, pushing stick forward yields negative values
    y = -raw  # flip sign if needed

    deadzone = 0.05
    if y < deadzone:
        return 0.0
    else:
        y = (y - deadzone) / (1.0 - deadzone)

    y = max(0.0, min(1.0, y))
    return y * MAX_THROTTLE


def gamepad_loop(state: TeleopState):
    print("[INFO] Gamepad loop started")
    print("[INFO] BTN_SOUTH (A/Cross) -> toggle recording, BTN_EAST (B/Circle) -> quit")

    while True:
        _, _, _, running = state.snapshot()
        if not running:
            print("[INFO] Gamepad loop exiting")
            break

        try:
            events = get_gamepad()
        except UnpluggedError:
            print("[WARN] No gamepad found on Jetson. Is it plugged in / paired?")
            time.sleep(2.0)
            continue

        for e in events:
            # Debug if needed:
            # print(e.ev_type, e.code, e.state)

            if e.ev_type == "Absolute":

                # --- STEERING: left stick horizontal, ABS_X (0..255) ---
                if e.code == "ABS_X":
                    v = float(e.state)  # 0..255
                    # center ~127 → map to [-1, 1]
                    norm = (v - 127.0) / 127.0
                    if norm < -1.0:
                        norm = -1.0
                    if norm > 1.0:
                        norm = 1.0
                    # Optional non-linear curve & deadzone
                    state.set_steering(steering_transform(norm))
                    
                # --- STRAIGHTEN: left stick vertical up, ABS_Y (0..255) ---
                elif e.code == "ABS_Y":
                    v = float(e.state)
                    norm = (127.0 - v) / 127.0
                    # If stick is pushed clearly UP (near 0), center steering
                    if norm > 1.0:
                        norm = 1.0   # threshold, adjust if you like
                        state.set_steering(0.0)

                # --- THROTTLE: right stick vertical on ABS_RZ (0..255) ---
                elif e.code == "ABS_RZ":
                    v = float(e.state)  # 0..255 from your debug

                    # up (v≈0)   -> norm≈1 (max forward)
                    # center(127)-> norm≈0
                    # down(255)  -> norm<0 (we clip to 0, no reverse)
                    norm = (127.0 - v) / 127.0

                    if norm < 0.0:
                        norm = 0.0      # no reverse
                    if norm > 1.0:
                        norm = 1.0

                    # scale by MAX_THROTTLE, but clamp to servo's [-1, 1]
                    throttle = norm * MAX_THROTTLE
                    if throttle > 1.0:
                        throttle = 1.0
                    if throttle < -1.0:
                        throttle = -1.0

                    state.set_throttle(throttle)

            elif e.ev_type == "Key":
                if e.code == "BTN_SOUTH" and e.state == 1:
                    state.toggle_recording()
                elif e.code == "BTN_EAST" and e.state == 1:
                    print("[INFO] Exit button pressed")
                    state.stop()
                    return

# ----------------------------
# FILE / CSV HELPERS
# ----------------------------

def create_session_folder(root: str, session_name=None):
    os.makedirs(root, exist_ok=True)
    if session_name is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_name = f"session_{ts}"
    session_path = os.path.join(root, session_name)
    images_path = os.path.join(session_path, "images")
    os.makedirs(images_path, exist_ok=True)
    csv_path = os.path.join(session_path, "labels.csv")
    return session_path, images_path, csv_path


def open_csv_writer(csv_path: str):
    csv_file = open(csv_path, mode="w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["frame_id", "filename", "steering", "throttle", "timestamp"])
    csv_file.flush()
    return csv_file, writer


# ----------------------------
# VISUAL ROI (CROPPING)
# ----------------------------

def wait_for_first_frame(camera, warmup_frames=20, timeout=5.0):
    """
    Wait until we have a few valid frames from camera.value.
    This helps avoid the noisy first buffer you saw.
    """
    start = time.time()
    count = 0
    frame = camera.value
    while (time.time() - start) < timeout:
        frame = camera.value
        if frame is not None:
            count += 1
            if count >= warmup_frames:
                return frame
        time.sleep(0.05)
    return frame


def select_roi_from_camera(camera: CSICamera):
    """
    Grab a frame from camera.value, open a window and let user draw ROI.
    Returns crop margins (top, bottom, left, right) in pixels.
    """
    print("[INFO] Grabbing a frame for ROI selection...")
    frame = wait_for_first_frame(camera, timeout=5.0)
    if frame is None:
        raise RuntimeError("Camera did not produce a frame for ROI selection.")

    h, w, _ = frame.shape
    win_name = "Select ROI (drag, ENTER when done)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.imshow(win_name, frame)

    print("[INFO] A window should appear on your Mac via XQuartz.")
    print("[INFO] Use mouse to drag a rectangle over the MAP area.")
    print("[INFO] Then press ENTER or SPACE to confirm, or ESC to cancel.")

    roi = cv2.selectROI(win_name, frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(win_name)

    x, y, rw, rh = roi
    if rw == 0 or rh == 0:
        print("[WARN] No ROI selected; using full frame.")
        return 0, 0, 0, 0

    crop_top = y
    crop_left = x
    crop_bottom = h - (y + rh)
    crop_right = w - (x + rw)

    print(f"[INFO] Selected ROI: x={x}, y={y}, w={rw}, h={rh}")
    print(f"[INFO] Crop margins: top={crop_top}, bottom={crop_bottom}, "
          f"left={crop_left}, right={crop_right}")

    return crop_top, crop_bottom, crop_left, crop_right


def crop_image(img, top, bottom, left, right):
    h, w, _ = img.shape

    y1 = max(0, top)
    y2 = h - max(0, bottom)
    x1 = max(0, left)
    x2 = w - max(0, right)

    if y1 >= y2 or x1 >= x2:
        return img

    return img[y1:y2, x1:x2]


# ----------------------------
# MAIN LOOP
# ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="./data",
                        help="Root folder where sessions will be stored")
    parser.add_argument("--session-name", type=str, default=None,
                        help="Optional session name (default: timestamp-based)")
    parser.add_argument("--save-interval", type=float, default=DEFAULT_SAVE_INTERVAL,
                        help="Seconds between saved frames while recording")
    args = parser.parse_args()

    # Prepare session folders
    session_path, images_path, csv_path = create_session_folder(
        args.data_root, args.session_name
    )
    csv_file, csv_writer = open_csv_writer(csv_path)

    print(f"[INFO] Session folder: {session_path}")
    print(f"[INFO] Images -> {images_path}")
    print(f"[INFO] Labels -> {csv_path}")

    # Init car
    car = NvidiaRacecar()

    # SAFE, calibrated-ish defaults (from official JetRacer / Waveshare docs)
    car.steering_gain   = -0.65   # do NOT make this bigger in magnitude
    car.steering_offset = -0.18   # your previously working value

    car.throttle_gain = 1.0       # OK, we'll clamp separately
    car.steering = 0.0
    car.throttle = 0.0

    # Init camera
    camera = CSICamera(
        width=224,
        height=224,
        capture_width=1280,
        capture_height=720,
        capture_fps=30,
    )
    camera.running = True  # start background capture

    # ROI selection (single frame from camera.value)
    crop_top, crop_bottom, crop_left, crop_right = select_roi_from_camera(camera)

    # Start gamepad thread
    state = TeleopState()
    gp_thread = Thread(target=gamepad_loop, args=(state,), daemon=True)
    gp_thread.start()

    frame_id = 0
    last_save_time = 0.0

    print("[INFO] Main loop started.")
    print("[INFO] Drive with gamepad. Press A/Cross to toggle recording.")

    try:
        while True:
            steering, throttle, recording, running = state.snapshot()
            if not running:
                print("[INFO] Main loop exiting...")
                break
            # extra safety: clamp before sending to hardware
            steering_cmd = max(-STEERING_LIMIT, min(STEERING_LIMIT, steering))
            throttle_cmd = max(-1.0, min(1.0, throttle))

            car.steering = steering_cmd
            car.throttle = throttle_cmd

            # Get latest frame from jetcam
            frame = camera.value
            if frame is None:
                time.sleep(0.01)
                continue

            # Crop according to selected ROI
            frame_cropped = crop_image(
                frame, crop_top, crop_bottom, crop_left, crop_right
            )

            # Show live preview (cropped)
            cv2.imshow("JetRacer Camera (cropped)", frame_cropped)
            # Needed to update the window; 1ms delay is fine
            cv2.waitKey(1)

            # Save if recording
            now = time.time()
            if recording and (now - last_save_time) >= args.save_interval:
                frame_id += 1
                filename = f"frame_{frame_id:06d}.jpg"
                filepath = os.path.join(images_path, filename)

                cv2.imwrite(filepath, frame_cropped)

                csv_writer.writerow([
                    frame_id,
                    filename,
                    float(steering),
                    float(throttle),
                    now,
                ])
                csv_file.flush()

                last_save_time = now

                print(f"[SAVE] {filename} | steering={steering:.3f} "
                      f"throttle={throttle:.3f}")

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("[INFO] KeyboardInterrupt, stopping...")

    finally:
        # Stop threads and hardware safely
        state.stop()
        car.throttle = 0.0
        car.steering = 0.0
        camera.running = False
        csv_file.close()
        cv2.destroyAllWindows()
        print("[INFO] Shutdown complete")


if __name__ == "__main__":
    main()
