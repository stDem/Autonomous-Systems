#!/usr/bin/env python3
"""
Offline click-to-steering labeler for an EXISTING JetRacer photo dataset.

Input folder structure (same as data_collection.py):
  <session>/
    images/
      frame_000001.jpg
      frame_000002.jpg
      ...
    labels.csv   (will be created / overwritten with backup)

Output:
  labels.csv with columns:
    frame_id, filename, steering, throttle, timestamp

Controls:
  - Left click: set target point (x,y) and auto-save current frame
  - D / Right arrow: next frame (requires click unless --allow-skip)
  - A / Left arrow: previous frame
  - Space: save current (if clicked) and go next
  - K: skip frame (only if --allow-skip)
  - Q / ESC: quit

Steering mapping:
  e = (x - W/2) / (W/2)  in [-1,1]
  steer = clip(k * e, -steering_limit, +steering_limit)
Use --invert if your steering sign should be flipped.
"""

from html import parser
import os
import csv
import time
import glob
import re
import argparse
from typing import Optional, Tuple

import cv2


def extract_frame_id(filename: str, fallback: int) -> int:
    m = re.search(r"(\d+)", filename)
    return int(m.group(1)) if m else fallback


def compute_steering(x: int, w: int, k: float, steering_limit: float,
                     invert: bool, deadzone: float, steer_bias: float) -> float:
    center = (w - 1) / 2.0
    e = (x - center) / center  # normalized ~[-1,1]

    # deadzone around center -> perfect straight
    if abs(e) < deadzone:
        steer = 0.0
    else:
        steer = k * e

    if invert:
        steer = -steer

    # bias correction (subtract constant bias)
    steer = steer - steer_bias

    # clamp
    if steer > steering_limit:
        steer = steering_limit
    if steer < -steering_limit:
        steer = -steering_limit

    return float(steer)



class ClickLabeler:
    def __init__(self):
        self.clicked_xy: Optional[Tuple[int, int]] = None

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicked_xy = (x, y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=str, required=True,
                        help="Path to session folder that contains an images/ subfolder")
    parser.add_argument("--images-subdir", type=str, default="images",
                        help="Images subfolder name (default: images)")
    parser.add_argument("--ext", type=str, default="jpg",
                        help="Image extension (default: jpg)")
    parser.add_argument("--throttle", type=float, default=0.18,
                        help="Constant throttle value to store in labels.csv (default: 0.18)")
    parser.add_argument("--k", type=float, default=1.0,
                        help="Steering gain from click offset (default: 1.0)")
    parser.add_argument("--steering-limit", type=float, default=0.6,
                        help="Clamp steering to [-limit, +limit] (default: 0.6)")
    parser.add_argument("--invert", action="store_true",
                        help="Flip steering sign (use if left/right is reversed)")
    parser.add_argument("--allow-skip", action="store_true",
                        help="Allow skipping frames without a click (writes no row for skipped frames)")
    parser.add_argument("--deadzone", type=float, default=0.04,
                    help="If abs(normalized error e) < deadzone => steer = 0 (default: 0.04)")
    parser.add_argument("--steer-bias", type=float, default=0.0,
                    help="Subtract this from steering label: steer = steer - bias (default: 0.0)")

    args = parser.parse_args()

    session = os.path.abspath(args.session)
    images_dir = os.path.join(session, args.images_subdir)
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Images folder not found: {images_dir}")

    pattern = os.path.join(images_dir, f"*.{args.ext}")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No images found: {pattern}")

    csv_path = os.path.join(session, "labels_new.csv")
# Do NOT touch existing labels.csv


    # We'll build rows in memory so you can go back/forward and edit.
    rows = {}  # filename -> (frame_id, filename, steering, throttle, timestamp)

    win = "Click label (ideal path) - Q to quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    clicker = ClickLabeler()
    cv2.setMouseCallback(win, clicker.on_mouse)

    idx = 0
    last_xy: Optional[Tuple[int, int]] = None

    print("[INFO] Start labeling.")
    print("[INFO] Left click to set point and auto-save current frame.")

    while 0 <= idx < len(files):
        path = files[idx]
        filename = os.path.basename(path)

        img = cv2.imread(path)
        if img is None:
            print(f"[WARN] Could not read {filename}, skipping.")
            idx += 1
            continue

        h, w = img.shape[:2]

        # If this frame already has a label, show it
        display = img.copy()
        if filename in rows:
            _, _, steer, _, _ = rows[filename]
            cv2.putText(display, f"{filename}  steer={steer:+.3f}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        else:
            cv2.putText(display, f"{filename}  (click to label)",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        # show last click as a hint
        if last_xy is not None:
            cv2.circle(display, last_xy, 6, (0, 255, 0), 2)

        cv2.imshow(win, display)

        key = cv2.waitKey(30) & 0xFF

        # Auto-save on new click
        if clicker.clicked_xy is not None:
            x, y = clicker.clicked_xy
            last_xy = (x, y)
            steer = compute_steering(
                x=x, w=w, k=args.k,
                steering_limit=args.steering_limit,
                invert=args.invert,
                deadzone=args.deadzone,
                steer_bias=args.steer_bias
            )

            frame_id = extract_frame_id(filename, fallback=idx + 1)
            ts = time.time()

            rows[filename] = (frame_id, filename, steer, float(args.throttle), ts)
            print(f"[SAVE] {filename} -> steer={steer:+.3f} throttle={args.throttle:.3f}")

            clicker.clicked_xy = None
            idx += 1
            continue

        # Navigation / commands
        if key in (ord('q'), 27):  # q or ESC
            break
        elif key in (ord('d'), 83):  # d or right arrow
            if filename in rows or args.allow_skip:
                idx += 1
            else:
                print("[INFO] Click first (or run with --allow-skip).")
        elif key in (ord('a'), 81):  # a or left arrow
            idx -= 1
        elif key == ord(' '):  # space = save (only if already clicked)
            if filename in rows:
                idx += 1
            else:
                print("[INFO] Click first to create label.")
        elif key == ord('k'):  # skip
            if args.allow_skip:
                print(f"[SKIP] {filename}")
                idx += 1
            else:
                print("[INFO] Skipping disabled. Use --allow-skip if you want this.")
        else:
            # no-op
            pass

    cv2.destroyAllWindows()

    # Write CSV in the same format as data_collection.py
    # Keep a stable order: sorted by filename
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_id", "filename", "steering", "throttle", "timestamp"])

        for fn in sorted(rows.keys()):
            frame_id, filename, steering, throttle, ts = rows[fn]
            writer.writerow([frame_id, filename, steering, throttle, ts])

    print(f"[INFO] Wrote labels: {csv_path}")
    print(f"[INFO] Labeled frames: {len(rows)} / {len(files)}")


if __name__ == "__main__":
    main()
