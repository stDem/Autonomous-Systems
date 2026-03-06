#!/usr/bin/env python3
from inputs import get_gamepad, UnpluggedError
import time

print("[INFO] Move each stick / trigger / button one by one.")
print("[INFO] We'll print the event code and value when it changes.")
print("[INFO] Ctrl+C to exit.\n")

last = {}

try:
    while True:
        try:
            events = get_gamepad()
        except UnpluggedError:
            print("[WARN] No gamepad found. Is it plugged into the Jetson?")
            time.sleep(2.0)
            continue

        for e in events:
            key = (e.ev_type, e.code)
            # only show when value changes
            if last.get(key) != e.state:
                last[key] = e.state
                print(f"{e.ev_type:8s} {e.code:10s} -> {e.state}")
except KeyboardInterrupt:
    print("\n[INFO] Stopped.")
