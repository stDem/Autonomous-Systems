#!/usr/bin/env python3
import time
from jetracer.nvidia_racecar import NvidiaRacecar

car = NvidiaRacecar()

# ---- SAFETY ----
car.throttle = 0.0

# ---- NEUTRAL CALIBRATION ----
# Start clean: no offset, no scaling
car.steering_gain = 1.0
car.steering_offset = 0.0

print("\n=== STEERING SIGN TEST ===")
print("Wheels OFF THE GROUND!")
print("Throttle is 0.0")
print("=========================\n")

time.sleep(2.0)

print("TEST: steering = +0.3")
car.steering = 0.3
time.sleep(2.0)

print("TEST: steering = 0.0")
car.steering = 0.0
time.sleep(1.5)

print("TEST: steering = -0.3")
car.steering = -0.3
time.sleep(2.0)

print("TEST: steering = 0.0")
car.steering = 0.0

print("\n=== TEST COMPLETE ===")
print("Observe:")
print("  • Did +0.3 turn LEFT or RIGHT?")
print("  • Did -0.3 turn the opposite way?")
print("=====================\n")
