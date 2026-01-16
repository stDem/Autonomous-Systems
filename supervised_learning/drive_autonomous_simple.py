import time, json, cv2
import numpy as np
import torch
import torch.nn as nn
from jetcam.csi_camera import CSICamera
from jetracer.nvidia_racecar import NvidiaRacecar

# ---- Dave2Small (must match training) ----
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

device = "cuda" if torch.cuda.is_available() else "cpu"

# load norm
with open("./models/control_norm.json", "r") as f:
    norm = json.load(f)
W, H = int(norm["input_w"]), int(norm["input_h"])
mean = np.array(norm["img_mean"], dtype=np.float32)
std  = np.array(norm["img_std"], dtype=np.float32)
steer_bias = float(norm.get("steer_bias", 0.0))

# load model
model = Dave2Small(dropout_p=0.6).to(device).eval()
model.load_state_dict(torch.load("./models/best_control_cnn.pth", map_location=device))

car = NvidiaRacecar()
camera = CSICamera(width=224, height=224, capture_width=1280, capture_height=720, capture_fps=30)
camera.running = True

STEERING_GAIN = 1.0
THROTTLE = 0.2
STEERING_CLAMP = 0.6

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

try:
    while True:
        frame_rgb = camera.value
        if frame_rgb is None:
            time.sleep(0.01)
            continue

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        img = cv2.resize(frame_bgr, (W, H))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - mean) / std
        img = np.transpose(img, (2, 0, 1))
        x = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(x)[0].detach().cpu().numpy()

        steer = float(out[0]) + steer_bias
        steer = clamp(steer * STEERING_GAIN, -STEERING_CLAMP, STEERING_CLAMP)

        car.steering = steer
        car.throttle = THROTTLE
finally:
    car.throttle = 0.0
    car.steering = 0.0
    camera.running = False
