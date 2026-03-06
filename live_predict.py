#!/usr/bin/env python3
import json
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
from jetcam.csi_camera import CSICamera

# ---- model must match training ----
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

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load normalization (image mean/std + resize)
    norm = json.load(open("./models/control_norm.json", "r"))
    W = int(norm["input_w"])
    H = int(norm["input_h"])
    mean = np.array(norm["img_mean"], dtype=np.float32)
    std = np.array(norm["img_std"], dtype=np.float32)

    # Load model
    model = Dave2Small(dropout_p=0.6).to(device)
    model.load_state_dict(torch.load("./models/best_control_cnn.pth", map_location=device))
    model.eval()

    # Camera (same config as your other scripts)
    camera = CSICamera(width=224, height=224, capture_width=1280, capture_height=720, capture_fps=30)
    camera.running = True

    print("[INFO] Live prediction running. Press 'q' to quit.")

    try:
        while True:
            frame_rgb = camera.value
            if frame_rgb is None:
                time.sleep(0.01)
                continue

            # For display
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # Preprocess exactly like training
            img = cv2.resize(frame_bgr, (W, H))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            img = (img - mean) / std
            img = np.transpose(img, (2, 0, 1))
            x = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                pred = model(x)[0].cpu().numpy()
            steer = float(pred[0])
            thr = float(pred[1])

            # Overlay text
            cv2.putText(frame_bgr, "steer={:+.3f} thr={:+.3f}".format(steer, thr),
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)

            cv2.imshow("Live Predictions (NO driving)", frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        camera.running = False
        cv2.destroyAllWindows()
        print("[INFO] Stopped.")

if __name__ == "__main__":
    main()
