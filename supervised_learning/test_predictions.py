#!/usr/bin/env python3
import os, json, random
import cv2
import numpy as np
import torch
import torch.nn as nn

# ----- model (must match training) -----
class Dave2Small(nn.Module):
    def __init__(self, dropout_p=0.0):
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
            nn.Linear(100, 50),
            nn.ReLU(inplace=True),
            nn.Linear(50, 10),
            nn.ReLU(inplace=True),
            nn.Linear(10, 2),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def parse_labels_semicolon(csv_path):
    # simple semicolon parser
    rows = []
    with open(csv_path, "r") as f:
        header = f.readline().strip().split(";")
        for line in f:
            parts = line.strip().split(";")
            if len(parts) != len(header):
                continue
            r = dict(zip(header, parts))
            rows.append(r)
    return rows

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    norm = json.load(open("./models/control_norm.json", "r"))
    W = int(norm["input_w"])
    H = int(norm["input_h"])
    mean = np.array(norm["img_mean"], dtype=np.float32)
    std = np.array(norm["img_std"], dtype=np.float32)

    model = Dave2Small().to(device)
    model.load_state_dict(torch.load("./models/best_control_cnn.pth", map_location=device))
    model.eval()

    data_dir = "./data/run_map_curves_main"
    img_dir = os.path.join(data_dir, "images")
    rows = parse_labels_semicolon(os.path.join(data_dir, "labels.csv"))

    sample_rows = random.sample(rows, min(50, len(rows)))

    for r in sample_rows:
        fn = r["filename"]
        img_path = os.path.join(img_dir, fn)
        img = cv2.imread(img_path)
        if img is None:
            continue

        gt_s = float(r["steering"])
        gt_t = float(r["throttle"])

        img = cv2.resize(img, (W, H))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - mean) / std
        img = np.transpose(img, (2, 0, 1))
        x = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(x)[0].cpu().numpy()
        ps, pt = float(pred[0]), float(pred[1])

        print("GT: steer={:+.3f} thr={:+.3f} | PRED: steer={:+.3f} thr={:+.3f} | {}".format(
            gt_s, gt_t, ps, pt, fn
        ))

if __name__ == "__main__":
    main()
