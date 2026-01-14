#!/usr/bin/env python3
import os, json
import numpy as np
import cv2
import torch
import torch.nn as nn

# same model as training
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

def read_labels_semicolon(path):
    rows = []
    with open(path, "r") as f:
        header = f.readline().strip().split(";")
        for line in f:
            parts = line.strip().split(";")
            if len(parts) != len(header):
                continue
            rows.append(dict(zip(header, parts)))
    return rows

def load_dataset(folder):
    img_dir = os.path.join(folder, "images")
    csv_path = os.path.join(folder, "labels.csv")
    rows = read_labels_semicolon(csv_path)
    samples = []
    for r in rows:
        fn = r.get("filename", "")
        if not fn:
            continue
        p = os.path.join(img_dir, os.path.basename(fn))
        if not os.path.isfile(p):
            continue
        s = float(r["steering"])
        t = float(r["throttle"])
        samples.append((p, s, t))
    return samples

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    norm = json.load(open("./models/control_norm.json", "r"))
    W, H = int(norm["input_w"]), int(norm["input_h"])
    mean = np.array(norm["img_mean"], dtype=np.float32)
    std  = np.array(norm["img_std"], dtype=np.float32)

    model = Dave2Small(dropout_p=0.6).to(device)
    model.load_state_dict(torch.load("./models/best_control_cnn.pth", map_location=device))
    model.eval()

    # evaluate on your validation run (change as needed)
    val_folder = "./data/run_manual"
    samples = load_dataset(val_folder)
    if not samples:
        print("No samples found in", val_folder)
        return

    gt_s, gt_t = [], []
    pr_s, pr_t = [], []

    for (img_path, s, t) in samples:
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (W, H))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - mean) / std
        img = np.transpose(img, (2, 0, 1))
        x = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(x)[0].cpu().numpy()
        gt_s.append(s); gt_t.append(t)
        pr_s.append(float(pred[0])); pr_t.append(float(pred[1]))

    gt_s = np.array(gt_s); gt_t = np.array(gt_t)
    pr_s = np.array(pr_s); pr_t = np.array(pr_t)

    mae_s = np.mean(np.abs(pr_s - gt_s))
    mae_t = np.mean(np.abs(pr_t - gt_t))
    print("VAL samples:", len(gt_s))
    print("MAE steering:", mae_s)
    print("MAE throttle:", mae_t)

    # straight vs turn
    straight = np.abs(gt_s) < 0.05
    turn     = np.abs(gt_s) > 0.15
    if straight.sum() > 0:
        print("\nSTRAIGHT frames:", int(straight.sum()))
        print("  mean GT steer:", float(gt_s[straight].mean()))
        print("  mean PR steer:", float(pr_s[straight].mean()))
        print("  MAE steer:", float(np.mean(np.abs(pr_s[straight] - gt_s[straight]))))
    if turn.sum() > 0:
        print("\nTURN frames:", int(turn.sum()))
        print("  mean |GT steer|:", float(np.mean(np.abs(gt_s[turn]))))
        print("  mean |PR steer|:", float(np.mean(np.abs(pr_s[turn]))))
        print("  MAE steer:", float(np.mean(np.abs(pr_s[turn] - gt_s[turn]))))

    # bias check
    print("\nPRED steer stats: mean={:+.4f}  std={:.4f}  min={:+.3f}  max={:+.3f}".format(
        float(pr_s.mean()), float(pr_s.std()), float(pr_s.min()), float(pr_s.max())
    ))

if __name__ == "__main__":
    main()
