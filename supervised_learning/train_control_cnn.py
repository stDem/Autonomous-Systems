#!/usr/bin/env python3
# train_control_cnn.py  (Python 3.6 compatible, Jetson Nano friendly)
#
# Dataset layout:
#   data/run_manual/
#     images/
#       *.jpg / *.png ...
#     labels.csv   (must include columns: steering, throttle, and filename OR image_path)
#
# Trains a DAVE-2 style CNN to predict [steering, throttle].
# Small-dataset friendly:
#   - computes dataset image mean/std (optional, default ON)
#   - uses ONLY road-safe augmentations (brightness/contrast/gamma/blur/noise)
#   - dropout + weight decay
#   - validation split + early stopping
#
# Output:
#   models/best_control_cnn.pth
#   models/control_norm.json   (image mean/std + resize info)

from __future__ import print_function
import os
import csv
import json
import random
import argparse

import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split


# -------------------------
# Reproducibility
# -------------------------
def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


# -------------------------
# Read labels.csv
# -------------------------
def read_labels_csv(labels_path):
    # Read a sample to detect delimiter
    with open(labels_path, "r") as f:
        sample = f.read(2048)
        f.seek(0)

        # detect ; vs ,
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=";,")
            delim = dialect.delimiter
        except Exception:
            # fallback: if semicolon appears more than comma in header, use ;
            header_line = sample.splitlines()[0] if sample else ""
            delim = ";" if header_line.count(";") > header_line.count(",") else ","

        reader = csv.DictReader(f, delimiter=delim)
        rows = [r for r in reader]

    if not rows:
        raise RuntimeError("labels.csv is empty: {}".format(labels_path))

    cols = list(rows[0].keys())

    # find image column
    if "filename" in cols:
        img_col = "filename"
    elif "image_path" in cols:
        img_col = "image_path"
    else:
        raise RuntimeError("labels.csv must contain 'filename' or 'image_path'. Found: {}".format(cols))

    # required label columns
    if ("steering" not in cols) or ("throttle" not in cols):
        raise RuntimeError("labels.csv must contain 'steering' and 'throttle'. Found: {}".format(cols))

    out = []
    for r in rows:
        out.append({
            "img": r[img_col],
            "steering": r["steering"],
            "throttle": r["throttle"],
        })
    return out



def build_samples(data_dir):
    labels_path = os.path.join(data_dir, "labels.csv")
    images_dir = os.path.join(data_dir, "images")

    if not os.path.isfile(labels_path):
        raise IOError("Missing labels.csv: {}".format(labels_path))
    if not os.path.isdir(images_dir):
        raise IOError("Missing images folder: {}".format(images_dir))

    rows = read_labels_csv(labels_path)

    samples = []
    for r in rows:
        img_rel = r["img"]

        # allow absolute or relative paths
        if os.path.isabs(img_rel) and os.path.isfile(img_rel):
            img_path = img_rel
        else:
            cand1 = os.path.join(images_dir, img_rel)
            cand2 = os.path.join(data_dir, img_rel)
            if os.path.isfile(cand1):
                img_path = cand1
            elif os.path.isfile(cand2):
                img_path = cand2
            else:
                img_path = os.path.join(images_dir, os.path.basename(img_rel))
                if not os.path.isfile(img_path):
                    print("[WARN] Missing image: {} (skipping)".format(img_rel))
                    continue

        steering = float(r["steering"])
        throttle = float(r["throttle"])
        samples.append((img_path, steering, throttle))

    if not samples:
        raise RuntimeError("No valid samples found. Check labels.csv paths.")
    return samples


# -------------------------
# Compute image mean/std
# -------------------------
def compute_image_mean_std(samples, resize_wh, max_images=500):
    w, h = resize_wh
    n = min(len(samples), max_images)
    # evenly sample indices
    if n <= 1:
        idxs = [0]
    else:
        idxs = np.linspace(0, len(samples) - 1, n).astype(np.int32).tolist()

    acc = []
    for i in idxs:
        img_path = samples[i][0]
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (w, h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        acc.append(img.reshape(-1, 3))

    if not acc:
        return [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

    pixels = np.concatenate(acc, axis=0)
    mean = pixels.mean(axis=0)
    std = pixels.std(axis=0) + 1e-6
    return mean.tolist(), std.tolist()


# -------------------------
# Dataset
# -------------------------
class DrivingDataset(Dataset):
    def __init__(self, samples, input_w, input_h, img_mean, img_std, train=True, aug_strength=0.8):
        self.samples = samples
        self.w = int(input_w)
        self.h = int(input_h)
        self.train = bool(train)
        self.aug_strength = float(aug_strength)

        self.img_mean = np.array(img_mean, dtype=np.float32)
        self.img_std = np.array(img_std, dtype=np.float32)

    def __len__(self):
        return len(self.samples)

    def _augment(self, img_bgr):
        """
        Road-safe augmentations: lighting + sensor effects ONLY.
        No rotation/shift/flip (these change correct steering unless labels are corrected).
        """
        img = img_bgr

        # brightness/contrast
        if random.random() < 0.9 * self.aug_strength:
            alpha = 1.0 + random.uniform(-0.20, 0.20) * self.aug_strength  # contrast
            beta = random.uniform(-15, 15) * self.aug_strength             # brightness
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        # gamma jitter
        if random.random() < 0.5 * self.aug_strength:
            gamma = 1.0 + random.uniform(-0.25, 0.25) * self.aug_strength
            gamma = max(0.6, min(1.6, gamma))
            lut = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)], dtype=np.uint8)
            img = cv2.LUT(img, lut)

        # mild blur
        if random.random() < 0.25 * self.aug_strength:
            img = cv2.GaussianBlur(img, (3, 3), 0)

        # mild noise
        if random.random() < 0.30 * self.aug_strength:
            noise = np.random.normal(0, 4 * self.aug_strength, img.shape).astype(np.float32)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        return img

    def __getitem__(self, idx):
        img_path, steering, throttle = self.samples[idx]
        img = cv2.imread(img_path)
        if img is None:
            raise RuntimeError("Failed to read image: {}".format(img_path))

        img = cv2.resize(img, (self.w, self.h))

        if self.train:
            img = self._augment(img)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - self.img_mean) / self.img_std
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        x = torch.tensor(img, dtype=torch.float32)

        # labels are NOT z-scored (simpler for first tests)
        y = torch.tensor([steering, throttle], dtype=torch.float32)

        return x, y


# -------------------------
# Model (DAVE-2 style + BN + Dropout)
# -------------------------
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

        # For input 66x200 -> output is typically 64x1x18
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


# -------------------------
# Train
# -------------------------
def train(args):
    seed_all(42)

    samples = build_samples(args.data_dir)
    print("[INFO] Samples: {}".format(len(samples)))

    if args.compute_img_stats:
        img_mean, img_std = compute_image_mean_std(samples, (args.input_w, args.input_h), max_images=500)
    else:
        img_mean, img_std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

    print("[INFO] img_mean={}, img_std={}".format(img_mean, img_std))

    dataset = DrivingDataset(
        samples=samples,
        input_w=args.input_w,
        input_h=args.input_h,
        img_mean=img_mean,
        img_std=img_std,
        train=True,
        aug_strength=args.aug_strength,
    )

    total_len = len(dataset)
    val_len = max(1, int(args.val_split * total_len))
    train_len = total_len - val_len

    train_ds, val_ds = random_split(dataset, [train_len, val_len])
    val_ds.dataset.train = False  # no aug in val

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    model = Dave2Small(dropout_p=args.dropout).to(args.device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    os.makedirs(args.out_dir, exist_ok=True)
    best_model_path = os.path.join(args.out_dir, "best_control_cnn.pth")
    norm_path = os.path.join(args.out_dir, "control_norm.json")

    best_val = 1e18
    no_improve = 0

    print("[INFO] Device: {}".format(args.device))
    print("[INFO] Train/Val: {}/{}".format(train_len, val_len))

    for epoch in range(1, args.epochs + 1):
        # ---- train ----
        model.train()
        train_sum = 0.0
        for x, y in train_loader:
            x = x.to(args.device, non_blocking=True)
            y = y.to(args.device, non_blocking=True)

            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            train_sum += loss.item() * x.size(0)
        train_loss = train_sum / float(train_len)

        # ---- val ----
        model.eval()
        val_sum = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(args.device, non_blocking=True)
                y = y.to(args.device, non_blocking=True)
                pred = model(x)
                loss = criterion(pred, y)
                val_sum += loss.item() * x.size(0)
        val_loss = val_sum / float(val_len)

        print("Epoch {}/{}  train={:.6f}  val={:.6f}".format(epoch, args.epochs, train_loss, val_loss))

        # ---- early stopping ----
        if val_loss + args.min_delta < best_val:
            best_val = val_loss
            no_improve = 0
            torch.save(model.state_dict(), best_model_path)

            with open(norm_path, "w") as f:
                json.dump({
                    "input_w": int(args.input_w),
                    "input_h": int(args.input_h),
                    "img_mean": img_mean,
                    "img_std": img_std,
                }, f, indent=2)

            print("  -> saved {}".format(best_model_path))
            print("  -> saved {}".format(norm_path))
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print("[INFO] Early stopping.")
                break

    print("[INFO] Best val loss: {:.6f}".format(best_val))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default="./data/run_manual")
    p.add_argument("--out-dir", type=str, default="./models")
    p.add_argument("--input-w", type=int, default=200)
    p.add_argument("--input-h", type=int, default=66)

    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)

    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--min-delta", type=float, default=1e-4)

    p.add_argument("--num-workers", type=int, default=2)

    p.add_argument("--dropout", type=float, default=0.6)
    p.add_argument("--aug-strength", type=float, default=0.8)

    # mean/std computation is useful for small dataset
    p.add_argument("--compute-img-stats", action="store_true", default=True)

    args = p.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args


if __name__ == "__main__":
    args = parse_args()
    train(args)
