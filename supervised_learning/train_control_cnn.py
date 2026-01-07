#!/usr/bin/env python3
import os
import csv
import json
import math
import random
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

# -------------------------
# Reproducibility
# -------------------------
def seed_all(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# -------------------------
# Data reading
# -------------------------
def read_labels_csv(labels_path: str) -> List[Dict[str, str]]:
    """
    Supports common variants:
      - filename, steering, throttle
      - image_path, steering, throttle
    Ignores extra columns.
    """
    with open(labels_path, "r") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]

    # basic check
    if not rows:
        raise RuntimeError(f"labels.csv is empty: {labels_path}")

    # find image column
    cols = rows[0].keys()
    if "filename" in cols:
        img_col = "filename"
    elif "image_path" in cols:
        img_col = "image_path"
    else:
        raise RuntimeError(f"labels.csv must contain 'filename' or 'image_path'. Found columns: {list(cols)}")

    # required label columns
    if "steering" not in cols or "throttle" not in cols:
        raise RuntimeError(f"labels.csv must contain 'steering' and 'throttle'. Found columns: {list(cols)}")

    # normalize into standard dict keys
    out = []
    for r in rows:
        out.append({
            "img": r[img_col],
            "steering": r["steering"],
            "throttle": r["throttle"],
        })
    return out

def build_samples(data_dir: str) -> List[Tuple[str, float, float]]:
    """
    data_dir = data/run_manual
    expects:
      data_dir/images/...
      data_dir/labels.csv
    """
    labels_path = os.path.join(data_dir, "labels.csv")
    images_dir = os.path.join(data_dir, "images")

    if not os.path.isfile(labels_path):
        raise FileNotFoundError(f"Missing labels.csv: {labels_path}")
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Missing images folder: {images_dir}")

    rows = read_labels_csv(labels_path)

    samples: List[Tuple[str, float, float]] = []
    for r in rows:
        img_rel = r["img"]
        # allow absolute, or relative to images_dir, or relative to data_dir
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
                # try basename fallback (some logs store full paths)
                img_path = os.path.join(images_dir, os.path.basename(img_rel))
                if not os.path.isfile(img_path):
                    print(f"[WARN] Missing image: {img_rel} (skipping)")
                    continue

        steering = float(r["steering"])
        throttle = float(r["throttle"])
        samples.append((img_path, steering, throttle))

    if not samples:
        raise RuntimeError("No valid samples found. Check labels.csv paths.")
    return samples

# -------------------------
# Normalization stats
# -------------------------
def compute_image_mean_std(samples: List[Tuple[str, float, float]], resize_wh: Tuple[int, int],
                           max_images: int = 500) -> Tuple[List[float], List[float]]:
    """
    Computes per-channel mean/std over up to max_images frames (for speed).
    Uses RGB after resize.
    """
    w, h = resize_wh
    n = min(len(samples), max_images)
    idxs = np.linspace(0, len(samples) - 1, n).astype(int)

    acc = []
    for i in idxs:
        img_path = samples[i][0]
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (w, h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        acc.append(img.reshape(-1, 3))
    if not acc:
        # fallback
        return [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

    pixels = np.concatenate(acc, axis=0)
    mean = pixels.mean(axis=0)
    std = pixels.std(axis=0) + 1e-6
    return mean.tolist(), std.tolist()

def compute_label_mean_std(samples: List[Tuple[str, float, float]]) -> Tuple[List[float], List[float]]:
    ys = np.array([[s, t] for _, s, t in samples], dtype=np.float32)
    mean = ys.mean(axis=0)
    std = ys.std(axis=0) + 1e-6
    return mean.tolist(), std.tolist()

# -------------------------
# Dataset
# -------------------------
class DrivingDataset(Dataset):
    def __init__(
        self,
        samples: List[Tuple[str, float, float]],
        input_wh: Tuple[int, int],
        img_mean: List[float],
        img_std: List[float],
        y_mean: List[float],
        y_std: List[float],
        train: bool = True,
        aug_strength: float = 1.0,
    ):
        self.samples = samples
        self.w, self.h = input_wh
        self.img_mean = np.array(img_mean, dtype=np.float32)
        self.img_std = np.array(img_std, dtype=np.float32)
        self.y_mean = np.array(y_mean, dtype=np.float32)
        self.y_std = np.array(y_std, dtype=np.float32)
        self.train = train
        self.aug_strength = aug_strength

    def __len__(self):
        return len(self.samples)

    def _augment(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Road-safe augmentations: lighting + sensor noise only.
        Avoid geometry transforms (rotation/shift/flip) because they change the correct steering
        unless you also correct labels.
        """
        img = img_bgr

        # Brightness / contrast
        if random.random() < 0.9 * self.aug_strength:
            alpha = 1.0 + random.uniform(-0.20, 0.20) * self.aug_strength  # contrast
            beta = random.uniform(-15, 15) * self.aug_strength             # brightness
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        # Gamma jitter (simulates exposure)
        if random.random() < 0.5 * self.aug_strength:
            gamma = 1.0 + random.uniform(-0.25, 0.25) * self.aug_strength
            gamma = max(0.6, min(1.6, gamma))
            lut = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).astype("uint8")
            img = cv2.LUT(img, lut)

        # Mild blur (vibration)
        if random.random() < 0.25 * self.aug_strength:
            img = cv2.GaussianBlur(img, (3, 3), 0)

        # Mild sensor noise
        if random.random() < 0.30 * self.aug_strength:
            noise = np.random.normal(0, 4 * self.aug_strength, img.shape).astype(np.float32)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        return img


    def __getitem__(self, idx: int):
        img_path, steering, throttle = self.samples[idx]
        img = cv2.imread(img_path)
        if img is None:
            raise RuntimeError(f"Failed to read image: {img_path}")

        img = cv2.resize(img, (self.w, self.h))

        if self.train:
            img = self._augment(img)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - self.img_mean) / self.img_std  # normalize with dataset stats

        # HWC -> CHW
        img = np.transpose(img, (2, 0, 1))
        x = torch.tensor(img, dtype=torch.float32)

        y = torch.tensor([steering, throttle], dtype=torch.float32)

        return x, y

# -------------------------
# Model: DAVE-2 style + BN + Dropout
# -------------------------
class Dave2Small(nn.Module):
    def __init__(self, dropout_p: float = 0.5):
        super().__init__()
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

        # for input 66x200 -> conv output should be 64x1x18 (like DAVE-2)
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
@dataclass
class TrainConfig:
    data_dir: str
    out_dir: str
    input_w: int
    input_h: int
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    val_split: float
    patience: int
    min_delta: float
    num_workers: int
    dropout: float
    aug_strength: float
    compute_img_stats: bool

def train(cfg: TrainConfig):
    seed_all(42)

    samples = build_samples(cfg.data_dir)
    print(f"[INFO] Samples: {len(samples)}")

    # normalization stats
    y_mean, y_std = compute_label_mean_std(samples)
    if cfg.compute_img_stats:
        img_mean, img_std = compute_image_mean_std(samples, (cfg.input_w, cfg.input_h), max_images=500)
    else:
        img_mean, img_std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

    print(f"[INFO] img_mean={img_mean}, img_std={img_std}")
    print(f"[INFO] y_mean={y_mean}, y_std={y_std}")

    dataset = DrivingDataset(
        samples=samples,
        input_wh=(cfg.input_w, cfg.input_h),
        img_mean=img_mean,
        img_std=img_std,
        y_mean=y_mean,
        y_std=y_std,
        train=True,
        aug_strength=cfg.aug_strength,
    )

    total_len = len(dataset)
    val_len = max(1, int(cfg.val_split * total_len))
    train_len = total_len - val_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])

    # disable aug for val
    val_ds.dataset.train = False

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True
    )

    model = Dave2Small(dropout_p=cfg.dropout).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    os.makedirs(cfg.out_dir, exist_ok=True)
    best_model_path = os.path.join(cfg.out_dir, "best_control_cnn.pth")
    norm_path = os.path.join(cfg.out_dir, "control_norm.json")

    best_val = float("inf")
    no_improve = 0

    print(f"[INFO] Device: {DEVICE}")
    print(f"[INFO] Train/Val: {train_len}/{val_len}")

    for epoch in range(1, cfg.epochs + 1):
        # ---- train ----
        model.train()
        train_sum = 0.0
        for x, y in train_loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            train_sum += loss.item() * x.size(0)
        train_loss = train_sum / train_len

        # ---- val ----
        model.eval()
        val_sum = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(DEVICE, non_blocking=True)
                y = y.to(DEVICE, non_blocking=True)
                pred = model(x)
                loss = criterion(pred, y)
                val_sum += loss.item() * x.size(0)
        val_loss = val_sum / val_len

        print(f"Epoch {epoch}/{cfg.epochs}  train={train_loss:.6f}  val={val_loss:.6f}")

        # ---- early stopping ----
        if val_loss + cfg.min_delta < best_val:
            best_val = val_loss
            no_improve = 0
            torch.save(model.state_dict(), best_model_path)

            # save normalization used for inference
            with open(norm_path, "w") as f:
                json.dump({
                    "input_w": cfg.input_w,
                    "input_h": cfg.input_h,
                    "img_mean": img_mean,
                    "img_std": img_std,
                    "y_mean": y_mean,
                    "y_std": y_std,
                }, f, indent=2)

            print(f"  -> saved {best_model_path}")
            print(f"  -> saved {norm_path}")
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                print("[INFO] Early stopping.")
                break

    print(f"[INFO] Best val loss: {best_val:.6f}")

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

    p.add_argument("--dropout", type=float, default=0.6)        # stronger for small dataset
    p.add_argument("--aug-strength", type=float, default=1.0)   # increase to 1.2 if overfitting fast
    p.add_argument("--compute-img-stats", action="store_true", default=True)

    args = p.parse_args()
    return TrainConfig(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        input_w=args.input_w,
        input_h=args.input_h,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_split=args.val_split,
        patience=args.patience,
        min_delta=args.min_delta,
        num_workers=args.num_workers,
        dropout=args.dropout,
        aug_strength=args.aug_strength,
        compute_img_stats=args.compute_img_stats,
    )

if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
