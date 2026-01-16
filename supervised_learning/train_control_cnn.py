#!/usr/bin/env python3
# train_control_cnn.py  (Python 3.6 compatible, Jetson Nano friendly)
#
# Dataset layout (each dir):
#   <run_dir>/
#     images/
#       *.jpg / *.png ...
#     labels.csv   (must include columns: steering, throttle, and filename OR image_path)
#
# Trains a DAVE-2 style CNN to predict [steering, throttle].
#
# Outputs (in --out-dir):
#   best_control_cnn.pth
#   control_norm.json
#   losses.csv
#   loss_curve.png            (if matplotlib available)
#   features_best.png         (feature maps from best model)
#   features_latest.png       (feature maps from last viz epoch, if enabled)

from __future__ import print_function
import os
import csv
import json
import random
import argparse
import math

import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# -------------------------
# Feature visualization helpers
# -------------------------
def _norm01(x):
    x = x.astype(np.float32)
    x = x - x.min()
    d = x.max() - x.min() + 1e-6
    return x / d

def _tile_feature_maps(feats_chw, tile_w, tile_h, max_maps=16, cols=4):
    """
    feats_chw: numpy [C,H,W]
    returns: BGR uint8 tiled image
    """
    C = feats_chw.shape[0]
    n = min(max_maps, C)

    tiles = []
    for i in range(n):
        fm = feats_chw[i]
        fm = _norm01(fm)
        fm = (fm * 255).astype(np.uint8)
        fm = cv2.resize(fm, (tile_w, tile_h))
        fm_bgr = cv2.cvtColor(fm, cv2.COLOR_GRAY2BGR)
        tiles.append(fm_bgr)

    rows = int(math.ceil(n / float(cols)))
    grid = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)

    for i, im in enumerate(tiles):
        r = i // cols
        c = i % cols
        y0, y1 = r * tile_h, (r + 1) * tile_h
        x0, x1 = c * tile_w, (c + 1) * tile_w
        grid[y0:y1, x0:x1] = im

    return grid

def _add_label(img_bgr, text):
    out = img_bgr.copy()
    cv2.putText(out, text, (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    return out

def visualize_features_multi(model, x_tensor, orig_bgr, max_maps=16, show_window=False, window_name="Learned features"):
    """
    Builds a merged image: input + conv1 + conv3 + conv5 grids.
    Returns merged BGR uint8 image.
    """
    tile_w, tile_h = 160, 52  # small tiles so everything fits

    with torch.no_grad():
        # conv is Sequential: [conv,bn,relu] repeated
        a1 = model.conv[0:3](x_tensor)       # conv1 block output
        a2 = model.conv[3:6](a1)
        a3 = model.conv[6:9](a2)            # conv3 block output
        a4 = model.conv[9:12](a3)
        a5 = model.conv[12:15](a4)          # conv5 block output

    f1 = a1[0].detach().cpu().numpy()
    f3 = a3[0].detach().cpu().numpy()
    f5 = a5[0].detach().cpu().numpy()

    orig_small = cv2.resize(orig_bgr, (tile_w, tile_h))
    orig_labeled = _add_label(orig_small, "input")

    g1 = _tile_feature_maps(f1, tile_w, tile_h, max_maps=max_maps, cols=4)
    g3 = _tile_feature_maps(f3, tile_w, tile_h, max_maps=max_maps, cols=4)
    g5 = _tile_feature_maps(f5, tile_w, tile_h, max_maps=max_maps, cols=4)

    g1 = _add_label(g1, "conv1")
    g3 = _add_label(g3, "conv3")
    g5 = _add_label(g5, "conv5")

    grid_width = g1.shape[1]
    top = np.zeros((tile_h, grid_width, 3), dtype=np.uint8)
    top[:, :tile_w] = orig_labeled

    merged = np.vstack([top, g1, g3, g5])

    if show_window:
        cv2.imshow(window_name, merged)
        cv2.waitKey(1)

    return merged


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
    with open(labels_path, "r") as f:
        sample = f.read(2048)
        f.seek(0)

        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=";,")
            delim = dialect.delimiter
        except Exception:
            header_line = sample.splitlines()[0] if sample else ""
            delim = ";" if header_line.count(";") > header_line.count(",") else ","

        reader = csv.DictReader(f, delimiter=delim)
        rows = [r for r in reader]

    if not rows:
        raise RuntimeError("labels.csv is empty: {}".format(labels_path))

    cols = list(rows[0].keys())

    if "filename" in cols:
        img_col = "filename"
    elif "image_path" in cols:
        img_col = "image_path"
    else:
        raise RuntimeError("labels.csv must contain 'filename' or 'image_path'. Found: {}".format(cols))

    if ("steering" not in cols) or ("throttle" not in cols):
        raise RuntimeError("labels.csv must contain 'steering' and 'throttle'. Found: {}".format(cols))

    out = []
    for r in rows:
        out.append({"img": r[img_col], "steering": r["steering"], "throttle": r["throttle"]})
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
        raise RuntimeError("No valid samples found in {}. Check labels.csv paths.".format(data_dir))
    return samples

def build_samples_multi(data_dirs):
    all_samples = []
    for d in data_dirs:
        s = build_samples(d)
        print("[INFO] {} samples from {}".format(len(s), d))
        all_samples.extend(s)
    if not all_samples:
        raise RuntimeError("No samples found in data dirs: {}".format(data_dirs))
    return all_samples


# -------------------------
# Compute image mean/std (train only)
# -------------------------
def compute_image_mean_std(samples, resize_wh, max_images=500):
    w, h = resize_wh
    n = min(len(samples), max_images)
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
    def __init__(self, samples, input_w, input_h, img_mean, img_std,
                 train=True, aug_strength=0.8, steer_bias=0.0):
        self.samples = samples
        self.w = int(input_w)
        self.h = int(input_h)
        self.train = bool(train)
        self.aug_strength = float(aug_strength)
        self.steer_bias = float(steer_bias)
        self.img_mean = np.array(img_mean, dtype=np.float32)
        self.img_std = np.array(img_std, dtype=np.float32)

    def __len__(self):
        return len(self.samples)

    def _augment(self, img_bgr):
        img = img_bgr

        # brightness/contrast
        if random.random() < 0.9 * self.aug_strength:
            alpha = 1.0 + random.uniform(-0.20, 0.20) * self.aug_strength
            beta = random.uniform(-15, 15) * self.aug_strength
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
        img = np.transpose(img, (2, 0, 1))
        x = torch.tensor(img, dtype=torch.float32)

        y = torch.tensor([steering - self.steer_bias, throttle], dtype=torch.float32)
        return x, y


# -------------------------
# Model (must match drive code exactly)
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
# Save losses (CSV + optional PNG plot)
# -------------------------
def save_losses(out_dir, loss_rows):
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "losses.csv")
    with open(csv_path, "w") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_loss"])
        for r in loss_rows:
            w.writerow([r["epoch"], r["train_loss"], r["val_loss"]])
    print("[INFO] Saved:", csv_path)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = [r["epoch"] for r in loss_rows]
        tr = [r["train_loss"] for r in loss_rows]
        va = [r["val_loss"] for r in loss_rows]

        plt.figure()
        plt.plot(epochs, tr, label="train")
        plt.plot(epochs, va, label="val")
        plt.xlabel("epoch")
        plt.ylabel("MSE loss")
        plt.title("Training / Validation Loss")
        plt.legend()
        plt.grid(True)

        png_path = os.path.join(out_dir, "loss_curve.png")
        plt.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close()
        print("[INFO] Saved:", png_path)
    except Exception as e:
        print("[WARN] Could not save loss plot (matplotlib missing?). CSV is saved. Error:", e)


# -------------------------
# Train
# -------------------------
def train(args):
    seed_all(42)
    os.makedirs(args.out_dir, exist_ok=True)

    # Build train/val from separate runs
    train_samples = build_samples_multi(args.train_dirs)
    val_samples   = build_samples_multi(args.val_dirs)

    print("[INFO] Train samples: {}".format(len(train_samples)))
    print("[INFO] Val samples:   {}".format(len(val_samples)))

    # Steering bias (you set to 0.0 currently; keep as-is)
    steer_bias = 0.0
    print("[INFO] Steering bias (train mean): {:.6f}".format(steer_bias))

    # Compute mean/std from TRAIN only (default ON unless disabled)
    compute_stats = (not args.no_compute_img_stats)
    if compute_stats:
        img_mean, img_std = compute_image_mean_std(train_samples, (args.input_w, args.input_h), max_images=500)
    else:
        img_mean, img_std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    print("[INFO] img_mean={}, img_std={}".format(img_mean, img_std))

    # Create datasets
    train_dataset = DrivingDataset(
        samples=train_samples,
        input_w=args.input_w,
        input_h=args.input_h,
        img_mean=img_mean,
        img_std=img_std,
        train=True,
        aug_strength=args.aug_strength,
        steer_bias=steer_bias,
    )
    val_dataset = DrivingDataset(
        samples=val_samples,
        input_w=args.input_w,
        input_h=args.input_h,
        img_mean=img_mean,
        img_std=img_std,
        train=False,
        aug_strength=0.0,
        steer_bias=steer_bias,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    model = Dave2Small(dropout_p=args.dropout).to(args.device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_model_path = os.path.join(args.out_dir, "best_control_cnn.pth")
    norm_path = os.path.join(args.out_dir, "control_norm.json")

    best_val = 1e18
    no_improve = 0

    train_len = len(train_dataset)
    val_len = len(val_dataset)

    loss_rows = []

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

        loss_rows.append({"epoch": epoch, "train_loss": float(train_loss), "val_loss": float(val_loss)})

        # ---- optional feature visualization + save latest ----
        if args.viz_features and (epoch % args.viz_every == 0):
            try:
                x_vis, _ = next(iter(val_loader))
                x0 = x_vis[0:1].to(args.device)

                x_cpu = x0[0].detach().cpu().numpy()        # [3,H,W] normalized
                x_cpu = np.transpose(x_cpu, (1, 2, 0))      # [H,W,3]
                disp = (x_cpu * np.array(img_std) + np.array(img_mean))
                disp = np.clip(disp * 255.0, 0, 255).astype(np.uint8)  # RGB
                disp_bgr = cv2.cvtColor(disp, cv2.COLOR_RGB2BGR)

                merged = visualize_features_multi(
                    model, x0, disp_bgr,
                    max_maps=16,
                    show_window=True,
                    window_name="Learned features"
                )
                cv2.imwrite(os.path.join(args.out_dir, "features_latest.png"), merged)
            except Exception as e:
                print("[WARN] Feature viz failed:", e)

        print("Epoch {}/{}  train={:.6f}  val={:.6f}".format(epoch, args.epochs, train_loss, val_loss))

        # ---- early stopping + save best ----
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
                    "steer_bias": steer_bias,
                }, f, indent=2)

            print("  -> saved {}".format(best_model_path))
            print("  -> saved {}".format(norm_path))
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print("[INFO] Early stopping.")
                break

    print("[INFO] Best val loss: {:.6f}".format(best_val))

    # Save losses CSV + optional plot
    save_losses(args.out_dir, loss_rows)

    # Save a "best features" image from the BEST model (always)
    try:
        # reload best weights
        model.load_state_dict(torch.load(best_model_path, map_location=args.device))
        model.eval()

        x_vis, _ = next(iter(val_loader))
        x0 = x_vis[0:1].to(args.device)

        x_cpu = x0[0].detach().cpu().numpy()
        x_cpu = np.transpose(x_cpu, (1, 2, 0))
        disp = (x_cpu * np.array(img_std) + np.array(img_mean))
        disp = np.clip(disp * 255.0, 0, 255).astype(np.uint8)
        disp_bgr = cv2.cvtColor(disp, cv2.COLOR_RGB2BGR)

        merged = visualize_features_multi(model, x0, disp_bgr, max_maps=16, show_window=False)
        feat_path = os.path.join(args.out_dir, "features_best.png")
        cv2.imwrite(feat_path, merged)
        print("[INFO] Saved:", feat_path)
    except Exception as e:
        print("[WARN] Could not save features_best.png:", e)


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--train-dirs", nargs="+", required=True,
                   help="One or more dataset folders for TRAINING (each has images/ and labels.csv)")
    p.add_argument("--val-dirs", nargs="+", required=True,
                   help="One or more dataset folders for VALIDATION (each has images/ and labels.csv)")

    p.add_argument("--out-dir", type=str, default="./models")
    p.add_argument("--input-w", type=int, default=200)
    p.add_argument("--input-h", type=int, default=66)

    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)

    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--min-delta", type=float, default=1e-5)

    p.add_argument("--num-workers", type=int, default=2)

    p.add_argument("--dropout", type=float, default=0.6)
    p.add_argument("--aug-strength", type=float, default=0.8)

    # Default ON (disable with flag)
    p.add_argument("--no-compute-img-stats", action="store_true",
                   help="Disable mean/std computation (default is to compute from TRAIN set)")

    p.add_argument("--viz-features", action="store_true",
                   help="Show feature maps in an OpenCV window during training")
    p.add_argument("--viz-every", type=int, default=2,
                   help="Visualize every N epochs (if --viz-features)")

    args = p.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args


if __name__ == "__main__":
    args = parse_args()
    train(args)
