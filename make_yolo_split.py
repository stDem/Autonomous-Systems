import os, random, shutil

SRC_IMG = "data/object_detection/images"
SRC_LBL = "data/object_detection/labels"
OUT = "datasets/od"

VAL_RATIO = 0.2
SEED = 42
EXTS = (".jpg", ".jpeg", ".png")

random.seed(SEED)

os.makedirs(f"{OUT}/images/train", exist_ok=True)
os.makedirs(f"{OUT}/images/val", exist_ok=True)
os.makedirs(f"{OUT}/labels/train", exist_ok=True)
os.makedirs(f"{OUT}/labels/val", exist_ok=True)

imgs = [f for f in os.listdir(SRC_IMG) if f.lower().endswith(EXTS)]
imgs.sort()
random.shuffle(imgs)

n_val = int(len(imgs) * VAL_RATIO)
val_set = set(imgs[:n_val])

def copy_pair(img_name, split):
    base = os.path.splitext(img_name)[0]
    img_src = os.path.join(SRC_IMG, img_name)
    lbl_src = os.path.join(SRC_LBL, base + ".txt")
    if not os.path.isfile(lbl_src):
        print("WARN missing label for", img_name)
        return
    shutil.copy2(img_src, f"{OUT}/images/{split}/{img_name}")
    shutil.copy2(lbl_src, f"{OUT}/labels/{split}/{base}.txt")

for im in imgs:
    split = "val" if im in val_set else "train"
    copy_pair(im, split)

print("Done. train =", len(imgs) - n_val, "val =", n_val)
