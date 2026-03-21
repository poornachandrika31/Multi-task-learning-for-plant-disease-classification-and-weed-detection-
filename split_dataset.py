# split_dataset.py
# Place in project/src/
import os
import shutil
import random
from pathlib import Path

# ----- EDIT THIS -----
# Set this to the folder that currently contains the PlantVillage files/folders.
# Example: "../data/PlantVillage"  (relative to src/) or absolute path
SOURCE_DIR = "../data/PlantVillage"
DEST_DIR   = "../data/PlantVillage_Split"
# ---------------------

RANDOM_SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

random.seed(RANDOM_SEED)

def find_class_folders(src):
    """
    Look for class folders (folders that contain image files).
    This handles nested folder like data/PlantVillage/PlantVillage/<class>/
    """
    p = Path(src)
    if not p.exists():
        raise FileNotFoundError(f"Source path does not exist: {src}")

    # If the source folder directly has class subfolders (each class -> images)
    subdirs = [d for d in p.iterdir() if d.is_dir()]
    # Heuristic: if subdirs contain many directories that themselves contain image files, choose the deeper level.
    class_dirs = []
    for d in subdirs:
        # if this directory contains image files -> assume it's class dir
        files = [f for f in d.iterdir() if f.suffix.lower() in IMAGE_EXTS]
        if files:
            class_dirs.append(d)
    if class_dirs:
        return class_dirs

    # otherwise search one level deeper (handle extra nesting)
    for d in subdirs:
        for dd in d.iterdir():
            if dd.is_dir():
                files = [f for f in dd.iterdir() if f.suffix.lower() in IMAGE_EXTS]
                if files:
                    class_dirs.append(dd)
    # if still nothing, do a deeper walk to find directories that contain images
    if not class_dirs:
        for root, dirs, files in os.walk(src):
            for filename in files:
                if Path(filename).suffix.lower() in IMAGE_EXTS:
                    class_dir = Path(root)
                    if class_dir not in class_dirs:
                        class_dirs.append(class_dir)
    # ensure uniqueness
    class_dirs = sorted(set(class_dirs))
    return class_dirs

def make_dirs(dest, splits, class_names):
    for split in splits:
        for cls in class_names:
            path = Path(dest) / split / cls
            path.mkdir(parents=True, exist_ok=True)

def split_and_copy(class_dir, dest_root):
    files = [p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    if not files:
        print(f"[WARN] No image files found in class folder: {class_dir}")
        return []

    random.shuffle(files)
    n = len(files)
    n_train = int(TRAIN_RATIO * n)
    n_val = int(VAL_RATIO * n)
    train_files = files[:n_train]
    val_files = files[n_train:n_train+n_val]
    test_files = files[n_train+n_val:]

    cls_name = class_dir.name

    copied = {"train":0, "val":0, "test":0}
    for f in train_files:
        dst = Path(dest_root) / "train" / cls_name / f.name
        try:
            shutil.copy2(str(f), str(dst))
            copied["train"] += 1
        except Exception as e:
            print(f"[ERROR copying train] {f} -> {dst} : {e}")

    for f in val_files:
        dst = Path(dest_root) / "val" / cls_name / f.name
        try:
            shutil.copy2(str(f), str(dst))
            copied["val"] += 1
        except Exception as e:
            print(f"[ERROR copying val] {f} -> {dst} : {e}")

    for f in test_files:
        dst = Path(dest_root) / "test" / cls_name / f.name
        try:
            shutil.copy2(str(f), str(dst))
            copied["test"] += 1
        except Exception as e:
            print(f"[ERROR copying test] {f} -> {dst} : {e}")

    return copied

def main():
    class_dirs = find_class_folders(SOURCE_DIR)
    if not class_dirs:
        print("No class folders with images found. Please check SOURCE_DIR.")
        return

    # class names (folder names)
    class_names = [d.name for d in class_dirs]
    print(f"Found {len(class_names)} class folders. Example classes: {class_names[:8]}")

    splits = ["train", "val", "test"]
    make_dirs(DEST_DIR, splits, class_names)

    summary = {}
    for class_dir in class_dirs:
        print(f"Processing class folder: {class_dir}")
        copied = split_and_copy(class_dir, DEST_DIR)
        summary[class_dir.name] = copied

    print("\nCopy summary (per-class):")
    for cls, counts in summary.items():
        print(f"  {cls}: train={counts['train']}, val={counts['val']}, test={counts['test']}")
    print("\nDone. New dataset located at:", DEST_DIR)

if __name__ == "__main__":
    main()
