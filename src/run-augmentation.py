# check if input folder is in yolo format


# ============================================================
# PVision Data Pipeline - Augmentation Module
# ============================================================

import os
import shutil
import numpy as np
import albumentations as A
from PIL import Image


# ── Augmentation pipeline (with YOLO bbox support) ──────────
aug_pipeline = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussianBlur(blur_limit=3, p=0.3),
        A.GaussNoise(p=0.3),
        A.Rotate(limit=15, p=0.5),
        A.RandomScale(scale_limit=0.1, p=0.3),
    ],
    bbox_params=A.BboxParams(
        format="yolo",
        label_fields=["labels"],
        min_visibility=0.3,
    ),
)

# ── YOLO label I/O ──────────────────────────────────────────
def load_yolo_labels(txt_path: str):
    """Load YOLO format labels from a .txt file."""
    labels, bboxes = [], []
    if not os.path.exists(txt_path):
        return labels, bboxes
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            labels.append(int(float(parts[0])))
            bboxes.append(list(map(float, parts[1:])))
    return labels, bboxes


def save_yolo_labels(txt_path: str, labels: list, bboxes: list) -> None:
    """Save YOLO format labels to a .txt file."""
    with open(txt_path, "w") as f:
        for label, bbox in zip(labels, bboxes):
            cx, cy, w, h = bbox
            f.write(f"{int(label)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


# ── Multiplier calculation ───────────────────────────────────
def calc_multiplier(current: int, target: int = 750, min_mult: int = 3, max_mult: int = 10) -> int:
    """
    Calculate augmentation multiplier to reach target image count.

    Uses ceiling division so the result always meets or exceeds target,
    then clamps to [min_mult, max_mult].
    """
    if current == 0:
        return 0
    mult = -(-target // current)   # ceiling division
    return max(min_mult, min(mult, max_mult))


# ── Per-part augmentation ────────────────────────────────────
def augment_part(src_images: str, src_labels: str, dst_dir: str, multiplier: int):
    """
    Augment all images in a part folder, preserving YOLO bbox annotations.

    Args:
        src_images  : Folder containing .jpg images
        src_labels  : Folder containing corresponding .txt labels
        dst_dir     : Output folder for originals + augmented images
        multiplier  : How many augmented copies per original image

    Returns:
        (original_count, augmented_count)
    """
    os.makedirs(dst_dir, exist_ok=True)
    images = sorted([f for f in os.listdir(src_images) if f.endswith(".jpg")])

    original, aug_saved = 0, 0

    for img_file in images:
        img_path = os.path.join(src_images, img_file)
        txt_path = os.path.join(src_labels, img_file.replace(".jpg", ".txt"))

        img            = np.array(Image.open(img_path))
        labels, bboxes = load_yolo_labels(txt_path)

        # Copy original
        shutil.copy(img_path, dst_dir)
        shutil.copy(txt_path, dst_dir)
        original += 1

        if not bboxes:
            print(f"  ⚠️ No bbox found: {img_file}. Skipping augmentation.")
            continue

        # Generate augmented copies
        for i in range(multiplier):
            try:
                augmented  = aug_pipeline(image=img, bboxes=bboxes, labels=labels)
                aug_img    = augmented["image"]
                aug_bboxes = augmented["bboxes"]
                aug_labels = augmented["labels"]

                if not aug_bboxes:
                    continue

                stem         = img_file.replace(".jpg", "")
                aug_img_name = f"{stem}_aug{i}.jpg"

                Image.fromarray(aug_img).save(os.path.join(dst_dir, aug_img_name))
                save_yolo_labels(
                    os.path.join(dst_dir, aug_img_name.replace(".jpg", ".txt")),
                    aug_labels,
                    aug_bboxes,
                )
                aug_saved += 1

            except Exception as e:
                print(f"  ⚠️ Augmentation error {img_file} aug{i}: {e}")

    return original, aug_saved


# ── Pipeline runner ──────────────────────────────────────────
def run_augment(
    annotation_dir: str,
    aug_dir: str,
    target: int = 750,
    min_mult: int = 3,
    max_mult: int = 10,
    exclude: set = None,
) -> dict:
    """
    Run augmentation for all annotated parts.

    Args:
        annotation_dir  : Root folder; each subfolder is a part and must
                          contain images/ and labels/ subdirectories
        aug_dir         : Output root for augmented data
        target          : Target image count per part (after augmentation)
        min_mult        : Minimum augmentation multiplier
        max_mult        : Maximum augmentation multiplier
        exclude         : Set of part folder names to skip

    Returns:
        Summary dict {part_name: total_image_count}
    """
    if exclude is None:
        exclude = set()

    summary = {}

    for part in sorted(os.listdir(annotation_dir)):
        if part in exclude:
            print(f"[{part}] Excluded. Skipping.\n")
            continue

        src_images = os.path.join(annotation_dir, part, "images")
        src_labels = os.path.join(annotation_dir, part, "labels")
        dst        = os.path.join(aug_dir, part)

        if not os.path.exists(src_images):
            print(f"  ⚠️ [{part}] images/ folder not found. Skipping.")
            continue

        current  = len([f for f in os.listdir(src_images) if f.endswith(".jpg")])
        mult     = calc_multiplier(current, target, min_mult, max_mult)
        original, augmented = augment_part(src_images, src_labels, dst, mult)
        total    = original + augmented

        summary[part] = total
        print(f"[{part}] {current} imgs × {mult} → total {total}")

    return summary