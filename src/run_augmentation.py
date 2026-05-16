# check if input folder is in yolo format


# ============================================================
# PVision Data Pipeline - Augmentation Module
# ============================================================

import os
import random
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


# ── Mosaic pipeline ──────────────────────────────────────────
# p=0.1 : applied at much lower probability than other techniques
mosaic_pipeline = A.Compose(
    [
        A.Mosaic(
            grid_yx=(2, 2),
            target_size=(512, 512),
            cell_shape=(512, 512),
            p=1.0,   # always apply inside Compose; probability controlled externally
        ),
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


# ── MixUp (custom implementation) ─────────────────────────
def apply_mixup(
    img1: np.ndarray, labels1: list, bboxes1: list,
    img2: np.ndarray, labels2: list, bboxes2: list,
    alpha: float = 0.35,
) -> tuple:
    """
    Blend two images using MixUp.
 
    alpha=0.35 : ~65% image1 / ~35% image2 on average
    Lambda sampled from Beta(0.35, 0.35) each time,
    keeping one image dominant for a natural blend
 
    Args:
        img1, img2      : numpy images (H, W, C)
        labels1/2       : YOLO class label lists
        bboxes1/2       : YOLO bbox lists
        alpha           : Beta distribution parameter
 
    Returns:
        (mixed_image, merged_labels, merged_bboxes)
    """
    lam   = np.random.beta(alpha, alpha)   # sample blend ratio each time
    # resize img2 to match img1 dimensions
    h, w  = img1.shape[:2]
    img2_resized = np.array(Image.fromarray(img2).resize((w, h)))
    mixed = np.clip(lam * img1 + (1 - lam) * img2_resized, 0, 255).astype(np.uint8)
 
    # merge bboxes and labels from both images
    merged_labels = labels1 + labels2
    merged_bboxes = bboxes1 + bboxes2
    return mixed, merged_labels, merged_bboxes


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

    # pre-load all images into pool for MixUp and Mosaic
    image_pool = []
    for f in images:
        img_p = os.path.join(src_images, f)
        txt_p = os.path.join(src_labels, f.replace(".jpg", ".txt"))
        img_arr        = np.array(Image.open(img_p))
        lbls, bxs      = load_yolo_labels(txt_p)
        image_pool.append((img_arr, lbls, bxs))
 
    original, aug_saved = 0, 0
 
    for idx, img_file in enumerate(images):
        img_path = os.path.join(src_images, img_file)
        txt_path = os.path.join(src_labels, img_file.replace(".jpg", ".txt"))
 
        img            = np.array(Image.open(img_path))
        labels, bboxes = load_yolo_labels(txt_path)
        stem           = img_file.replace(".jpg", "")
 
        # copy original image and label
        shutil.copy(img_path, dst_dir)
        shutil.copy(txt_path, dst_dir)
        original += 1
 
        if not bboxes:
            print(f"  ⚠️ No bbox found: {img_file}. Skipping augmentation.")
            continue

        # counters for MixUp and Mosaic per original image
        mixup_count  = 0
        mosaic_count = 0

         # ── 1. Standard augmentation ──────────────────────────
        for i in range(multiplier):
            try:
                augmented  = aug_pipeline(image=img, bboxes=bboxes, labels=labels)
                aug_img    = augmented["image"]
                aug_bboxes = augmented["bboxes"]
                aug_labels = augmented["labels"]
 
                if not aug_bboxes:
                    continue
 
                aug_name = f"{stem}_aug{i}.jpg"
                Image.fromarray(aug_img).save(os.path.join(dst_dir, aug_name))
                save_yolo_labels(
                    os.path.join(dst_dir, aug_name.replace(".jpg", ".txt")),
                    aug_labels, aug_bboxes,
                )
                aug_saved += 1
 
            except Exception as e:
                print(f"  ⚠️ Standard aug error {img_file} aug{i}: {e}")
 
        # ── 2. MixUp (p=0.1) ─────────────────────────────────
        if random.random() < 0.1 and len(image_pool) >= 2:
            try:
                other_idx              = random.choice([j for j in range(len(image_pool)) if j != idx])
                img2, labels2, bboxes2 = image_pool[other_idx]
 
                mixed, m_labels, m_bboxes = apply_mixup(
                    img, labels, bboxes,
                    img2, labels2, bboxes2,
                )
 
                if m_bboxes:
                    mix_name = f"{stem}_mixup{mixup_count}.jpg"
                    mixup_count += 1
                    Image.fromarray(mixed).save(os.path.join(dst_dir, mix_name))
                    save_yolo_labels(
                        os.path.join(dst_dir, mix_name.replace(".jpg", ".txt")),
                        m_labels, m_bboxes,
                    )
                    aug_saved += 1
 
            except Exception as e:
                print(f"  ⚠️ MixUp error {img_file}: {e}")
 
        # ── 3. Mosaic (p=0.1, requires 4 images) ───────────────
        if random.random() < 0.1 and len(image_pool) >= 4:
            try:
                other_indices = random.sample([j for j in range(len(image_pool)) if j != idx], 3)
                mosaic_meta   = [
                    {
                        "image"  : image_pool[j][0],
                        "bboxes" : np.array(image_pool[j][2]) if image_pool[j][2] else np.zeros((0, 4)),
                        "labels" : image_pool[j][1],
                    }
                    for j in other_indices
                ]
 
                h, w = img.shape[:2]
                result = mosaic_pipeline(
                    image           = img,
                    bboxes          = bboxes,
                    labels          = labels,
                    mosaic_metadata = mosaic_meta,
                )
 
                mos_img    = result["image"]
                mos_bboxes = result["bboxes"]
                mos_labels = result["labels"]
 
                if mos_bboxes:
                    mos_name = f"{stem}_mosaic{mosaic_count}.jpg"
                    mosaic_count += 1
                    Image.fromarray(mos_img).save(os.path.join(dst_dir, mos_name))
                    save_yolo_labels(
                        os.path.join(dst_dir, mos_name.replace(".jpg", ".txt")),
                        mos_labels, mos_bboxes,
                    )
                    aug_saved += 1
 
            except Exception as e:
                print(f"  ⚠️ Mosaic error {img_file}: {e}")
 
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