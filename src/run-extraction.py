# check if video are are in correct folder and correct format
# ============================================================
# PVision Data Pipeline - Frame Extraction Module
# ============================================================

import cv2
import os
import uuid
import re
import imagehash
import openpyxl
from PIL import Image
 
 
# ── Part name lookup ─────────────────────────────────────────
 
def load_part_names(excel_path: str) -> dict:
    """
    Load Part number → Part name mapping from HP parts list Excel file.
 
    Expected Excel structure (Sheet1):
        Row index (1-based) | SVC Part Number | Mfg. Part Number | Part Description
 
    Args:
        excel_path : Path to the HP parts list .xlsx file
 
    Returns:
        Dict { part_number (int): sanitized_part_name (str) }
        e.g. { 1: "SVC_HP-LaserJet-Fuser-220V-Kit", 2: ... }
    """
    wb  = openpyxl.load_workbook(excel_path, read_only=True, data_only=True)
    ws  = wb.active
 
    part_names = {}
    for row in ws.iter_rows(min_row=2, values_only=True):   # skip header
        row_idx, scv_part_number, _ = row[0], row[1], row[2]
        if row_idx is None or scv_part_number is None:
            continue
        part_num  = int(row_idx)
        safe_name = _sanitize(str(scv_part_number))
        part_names[part_num] = safe_name
 
    wb.close()
    return part_names
 
 
def _sanitize(name: str) -> str:
    """
    Convert a part description into a filesystem-safe string.
    Replaces spaces and special chars with hyphens; collapses runs.
    """
    name = name.strip()
    name = re.sub(r"[^\w\-]", "-", name)   # keep word chars and hyphens
    name = re.sub(r"-{2,}", "-", name)      # collapse consecutive hyphens
    name = name.strip("-")
    return name
 
 
def _part_folder_to_number(folder_name: str) -> int | None:
    """
    Extract part number from a folder name like 'Part1', 'part_3', 'Part 12'.
    Returns None if no number found.
    """
    match = re.search(r"\d+", folder_name)
    return int(match.group()) if match else None
 
 
# ── Frame extraction ─────────────────────────────────────────
 
def extract_frames(
    video_path: str,
    save_dir: str,
    frame_rate: float,
    part_name: str,
    max_frames: int = None,
    hash_threshold: int = None,
    seen_hashes: list = None,
) -> int:
    """
    Extract frames from a single video file.
 
    Saved filename format:
        {part_name}-{uuid4}.jpg
        e.g. SVC_HP-LaserJet-Fuser-220V-Kit-3f2a1b4c-....jpg
 
    Args:
        video_path      : Path to input video file
        save_dir        : Directory to save extracted frames
        frame_rate      : Seconds per frame (e.g. 0.25 = 4 fps)
        part_name       : Sanitized part name used as filename prefix
        max_frames      : Maximum number of frames to extract (None = no limit)
        hash_threshold  : Perceptual hash threshold for dedup (None = skip)
        seen_hashes     : Shared hash list across videos in same part
 
    Returns:
        Number of frames saved
    """
    if seen_hashes is None:
        seen_hashes = []
 
    cap      = cv2.VideoCapture(video_path)
    fps      = cap.get(cv2.CAP_PROP_FPS) or 30
    interval = max(int(fps * frame_rate), 1)
 
    count, saved = 0, 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames is not None and saved >= max_frames:
            break
 
        if count % interval == 0:
            try:
                if hash_threshold is not None:
                    pil_img      = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    h            = imagehash.phash(pil_img)
                    is_duplicate = any(abs(h - seen) <= hash_threshold for seen in seen_hashes)
                else:
                    is_duplicate = False
 
                if not is_duplicate:
                    fname = f"{part_name}-{uuid.uuid1()}.jpg"
                    cv2.imwrite(os.path.join(save_dir, fname), frame)
                    if hash_threshold is not None:
                        seen_hashes.append(h)
                    saved += 1
 
            except Exception as e:
                print(f"  ⚠️ Error at frame {count}: {e}")
 
        count += 1
 
    cap.release()
    return saved
 
 
# ── Pipeline runner ──────────────────────────────────────────
 
def run_extract(
    raw_path: str,
    extract_dir: str,
    frame_rate: float,
    target_max: int,
    part_names: dict,
    hash_threshold: int = None,
) -> dict:
    """
    Run frame extraction for all parts in raw_path.
 
    Folder name must contain the part number (e.g. 'Part1', 'Part_1').
    The corresponding part name is looked up from part_names dict and
    used as the filename prefix for every extracted frame.
 
    Args:
        raw_path        : Root folder containing part subfolders with videos
        extract_dir     : Output directory for extracted frames
        frame_rate      : Seconds per frame
        target_max      : Max frames to extract per part
        part_names      : {part_number: part_name} from load_part_names()
        hash_threshold  : Perceptual hash threshold for dedup (None = skip)
 
    Returns:
        Summary dict {part_folder_name: frame_count}
    """
    summary = {}
 
    for part_folder in sorted(os.listdir(raw_path)):
        src = os.path.join(raw_path, part_folder)
        if not os.path.isdir(src):
            continue
 
        dst = os.path.join(extract_dir, part_folder)
        os.makedirs(dst, exist_ok=True)
 
        # Resolve part name from folder name
        part_num  = _part_folder_to_number(part_folder)
        part_name = part_names.get(part_num, f"Part{part_num}")
 
        if part_num is None:
            print(f"  ⚠️ [{part_folder}] Cannot parse part number. Using folder name.")
            part_name = _sanitize(part_folder)
        elif part_num not in part_names:
            print(f"  ⚠️ [{part_folder}] Part{part_num} not found in Excel. Using fallback name.")
 
        print(f"[{part_folder}] → name: {part_name}")
 
        part_saved  = 0
        seen_hashes = []
 
        for file in os.listdir(src):
            if not file.endswith((".mp4", ".avi", ".mov")):
                continue
            if part_saved >= target_max:
                print(f"  [{part_folder}] Limit reached. Skipping {file}")
                continue
 
            remaining = target_max - part_saved
            n = extract_frames(
                video_path      = os.path.join(src, file),
                save_dir        = dst,
                frame_rate      = frame_rate,
                part_name       = part_name,
                max_frames      = remaining,
                hash_threshold  = hash_threshold,
                seen_hashes     = seen_hashes,
            )
            part_saved += n
            print(f"  {file}: {n} frames")
 
        summary[part_folder] = part_saved
        print(f"[{part_folder}] Total : {part_saved}\n")
 
    return summary
