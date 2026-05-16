# check if video are are in correct folder and correct format
# ============================================================
# PVision Data Pipeline - Frame Extraction Module
# ============================================================

import cv2
import os
import uuid
import re
import imagehash
import pandas as pd
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
        Dict { part_number (int): SVC Part Number (str) }
        e.g. { 1: "5PN77-67001", 2: ... }
    """
    df = pd.read_excel(excel_path, headr = 0 )
    return {
        int(row[0]): _sanitize(str(row[1]))
        for _, row in df.iterrows()
        if pd.notna(row[0]) and pd.notna(row[1])
    }

 
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
 
# ── Version management ───────────────────────────────────────
 
def get_next_version(base_extract_dir: str) -> str:
    """
    Scan existing Version_N folders inside base_extract_dir and
    return the next version name (e.g. 'Version_3' if 1 and 2 exist).
    """
    existing = [
        d for d in os.listdir(base_extract_dir)
        if os.path.isdir(os.path.join(base_extract_dir, d))
        and d.startswith("Version_")
        and d.split("_")[-1].isdigit()
    ]
    if not existing:
        return "Version_1"
    latest = max(int(d.split("_")[-1]) for d in existing)
    return f"Version_{latest + 1}"
 
 
def get_latest_version(base_extract_dir: str) -> str:
    """
    Return the latest existing Version_N folder name.
    Falls back to 'Version_1' if none exist.
    """
    existing = [
        d for d in os.listdir(base_extract_dir)
        if os.path.isdir(os.path.join(base_extract_dir, d))
        and d.startswith("Version_")
        and d.split("_")[-1].isdigit()
    ]
    if not existing:
        return "Version_1"
    latest = max(int(d.split("_")[-1]) for d in existing)
    return f"Version_{latest}"
 
 
def setup_extract_dirs(base_extract_dir: str, version_mode: str = "new") -> dict:
    """
    Create and return extraction directory structure for a given version.
 
    Output structure:
        base_extract_dir/
        └── Version_N/
            ├── all/              <- all frames merged (used for augmentation)
            └── individual/       <- per-part folders (used for annotation)
                ├── Part1/
                ├── Part2/
                └── ...
 
    Args:
        base_extract_dir : Root directory for all extraction versions
        version_mode     : "new" creates next version; "latest" reuses current
 
    Returns:
        Dict with keys: version, all_dir, individual_dir
    """
    os.makedirs(base_extract_dir, exist_ok=True)
 
    if version_mode == "new":
        version = get_next_version(base_extract_dir)
    elif version_mode == "latest":
        version = get_latest_version(base_extract_dir)
    else:
        raise ValueError(f"version_mode must be 'new' or 'latest', got '{version_mode}'")
 
    all_dir        = os.path.join(base_extract_dir, version, "all")
    individual_dir = os.path.join(base_extract_dir, version, "individual")
 
    os.makedirs(all_dir,        exist_ok=True)
    os.makedirs(individual_dir, exist_ok=True)
 
    print(f"Extract version : {version}")
    print(f"  all/          : {all_dir}")
    print(f"  individual/   : {individual_dir}\n")
 
    return {
        "version"       : version,
        "all_dir"       : all_dir,
        "individual_dir": individual_dir,
    }

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
 
    count       = 0
    saved_paths = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames is not None and len(saved_paths) >= max_frames:
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
                    fpath = os.path.join(save_dir,fname)
                    cv2.imwrite(os.path.join(save_dir, fname), frame)
                    if hash_threshold is not None:
                        seen_hashes.append(h)
                    saved_paths.append(fpath)
 
            except Exception as e:
                print(f"  ⚠️ Error at frame {count}: {e}")
 
        count += 1
 
    cap.release()
    return saved_paths
 
 
# ── Pipeline runner ──────────────────────────────────────────
 
# ── Pipeline runner ──────────────────────────────────────────
 
def run_extract(
    raw_path: str,
    all_dir: str,
    individual_dir: str,
    frame_rate: float,
    target_max: int,
    part_names: dict,
    hash_threshold: int = None,
) -> dict:
    """
    Run frame extraction for all parts in raw_path.
 
    Each frame is saved to:
      - all/            : merged folder for all parts
      - individual/PartN/ : per-part folder for annotation
 
    Args:
        raw_path        : Root folder containing part subfolders with videos
        all_dir         : Output directory for merged frames
        individual_dir  : Output directory for per-part annotation folders
        frame_rate      : Seconds per frame
        target_max      : Max frames to extract per part
        part_names      : {part_number: part_name} from load_part_names()
        hash_threshold  : Perceptual hash threshold for dedup (None = skip)
 
    Returns:
        Summary dict {"all": total_count, "Part1": count, "Part2": count, ...}
    """
    os.makedirs(all_dir,        exist_ok=True)
    os.makedirs(individual_dir, exist_ok=True)
 
    summary = {"all": 0}
 
    for part_folder in sorted(os.listdir(raw_path)):
        src = os.path.join(raw_path, part_folder)
        if not os.path.isdir(src):
            continue
 
        # Resolve part name from folder name
        part_num  = _part_folder_to_number(part_folder)
        part_name = part_names.get(part_num, f"Part{part_num}")
 
        if part_num is None:
            print(f"  ⚠️ [{part_folder}] Cannot parse part number. Using folder name.")
            part_name = _sanitize(part_folder)
        elif part_num not in part_names:
            print(f"  ⚠️ [{part_folder}] Part{part_num} not found in Excel. Using fallback name.")
 
        # individual/ subfolder for this part
        dst_individual = os.path.join(individual_dir, part_folder)
        os.makedirs(dst_individual, exist_ok=True)
 
        print(f"[{part_folder}] → {part_name}")
 
        part_saved  = 0
        seen_hashes = []
 
        for file in sorted(os.listdir(src)):
            if not file.endswith((".mp4", ".avi", ".mov")):
                continue
            if part_saved >= target_max:
                print(f"  [{part_folder}] Limit reached. Skipping {file}")
                continue
 
            remaining   = target_max - part_saved
            saved_paths = extract_frames(
                video_path      = os.path.join(src, file),
                save_dir        = all_dir,       # save to all/
                frame_rate      = frame_rate,
                part_name       = part_name,
                max_frames      = remaining,
                hash_threshold  = hash_threshold,
                seen_hashes     = seen_hashes,
            )
 
            # copy extracted frames to individual/PartN/
            for fpath in saved_paths:
                shutil.copy(fpath, dst_individual)
 
            part_saved += len(saved_paths)
            print(f"  {file}: {len(saved_paths)} frames")
 
        summary[part_folder]  = part_saved
        summary["all"]       += part_saved
        print(f"[{part_folder}] Total : {part_saved}\n")
 
    print(f"Total frames extracted : {summary['all']}")
    return summary
