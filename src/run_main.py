# ============================================================
# PVision Data Pipeline - Main Entry Point
# ============================================================
# Usage:
#   python run_main.py
#   python run_main.py --config config.yml
# ============================================================

import yaml
import os
import sys
import argparse

# ── Allow imports from src/ ──────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from run_extraction  import load_part_names, setup_extract_dirs, run_extract
from run_augmentation import run_augment


# ── Argument parser ──────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="PVision Data Pipeline")
    parser.add_argument(
        "--config",
        type    = str,
        default = os.path.join(os.path.dirname(__file__), "..", "config.yml"),
        help    = "Path to config file (default: config.yml)",
    )
    return parser.parse_args()


# ── Config loader ────────────────────────────────────────────
def load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ── Main ─────────────────────────────────────────────────────
def main():
    args   = parse_args()
    cfg    = load_config(args.config)

    PATH_CFG = cfg["path"]
    EXT_CFG  = cfg["extraction"]
    AUG_CFG  = cfg["augmentation"]
    RUN_CFG  = cfg["run"]

    print("=" * 60)
    print("  PVision Data Pipeline")
    print("=" * 60)
    print(f"Config : {args.config}\n")

    # ── Extract ──────────────────────────────────────────────
    if RUN_CFG.get("extract", False):
        print("▶ STAGE 1 : EXTRACTION")
        print("-" * 40)

        part_names = load_part_names(PATH_CFG["parts_excel"])
        print(f"Loaded {len(part_names)} part names from Excel\n")

        extract_dirs = setup_extract_dirs(
            base_extract_dir = PATH_CFG["base_extract_dir"],
            version_mode     = EXT_CFG["version_mode"],
        )

        summary = run_extract(
            raw_path        = PATH_CFG["raw_path"],
            all_dir         = extract_dirs["all_dir"],
            individual_dir  = extract_dirs["individual_dir"],
            frame_rate      = EXT_CFG["frame_rate"],
            target_max      = EXT_CFG["target_max"],
            part_names      = part_names,
            hash_threshold  = EXT_CFG["hash_threshold"],
        )

        print(f"\nExtraction complete : {summary['all']} total frames")
        print("=" * 60)

    # ── Augment ──────────────────────────────────────────────
    if RUN_CFG.get("augmentation", False):
        print("▶ STAGE 2 : AUGMENTATION")
        print("-" * 40)

        summary = run_augment(
            annotation_dir = PATH_CFG["annotation_dir"],
            aug_dir        = PATH_CFG["aug_dir"],
            target         = AUG_CFG["aug_target"],
            min_mult       = AUG_CFG["aug_min_mult"],
            max_mult       = AUG_CFG["aug_max_mult"],
            exclude        = set(AUG_CFG["exclude_parts"]),
        )

        print(f"\nAugmentation complete")
        print("=" * 60)

    print("\nPipeline finished.")


if __name__ == "__main__":
    main()