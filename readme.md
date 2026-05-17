# PVision Data Pipeline

A data preparation pipeline for HP printer part image datasets.  
Handles frame extraction from raw videos and image augmentation for model training.

---

## Project Structure

```
PV-DP-HPPK/
├── config.yml
├── requirements.txt
└── src/
    ├── run_main.py
    ├── run_extraction.py
    ├── run_augmentation.py
    ├── annotate.py
    ├── augment.py
    └── utils.py
```

---

## Setup

### 1. Clone the repository
```bash
git clone <repo-url>
cd PV-DP-HPPK
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare your local data folders

```
Capstone/
├── PV-DP-HPPK/                  ← this repo
├── Local/
│   ├── Capstone_Raw_Dataset/    ← place raw videos here (required)
│   │   ├── Part1/
│   │   │   ├── video1.mp4
│   │   │   └── video2.mp4
│   │   ├── Part2/
│   │   └── ...
│   ├── Run-Extracted/           ← auto-created by Stage 1
│   ├── Run-Annotated/           ← create manually, place YOLO-annotated data here
│   │   ├── Part1/
│   │   │   ├── images/
│   │   │   └── labels/
│   │   └── ...
│   └── Run-Augmented/           ← auto-created by Stage 2
└── HP_dataset/
    └── AI Parts Finder_Parts List_Jasper_rev.xlsx
```

---

## Configuration

Edit `config.yml` before running:

```yaml
extraction:
  version_mode  : "new"   # "new" = create next Version_N | "latest" = reuse current
  frame_rate    : 0.25    # seconds per frame (0.25 = 4 fps)
  target_max    : 400     # max frames to extract per part

augmentation:
  aug_target    : 750     # target image count per part after augmentation
  aug_min_mult  : 3       # minimum augmentation multiplier
  aug_max_mult  : 10      # maximum augmentation multiplier
  exclude_parts : []      # list of part folder names to skip (e.g. ["Part5"])

run:
  extract       : true    # set to true to run extraction
  augmentation  : false   # set to true to run augmentation
```

> The default relative paths assume this structure:
> ```
> Capstone/
> ├── PV-DP-HPPK/   ← repo
> ├── Local/         ← data folders
> └── HP_dataset/   ← Excel file
> ```
> If your folder structure is different, update the paths in `config.yml` to point to your data locations. Absolute paths (e.g. `C:/Users/yourname/...`) are also supported.

---

## Usage

Run from the project root (`PV-DP-HPPK/`):

```bash
python src/run_main.py
```

To specify a custom config file:
```bash
python src/run_main.py --config config.yml
```

---

## Pipeline Stages

### Stage 1 — Frame Extraction
- Reads videos from `Capstone_Raw_Dataset/PartN/`
- Maps part folders to HP part names via the Excel file
- Extracts frames at 4 fps, up to `target_max` per part
- Output → `Run-Extracted/Version_N/all/` and `individual/PartN/`

### Stage 2 - Annotation

### Stage 3 — Augmentation
- Reads annotated images from `Run-Annotated/` (YOLO format)
- Applies flip, brightness, blur, noise, rotate, MixUp, Mosaic
- Output → `Run-Augmented/`

---

## Dependencies

| Package | Purpose |
|---|---|
| `opencv-python` | Video reading and frame extraction |
| `Pillow` | Image loading and saving |
| `numpy` | Array operations for augmentation |
| `pandas` | Reading the HP parts Excel file |
| `openpyxl` | Excel `.xlsx` support for pandas |
| `pyyaml` | Parsing `config.yml` |
| `imagehash` | Perceptual hash deduplication |
| `albumentations` | Image augmentation pipeline |
