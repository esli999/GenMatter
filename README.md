# GenMatter

Probabilistic 3D particle tracking for motion segmentation (Gestalt stimuli + TAP-Vid DAVIS).

## Requirements

- CUDA 12.4+ and a compatible NVIDIA driver  
- GPU with ≥ 24 GB memory (as used in the paper setup)  
- Python 3.11  

## Install

```bash
conda create -n genmatter python=3.11
conda activate genmatter
pip install -r requirements.txt
```

## Data layout (defaults)

All paths are set in **`config.py`** (overridable with env vars below).

| Location | Contents |
|----------|----------|
| **`genmatter_data/assets/from_thomas/`** | Gestalt stimuli (see *Required assets* below) |
| **`genmatter_data/assets/raft_flows/`** | Trimmed RAFT optical flow `*.npz` (5 frames, compressed) |
| **`<GENMATTER_DAVIS_DIR>/tapvid_davis_30_videos_processed/`** | DAVIS TAP-Vid bundle (RGB, motion, masks, DINO, SAM) |

**DAVIS parent directory** (where `tapvid_davis_30_videos_processed/` lives):

1. If **`GENMATTER_DAVIS_DIR`** is set → use that.
2. Else if **`<repo>/assets`** exists as a **real** directory (not a symlink) → use it.
3. Else → **`$GENMATTER_LEGACY_DATA_ROOT/assets`** (default: `/home/esli/GenMatter_neural_stimulus/assets`).

So DAVIS usually resolves to  
`…/assets/tapvid_davis_30_videos_processed/` without symlinks in the repo.

Gestalt does **not** use that tree; it only uses **`genmatter_data/assets/`**.

**Populate defaults** (`run_experiments.py populate-data`): `--source` and `--raft-source` follow the same rule (real `<repo>/assets` and `<repo>/raft_flows`, else legacy root). Notebooks or scripts that pointed at old top-level symlinks (`assets`, `raft_flows`, `final_cvpr_results/…`) should use the real paths under your **`GENMATTER_LEGACY_DATA_ROOT`** or **`GenMatter_NeurIPS`** checkout instead.

### If you already had `genmatter_data/from_thomas/` (old layout)

```bash
mkdir -p genmatter_data/assets
mv genmatter_data/from_thomas genmatter_data/assets/
mv genmatter_data/raft_flows genmatter_data/assets/
```

### Populate Gestalt from a source tree with `from_thomas/` + full RAFT files

Copies the **minimal** Gestalt file set and writes **trimmed** RAFT npz under `genmatter_data/assets/`:

```bash
conda activate genmatter
python run_experiments.py populate-data
# optional: only refresh RAFT
python run_experiments.py populate-data --raft-only
```

Source defaults: same resolution as `config.POPULATE_*` (real repo dirs, else `GENMATTER_LEGACY_DATA_ROOT`).  
Override: `python scripts/populate_genmatter_data.py --source /path --raft-source /path`

---

## Required extracted assets

### Gestalt (`genmatter_data/assets/`)

**Scenes:** `scene_00000` … `scene_00019`  
**Textures:** `texture_00`, `texture_07`, `texture_13`, `texture_16`, `texture_21`, `texture_22`, `texture_25`

Per **scene**:

- `render_passes/masks/Image0001.png` … `Image0006.png` (GT masks)

Per **scene / texture**:

- `output_six_frame_depths.npz` (6 inverse-depth frames; key `depths` or first array)
- Optional for `postprocess-gestalt` vs FlowSAM:  
  `masks/flowsam_matched_reprocessed/frame_00000_matched.png` … `frame_00004_matched.png`

**RAFT** (`genmatter_data/assets/raft_flows/`):

- One file per (scene, texture): `raft_flows_{scene}_{texture}.npz`  
- After `populate-data`: contains **5** flow frames, key **`flow`**, zlib-compressed.

### DAVIS TAP-Vid (`tapvid_davis_30_videos_processed/`)

**Videos:** exactly the 30 names in **`config.TAPVID_DAVIS_VIDEO_NAMES`**.

Under **`tapvid_davis_30_videos_processed/`** you need:

| Subdirectory | Per-video files |
|--------------|-----------------|
| `tapvid_davis_rgb_frames/{video}/` | RGB frames (`*.jpg`) |
| `tapvid_davis_npzs/` | `{video}_3d_motion.npz` |
| `tapvid_davis_segmasks/{video}/` | Segmentation masks |
| `tapvid_davis_dino/` | `{video}_dino_pca_per_pixel.npz` (from `davis-extract-dino`) |
| `tapvid_davis_SAM_frame0/` | `{video}_SAM_frame0.png` |

---

## How to run experiments

Use the same conda env you installed into (**JAX / PyTorch / genmatter live there**):

```bash
conda activate genmatter
cd /path/to/GenMatter   # repo root
```

Then:

```bash
# 0) One-time: DINO features for DAVIS (needs RGB frames + 3d motion npzs already)
python run_experiments.py davis-extract-dino

# Gestalt (needs genmatter_data/assets/ populated)
python run_experiments.py gestalt
python run_experiments.py gestalt-depth-ablation

# DAVIS HDGMM + baselines
python run_experiments.py davis-tracking
python run_experiments.py davis-subsampling
python run_experiments.py davis-ablation
python run_experiments.py cotracker
```

**Postprocessing** (after the matching runs have written under `results/`; same env):

```bash
conda activate genmatter
python run_experiments.py postprocess-gestalt          # + SegAnyMo / FlowSAM paths in config
python run_experiments.py postprocess-gestalt-ablation
python run_experiments.py postprocess-davis
```

Outputs: `results/postprocessing/*.json` and `*.csv`.

---

## Configuration (env vars)

| Env var | Role |
|---------|------|
| `GENMATTER_DATA_DIR` | Root for `assets/` subfolder (default `<repo>/genmatter_data`) |
| `GENMATTER_DAVIS_DIR` | Parent of `tapvid_davis_30_videos_processed/` (overrides auto-resolve) |
| `GENMATTER_LEGACY_DATA_ROOT` | Fallback when `<repo>/assets` or `<repo>/raft_flows` are missing or symlinks (default `…/GenMatter_neural_stimulus`) |
| `GENMATTER_RAFT_SOURCE_DIR` | Full-length RAFT npz dir for `populate-data` (overrides auto-resolve) |
| `GENMATTER_RAFT_FLOWS_PATH` | Trimmed RAFT output directory (default `genmatter_data/assets/raft_flows`) |
| `GENMATTER_RAFT_NUM_FLOW_FRAMES` | Flow frames kept in trimmed RAFT (default `5`) |
| `GENMATTER_RESULTS_DIR` | Where `results/` lives |
| `SEGANYMO_BASE_PATH` | For `postprocess-gestalt` SegAnyMo masks |

---

## Repo map

```
config.py                 # Paths + TAPVID_DAVIS_VIDEO_NAMES + GESTALT_SCENES/TEXTURES
run_experiments.py        # CLI: populate-data, gestalt, davis-*, cotracker, postprocess-*
scripts/populate_genmatter_data.py
experiments/gestalt/      # Mask-propagation Gestalt
experiments/davis/        # DINO extract, tracking, subsampling, ablation
experiments/baselines/    # CoTracker3
postprocessing/           # Metrics aggregation
genmatter/                # Core library
```

**CoTracker3** weights load from PyTorch Hub on first use.

**DAVIS depth / 3D motion:** use your Video-Depth-Anything + `preprocessing/motion_extraction_3d/davis_motion_extraction.py` pipeline to produce the `tapvid_davis_npzs` files before tracking.
