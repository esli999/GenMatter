# GenMatter

Probabilistic 3D particle tracking for motion segmentation (Gestalt stimuli + TAP-Vid DAVIS).

## Requirements

- CUDA 12.4+ and a compatible NVIDIA driver  
- GPU with <= 24 GB memory (as used in the paper setup)  
- Python 3.11 (pinned via `.python-version`; uv will download it if missing)  
- [uv](https://docs.astral.sh/uv/getting-started/installation/) for the virtualenv and locked dependencies  

## Install

From the repository root:

```bash
uv sync
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
3. Else if **`GENMATTER_LEGACY_DATA_ROOT`** is set and **`$GENMATTER_LEGACY_DATA_ROOT/assets`** exists → use that.
4. Else → **`<repo>/assets`** (even if missing or a symlink).

So DAVIS usually resolves to  
`…/assets/tapvid_davis_30_videos_processed/` without symlinks in the repo.

Gestalt does **not** use that tree; it only uses **`genmatter_data/assets/`**.

**Populate defaults** (`run_experiments.py populate-data`): `--source` and `--raft-source` follow the same rule (real `<repo>/assets` and `<repo>/raft_flows`, else **`GENMATTER_LEGACY_DATA_ROOT`** if set). Notebooks or scripts that pointed at old top-level symlinks (`assets`, `raft_flows`, `final_cvpr_results/…`) should use the real paths under your **`GENMATTER_LEGACY_DATA_ROOT`** or **`GenMatter_NeurIPS`** checkout instead.

### If you already had `genmatter_data/from_thomas/` (old layout)

```bash
mkdir -p genmatter_data/assets
mv genmatter_data/from_thomas genmatter_data/assets/
mv genmatter_data/raft_flows genmatter_data/assets/
```

### Populate Gestalt from a source tree with `from_thomas/` + full RAFT files

Copies the **minimal** Gestalt file set and writes **trimmed** RAFT npz under `genmatter_data/assets/`:

```bash
uv run python run_experiments.py populate-data
# optional: only refresh RAFT
uv run python run_experiments.py populate-data --raft-only
```

Source defaults: same resolution as `config.POPULATE_*` (real repo dirs, else `GENMATTER_LEGACY_DATA_ROOT` when set).  
Override: `uv run python scripts/populate_genmatter_data.py --source /path --raft-source /path`

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

**Videos:** the 30 names in **`config.TAPVID_DAVIS_VIDEO_NAMES`**. Parent directory is **`GENMATTER_DAVIS_DIR`** when set; otherwise **`config`** resolves **`assets/`** (see `config.py`).

Under **`tapvid_davis_30_videos_processed/`** you need:

| Subdirectory | Per-video files |
|--------------|-----------------|
| `tapvid_davis_rgb_frames/{video}/` | RGB frames (`*.jpg`) |
| `tapvid_davis_npzs/` | `{video}_3d_motion.npz` |
| `tapvid_davis_segmasks/{video}/` | Segmentation masks |
| `tapvid_davis_dino/` | `{video}_dino_pca_per_pixel.npz` |
| `tapvid_davis_SAM_frame0/` | `{video}_SAM_frame0.png` |

We provide 2 ways to download the data. The first is to download all pre-processed data directly. The second is to download the raw TAPVID-DAVIS dataset online and preprocess the data yourself. Option A will take a few minutes, but is faster than option B.

**Option A — S3 mirror (directly downloads preprocessed data)**

`scripts/tapvid_davis_urls.txt` lists objects under **`tapvid_davis_30_videos_processed/`** (RGB, segmasks, npzs, DINO, SAM). From the repository root:

```bash
wget -x -nH -c -P assets -i scripts/tapvid_davis_urls.txt
```

**Option B — downloads tapvid + local preprocess**

1. Downloads DAVIS 2017 full-resolution trainval and copies the TAP-Vid subset into **`tapvid_davis_rgb_frames/`** and **`tapvid_davis_segmasks/`**.
2. Runs **`davis-extract-3d-motion`** (VDA + RAFT 3D motion npzs), then **`davis-extract-dino`**, then **`davis-extract-sam-frame0`**, in that order. The first 3D-motion run clones Video-Depth-Anything and the default **`vitl`** checkpoint (needs **git** + network); **`GENMATTER_VIDEO_DEPTH_ANYTHING_PATH`** overrides the clone. Other encoders (`--encoder vits` / `vitb`) fetch weights on first use. For flags on a single step, run that command alone instead of **`davis-preprocess`**.

```bash
uv run python run_experiments.py download-tapvid-davis
uv run python run_experiments.py davis-preprocess
```

### RDK psychophysics (`assets/RDK/`)

**Frozen reference bundle:** **`assets/RDK_groundtruth/`** holds the previous **`assets/RDK`** JSON files (and optional full trees) for comparison. The active benchmark directory remains **`GENMATTER_RDK_DIR`** (default **`<repo>/assets/RDK`**).

There are 2 ways to get the data: download preprocessed files from S3, or build them locally with the preprocess command. Option A only takes a few seconds and is much faster.

**Option A — S3 mirror (directly downloads preprocessed data)**

`scripts/rdk_urls.txt` lists objects under **`RDK/`**. From the repository root:

```bash
wget -x -nH -c -P assets -i scripts/rdk_urls.txt
```

**Option B — local preprocess** (physics MP4s → RAFT `data.npz` → canonical JSON → GIFs)

```bash
uv run python run_experiments.py rdk-preprocess
```

| Path | Contents |
|------|----------|
| **`RDK_configs.json`** | Stimulus IDs, frames, probe, points, `ground_truth` |
| **`config_<n>/data.npz`** | `points_data` (and any fields your preprocessing writes) |
| **`reproducibility_keys.json`** (optional) | Map `stim_id` → integer used as `jkey(...)` root per stimulus; if present, the benchmark uses it by default (legacy name **`keys.json`** is still accepted). |

Benchmark runs and the correlation plot are listed under **How to run experiments** (same `uv run` pattern as Gestalt and DAVIS). Optional flags for the benchmark: `--outer-trials`, `--num-runs`, `--output-dir`, `--no-repro-keys`, `--repro-keys-path`, `--require-repro-keys` after `--`. The correlation postprocess scans **`results/psychophysics/*_benchmark.json`** (full run and ablations) and writes one **`results/postprocessing/psychophysics_correlation_<stem>.png`** per file. Use `--input` after `--` for a single JSON; optional **`--psychophysics-dir`** to scan a different folder.

---

## How to run experiments

DAVIS TAP-Vid inputs: use **Option A** or **Option B** under [DAVIS TAP-Vid](#davis-tap-vid-tapvid_davis_30_videos_processed) above.

```bash
# Gestalt (needs genmatter_data/assets/ populated)
uv run python run_experiments.py gestalt
uv run python run_experiments.py gestalt-depth-ablation

# DAVIS HDGMM + baselines
uv run python run_experiments.py davis-tracking
uv run python run_experiments.py davis-subsampling
uv run python run_experiments.py davis-ablation
uv run python run_experiments.py cotracker

# RDK psychophysics (needs assets/RDK/; see data layout above)
uv run python run_experiments.py psychophysics-benchmark
uv run python run_experiments.py psychophysics-rdk-ablation-fixed
uv run python run_experiments.py psychophysics-rdk-ablation-adaptive
```

**Postprocessing** (after the matching runs have written under `results/`):

```bash
uv run python run_experiments.py postprocess-gestalt          # + SegAnyMo / FlowSAM paths in config
uv run python run_experiments.py postprocess-gestalt-ablation
uv run python run_experiments.py postprocess-davis
uv run python run_experiments.py postprocess-psychophysics-correlation
```

Outputs: `results/postprocessing/*.json`, `*.csv`, and psychophysics `*.png` when using the correlation command.

---

## Configuration (env vars)

| Env var | Role |
|---------|------|
| `GENMATTER_DATA_DIR` | Root for `assets/` subfolder (default `<repo>/genmatter_data`) |
| `GENMATTER_DAVIS_DIR` | Parent of `tapvid_davis_30_videos_processed/` (overrides auto-resolve) |
| `GENMATTER_DEEPLEARNING_WEIGHTS_DIR` | Ultralytics SAM weights directory (default `<repo>/assets/deeplearning_weights`) |
| `GENMATTER_VIDEO_DEPTH_ANYTHING_PATH` | Path to Video-Depth-Anything clone (default `<repo>/external/video-depth-anything`) |
| `GENMATTER_LEGACY_DATA_ROOT` | Optional fallback when `<repo>/assets` or `<repo>/raft_flows` are missing or symlinks (no default; set explicitly if needed) |
| `GENMATTER_RAFT_SOURCE_DIR` | Full-length RAFT npz dir for `populate-data` (overrides auto-resolve) |
| `GENMATTER_RAFT_FLOWS_PATH` | Trimmed RAFT output directory (default `genmatter_data/assets/raft_flows`) |
| `GENMATTER_RAFT_NUM_FLOW_FRAMES` | Flow frames kept in trimmed RAFT (default `5`) |
| `GENMATTER_RESULTS_DIR` | Where `results/` lives |
| `GENMATTER_RDK_DIR` | Root for RDK psychophysics (`RDK_configs.json`, `config_*/data.npz`; default `<repo>/assets/RDK`) |
| `GENMATTER_RDK_REPRO_KEYS_PATH` | Explicit path to `reproducibility_keys.json` (overrides default `<GENMATTER_RDK_DIR>/reproducibility_keys.json`) |
| `SEGANYMO_BASE_PATH` | For `postprocess-gestalt` SegAnyMo masks (optional; unset skips SegAnyMo paths) |

---

## Repo map

```
pyproject.toml / uv.lock  # Dependencies (uv); CUDA PyTorch + JAX via configured indexes
.python-version           # 3.11 (used by uv)
config.py                 # Paths + TAPVID_DAVIS_VIDEO_NAMES + GESTALT_SCENES/TEXTURES
run_experiments.py        # CLI: populate-data, download-tapvid-davis, davis-preprocess, gestalt, davis-*, …
scripts/download_tapvid_davis.sh  # ETH DAVIS zip → RGB + seg (also: run_experiments download-tapvid-davis)
scripts/davis_preprocess.py       # 3D motion + DINO + SAM (also: run_experiments davis-preprocess)
scripts/populate_genmatter_data.py
experiments/gestalt/      # Mask-propagation Gestalt
experiments/davis/        # DINO + SAM frame0 extract, tracking, subsampling, ablation
preprocessing/motion_extraction_3d/  # VDA+RAFT; ``ensure_vda_assets`` clones + checkpoints on first 3D-motion run
experiments/psychophysics/ # RDK human–model benchmark (full + ablations)
experiments/baselines/    # CoTracker3
postprocessing/           # Metrics aggregation
genmatter/                # Core library
```

**CoTracker3** weights load from PyTorch Hub on first use.

**SAM frame 0 (`davis-extract-sam-frame0`):** The default checkpoint `sam2.1_l.pt` is stored under **`assets/deeplearning_weights/`** (created on first run). Pass `--model /path/to.pt` to use an existing file, or `--weights-dir` to override that directory.

**DAVIS 3D motion:** **`davis-preprocess`** (or **`davis-extract-3d-motion`** alone) sets up VDA on first run and writes **`tapvid_davis_npzs/{video}_3d_motion.npz`**.
