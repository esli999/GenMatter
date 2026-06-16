# GenMatter

Probabilistic 3D particle tracking for motion segmentation (Gestalt stimuli + TAP-Vid DAVIS + RDK Psychophysics).

> 🆕 **NEW · [StreamingVision](https://github.com/esli999/StreamingVision)** — an improved version of GenMatter built for real-time applications.

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

## Data

The paper has **three** experiments. Put all data for the experiments under **`assets/`** at the repository root (that directory is gitignored).


| Setup | Role | Root under `assets/` |
|-------|------|------------------------|
| **Gestalt** | Synthetic rendered scenes: mask propagation, depth, RAFT flow | `gestalt_stimuli/` |
| **DAVIS (TAP-Vid)** | 30 real DAVIS videos with 3D motion, DINO, SAM, etc. | `tapvid_davis_30_videos_processed/` |
| **RDK psychophysics** | Random-dot stimuli for human–model comparison | `RDK/` |

### Gestalt (`assets/gestalt_stimuli/`)

**What’s inside**

- **`gestalt_scenes/`** — One folder per scene (`scene_00000` … `scene_00019`). Each scene has `render_passes/masks/` (PNG masks), and per **texture** (`texture_00`, `texture_07`, … — seven in total) depth in `output_six_frame_depths.npz` and FlowSAM masks under `masks/flowsam_matched_reprocessed/`.
- **`raft_flows/`** — One `raft_flows_<scene>_<texture>.npz` per condition (trimmed optical flow).

**Scale:** 20 scenes × 7 textures (see `config.py`).

**Download (from AWS S3)**

```bash
uv run python scripts/download_url_list.py -P assets -c -i scripts/gestalt_stimuli_urls.txt
```

### DAVIS TAP-Vid (`assets/tapvid_davis_30_videos_processed/`)

**What’s inside** — For each of the **30** videos in `config.TAPVID_DAVIS_VIDEO_NAMES`, the pipeline reads from these subfolders:

| Subfolder | Contents |
|-----------|----------|
| `tapvid_davis_rgb_frames/` | RGB frames |
| `tapvid_davis_segmasks/` | Segmentation masks |
| `tapvid_davis_npzs/` | Npz bundles (e.g. `{video}_3d_motion.npz` after `davis-preprocess`) |
| `tapvid_davis_dino/` | DINO features |
| `tapvid_davis_SAM_frame0/` | SAM output at frame 0 |

**Download**

**Option A — AWS S3 mirror**

```bash
uv run python scripts/download_url_list.py -P assets -c -i scripts/tapvid_davis_urls.txt
```

Expect roughly **5–7 minutes** download time on a typical connection. The list begins with **large npz files**, so the progress bar (pegged to number of files) looks slow at first; but picks up later rapidly.

**Option B — Download DAVIS + preprocess** (downloads DAVIS, then builds npzs, DINO, SAM, 3D motion)

```bash
uv run python run_experiments.py download-tapvid-davis
uv run python run_experiments.py davis-preprocess
```

### RDK psychophysics (`assets/RDK/`)

**What’s inside**

- **`RDK_configs.json`** — Stimulus definitions  
- **`config_<n>/data.npz`** — Per-configuration stimulus data  
- **`reproducibility_keys.json`** (optional) — Per-stimulus seeds for JAX  

**Download**

**Option A — AWS S3**

```bash
uv run python scripts/download_url_list.py -P assets -c -i scripts/rdk_urls.txt
```

**Option B — local preprocess**

```bash
uv run python run_experiments.py rdk-preprocess
```

---

## Running the Experiments

After [data](#data) is in place:

```bash
# Gestalt
uv run python run_experiments.py gestalt
uv run python run_experiments.py gestalt-depth-ablation

# DAVIS — SAM vs GT frame-0 init (pair as you need)
uv run python run_experiments.py davis-tracking-sam
uv run python run_experiments.py davis-tracking-gt-init
uv run python run_experiments.py davis-ablation-sam
uv run python run_experiments.py davis-ablation-gt-init
uv run python run_experiments.py cotracker

# optional -- if you want to run a spped test (FPS) at different subsampling rates
uv run python run_experiments.py davis-subsampling-sam
uv run python run_experiments.py davis-subsampling-gt-init

# RDK / psychophysics
uv run python run_experiments.py psychophysics-benchmark
uv run python run_experiments.py psychophysics-rdk-ablation-fixed
uv run python run_experiments.py psychophysics-rdk-ablation-adaptive
```

**Postprocessing** (run after the corresponding experiments have written JSON under `results/`):

```bash
uv run python run_experiments.py postprocess-gestalt
uv run python run_experiments.py postprocess-davis
uv run python run_experiments.py postprocess-psychophysics
```

Aggregated tables and CSVs: `results/postprocessing/`.

---

## The `run_experiments` CLI

Every experiment and postprocessing step is invoked the same way:

```bash
uv run python run_experiments.py <command>
```

`<command>` is one of the keys in `run_experiments.py` (for a full list, run `uv run python run_experiments.py -h`). Common groups:

| Area | Commands | Default output under `results/` (see `config.py` to override) |
|------|----------|----------------------------------------------------------------|
| Gestalt | `gestalt`, `gestalt-depth-ablation` | `gestalt/`, `gestalt_depth_ablation/` |
| DAVIS preprocess | `download-tapvid-davis`, `davis-preprocess`, `davis-extract-dino`, … | Writes under `assets/tapvid_davis_30_videos_processed/` |
| DAVIS experiments | `davis-tracking-sam`, `davis-tracking-gt-init`, `davis-subsampling-sam`, `davis-subsampling-gt-init`, `davis-ablation-sam`, `davis-ablation-gt-init`, `cotracker` | `davis_tracking/`, `davis_subsampling/`, `davis_ablation/`, `cotracker_baseline/`, and `*_gt_init` variants for TAP-Vid frame-0 init |
| RDK | `rdk-preprocess`, `psychophysics-benchmark`, `psychophysics-rdk-ablation-fixed`, `psychophysics-rdk-ablation-adaptive` | `psychophysics/` (and assets under `assets/RDK/`) |
| Postprocessing | `postprocess-gestalt`, `postprocess-davis`, `postprocess-psychophysics` | `postprocessing/` |


## Configuration

Paths, defaults, and experiment constants live in **`config.py`**, including environment variables you can set to override locations (data roots, results, optional SegAnyMo paths, and similar). Edit that file or export the variables it reads to match your machine.

## License

This project is licensed under the MIT License; see [`LICENSE`](LICENSE).
