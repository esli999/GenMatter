# GenMatter

Probabilistic 3D particle tracking for motion segmentation (Gestalt stimuli + TAP-Vid DAVIS + RDK Psychophysics).

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

- **`from_thomas/`** — One folder per scene (`scene_00000` … `scene_00019`). Each scene has `render_passes/masks/` (PNG masks), and per **texture** (`texture_00`, `texture_07`, … — seven in total) depth in `output_six_frame_depths.npz` and FlowSAM masks under `masks/flowsam_matched_reprocessed/`.
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

## How to run experiments

```bash
# Gestalt (after Gestalt wget under [Data](#data))
uv run python run_experiments.py gestalt
uv run python run_experiments.py gestalt-depth-ablation

# DAVIS + baselines
# Full-grid tracking, three DAVIS ablations, subsampling tradeoff — each with SAM on or off:
uv run python run_experiments.py davis-tracking-sam
uv run python run_experiments.py davis-tracking-no-sam
# Ablation-1 (K=9): frozen hyperblob Gibbs, zero-mean blob mean/vel priors → results/davis_ablation/
uv run python run_experiments.py davis-ablation-sam
uv run python run_experiments.py davis-ablation-no-sam
# Ablation-2: same hypers as the large-Ψ_H setup (K=1, large Ψ_H, inflated k-means hyperblob covs), full hyperblob Gibbs during tracking → results/davis_ablation_2/
uv run python run_experiments.py davis-ablation-2-sam
uv run python run_experiments.py davis-ablation-2-no-sam
# Ablation-3: same init/hypers as ablation-2, but no hyperblob Gibbs during tracking (hyperblobs fixed after init) → results/davis_ablation_3/
uv run python run_experiments.py davis-ablation-3-sam
uv run python run_experiments.py davis-ablation-3-no-sam
uv run python run_experiments.py davis-subsampling-sam
uv run python run_experiments.py davis-subsampling-no-sam

uv run python run_experiments.py cotracker

# RDK (Table 1 Results from Paper)
uv run python run_experiments.py psychophysics-benchmark
uv run python run_experiments.py psychophysics-rdk-ablation-fixed
uv run python run_experiments.py psychophysics-rdk-ablation-adaptive
```

**Postprocessing** (after the matching runs have written under `results/`):

```bash
uv run python run_experiments.py postprocess-gestalt
uv run python run_experiments.py postprocess-davis
uv run python run_experiments.py postprocess-psychophysics
```

Outputs: `results/postprocessing/`.

---

## Configuration

Paths, defaults, and experiment constants live in **`config.py`**, including environment variables you can set to override locations (data roots, results, optional SegAnyMo paths, and similar). Edit that file or export the variables it reads to match your machine.
