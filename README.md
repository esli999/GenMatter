# GenMatter

Probabilistic 3D particle tracking for motion segmentation. Code release for reproducing the Gestalt segmentation and TAP-Vid DAVIS tracking experiments.

## Requirements

- CUDA 12.4+ with a compatible NVIDIA driver
- Single NVIDIA GPU with at least 24 GB memory
- Python 3.11

## Setup

```bash
conda create -n genparticles python=3.11
conda activate genparticles
pip install -r requirements.txt
```

## Repository Structure

```
GenMatter/
  config.py                          # Central path configuration
  run_experiments.py                 # CLI dispatcher

  genparticles/                      # Core library (unchanged)

  experiments/
    gestalt/
      algorithm.py                   # HDGMM algorithm for Gestalt
      run_gestalt.py                 # Mask-propagation Gestalt experiment
      run_gestalt_depth_ablation.py  # Depth ablation variant
    davis/
      dino_extractor.py              # DINO feature extraction for DAVIS
      run_davis_tracking.py          # HDGMM tracking on DAVIS
      run_davis_subsampling.py       # Subsampling evaluation
      run_davis_ablation.py          # Hyperblob clustering ablation
    baselines/
      run_cotracker.py               # CoTracker3 baseline

  postprocessing/
    postprocess_gestalt.py           # Aggregate Gestalt metrics
    postprocess_gestalt_ablation.py  # Baseline vs depth-ablation comparison
    postprocess_davis.py             # Aggregate DAVIS metrics

  preprocessing/                     # Depth/motion extraction (unchanged)
  assets/                            # Data directory (gitignored)
  raft_flows/                        # Precomputed optical flows (gitignored)
  results/                           # Experiment outputs
```

## Configuration

All paths are centralized in `config.py` using repo-relative defaults with environment variable overrides:

| Variable | Default | Override env var |
|---|---|---|
| `ASSETS_DIR` | `<repo>/assets` | `GENMATTER_ASSETS_DIR` |
| `RESULTS_DIR` | `<repo>/results` | `GENMATTER_RESULTS_DIR` |
| `SEGANYMO_BASE_PATH` | `/home/esli/SegAnyMo/gestalt_SegAnyMo_outputs` | `SEGANYMO_BASE_PATH` |

## Preprocessing

### 1. Depth and Motion Estimates

Install [Video-Depth-Anything](https://github.com/DepthAnything/Video-Depth-Anything) and use `davis_script.sh` to preprocess depth files. Then extract 3D motion:

```bash
python preprocessing/motion_extraction_3d/davis_motion_extraction.py
```

### 2. DINO Feature Extraction

```bash
python run_experiments.py davis-extract-dino
```

## Running Experiments

Use the CLI dispatcher to run any experiment:

```bash
# Gestalt segmentation
python run_experiments.py gestalt
python run_experiments.py gestalt-depth-ablation

# DAVIS tracking
python run_experiments.py davis-tracking
python run_experiments.py davis-subsampling
python run_experiments.py davis-ablation

# CoTracker baseline
python run_experiments.py cotracker
```

## Postprocessing

After experiments complete, aggregate metrics:

```bash
python run_experiments.py postprocess-gestalt
python run_experiments.py postprocess-gestalt-ablation
python run_experiments.py postprocess-davis
```

Outputs are written to `results/postprocessing/` as JSON and CSV files.

## Baseline Comparisons


**CoTracker3:** The `run_cotracker.py` experiment downloads the offline model from PyTorch Hub automatically.
