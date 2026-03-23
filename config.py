"""
Central paths and shared experiment constants for GenMatter.

Gestalt inputs live under ``genmatter_data/assets/`` (see README for the exact
file tree). DAVIS TAP-Vid data stays under ``<GENMATTER_DAVIS_DIR>/tapvid_davis_30_videos_processed/``.

**No symlinks required:** if ``<repo>/assets`` or ``<repo>/raft_flows`` are missing
or are symlinks, defaults fall back to ``GENMATTER_LEGACY_DATA_ROOT`` (see below).
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).parent.resolve()

# Sibling checkout with full ``assets/`` and ``raft_flows/`` (override per machine).
LEGACY_DATA_ROOT = Path(
    os.environ.get("GENMATTER_LEGACY_DATA_ROOT", "/home/esli/GenMatter_neural_stimulus")
)


def _resolve_assets_parent() -> Path:
    """Directory that contains ``from_thomas/`` and (for DAVIS) ``tapvid_davis_30_videos_processed/``."""
    if os.environ.get("GENMATTER_DAVIS_DIR"):
        return Path(os.environ["GENMATTER_DAVIS_DIR"])
    local = REPO_ROOT / "assets"
    legacy = LEGACY_DATA_ROOT / "assets"
    if local.is_dir() and not local.is_symlink():
        return local.resolve()
    if legacy.is_dir():
        return legacy
    return local


def _resolve_full_raft_source_dir() -> Path:
    """Directory of *full-length* RAFT ``*.npz`` (input to ``populate_genmatter_data``)."""
    if os.environ.get("GENMATTER_RAFT_SOURCE_DIR"):
        return Path(os.environ["GENMATTER_RAFT_SOURCE_DIR"])
    local = REPO_ROOT / "raft_flows"
    legacy = LEGACY_DATA_ROOT / "raft_flows"
    if local.is_dir() and not local.is_symlink():
        return local.resolve()
    if legacy.is_dir():
        return legacy
    return local


def _resolve_data_dir() -> Path:
    if os.environ.get("GENMATTER_DATA_DIR"):
        return Path(os.environ["GENMATTER_DATA_DIR"])
    if os.environ.get("GENMATTER_ASSETS_DIR"):
        return Path(os.environ["GENMATTER_ASSETS_DIR"])
    return REPO_ROOT / "genmatter_data"


GENMATTER_DATA_DIR = _resolve_data_dir()

# Gestalt stimuli + RAFT (populate into this folder, or unzip your bundle here)
GENMATTER_LOCAL_ASSETS = GENMATTER_DATA_DIR / "assets"
GESTALT_BASE_PATH = GENMATTER_LOCAL_ASSETS / "from_thomas"

# First N flow frames match ``load_six_frame_gestalt_data`` / HDGMM pairing with 6 depth frames
GESTALT_RAFT_NUM_FLOW_FRAMES = int(os.environ.get("GENMATTER_RAFT_NUM_FLOW_FRAMES", "5"))
RAFT_FLOWS_PATH = Path(
    os.environ.get("GENMATTER_RAFT_FLOWS_PATH", GENMATTER_LOCAL_ASSETS / "raft_flows")
)

# Scenes / textures used by Gestalt experiments and postprocessing (20 × 7)
GESTALT_SCENES = tuple(f"scene_{i:05d}" for i in range(20))
GESTALT_TEXTURES = (
    "texture_00",
    "texture_07",
    "texture_13",
    "texture_16",
    "texture_21",
    "texture_22",
    "texture_25",
)

# 30 TAP-Vid DAVIS videos (DINO, tracking, CoTracker, populate docs)
TAPVID_DAVIS_VIDEO_NAMES = (
    "blackswan",
    "bike-packing",
    "bmx-trees",
    "breakdance",
    "camel",
    "car-roundabout",
    "car-shadow",
    "cows",
    "dance-twirl",
    "dog",
    "dogs-jump",
    "drift-chicane",
    "drift-straight",
    "goat",
    "gold-fish",
    "horsejump-high",
    "india",
    "judo",
    "kite-surf",
    "lab-coat",
    "libby",
    "loading",
    "mbike-trick",
    "motocross-jump",
    "paragliding-launch",
    "parkour",
    "pigs",
    "scooter-black",
    "shooting",
    "soapbox",
)

RESULTS_DIR = Path(os.environ.get("GENMATTER_RESULTS_DIR", REPO_ROOT / "results"))

# DAVIS: parent directory must contain ``tapvid_davis_30_videos_processed/``
DAVIS_PARENT_DIR = _resolve_assets_parent()
DAVIS_BASE = DAVIS_PARENT_DIR / "tapvid_davis_30_videos_processed"

# Populate script defaults (full RAFT + source ``from_thomas`` tree)
POPULATE_GESTALT_SOURCE_DIR = DAVIS_PARENT_DIR
POPULATE_FULL_RAFT_SOURCE_DIR = _resolve_full_raft_source_dir()
DAVIS_3D_MOTION_PATH = DAVIS_BASE / "tapvid_davis_npzs"
DAVIS_SEGMASKS_PATH = DAVIS_BASE / "tapvid_davis_segmasks"
DAVIS_RGB_PATH = DAVIS_BASE / "tapvid_davis_rgb_frames"
DAVIS_DINO_PATH = DAVIS_BASE / "tapvid_davis_dino"
DAVIS_SAM_FRAME0_PATH = DAVIS_BASE / "tapvid_davis_SAM_frame0"

GESTALT_OUTPUT_DIR = RESULTS_DIR / "gestalt"
GESTALT_DEPTH_ABLATION_OUTPUT_DIR = RESULTS_DIR / "gestalt_depth_ablation"
DAVIS_TRACKING_OUTPUT_DIR = RESULTS_DIR / "davis_tracking"
DAVIS_SUBSAMPLING_OUTPUT_DIR = RESULTS_DIR / "davis_subsampling"
DAVIS_ABLATION_OUTPUT_DIR = RESULTS_DIR / "davis_ablation"
COTRACKER_OUTPUT_DIR = RESULTS_DIR / "cotracker_baseline"
POSTPROCESSING_OUTPUT_DIR = RESULTS_DIR / "postprocessing"

SEGANYMO_BASE_PATH = Path(
    os.environ.get(
        "SEGANYMO_BASE_PATH",
        "/home/esli/SegAnyMo/gestalt_SegAnyMo_outputs",
    )
)
FLOWSAM_MASKS_SUBPATH = "masks/flowsam_matched_reprocessed"
