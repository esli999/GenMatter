"""
Central paths and shared experiment constants for GenMatter.

With default ``GENMATTER_DATA_DIR`` (repository root), inputs live under ``<repo>/assets/``:
``gestalt_stimuli/{from_thomas,raft_flows}/``, ``tapvid_davis_30_videos_processed/``, ``RDK/``.
See README for layout. Optional per-stimulus JAX seeds: ``GENMATTER_RDK_REPRO_KEYS_PATH`` or
``reproducibility_keys.json`` under the RDK directory.

If ``<repo>/assets`` or ``<repo>/raft_flows`` are missing or are symlinks, you can set
``GENMATTER_LEGACY_DATA_ROOT`` to a sibling checkout that contains those trees.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).parent.resolve()

# SAM / Ultralytics checkpoints (see ``experiments/davis/sam_frame0_extractor.py``)
DEEPLEARNING_WEIGHTS_DIR = (
    Path(os.environ["GENMATTER_DEEPLEARNING_WEIGHTS_DIR"]).expanduser().resolve()
    if os.environ.get("GENMATTER_DEEPLEARNING_WEIGHTS_DIR")
    else (REPO_ROOT / "assets" / "deeplearning_weights").resolve()
)


def _legacy_data_root() -> Path | None:
    """Optional sibling checkout with ``assets/`` and ``raft_flows/`` (``GENMATTER_LEGACY_DATA_ROOT``)."""
    p = os.environ.get("GENMATTER_LEGACY_DATA_ROOT")
    return Path(p).expanduser().resolve() if p else None


def _resolve_assets_parent() -> Path:
    """Directory that contains ``from_thomas/`` and (for DAVIS) ``tapvid_davis_30_videos_processed/``."""
    if os.environ.get("GENMATTER_DAVIS_DIR"):
        return Path(os.environ["GENMATTER_DAVIS_DIR"])
    local = REPO_ROOT / "assets"
    legacy_root = _legacy_data_root()
    legacy = legacy_root / "assets" if legacy_root is not None else None
    if local.is_dir() and not local.is_symlink():
        return local.resolve()
    if legacy is not None and legacy.is_dir():
        return legacy
    return local


def _resolve_full_raft_source_dir() -> Path:
    """Directory of *full-length* RAFT ``*.npz`` (input to ``populate_genmatter_data``)."""
    if os.environ.get("GENMATTER_RAFT_SOURCE_DIR"):
        return Path(os.environ["GENMATTER_RAFT_SOURCE_DIR"])
    local = REPO_ROOT / "raft_flows"
    legacy_root = _legacy_data_root()
    legacy = legacy_root / "raft_flows" if legacy_root is not None else None
    if local.is_dir() and not local.is_symlink():
        return local.resolve()
    if legacy is not None and legacy.is_dir():
        return legacy
    return local


def _resolve_data_dir() -> Path:
    if os.environ.get("GENMATTER_DATA_DIR"):
        return Path(os.environ["GENMATTER_DATA_DIR"])
    if os.environ.get("GENMATTER_ASSETS_DIR"):
        return Path(os.environ["GENMATTER_ASSETS_DIR"])
    return REPO_ROOT


GENMATTER_DATA_DIR = _resolve_data_dir()

# Gestalt stimuli + RAFT under ``assets/gestalt_stimuli/{from_thomas,raft_flows}``
GENMATTER_LOCAL_ASSETS = GENMATTER_DATA_DIR / "assets"
GESTALT_STIMULI_DIR = (
    Path(os.environ["GENMATTER_GESTALT_STIMULI_DIR"]).expanduser().resolve()
    if os.environ.get("GENMATTER_GESTALT_STIMULI_DIR")
    else (GENMATTER_LOCAL_ASSETS / "gestalt_stimuli")
)
GESTALT_BASE_PATH = GESTALT_STIMULI_DIR / "from_thomas"

# First N flow frames match ``load_six_frame_gestalt_data`` / GenMatter pairing with 6 depth frames
GESTALT_RAFT_NUM_FLOW_FRAMES = int(os.environ.get("GENMATTER_RAFT_NUM_FLOW_FRAMES", "5"))
RAFT_FLOWS_PATH = Path(
    os.environ.get("GENMATTER_RAFT_FLOWS_PATH", GESTALT_STIMULI_DIR / "raft_flows")
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

# RDK psychophysics: ``<RDK_ROOT>/config_<n>/data.npz`` and ``RDK_configs.json``
RDK_ROOT = Path(os.environ.get("GENMATTER_RDK_DIR", REPO_ROOT / "assets" / "RDK")).resolve()


def rdk_npz_path(config_num: int) -> Path:
    """Path to ``data.npz`` for a given RDK configuration index."""
    return RDK_ROOT / f"config_{config_num}" / "data.npz"


def rdk_configs_json_path() -> Path:
    return RDK_ROOT / "RDK_configs.json"


def rdk_reproducibility_keys_json_path() -> Path:
    """Per-stimulus integer seeds for JAX ``jkey(...)`` (optional; see psychophysics benchmark)."""
    if os.environ.get("GENMATTER_RDK_REPRO_KEYS_PATH"):
        return Path(os.environ["GENMATTER_RDK_REPRO_KEYS_PATH"]).resolve()
    return RDK_ROOT / "reproducibility_keys.json"


PSYCHOPHYSICS_OUTPUT_DIR = RESULTS_DIR / "psychophysics"

# DAVIS: parent directory must contain ``tapvid_davis_30_videos_processed/``
DAVIS_PARENT_DIR = _resolve_assets_parent()
DAVIS_BASE = DAVIS_PARENT_DIR / "tapvid_davis_30_videos_processed"

# Populate script defaults (full RAFT + source ``from_thomas`` tree under ``--source``)
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

def _resolve_seganymo_base_path() -> Path | None:
    """SegAnyMo mask root for ``postprocess-gestalt``; unset unless ``SEGANYMO_BASE_PATH`` is set."""
    p = os.environ.get("SEGANYMO_BASE_PATH")
    return Path(p).expanduser().resolve() if p else None


SEGANYMO_BASE_PATH: Path | None = _resolve_seganymo_base_path()
FLOWSAM_MASKS_SUBPATH = "masks/flowsam_matched_reprocessed"
