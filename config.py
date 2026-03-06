import os
from pathlib import Path

REPO_ROOT = Path(__file__).parent.resolve()
ASSETS_DIR = Path(os.environ.get("GENMATTER_ASSETS_DIR", REPO_ROOT / "assets"))
RESULTS_DIR = Path(os.environ.get("GENMATTER_RESULTS_DIR", REPO_ROOT / "results"))

# Gestalt
GESTALT_BASE_PATH = ASSETS_DIR / "from_thomas"
RAFT_FLOWS_PATH = REPO_ROOT / "raft_flows"

# DAVIS
DAVIS_BASE = ASSETS_DIR / "tapvid_davis_30_videos_processed"
DAVIS_3D_MOTION_PATH = DAVIS_BASE / "tapvid_davis_npzs"
DAVIS_SEGMASKS_PATH = DAVIS_BASE / "tapvid_davis_segmasks"
DAVIS_RGB_PATH = DAVIS_BASE / "tapvid_davis_rgb_frames"
DAVIS_DINO_PATH = DAVIS_BASE / "tapvid_davis_dino"
DAVIS_SAM_FRAME0_PATH = DAVIS_BASE / "tapvid_davis_SAM_frame0"

# Output dirs
GESTALT_OUTPUT_DIR = RESULTS_DIR / "gestalt"
GESTALT_DEPTH_ABLATION_OUTPUT_DIR = RESULTS_DIR / "gestalt_depth_ablation"
DAVIS_TRACKING_OUTPUT_DIR = RESULTS_DIR / "davis_tracking"
DAVIS_SUBSAMPLING_OUTPUT_DIR = RESULTS_DIR / "davis_subsampling"
DAVIS_ABLATION_OUTPUT_DIR = RESULTS_DIR / "davis_ablation"
COTRACKER_OUTPUT_DIR = RESULTS_DIR / "cotracker_baseline"
POSTPROCESSING_OUTPUT_DIR = RESULTS_DIR / "postprocessing"

# External baselines (postprocessing only)
SEGANYMO_BASE_PATH = Path(os.environ.get(
    "SEGANYMO_BASE_PATH", "/home/esli/SegAnyMo/gestalt_SegAnyMo_outputs"))
FLOWSAM_MASKS_SUBPATH = "masks/flowsam_matched_reprocessed"
