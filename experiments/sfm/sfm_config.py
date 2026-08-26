"""
Central configuration for the SFM ephys-stimuli experiment.

All paths under TPO_PREFIX are READ-ONLY (another researcher's data). Every writer in
this module tree must route output paths through `assert_writable_path`.
"""
import os
import hashlib
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path

# ----------------------------------------------------------------------------- paths
TPO_PREFIX = "/orcd/data/jbt/001/tpo"

STIM_ROOT = Path(TPO_PREFIX) / "sfm_experiment/ephys_stimuli"
VIDEOS_DIR = STIM_ROOT / "SFM_parafoveal_videos_2/stimuli/mworks_videos"
META_CSV = STIM_ROOT / "SFM_parafoveal_videos_2/meta/SFM.foveal.videos_2.csv"
# RGBA alpha-mask anchor images (foveal and parafoveal sets are equivalent content)
IMAGES_DIR = STIM_ROOT / "SFM_foveal_images/stim_from_rig/mworks_images"
IMAGES_CSV = STIM_ROOT / "SFM_foveal_images/stim_from_rig/SFM.foveal.images.csv"

WORK_ROOT = Path(os.environ.get("SFM_WORK_ROOT", "/orcd/data/jbt/001/esli/genmatter_sfm"))
SCRATCH_ROOT = Path(os.environ.get("SFM_SCRATCH_ROOT", "/orcd/scratch/bcs/001/esli/genmatter_sfm"))

RESULTS_DIR = WORK_ROOT / "results"
BUNDLES_DIR = SCRATCH_ROOT / "bundles"
# Durable archive of the FINAL (mfull) experiment: the self-contained release
# folder (inputs/ = depth/flow/points/motion bundles, windows/ = outputs +
# latent traces, README.md = full documentation). Scratch bundles are
# regenerable and PURGEABLE, so bundle READS resolve the release archive first
# and fall back to scratch; preprocessing still WRITES new bundles to scratch.
RELEASE_DIR = Path(os.environ.get("SFM_RELEASE_DIR", str(WORK_ROOT / "release_mfull")))
BUNDLE_READ_DIRS = (RELEASE_DIR / "inputs", BUNDLES_DIR)
MASKS_DIR = SCRATCH_ROOT / "masks"
DEPTH_CACHE_DIR = SCRATCH_ROOT / "depth_cache"
JAX_CACHE_DIR = WORK_ROOT / "GenMatter" / ".jax_cache"

VDA_REPO = Path(os.environ.get(
    "GENMATTER_VIDEO_DEPTH_ANYTHING_PATH", "/orcd/data/jbt/001/esli/Video-Depth-Anything"))


def assert_writable_path(p) -> Path:
    """Refuse any output path inside the read-only tpo tree."""
    p = Path(p).resolve()
    if str(p).startswith(TPO_PREFIX):
        raise PermissionError(f"REFUSING to write inside read-only tree {TPO_PREFIX}: {p}")
    return p


# ------------------------------------------------------------------- stimulus layout
# Generation order verified against generate_new_hi_res_videos_2.ipynb and the CSV:
#   stim_id = size_idx*840 + object_idx*42 + texture_idx*7 + viewpoint_id
TEXTURES = ("shaded", "texture_00", "texture_07", "texture_13", "texture_21", "texture_22")
SIZES_DEG = (6.0, 16.0, 27.0)
N_OBJECTS = 20
N_VIEWPOINTS = 7
N_STIMULI = 2520            # stim ids 0..2519; 2520.mp4 is the blank clip (excluded)
BLANK_STIM_ID = 2520

VIDEO_RES = 1024
N_FRAMES = 24
MOVING_FRAMES = tuple(range(6, 18))   # video frames 6..17 hold render frames init..init+11
N_MOVING = 12

# Presentation protocol, verified against stimuli/SFM.parafoveal.videos_2.csv, the
# MWorks .mwel rig protocol, the generation notebook, and the lab's paper
# (biorxiv 10.64898/2026.05.13.724755): each video autoplays ONCE for 400 ms
# (100 ms static hold -> 200 ms rotation -> 100 ms static hold at 60 fps);
# 7 viewpoints evenly spaced on a 90-frame orbit => 4 deg/frame, 48 deg arc per
# clip, non-overlapping; the paper's primary neural analysis window is
# 100-300 ms = the motion epoch = video frames 6..17.
PROTOCOL = {
    "duration_ms": 400, "fps": 60,
    "static_hold_ms": 100, "motion_ms": 200,
    "motion_onset_frame": 6, "motion_offset_frame": 18,
    "rotation_deg_per_frame": 4.0, "orbit_frames": 90,
    "neural_analysis_window_ms": (100, 300),
    "stim_on_delay_ms": 150, "iti_ms": (500, 1000),
}

# Model grid: round32(max(H,W)//8) at native 1024 -> 128
GRID_HW = 128
N_DATAPOINTS = GRID_HW * GRID_HW      # 16384 (divisible by the 1024 logprob chunk)


def stim_id_of(size_idx: int, object_idx: int, texture_idx: int, viewpoint: int) -> int:
    return size_idx * 840 + object_idx * 42 + texture_idx * 7 + viewpoint


def condition_of(stim_id: int):
    """stim_id -> (size_idx, object_idx, texture_idx, viewpoint)."""
    size_idx, r = divmod(stim_id, 840)
    object_idx, r = divmod(r, 42)
    texture_idx, viewpoint = divmod(r, 7)
    return size_idx, object_idx, texture_idx, viewpoint


def shaded_sibling(stim_id: int) -> int:
    """The shaded video sharing this stimulus's (size, object, viewpoint) — the GT-mask source."""
    s, o, _, v = condition_of(stim_id)
    return stim_id_of(s, o, 0, v)


def mask_key(stim_id: int) -> str:
    """Masks are shared across textures: key by (size, object, viewpoint)."""
    s, o, _, v = condition_of(stim_id)
    return f"s{s}_o{o:02d}_v{v}"


def video_path(stim_id: int) -> Path:
    return VIDEOS_DIR / f"{stim_id}.mp4"


def load_meta_rows():
    """Read the stimulus CSV with stdlib csv (no pandas in the pinned env)."""
    import csv
    rows = {}
    with open(META_CSV) as f:
        for row in csv.DictReader(f):
            rows[int(row["stim_id"])] = row
    return rows


def verify_metadata_formula(rows) -> list:
    """Cross-check the stim_id arithmetic against every CSV row; return mismatches."""
    bad = []
    for sid, row in rows.items():
        if sid == BLANK_STIM_ID or not row["object_id"]:
            continue
        s, o, t, v = condition_of(sid)
        ok = (row["object_id"] == f"scene_{o:05d}"
              and row["texture_id"] == TEXTURES[t]
              and int(float(row["viewpoint_id"])) == v
              and float(row["size_deg"]) == SIZES_DEG[s])
        if not ok:
            bad.append(sid)
    return bad


# --------------------------------------------------------------------- model configs
@dataclass(frozen=True)
class SfmModelConfig:
    name: str
    n_blobs: int
    n_roi_blobs: int      # fixed ROI blob count -> pins actual_num_blobs == n_blobs
    n_hyperblobs: int
    init_sweeps: int
    vel_sweeps: int
    track_sweeps: int
    inner_loops: int
    # SE(3) proposal grids (STATIC in the jit: each distinct grid = one compilation)
    trans_gaussian_scale: float
    trans_max_radius: float
    trans_num_radii_cells: int
    trans_theta_step_deg: int
    rot_vmf_kappa: float
    rot_angle_max_deg: float
    rot_angle_step_deg: float
    # outlier / priors
    outlier_prob: float = 0.001
    outlier_gamma_shape: float = 7.5
    outlier_gamma_rate: float = 0.5
    sigma_H: float = 25.0                 # (10*0.5)**2, as in run_gestalt.py
    sigma_V: float = 10e14
    # preprocessing knobs
    focal_length_scale: float = 2.0
    min_motion_magnitude: float = 0.05
    # sampling / selection
    seed: int = 42
    rescore_stride: int = 1               # 1 = score every sweep (dominates strided picks)
    keep_last_samples: int = 50           # assignment history retained for count maps
    # ROI heuristic feeding k-means init + eval reference:
    #   "gestalt"  = extract_gestalt_segmentation (paper path; over-segments small
    #                SFM objects ~20:1 -> ROI hyperblob never isolates them)
    #   "sfm_flow" = tight 2D-flow-magnitude mask (mag > max(floor, 6x median))
    roi_heuristic: str = "gestalt"
    roi_flow_floor: float = 0.35
    # Inference-trace logging: save thinned per-sweep values of ALL latent variables
    # (hyperblob weights/means/covs/transforms, blob weights/means/covs/velocities,
    # blob->hyperblob and datapoint->blob assignments, joint scores) to traces.npz.
    log_traces: bool = False
    trace_thin: int = 5

    def content_hash(self) -> str:
        d = asdict(self)
        for k in ("log_traces", "trace_thin"):  # logging-only: the chain is unchanged
            d.pop(k, None)
        return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:10]


# The published gestalt configuration (run_gestalt.py), used for parity testing.
PAPER_CONFIG = SfmModelConfig(
    name="paper", n_blobs=100, n_roi_blobs=50, n_hyperblobs=5,
    init_sweeps=50, vel_sweeps=20, track_sweeps=500, inner_loops=5,
    trans_gaussian_scale=0.2, trans_max_radius=0.35,
    trans_num_radii_cells=15, trans_theta_step_deg=15,
    rot_vmf_kappa=5.0, rot_angle_max_deg=15.0, rot_angle_step_deg=1.0,
    rescore_stride=25,
)

# SFM-tuned base: 3 hyperblobs (object/background/slack), more blobs for the 128^2 grid,
# rotation proposal grid narrowed+refined to the ~4 deg/frame orbit statistics.
SFM_BASE = SfmModelConfig(
    name="sfm_base", n_blobs=150, n_roi_blobs=75, n_hyperblobs=3,
    init_sweeps=50, vel_sweeps=20, track_sweeps=150, inner_loops=5,
    trans_gaussian_scale=0.2, trans_max_radius=0.35,
    trans_num_radii_cells=15, trans_theta_step_deg=15,
    rot_vmf_kappa=5.0, rot_angle_max_deg=12.0, rot_angle_step_deg=0.6,
)

CONFIGS = {c.name: c for c in (PAPER_CONFIG, SFM_BASE)}


def pilot_config_grid():
    """Config variants the pilot compares. v1 = gestalt ROI heuristic (baseline);
    v2 = tight flow ROI + small-object-scaled ROI blob count. All shapes are shared
    with SFM_BASE except sfm_hb5* -> compiled programs are reused from the cache."""
    import dataclasses
    grid = [SFM_BASE]
    grid.append(dataclasses.replace(SFM_BASE, name="sfm_hb5", n_hyperblobs=5))
    grid.append(dataclasses.replace(SFM_BASE, name="sfm_track300", track_sweeps=300))
    grid.append(dataclasses.replace(SFM_BASE, name="sfm_track500", track_sweeps=500))
    grid.append(dataclasses.replace(SFM_BASE, name="sfm_papergrid",
                                    rot_angle_max_deg=15.0, rot_angle_step_deg=1.0))
    v2 = dataclasses.replace(SFM_BASE, name="sfm_v2",
                             roi_heuristic="sfm_flow", n_roi_blobs=40)
    grid.append(v2)
    grid.append(dataclasses.replace(v2, name="sfm_v2_hb5", n_hyperblobs=5))
    grid.append(dataclasses.replace(v2, name="sfm_v2_track300", track_sweeps=300))
    # rotation-identifiability sweep: sigma_V is the variance of blob velocities
    # around the hyperblob transform's predicted field. At the inherited 1e15 the
    # transform is DECOUPLED from the data (its posterior is the prior — per-sweep
    # rotation samples are diffuse and key-dominated). Tightening it couples blob
    # velocities to the transform; SFM objects are rigid, so the coupling is
    # physically right. Blob-velocity magnitudes are ~0.3-0.9 world units/frame.
    # 0.001 extends the sweep below the adopted 0.01 for the robustness check
    # (is the optimum a plateau or a knife-edge?)
    for sv in (1.0, 0.1, 0.01, 0.001):
        grid.append(dataclasses.replace(v2, name=f"sfm_v2_sv{sv:g}", sigma_V=sv))
    return {c.name: c for c in grid}


# Depth conversion (load_six_frame_gestalt_data convention). If VDA inverse depth does
# not sit in the clip band, `auto_rescale` maps its median to the band center first;
# the factor is recorded in the bundle metadata.
DEPTH_INV_CLIP = (5.0, 1500.0)
DEPTH_SCALE = 3000.0
DEPTH_METRIC_CLIP = (0.25, 20.0)
DEPTH_AUTO_RESCALE_TARGET = 750.0
