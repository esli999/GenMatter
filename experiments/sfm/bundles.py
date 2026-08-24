"""Model-ready per-(video, window) bundle I/O with strict schema validation."""
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from . import sfm_config as cfg

# Model timesteps per window T = len(window frames) - 1 (one per flow pair); phase-1
# 6-frame variants have T=5, the whole-video seg0..seg5 variants have T=3. Validation
# infers T from points_3d and checks every field against it.


def bundle_path(bundles_dir, stim_id: int, variant: str) -> Path:
    return Path(bundles_dir) / variant / f"{stim_id:04d}.npz"


def save_bundle(path, *, points_3d, motion_3d, motion_valid, depth_sq, flow_sq,
                gt_masks, meta: dict):
    path = cfg.assert_writable_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = dict(
        points_3d=np.asarray(points_3d, np.float32),
        motion_3d=np.asarray(motion_3d, np.float32),
        motion_valid=np.asarray(motion_valid, bool),
        depth_sq=np.asarray(depth_sq, np.float16),
        flow_sq=np.asarray(flow_sq, np.float16),
        gt_masks=np.asarray(gt_masks, bool),
        meta_json=np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8),
    )
    _validate(arrays)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp.npz")
    os.close(fd)
    try:
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def load_bundle(path):
    d = np.load(path)
    arrays = {k: d[k] for k in d.files}
    _validate(arrays)
    meta = json.loads(bytes(arrays.pop("meta_json")).decode())
    return arrays, meta


def _validate(a):
    G, N = cfg.GRID_HW, cfg.N_DATAPOINTS
    T = a["points_3d"].shape[0]
    if T < 1:
        raise ValueError(f"bundle has {T} timesteps")
    expect = {
        "points_3d": (T, N, 3),
        "motion_3d": (T, N, 3),
        "motion_valid": (T, N),
        "depth_sq": (T + 1, G, G),
        "flow_sq": (T, G, G, 2),
        "gt_masks": (T + 1, G, G),
    }
    for k, shape in expect.items():
        if tuple(a[k].shape) != shape:
            raise ValueError(f"bundle field {k}: shape {a[k].shape} != expected {shape}")
    for k in ("points_3d", "motion_3d"):
        if not np.isfinite(a[k]).all():
            raise ValueError(f"bundle field {k} contains non-finite values")
