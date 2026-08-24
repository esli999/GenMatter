"""
Phase 2: derive per-frame GT segmentation masks from the SHADED videos.

The shaded stimuli render the object on a black background, so a luminance threshold +
morphology yields per-frame object silhouettes. Geometry/camera are identical across the
6 textures of a (size, object, viewpoint) triple, so each mask stack serves all of them.

Calibration/QA anchor: the RGBA alphas in SFM_foveal_images/stim_from_rig/mworks_images
are true renderer masks for each video's initial-viewpoint frame (sizes 16/27 deg map
onto the videos by a single global canvas transform; 6 deg has no anchor and is covered
by the cross-size consistency check in verify_masks.py).

Usage:
  uv run python preprocessing/sfm/derive_masks.py --calibrate         # once, writes calibration.json
  uv run python preprocessing/sfm/derive_masks.py --task-id 0 --num-tasks 4
"""
import argparse
import csv
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.sfm import sfm_config as cfg
from preprocessing.sfm.video_io import decode_video, decode_moving_gray

CAL_PATH = cfg.MASKS_DIR / "calibration.json"
DEFAULT_TAU = 12


def silhouette(gray: np.ndarray, tau: int) -> np.ndarray:
    """Threshold + open(3) + close(5) + hole fill."""
    m = (gray > tau).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return binary_fill_holes(m.astype(bool))


def to_grid(mask: np.ndarray) -> np.ndarray:
    """1024 -> 128 by area-average majority vote."""
    g = cv2.resize(mask.astype(np.float32), (cfg.GRID_HW, cfg.GRID_HW),
                   interpolation=cv2.INTER_AREA)
    return g > 0.5


def iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union else 1.0


# ------------------------------------------------------------------ anchor alphas
def load_image_index():
    """(object_idx, viewpoint, size_deg) -> image stim_id, from the images CSV."""
    idx = {}
    with open(cfg.IMAGES_CSV) as f:
        for row in csv.DictReader(f):
            if not row["object_id"]:
                continue
            o = int(row["object_id"].split("_")[1])
            v = int(float(row["viewpoint_id"]))
            s = float(row["size_deg"])
            idx[(o, v, s)] = int(row["stim_id"])
    return idx


def load_alpha(image_stim_id: int) -> np.ndarray:
    img = cv2.imread(str(cfg.IMAGES_DIR / f"{image_stim_id}.png"), cv2.IMREAD_UNCHANGED)
    if img is None or img.shape[2] != 4:
        raise IOError(f"image {image_stim_id}.png missing or not RGBA")
    return img[:, :, 3] > 127


def warp_alpha(alpha: np.ndarray, scale: float, dx: int, dy: int) -> np.ndarray:
    """Map the 1000x1000 alpha onto the 1024x1024 video canvas (center-anchored)."""
    R = cfg.VIDEO_RES
    side = int(round(alpha.shape[0] * scale))
    a = cv2.resize(alpha.astype(np.uint8), (side, side), interpolation=cv2.INTER_NEAREST)
    out = np.zeros((R, R), np.uint8)
    off = (R - side) // 2
    sy, sx = off + dy, off + dx
    ys0, ys1 = max(sy, 0), min(sy + side, R)
    xs0, xs1 = max(sx, 0), min(sx + side, R)
    out[ys0:ys1, xs0:xs1] = a[ys0 - sy:ys1 - sy, xs0 - sx:xs1 - sx]
    return out.astype(bool)


def anchor_pairs(image_index, n_per_size=20, rng_seed=0):
    """Sample (video_stim_id, image_stim_id) anchor pairs at sizes 16/27 deg."""
    rng = np.random.default_rng(rng_seed)
    pairs = []
    for size_idx, size_deg in ((1, 16.0), (2, 27.0)):
        combos = [(o, v) for o in range(cfg.N_OBJECTS) for v in range(cfg.N_VIEWPOINTS)]
        rng.shuffle(combos)
        taken = 0
        for o, v in combos:
            key = (o, v, size_deg)
            if key not in image_index:
                continue
            pairs.append((cfg.stim_id_of(size_idx, o, 0, v), image_index[key], size_deg))
            taken += 1
            if taken >= n_per_size:
                break
    return pairs


def calibrate(n_per_size=20):
    """Fit the global alpha->video transform, then pick tau maximizing anchor IoU."""
    image_index = load_image_index()
    pairs = anchor_pairs(image_index, n_per_size=n_per_size)
    # Initial-viewpoint frame = any static frame; use frame 0.
    sils, alphas = [], []
    for vid_sid, img_sid, _ in pairs:
        frame0 = cv2.cvtColor(decode_video(vid_sid)[0], cv2.COLOR_BGR2GRAY)
        sils.append(frame0)
        alphas.append(load_alpha(img_sid))

    # 1) transform fit at a provisional tau, on a subset (silhouettes cached)
    tau0 = DEFAULT_TAU
    sub = list(range(0, len(pairs), max(1, len(pairs) // 12)))
    sil_cache = {i: silhouette(sils[i], tau0) for i in sub}
    best = (-1.0, 1.024, 0, 0)
    for scale in np.arange(1.000, 1.061, 0.006):
        for dx in range(-6, 7, 2):
            for dy in range(-6, 7, 2):
                score = np.mean([
                    iou(sil_cache[i], warp_alpha(alphas[i], scale, dx, dy))
                    for i in sub])
                if score > best[0]:
                    best = (float(score), float(scale), dx, dy)
    _, scale, dx, dy = best
    warped = [warp_alpha(a, scale, dx, dy) for a in alphas]

    # 2) tau sweep with the fitted transform, on all sampled pairs
    tau_scores = {}
    for tau in range(4, 33, 2):
        tau_scores[tau] = float(np.median([
            iou(silhouette(sils[i], tau), warped[i]) for i in range(len(pairs))]))
    tau_star = max(tau_scores, key=tau_scores.get)

    cal = {
        "scale": scale, "dx": dx, "dy": dy, "tau": int(tau_star),
        "fit_iou_at_tau0": best[0], "tau_scores": tau_scores,
        "n_pairs": len(pairs), "median_iou": tau_scores[tau_star],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    CAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    cfg.assert_writable_path(CAL_PATH).write_text(json.dumps(cal, indent=2))
    print(f"calibration: tau={tau_star} scale={scale:.3f} shift=({dx},{dy}) "
          f"median anchor IoU={tau_scores[tau_star]:.4f}")
    return cal


def load_calibration():
    if CAL_PATH.exists():
        return json.loads(CAL_PATH.read_text())
    raise FileNotFoundError(f"run --calibrate first ({CAL_PATH} missing)")


# ------------------------------------------------------------------ mask derivation
def mask_stack_path(mask_key: str) -> Path:
    return cfg.MASKS_DIR / f"{mask_key}.npz"


def derive_one(size_idx: int, object_idx: int, viewpoint: int, tau: int,
               image_index=None, cal=None) -> dict:
    shaded_sid = cfg.stim_id_of(size_idx, object_idx, 0, viewpoint)
    gray = decode_moving_gray(shaded_sid)                       # (12, 1024, 1024)
    masks_full = np.stack([silhouette(g, tau) for g in gray])   # (12, 1024, 1024) bool
    masks_grid = np.stack([to_grid(m) for m in masks_full])     # (12, 128, 128) bool

    qa = {"shaded_stim_id": shaded_sid, "tau": tau,
          "area_px": [int(m.sum()) for m in masks_full]}
    # anchor IoU on the initial frame (moving frame 0 = render init), where available
    if image_index is not None and cal is not None:
        key = (object_idx, viewpoint, cfg.SIZES_DEG[size_idx])
        if key in image_index:
            alpha = warp_alpha(load_alpha(image_index[key]),
                               cal["scale"], cal["dx"], cal["dy"])
            qa["anchor_iou_frame0"] = iou(masks_full[0], alpha)
    return {"masks_full": masks_full, "masks_grid": masks_grid, "qa": qa}


def save_stack(path: Path, stack: dict):
    path = cfg.assert_writable_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp.npz")
    os.close(fd)
    try:
        np.savez_compressed(tmp, masks_full=stack["masks_full"],
                            masks_grid=stack["masks_grid"],
                            qa_json=np.frombuffer(json.dumps(stack["qa"]).encode(), np.uint8))
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def load_mask_grid(stim_id: int) -> np.ndarray:
    """(12, 128, 128) bool mask stack for this stimulus's (size, object, viewpoint)."""
    d = np.load(mask_stack_path(cfg.mask_key(stim_id)))
    return d["masks_grid"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--num-tasks", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tau", type=int, default=0, help="override calibrated tau")
    args = ap.parse_args()

    if args.calibrate:
        calibrate()
        return

    cal = load_calibration()
    tau = args.tau or cal["tau"]
    image_index = load_image_index()

    triples = [(s, o, v) for s in range(len(cfg.SIZES_DEG))
               for o in range(cfg.N_OBJECTS) for v in range(cfg.N_VIEWPOINTS)]
    mine = triples[args.task_id::args.num_tasks]
    if args.limit:
        mine = mine[:args.limit]

    t0, done, skipped = time.time(), 0, 0
    for s, o, v in mine:
        key = f"s{s}_o{o:02d}_v{v}"
        out = mask_stack_path(key)
        if out.exists():
            skipped += 1
            continue
        save_stack(out, derive_one(s, o, v, tau, image_index, cal))
        done += 1
        if done % 20 == 0:
            print(f"[{done}/{len(mine)}] {key} ({time.time()-t0:.0f}s)", flush=True)
    print(f"done: {done} derived, {skipped} skipped, {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
