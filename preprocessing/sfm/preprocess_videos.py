"""
Phase 3: depth (Video-Depth-Anything vitl) + flow (torchvision RAFT-large) at native
1024x1024 -> model-ready per-(video, window) bundles.

Torch owns the GPU here; the small JAX pieces (resize/unproject inside the reused
gestalt code path) are pinned to CPU via JAX_PLATFORMS below, so there is no
torch/JAX GPU co-residency in this stage.

Usage:
  uv run python preprocessing/sfm/preprocess_videos.py \
      --stim-file <ids.txt> --variants tiledA,tiledB,centered,stride2 \
      --task-id T --num-tasks N [--save-raw] [--batch-size 4]
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")          # keep JAX off the GPU in this stage
os.environ.setdefault("OMP_NUM_THREADS", "8")

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from experiments.sfm import sfm_config as cfg
from experiments.sfm import windows as W
from experiments.sfm import bundles
from preprocessing.sfm.video_io import decode_video
from preprocessing.sfm.derive_masks import load_mask_grid

sys.path.insert(0, str(cfg.VDA_REPO))
from video_depth_anything.video_depth import VideoDepthAnything  # noqa: E402

# The reused gestalt code path (JAX on CPU): resize_to_square + 3D unprojection.
from experiments.gestalt.algorithm import compute_3d_points_and_motion  # noqa: E402

VDA_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}


def load_vda(encoder="vitl", device="cuda"):
    ckpt = cfg.VDA_REPO / "checkpoints" / f"video_depth_anything_{encoder}.pth"
    model = VideoDepthAnything(**VDA_CONFIGS[encoder], metric=False)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
    return model.to(device).eval()


def load_raft(device="cuda"):
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
    model = raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2, progress=False)
    return model.to(device).eval()


def raft_preprocess(frames_u8: torch.Tensor) -> torch.Tensor:
    """(N, H, W, 3) uint8 RGB -> normalized (N, 3, H, W); native res (1024 is /8)."""
    x = frames_u8.permute(0, 3, 1, 2).float() / 255.0
    return (x - 0.5) / 0.5


@torch.inference_mode()
def compute_flows(raft, frames_rgb: np.ndarray, pairs, device="cuda", batch_size=4):
    """Flow for the given (i, j) frame-index pairs -> dict[(i, j)] = (H, W, 2) float32."""
    out = {}
    pairs = list(pairs)
    for b0 in range(0, len(pairs), batch_size):
        chunk = pairs[b0:b0 + batch_size]
        a = torch.from_numpy(frames_rgb[[i for i, _ in chunk]]).to(device)
        b = torch.from_numpy(frames_rgb[[j for _, j in chunk]]).to(device)
        flows = raft(raft_preprocess(a), raft_preprocess(b))[-1]      # (B, 2, H, W)
        flows = flows.permute(0, 2, 3, 1).float().cpu().numpy()
        for k, p in enumerate(chunk):
            out[p] = flows[k]
    return out


def convert_depth(inv_depth: np.ndarray):
    """VDA inverse depth -> metric-ish depth via the gestalt convention, with an
    auto-rescale into the clip band when the raw scale sits outside it."""
    lo, hi = cfg.DEPTH_INV_CLIP
    med = float(np.median(inv_depth))
    factor = 1.0
    if not (lo * 2 < med < hi / 2):
        factor = cfg.DEPTH_AUTO_RESCALE_TARGET / max(med, 1e-6)
        inv_depth = inv_depth * factor
    depth = cfg.DEPTH_SCALE / (np.clip(inv_depth, lo, hi) + 1e-6)
    return np.clip(depth, *cfg.DEPTH_METRIC_CLIP).astype(np.float32), factor, med


def needed_pairs(variants):
    s = set()
    for v in variants:
        s.update(W.flow_pairs(v))
    return sorted(s)


def resize_grid(arr: np.ndarray) -> np.ndarray:
    """(T, H, W[, C]) -> (T, G, G[, C]) float16 diagnostic copies for the bundle."""
    G = cfg.GRID_HW
    out = np.stack([cv2.resize(a.astype(np.float32), (G, G), interpolation=cv2.INTER_AREA)
                    for a in arr])
    return out.astype(np.float16)


def process_video(stim_id, variants, vda, raft, model_cfg, device, batch_size,
                  save_raw=False):
    frames_bgr = decode_video(stim_id)[list(cfg.MOVING_FRAMES)]      # (12, 1024, 1024, 3)
    frames_rgb = np.ascontiguousarray(frames_bgr[:, :, :, ::-1])

    with torch.inference_mode():
        inv_depth, _ = vda.infer_video_depth(
            frames_rgb, target_fps=-1, input_size=518, device=device, fp32=False)
    inv_depth = np.asarray(inv_depth, np.float32)                    # (12, 1024, 1024)

    # flow pairs are defined in video-frame indices (6..17); frames_rgb holds only
    # the 12 moving frames, so translate to moving indices (0..11) for array access
    vid_pairs = needed_pairs(variants)
    mov_pairs = [(W.moving_index(a), W.moving_index(b)) for a, b in vid_pairs]
    flows_m = compute_flows(raft, frames_rgb, mov_pairs, device, batch_size)
    flows = {vp: flows_m[mp] for vp, mp in zip(vid_pairs, mov_pairs)}

    if save_raw:
        raw = cfg.assert_writable_path(cfg.DEPTH_CACHE_DIR / f"{stim_id:04d}_raw.npz")
        raw.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(raw, inv_depth=resize_grid(inv_depth),
                            **{f"flow_{i}_{j}": resize_grid(f[None])[0]
                               for (i, j), f in flows.items()})

    mask_grid12 = load_mask_grid(stim_id)                            # (12, 128, 128)
    meta_rows = process_video.meta_rows

    for variant in variants:
        out = bundles.bundle_path(cfg.BUNDLES_DIR, stim_id, variant)
        if out.exists():
            continue
        fidx = [W.moving_index(f) for f in W.window_frames(variant)]  # 6 indices into 0..11
        depth_win, factor, med = convert_depth(inv_depth[fidx])       # (6, 1024, 1024)
        flow_win = np.stack([flows[p] for p in W.flow_pairs(variant)])  # (5, 1024, 1024, 2)

        points_3d, motion_3d, motion_valid = compute_3d_points_and_motion(
            depth_win, flow_win, downsample_factor=8,
            min_motion_magnitude=model_cfg.min_motion_magnitude,
            focal_length_scale=model_cfg.focal_length_scale)

        gt = mask_grid12[fidx]                                        # (6, 128, 128)
        row = meta_rows[stim_id]
        bundles.save_bundle(
            out,
            points_3d=points_3d, motion_3d=motion_3d, motion_valid=motion_valid,
            depth_sq=resize_grid(depth_win),
            flow_sq=resize_grid(flow_win),
            gt_masks=gt,
            meta={
                "stim_id": stim_id, "variant": variant,
                "object_id": row["object_id"], "texture_id": row["texture_id"],
                "viewpoint_id": row["viewpoint_id"], "size_deg": row["size_deg"],
                "depth_rescale_factor": factor, "inv_depth_median_raw": med,
                "min_motion_magnitude": model_cfg.min_motion_magnitude,
                "focal_length_scale": model_cfg.focal_length_scale,
                "git_rev": GIT_REV,
            })


GIT_REV = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                         capture_output=True, text=True).stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stim-file", type=str, default="",
                    help="file with one stim_id per line; default = all 0..2519")
    ap.add_argument("--variants", type=str, default="tiledA")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--num-tasks", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--config", type=str, default="sfm_base")
    ap.add_argument("--save-raw", action="store_true")
    args = ap.parse_args()

    variants = args.variants.split(",")
    model_cfg = cfg.CONFIGS.get(args.config) or cfg.pilot_config_grid()[args.config]

    if args.stim_file:
        ids = [int(x) for x in Path(args.stim_file).read_text().split()]
    else:
        ids = list(range(cfg.N_STIMULI))
    ids = ids[args.task_id::args.num_tasks]
    if args.limit:
        ids = ids[:args.limit]

    process_video.meta_rows = cfg.load_meta_rows()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} variants={variants} n_videos={len(ids)}", flush=True)
    vda = load_vda(device=device)
    raft = load_raft(device=device)

    t0, done, skipped = time.time(), 0, 0
    for n, sid in enumerate(ids):
        if all(bundles.bundle_path(cfg.BUNDLES_DIR, sid, v).exists() for v in variants):
            skipped += 1
            continue
        process_video(sid, variants, vda, raft, model_cfg, device, args.batch_size,
                      save_raw=args.save_raw)
        done += 1
        if done % 25 == 0:
            rate = (time.time() - t0) / max(done, 1)
            print(f"[{n+1}/{len(ids)}] {rate:.1f}s/video", flush=True)
    print(f"done: {done} processed, {skipped} skipped, {time.time()-t0:.0f}s total")


if __name__ == "__main__":
    main()
