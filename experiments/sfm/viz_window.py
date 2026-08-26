"""Render QA panels for a processed window: RGB / depth / flow / GT / assignment / ROI.

Usage: uv run python experiments/sfm/viz_window.py --stim-id 42 --variant tiledA \
           [--config sfm_base] [--out results/viz]
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.sfm import sfm_config as cfg
from experiments.sfm import bundles, worklist, windows as W
from preprocessing.sfm.video_io import decode_video

TILE = 192
PALETTE = np.array([[66, 135, 245], [52, 168, 83], [234, 67, 53], [251, 188, 5],
                    [171, 71, 188], [0, 172, 193]], np.uint8)


def norm_img(x):
    x = x.astype(np.float32)
    lo, hi = np.percentile(x, 2), np.percentile(x, 98)
    return np.clip((x - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)


def flow_to_bgr(flow):
    mag = np.linalg.norm(flow, axis=-1)
    ang = (np.arctan2(flow[..., 1], flow[..., 0]) + np.pi) / (2 * np.pi) * 179
    hsv = np.stack([ang, np.full_like(ang, 255), norm_img(mag).astype(np.float32)],
                   axis=-1).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def assign_to_bgr(pixel_hb, n_hyper):
    img = np.zeros((*pixel_hb.shape, 3), np.uint8)
    for k in range(n_hyper):
        img[pixel_hb == k] = PALETTE[k % len(PALETTE)]
    img[pixel_hb < 0] = (128, 128, 128)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stim-id", type=int, required=True)
    ap.add_argument("--variant", default="tiledA")
    ap.add_argument("--config", default="sfm_base")
    ap.add_argument("--out", default=str(cfg.RESULTS_DIR / "viz"))
    args = ap.parse_args()

    arrays, meta = bundles.load_bundle(
        bundles.find_bundle(args.stim_id, args.variant))
    rdir = worklist.result_dir(cfg.RESULTS_DIR / "windows", args.config,
                               args.variant, args.stim_id)
    res = json.loads((rdir / "results.json").read_text())
    assign = np.load(rdir / "assignments.npz")

    frames = decode_video(args.stim_id)
    win = W.window_frames(args.variant)
    G = cfg.GRID_HW
    rows = []
    n_hyper = int(assign["hyperblob_sample_counts"].shape[-1] - 1)
    for t in range(5):
        rgb = cv2.resize(frames[win[t]], (TILE, TILE))
        depth = cv2.resize(cv2.applyColorMap(norm_img(arrays["depth_sq"][t]),
                                             cv2.COLORMAP_TURBO), (TILE, TILE))
        flow = cv2.resize(flow_to_bgr(arrays["flow_sq"][t].astype(np.float32)),
                          (TILE, TILE), interpolation=cv2.INTER_NEAREST)
        gt = cv2.resize((arrays["gt_masks"][t] * 255).astype(np.uint8), (TILE, TILE),
                        interpolation=cv2.INTER_NEAREST)
        gt = cv2.cvtColor(gt, cv2.COLOR_GRAY2BGR)
        hb = cv2.resize(assign_to_bgr(assign["pixel_hyperblob_assignments"][t], n_hyper),
                        (TILE, TILE), interpolation=cv2.INTER_NEAREST)
        roi = (assign["pixel_hyperblob_assignments"][t]
               == assign["roi_hyperblob_ids"][t]).astype(np.uint8) * 255
        roi = cv2.cvtColor(cv2.resize(roi, (TILE, TILE),
                                      interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
        row = np.hstack([rgb, depth, flow, gt, hb, roi])
        f = res["frames"][t]
        cv2.putText(row, f"f{t} acc={f['probe_accuracy']:.2f} jacc={f['roi_jaccard']:.2f}",
                    (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        rows.append(row)

    panel = np.vstack(rows)
    header = np.zeros((22, panel.shape[1], 3), np.uint8)
    cv2.putText(header, f"stim {args.stim_id} {meta['object_id']}/{meta['texture_id']}"
                        f"/vp{meta['viewpoint_id']}/{meta['size_deg']}deg  {args.variant}"
                        f"  [RGB|depth|flow|GT|hyperblobs|ROI]",
                (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    out = cfg.assert_writable_path(
        Path(args.out) / f"{args.stim_id:04d}_{args.variant}_{args.config}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack([header, panel]))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
