"""
Phase 2b: SEPARATE mask<->video correspondence verification (gates all downstream work).

Per video (all 2,520):
  1. temporal-offset check — the stored 12-frame mask stack must best-align with the
     video's moving frames at offset 0 (catches indexing bugs around the duplicated
     first moving frame);
  2. motion-energy check — |frame_{t+1}-frame_t| must concentrate INSIDE mask_t∪mask_{t+1}
     (backgrounds are static), verifying that shaded-derived masks match texture videos;
  3. anchor IoU — initial-frame mask vs true RGBA alpha (where an anchor exists);
  4. metadata mapping — stim_id arithmetic vs the CSV, all rows (once, in task 0).

Usage:
  uv run python preprocessing/sfm/verify_masks.py --task-id T --num-tasks N
  uv run python preprocessing/sfm/verify_masks.py --merge      # after all tasks
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.sfm import sfm_config as cfg
from preprocessing.sfm.video_io import decode_moving_gray
from preprocessing.sfm.derive_masks import (
    silhouette, iou, load_calibration, load_image_index, load_alpha, warp_alpha,
    mask_stack_path)

VERIF_DIR = cfg.MASKS_DIR / "verification"
GALLERY_DIR = cfg.MASKS_DIR / "gallery"
REPORT_PATH = cfg.MASKS_DIR / "verification_report.json"

MOTION_RATIO_MIN = 3.0     # median inside/outside motion-energy ratio to pass
SHADED_IOU_MIN = 0.90      # per-frame IoU of stored mask vs recomputed silhouette
OFFSETS = (-2, -1, 0, 1, 2)


def motion_ratio(gray: np.ndarray, masks: np.ndarray, offset: int = 0):
    """Median over frame pairs of mean(|diff|) inside mask-union vs outside."""
    ratios = []
    T = gray.shape[0]
    for t in range(T - 1):
        mt = t + offset
        if not (0 <= mt < T - 1):
            continue
        diff = cv2.absdiff(gray[t + 1], gray[t]).astype(np.float32)
        union = np.logical_or(masks[mt], masks[mt + 1])
        if union.sum() < 50 or (~union).sum() < 50:
            continue
        inside = float(diff[union].mean())
        outside = float(diff[~union].mean()) + 1e-3
        ratios.append(inside / outside)
    return float(np.median(ratios)) if ratios else 0.0


def verify_one(stim_id: int, tau: int, image_index, cal) -> dict:
    s, o, t, v = cfg.condition_of(stim_id)
    gray = decode_moving_gray(stim_id)                      # (12, 1024, 1024)
    stack = np.load(mask_stack_path(cfg.mask_key(stim_id)))
    masks = stack["masks_full"]                             # (12, 1024, 1024) bool

    rec = {"stim_id": stim_id, "texture": cfg.TEXTURES[t],
           "size_deg": cfg.SIZES_DEG[s], "checks": {}}

    if t == 0:  # shaded: direct per-frame silhouette agreement + temporal offset
        sils = np.stack([silhouette(g, tau) for g in gray])
        per_frame = [iou(masks[i], sils[i]) for i in range(12)]
        rec["checks"]["shaded_frame_iou_min"] = float(min(per_frame))
        offset_scores = {off: float(np.mean(
            [iou(masks[i + off], sils[i]) for i in range(12) if 0 <= i + off < 12]))
            for off in OFFSETS}
        best_off = max(offset_scores, key=offset_scores.get)
        rec["checks"]["best_offset"] = int(best_off)
        rec["pass"] = (min(per_frame) >= SHADED_IOU_MIN) and (best_off == 0)
    else:       # texture: motion energy must sit inside the shaded-derived masks
        ratios = {off: motion_ratio(gray, masks, off) for off in OFFSETS}
        best_off = max(ratios, key=ratios.get)
        rec["checks"]["motion_ratio"] = ratios[0]
        rec["checks"]["best_offset"] = int(best_off)
        rec["checks"]["offset0_vs_best"] = float(ratios[0] / max(ratios[best_off], 1e-9))
        # Temporal offset is only identifiable when the mask actually moves across
        # frames. In-place-rotating small objects have lag-2 mask IoU near 1, inside
        # energy is then offset-invariant, and the argmax is decided by the
        # compression-noise floor in the denominator (measured: inside ~26 grey-levels
        # at every offset, outside 0.01-0.24). Apply the offset test only when
        # identifiable; spatial correspondence (ratio at offset 0) always applies,
        # and shaded videos carry exact per-frame temporal checks.
        lag2 = float(np.median([iou(masks[i], masks[i + 2]) for i in range(10)]))
        identifiable = lag2 < 0.85
        rec["checks"]["mask_lag2_iou"] = lag2
        rec["checks"]["offset_identifiable"] = identifiable
        ok_offset = (ratios[0] >= 0.85 * ratios[best_off]) if identifiable else True
        rec["pass"] = (ratios[0] >= MOTION_RATIO_MIN) and ok_offset

    # anchor IoU where an alpha exists (initial frame = moving frame 0)
    key = (o, v, cfg.SIZES_DEG[s])
    if t == 0 and key in image_index:
        alpha = warp_alpha(load_alpha(image_index[key]), cal["scale"], cal["dx"], cal["dy"])
        rec["checks"]["anchor_iou"] = iou(masks[0], alpha)

    return rec


def save_gallery(stim_id: int, out_dir: Path):
    gray = decode_moving_gray(stim_id)
    masks = np.load(mask_stack_path(cfg.mask_key(stim_id)))["masks_full"]
    tiles = []
    for i in (0, 5, 11):
        img = cv2.cvtColor(gray[i], cv2.COLOR_GRAY2BGR)
        contours, _ = cv2.findContours(masks[i].astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (0, 0, 255), 2)
        tiles.append(cv2.resize(img, (341, 341)))
    out = cfg.assert_writable_path(out_dir / f"{stim_id:04d}_overlay.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.hstack(tiles))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--num-tasks", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()

    if args.merge:
        merge()
        return

    cal = load_calibration()
    image_index = load_image_index()
    tau = cal["tau"]

    records, meta_fail = [], []
    if args.task_id == 0:  # metadata mapping check, once
        rows = cfg.load_meta_rows()
        meta_fail = cfg.verify_metadata_formula(rows)
        print(f"metadata formula mismatches: {len(meta_fail)}")

    ids = list(range(cfg.N_STIMULI))[args.task_id::args.num_tasks]
    if args.limit:
        ids = ids[:args.limit]

    t0 = time.time()
    gallery_ids = set(ids[:: max(1, len(ids) // 6)][:6])
    for n, sid in enumerate(ids):
        try:
            records.append(verify_one(sid, tau, image_index, cal))
            if sid in gallery_ids:
                save_gallery(sid, GALLERY_DIR)
        except Exception as e:  # keep going; failures surface in the report
            records.append({"stim_id": sid, "pass": False, "error": str(e)})
        if (n + 1) % 100 == 0:
            print(f"[{n+1}/{len(ids)}] {time.time()-t0:.0f}s", flush=True)

    out = cfg.assert_writable_path(VERIF_DIR / f"report_task{args.task_id}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"records": records, "metadata_mismatches": meta_fail}))
    n_pass = sum(1 for r in records if r.get("pass"))
    print(f"task {args.task_id}: {n_pass}/{len(records)} pass, {time.time()-t0:.0f}s")


def merge():
    records, meta_fail = [], []
    for p in sorted(VERIF_DIR.glob("report_task*.json")):
        d = json.loads(p.read_text())
        records.extend(d["records"])
        meta_fail.extend(d.get("metadata_mismatches", []))
    failures = [r for r in records if not r.get("pass")]
    anchor = [r["checks"]["anchor_iou"] for r in records
              if r.get("checks", {}).get("anchor_iou") is not None]
    motion = [r["checks"]["motion_ratio"] for r in records
              if r.get("checks", {}).get("motion_ratio") is not None]
    summary = {
        "n_videos": len(records),
        "n_pass": len(records) - len(failures),
        "n_fail": len(failures),
        "metadata_mismatches": meta_fail,
        "anchor_iou": {"n": len(anchor), "median": float(np.median(anchor)) if anchor else None,
                       "min": float(np.min(anchor)) if anchor else None},
        "motion_ratio": {"n": len(motion), "median": float(np.median(motion)) if motion else None,
                         "p05": float(np.percentile(motion, 5)) if motion else None},
        "failed_stim_ids": [r["stim_id"] for r in failures],
    }
    out = cfg.assert_writable_path(REPORT_PATH)
    out.write_text(json.dumps({"summary": summary, "records": records}, indent=1))
    print(json.dumps(summary, indent=2))
    if failures or meta_fail:
        print("VERIFICATION GATE: FAILURES PRESENT — do not consume masks for failed ids.")
        sys.exit(1)
    print("VERIFICATION GATE: all pass.")


if __name__ == "__main__":
    main()
