"""
Rotation-identifiability analysis over the sigma_V sweep: for each config, find
the OBJECT hyperblob of every traced m1x window (blobs whose pixels sit mostly
inside GT, then the hyperblob owning the plurality of them), and compare its
per-sweep rotation-angle samples against the 4 deg/frame the generation metadata
implies. A tight, data-driven posterior concentrates near 4; the decoupled
(sigma_V = 1e15) reference is diffuse over the proposal grid.

  uv run python experiments/sfm/analyze_rotation.py \
      --configs sfm_v2,sfm_v2_sv1,sfm_v2_sv0.1,sfm_v2_sv0.01
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.sfm import sfm_config as cfg
from experiments.sfm import windows as W
from preprocessing.sfm.derive_masks import load_mask_grid

TRUE_DEG = cfg.PROTOCOL["rotation_deg_per_frame"]


def rot_angle_deg(R):
    tr = np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(tr))


def object_hyperblob(dd, gt_flat):
    last_track = sorted(p.split(".")[0] for p in dd.files if "track" in p)[-1]
    da = dd[f"{last_track}.datapoint_assignments"][-1].astype(np.int64)
    ha = dd[f"{last_track}.blob_hyperblob_assignments"][-1].astype(np.int64)
    L = ha.shape[0]
    counts = np.bincount(da, minlength=L + 1)[:L]
    inside = np.bincount(da, weights=gt_flat, minlength=L + 1)[:L]
    frac = inside / np.maximum(counts, 1)
    obj = np.where((counts >= 20) & (frac > 0.5))[0]
    if len(obj) == 0:
        obj = np.argsort(inside)[-10:]
    return int(np.bincount(ha[obj]).argmax())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=str(cfg.RESULTS_DIR / "windows"))
    ap.add_argument("--configs", required=True)
    ap.add_argument("--variant", default="m1x")
    args = ap.parse_args()
    out_root = Path(args.out_root)

    frames = W.window_frames(args.variant)
    gt_frame = W.frame_to_mask_index(frames[-2])

    for config in args.configs.split(","):
        errs, meds, jaccs = [], [], []
        for rdir in sorted((out_root / config / args.variant).glob("*")):
            tnpz, rjson = rdir / "traces.npz", rdir / "results.json"
            if not (tnpz.exists() and rjson.exists()):
                continue
            r = json.loads(rjson.read_text())
            if r.get("error"):
                continue
            sid = int(rdir.name)
            gt = load_mask_grid(sid)[gt_frame].reshape(-1).astype(np.float64)
            dd = np.load(tnpz)
            hb = object_hyperblob(dd, gt)
            angs = np.concatenate([
                rot_angle_deg(dd[f"{ph}.hyperblob_rot_vels"].astype(np.float64)[:, hb])
                for ph in sorted(p.split(".")[0] for p in dd.files if "track" in p)])
            med = float(np.median(angs))
            meds.append(med)
            errs.append(abs(med - TRUE_DEG))
            jaccs.append(float(r["mean_roi_jaccard"]))
        if not meds:
            print(f"{config}: no traced windows found")
            continue
        print(f"{config}: n={len(meds)}  median rot={np.median(meds):.2f} deg/frame "
              f"(true {TRUE_DEG:g})  mean|err|={np.mean(errs):.2f}  "
              f"frac within 2 deg={np.mean(np.array(errs) < 2.0):.2f}  "
              f"mean Jaccard={np.mean(jaccs):.3f}")


if __name__ == "__main__":
    main()
