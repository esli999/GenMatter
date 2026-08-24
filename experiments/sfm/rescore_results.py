"""
Uniformly re-score saved windows from their assignments.npz + bundle (CPU, no GPU).

Inference saves per-frame pixel-hyperblob assignments, so ROI selection, Jaccard,
oracle Jaccard, and probe accuracy are all recomputable offline. This normalizes
results produced by different code revisions of the *evaluation* (the Gibbs outputs
themselves are untouched). Original values are preserved under eval_orig.

Usage: uv run python experiments/sfm/rescore_results.py [--out-root ...]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.sfm import sfm_config as cfg
from experiments.sfm import bundles as B
from experiments.sfm import counts as C
from experiments.sfm.algorithm import sfm_flow_roi
from experiments.gestalt.algorithm import extract_gestalt_segmentation


def heuristic_ref(mcfg, arrays):
    if mcfg.roi_heuristic == "sfm_flow":
        return sfm_flow_roi(arrays["flow_sq"].astype(np.float32), mcfg.roi_flow_floor)
    first = extract_gestalt_segmentation(arrays["depth_sq"].astype(np.float32),
                                         arrays["flow_sq"].astype(np.float32),
                                         arrays["points_3d"].astype(np.float32))
    return first & arrays["motion_valid"][0]


def rescore_one(rdir: Path, mcfg, arrays):
    res = json.loads((rdir / "results.json").read_text())
    if "error" in res:
        return False
    a = np.load(rdir / "assignments.npz")
    ph_all = a["pixel_hyperblob_assignments"]      # (5, G, G)
    gt = arrays["gt_masks"]
    ref = heuristic_ref(mcfg, arrays)
    rng = np.random.default_rng(res.get("seed", mcfg.seed))
    roi_id, prev = None, None
    new_frames, rois = [], []
    for t in range(ph_all.shape[0]):
        ph = ph_all[t].ravel()
        r = ref if t == 0 else prev
        roi_id = C.pick_roi_hyperblob(ph, r, mcfg.n_hyperblobs,
                                      fallback=roi_id or 0, by_iou=True)
        pred = (ph == roi_id)
        prev = pred
        g = gt[t].ravel()
        acc, _ = C.probe_accuracy(g, ph, num_probes=100, rng=rng)
        new_frames.append({
            "frame": t, "probe_accuracy": acc,
            "roi_jaccard": C.jaccard(pred, g),
            "oracle_jaccard": max(C.jaccard(ph == k, g) for k in range(mcfg.n_hyperblobs)),
            "roi_hyperblob": int(roi_id), "gt_area": int(g.sum()),
            "pred_area": int(pred.sum()),
        })
        rois.append(int(roi_id))
    res.setdefault("eval_orig", {
        "mean_probe_accuracy": res.get("mean_probe_accuracy"),
        "mean_roi_jaccard": res.get("mean_roi_jaccard"),
    })
    # merge: keep inline-only fields (uncertainty, scores) from the original frames
    for t, fr in enumerate(res.get("frames", [])):
        if t < len(new_frames):
            for k in ("uncertainty_mean", "final_score"):
                if k in fr:
                    new_frames[t][k] = fr[k]
    res["frames"] = new_frames
    res["mean_probe_accuracy"] = float(np.mean([f["probe_accuracy"] for f in new_frames]))
    res["mean_roi_jaccard"] = float(np.mean([f["roi_jaccard"] for f in new_frames]))
    res["mean_oracle_jaccard"] = float(np.mean([f["oracle_jaccard"] for f in new_frames]))
    res["rescored"] = True
    (rdir / "results.json").write_text(json.dumps(res, indent=1))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=str(cfg.RESULTS_DIR / "windows"))
    args = ap.parse_args()
    all_cfgs = dict(cfg.CONFIGS)
    all_cfgs.update(cfg.pilot_config_grid())
    n = 0
    cache = {}
    for rj in sorted(Path(args.out_root).glob("*/*/*/results.json")):
        config, variant, stim = rj.parts[-4], rj.parts[-3], int(rj.parts[-2])
        if config not in all_cfgs:
            continue
        key = (stim, variant)
        if key not in cache:
            bp = B.bundle_path(cfg.BUNDLES_DIR, stim, variant)
            if not bp.exists():
                continue
            cache.clear()  # bound memory: one bundle at a time is enough (sorted order)
            cache[key] = B.load_bundle(bp)[0]
        if rescore_one(rj.parent, all_cfgs[config], cache[key]):
            n += 1
    print(f"rescored {n} windows")


if __name__ == "__main__":
    main()
