"""
Whole-video aggregation over the seg0..seg5 results: per-config 24-frame Jaccard
time courses (global video frame = 4*K + window frame; each 4-frame segment
evaluates its first 3 frames, so 18/24 frames carry model output), headline
motion-grouping accuracy over the moving segments (seg1..seg4), and the static
holds (seg0/seg5) reported separately.

Usage: uv run python experiments/sfm/aggregate_video.py --config sfm_v2
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.sfm import sfm_config as cfg
from genmatter.bootstrap_stats import bootstrap_mean_ci_95

N_FRAMES = cfg.N_FRAMES


def collect(out_root: Path, config: str):
    """{stim_id: {"texture":…, "curve": [24 floats/nan], "seg_mean": {segK: jacc}}}"""
    vids = {}
    n_err = 0
    for k in range(6):
        for rj in sorted((out_root / config / f"seg{k}").glob("*/results.json")):
            d = json.loads(rj.read_text())
            if d.get("error"):
                n_err += 1
                continue
            sid = int(d["stim_id"])
            v = vids.setdefault(sid, {
                "texture": cfg.TEXTURES[cfg.condition_of(sid)[2]],
                "viewpoint": cfg.condition_of(sid)[3],
                "object": cfg.condition_of(sid)[1],
                "curve": [float("nan")] * N_FRAMES,
                "oracle_curve": [float("nan")] * N_FRAMES,
                "seg_mean": {},
            })
            for fr in d["frames"]:
                g = 4 * k + fr["frame"]
                v["curve"][g] = fr["roi_jaccard"]
                v["oracle_curve"][g] = fr["oracle_jaccard"]
            v["seg_mean"][f"seg{k}"] = d["mean_roi_jaccard"]
    if n_err:
        print(f"note: {n_err} error windows excluded")
    return vids


def mean_curves(vids, pred):
    sel = [v for v in vids.values() if pred(v)]
    if not sel:
        return None
    arr = np.array([v["curve"] for v in sel])            # [n, 24] with NaNs
    orc = np.array([v["oracle_curve"] for v in sel])
    with np.errstate(all="ignore"):
        return {"n": len(sel),
                "jaccard": [round(float(x), 4) if np.isfinite(x) else None
                            for x in np.nanmean(arr, axis=0)],
                "oracle": [round(float(x), 4) if np.isfinite(x) else None
                           for x in np.nanmean(orc, axis=0)]}


def headline(vids, pred, segs):
    vals = []
    for v in vids.values():
        if not pred(v):
            continue
        xs = [v["seg_mean"][s] for s in segs if s in v["seg_mean"]]
        if xs:
            vals.append(float(np.mean(xs)))
    if not vals:
        return None
    m, lo, hi = bootstrap_mean_ci_95(vals)
    return {"n": len(vals), "jaccard": round(float(m), 4),
            "ci": [round(float(lo), 4), round(float(hi), 4)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=str(cfg.RESULTS_DIR / "windows"))
    ap.add_argument("--config", default="sfm_v2")
    args = ap.parse_args()

    vids = collect(Path(args.out_root), args.config)
    print(f"{args.config}: {len(vids)} videos with segment results")
    if not vids:
        return
    moving = [f"seg{k}" for k in (1, 2, 3, 4)]
    static = ["seg0", "seg5"]
    out = {"config": args.config, "n_videos": len(vids),
           "headline_moving_seg1_4": {}, "static_seg0_5": {},
           "curves_by_texture": {}}
    for tex in ("ALL",) + cfg.TEXTURES:
        pred = (lambda v: True) if tex == "ALL" else (lambda v, t=tex: v["texture"] == t)
        out["headline_moving_seg1_4"][tex] = headline(vids, pred, moving)
        out["static_seg0_5"][tex] = headline(vids, pred, static)
        out["curves_by_texture"][tex] = mean_curves(vids, pred)
    agg = cfg.assert_writable_path(cfg.RESULTS_DIR / "aggregated" /
                                   f"video_{args.config.replace('+','_')}.json")
    agg.parent.mkdir(parents=True, exist_ok=True)
    agg.write_text(json.dumps(out, indent=1))
    for tex in ("ALL",) + cfg.TEXTURES:
        h, s = out["headline_moving_seg1_4"][tex], out["static_seg0_5"][tex]
        if h:
            print(f"{tex:>12}: moving(seg1-4) jacc={h['jaccard']:.3f} "
                  f"[{h['ci'][0]:.3f},{h['ci'][1]:.3f}] n={h['n']}   "
                  f"static(seg0/5) jacc={s['jaccard']:.3f}" if s else "")
    print(f"wrote {agg}")


if __name__ == "__main__":
    main()
