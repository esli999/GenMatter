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
from experiments.sfm import windows as W
from genmatter.bootstrap_stats import bootstrap_mean_ci_95

N_FRAMES = cfg.N_FRAMES

SEGSETS = {
    # blind 4-frame tiles (phase-2 first pass)
    "seg": {"segments": [f"seg{i}" for i in range(6)],
            "moving": ["seg1", "seg2", "seg3", "seg4"],
            "static": ["seg0", "seg5"]},
    # metadata-aligned: boundaries at motion onset (frame 6) and offset (frame 18)
    "exp": {"segments": list(W.EXP_SEGMENTS),
            "moving": ["m0", "m1", "m2"],
            "static": ["hold1", "hold2"]},
}


def collect(out_root: Path, config: str, segments):
    """{stim_id: {"texture":…, "curve": [24 floats/nan], "seg_mean": {seg: jacc}}}"""
    vids = {}
    n_err = 0
    for sv in segments:
        frames = W.window_frames(sv)
        for rj in sorted((out_root / config / sv).glob("*/results.json")):
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
                g = frames[fr["frame"]]
                v["curve"][g] = fr["roi_jaccard"]
                v["oracle_curve"][g] = fr["oracle_jaccard"]
            v["seg_mean"][sv] = d["mean_roi_jaccard"]
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
    ap.add_argument("--segset", default="seg", choices=sorted(SEGSETS))
    args = ap.parse_args()
    ss = SEGSETS[args.segset]

    vids = collect(Path(args.out_root), args.config, ss["segments"])
    print(f"{args.config} [{args.segset}]: {len(vids)} videos with segment results")
    if not vids:
        return
    out = {"config": args.config, "segset": args.segset, "n_videos": len(vids),
           "headline_moving": {}, "static_holds": {}, "curves_by_texture": {}}
    for tex in ("ALL",) + cfg.TEXTURES:
        pred = (lambda v: True) if tex == "ALL" else (lambda v, t=tex: v["texture"] == t)
        out["headline_moving"][tex] = headline(vids, pred, ss["moving"])
        out["static_holds"][tex] = headline(vids, pred, ss["static"])
        out["curves_by_texture"][tex] = mean_curves(vids, pred)
    suffix = "" if args.segset == "seg" else f"_{args.segset}"
    agg = cfg.assert_writable_path(
        cfg.RESULTS_DIR / "aggregated" /
        f"video_{args.config.replace('+','_')}{suffix}.json")
    agg.parent.mkdir(parents=True, exist_ok=True)
    agg.write_text(json.dumps(out, indent=1))
    for tex in ("ALL",) + cfg.TEXTURES:
        h, s = out["headline_moving"][tex], out["static_holds"][tex]
        if h:
            print(f"{tex:>12}: moving jacc={h['jaccard']:.3f} "
                  f"[{h['ci'][0]:.3f},{h['ci'][1]:.3f}] n={h['n']}   "
                  f"static jacc={s['jaccard']:.3f}" if s else "")
    print(f"wrote {agg}")


if __name__ == "__main__":
    main()
