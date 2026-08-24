"""
Handoff bake-off analysis: whole-video (seg0..seg5) memory runs vs the independent
segment baseline, paired per video on the SAME stims. Reports the plan's handoff
gate (frames 6..23 Jaccard up AND guarded acceptance > 30%) plus the specific
targets the baseline exposed: the seg4 motion-cessation collapse (frames 17-18)
and the static tail (seg5).

  uv run python experiments/sfm/analyze_handoff.py \
      --memories sticky1_filt0.5_hoG,hoG [--baseline-config sfm_v2]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from genmatter.bootstrap_stats import bootstrap_mean_ci_95
from experiments.sfm import sfm_config as cfg


def collect(out_root: Path, name: str):
    vids = {}
    for k in range(6):
        seg_dir = out_root / name / f"seg{k}"
        if not seg_dir.exists():
            continue
        for rj in sorted(seg_dir.glob("*/results.json")):
            d = json.loads(rj.read_text())
            if d.get("error"):
                continue
            sid = int(d["stim_id"])
            v = vids.setdefault(sid, {"curve": [np.nan] * 24, "accepted": [],
                                      "texture": cfg.TEXTURES[cfg.condition_of(sid)[2]]})
            for fr in d["frames"]:
                v["curve"][4 * k + fr["frame"]] = fr["roi_jaccard"]
            if k > 0 and "handoff_used" in d:
                v["accepted"].append(bool(d["handoff_used"]))
    return vids


def paired_windows(base, mem, sids, frames):
    d = []
    for s in sids:
        b = np.array(base[s]["curve"])[frames]
        m = np.array(mem[s]["curve"])[frames]
        ok = np.isfinite(b) & np.isfinite(m)
        if ok.any():
            d.append(float(np.mean(m[ok] - b[ok])))
    return bootstrap_mean_ci_95(d) if d else (np.nan,) * 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=str(cfg.RESULTS_DIR / "windows"))
    ap.add_argument("--baseline-config", default="sfm_v2")
    ap.add_argument("--memories", required=True)
    args = ap.parse_args()
    out_root = Path(args.out_root)
    base = collect(out_root, args.baseline_config)
    print(f"baseline {args.baseline_config}: {len(base)} videos")

    EVAL = {"frames 6-23": list(range(6, 24)),
            "moving seg2-3 (8-14)": [8, 9, 10, 12, 13, 14],
            "cessation (17-18)": [17, 18],
            "static tail seg5 (20-22)": [20, 21, 22],
            "static head seg0-1 (0-6)": [0, 1, 2, 4, 5, 6]}
    for name in args.memories.split(","):
        mem = collect(out_root, f"{args.baseline_config}+{name}")
        sids = sorted(set(base) & set(mem))
        acc = [a for s in sids for a in mem[s]["accepted"]]
        rate = float(np.mean(acc)) if acc else float("nan")
        print(f"\n{name}: n={len(sids)} videos, handoff acceptance={rate:.2f} "
              f"({sum(acc)}/{len(acc)})")
        for lab, fr in EVAL.items():
            m, lo, hi = paired_windows(base, mem, sids, fr)
            print(f"  {lab:26} delta={m:+.3f} [{lo:+.3f},{hi:+.3f}]")
        gate = rate > 0.30
        m, lo, hi = paired_windows(base, mem, sids, EVAL["frames 6-23"])
        gate = gate and lo > 0
        print(f"  HANDOFF-GATE (6-23 up, accept>30%): {'PASS' if gate else 'fail'}")


if __name__ == "__main__":
    main()
