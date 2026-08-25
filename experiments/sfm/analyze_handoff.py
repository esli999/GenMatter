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


def collect(out_root: Path, name: str, segments):
    from experiments.sfm import windows as W
    vids = {}
    for i, sv in enumerate(segments):
        seg_dir = out_root / name / sv
        if not seg_dir.exists():
            continue
        frames = W.window_frames(sv)
        for rj in sorted(seg_dir.glob("*/results.json")):
            d = json.loads(rj.read_text())
            if d.get("error"):
                continue
            sid = int(d["stim_id"])
            v = vids.setdefault(sid, {"curve": [np.nan] * 24, "accepted": [],
                                      "texture": cfg.TEXTURES[cfg.condition_of(sid)[2]]})
            for fr in d["frames"]:
                v["curve"][frames[fr["frame"]]] = fr["roi_jaccard"]
            if "handoff_used" in d:
                v["accepted"].append(bool(d["handoff_used"]))
                # acceptance denominator = windows actually OFFERED a handoff
                # (older results lack the field: fall back to counting all)
                v.setdefault("offered", []).append(
                    bool(d.get("handoff_offered", True)))
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
    ap.add_argument("--segset", default="seg", choices=["seg", "exp", "exp2"])
    args = ap.parse_args()
    out_root = Path(args.out_root)
    from experiments.sfm import windows as W
    segments = (list(W.EXP2_SEGMENTS) if args.segset == "exp2"
                else list(W.EXP_SEGMENTS) if args.segset == "exp"
                else [f"seg{i}" for i in range(6)])
    base = collect(out_root, args.baseline_config, segments)
    print(f"baseline {args.baseline_config} [{args.segset}]: {len(base)} videos")

    if args.segset == "exp2":       # full 24/24 coverage
        EVAL = {"frames 6-23": list(range(6, 24)),
                "motion (6-17)": list(range(6, 18)),
                "post-offset hold2 (18-23)": list(range(18, 24)),
                "static head hold1 (0-5)": list(range(0, 6))}
    elif args.segset == "exp":
        EVAL = {"frames 6-23": list(range(6, 24)),
                "motion (6-16)": list(range(6, 17)),
                "post-offset hold2 (18-22)": [18, 19, 20, 21, 22],
                "static head hold1 (0-4)": [0, 1, 2, 3, 4]}
    else:
        EVAL = {"frames 6-23": list(range(6, 24)),
                "moving seg2-3 (8-14)": [8, 9, 10, 12, 13, 14],
                "cessation (17-18)": [17, 18],
                "static tail seg5 (20-22)": [20, 21, 22],
                "static head seg0-1 (0-6)": [0, 1, 2, 4, 5, 6]}
    for name in args.memories.split(","):
        mem = collect(out_root, f"{args.baseline_config}+{name}", segments)
        sids = sorted(set(base) & set(mem))
        pairs = [(a, o) for s in sids
                 for a, o in zip(mem[s]["accepted"], mem[s].get("offered", []))]
        n_acc = sum(a for a, _ in pairs)
        n_off = sum(o for _, o in pairs)
        rate = n_acc / n_off if n_off else float("nan")
        print(f"\n{name}: n={len(sids)} videos, handoff acceptance={rate:.2f} "
              f"({n_acc}/{n_off} offered; {len(pairs)} windows)")
        for lab, fr in EVAL.items():
            m, lo, hi = paired_windows(base, mem, sids, fr)
            print(f"  {lab:26} delta={m:+.3f} [{lo:+.3f},{hi:+.3f}]")
        gate = rate > 0.30
        m, lo, hi = paired_windows(base, mem, sids, EVAL["frames 6-23"])
        gate = gate and lo > 0
        print(f"  HANDOFF-GATE (6-23 up, accept>30%): {'PASS' if gate else 'fail'}")


if __name__ == "__main__":
    main()
