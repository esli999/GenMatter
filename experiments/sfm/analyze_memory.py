"""
Paired bake-off analysis: baseline (sfm_v2) vs memory variants on the SAME
(stim, variant) windows, with the plan's numeric adoption gates.

  uv run python experiments/sfm/analyze_memory.py \
      --variant tiledA_g --baseline sfm_v2 --memories sticky0.5,sticky1,sticky2,sticky4

Gates (plan Workstream 4):
- sticky:    final-frame delta >= +0.05 with 95% CI excluding 0, AND mean
             delta >= -0.01
- filtering: additional final-frame delta >= +0.03 over its sticky base, AND
             frame-1 regression <= 0.02
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from genmatter.bootstrap_stats import bootstrap_mean_ci_95
from experiments.sfm import sfm_config as cfg


def load(out_root, name, variant):
    rows = {}
    base = Path(out_root) / name / variant
    if not base.exists():
        return rows
    for rj in sorted(base.glob("*/results.json")):
        r = json.loads(rj.read_text())
        if r.get("error"):
            continue
        jacs = [f["roi_jaccard"] for f in r["frames"]]
        rows[int(r["stim_id"])] = {
            "mean": r["mean_roi_jaccard"], "per_frame": jacs,
            "texture": r["bundle_meta"]["texture_id"],
            "handoff": r.get("handoff_used"),
        }
    return rows


def paired(base_rows, mem_rows):
    sids = sorted(set(base_rows) & set(mem_rows))
    out = {"n": len(sids)}
    if not sids:
        return out
    d_mean = [mem_rows[s]["mean"] - base_rows[s]["mean"] for s in sids]
    d_final = [mem_rows[s]["per_frame"][-1] - base_rows[s]["per_frame"][-1]
               for s in sids]
    d_first = [mem_rows[s]["per_frame"][0] - base_rows[s]["per_frame"][0]
               for s in sids]
    nfr = min(len(mem_rows[sids[0]]["per_frame"]), len(base_rows[sids[0]]["per_frame"]))
    curves = {
        "base": [float(np.mean([base_rows[s]["per_frame"][f] for s in sids]))
                 for f in range(nfr)],
        "mem": [float(np.mean([mem_rows[s]["per_frame"][f] for s in sids]))
                for f in range(nfr)],
    }
    for label, d in (("mean", d_mean), ("final", d_final), ("first", d_first)):
        m, lo, hi = bootstrap_mean_ci_95(d)
        out[f"d_{label}"] = {"mean": float(m), "lo": float(lo), "hi": float(hi)}
    out["curves"] = curves
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=str(cfg.RESULTS_DIR / "windows"))
    ap.add_argument("--variant", default="tiledA_g")
    ap.add_argument("--model", default="sfm_v2")
    ap.add_argument("--baseline", default="null",
                    help="memory name of the matched baseline ('off' = the plain "
                         "model run)")
    ap.add_argument("--memories", required=True, help="comma-separated memory names")
    ap.add_argument("--textured-only", action="store_true",
                    help="restrict pairing to non-shaded stimuli")
    args = ap.parse_args()

    base_name = (args.model if args.baseline == "off"
                 else f"{args.model}+{args.baseline}")
    base_rows = load(args.out_root, base_name, args.variant)
    print(f"baseline {base_name}/{args.variant}: {len(base_rows)} windows")
    report = {}
    for name in args.memories.split(","):
        mem_rows = load(args.out_root, f"{args.model}+{name}", args.variant)
        b = base_rows
        m = mem_rows
        if args.textured_only:
            b = {s: r for s, r in b.items() if r["texture"] != "shaded"}
            m = {s: r for s, r in m.items() if r["texture"] != "shaded"}
        p = paired(b, m)
        report[name] = p
        if p["n"] == 0:
            print(f"{name}: no paired windows yet")
            continue
        dm, df, d0 = p["d_mean"], p["d_final"], p["d_first"]
        gate = (df["mean"] >= 0.05 and df["lo"] > 0.0 and dm["mean"] >= -0.01)
        print(f"{name}: n={p['n']}  d_mean={dm['mean']:+.3f} [{dm['lo']:+.3f},{dm['hi']:+.3f}]"
              f"  d_final={df['mean']:+.3f} [{df['lo']:+.3f},{df['hi']:+.3f}]"
              f"  d_first={d0['mean']:+.3f}  STICKY-GATE={'PASS' if gate else 'fail'}")
        print(f"    curve base {['%.3f' % v for v in p['curves']['base']]}")
        print(f"    curve mem  {['%.3f' % v for v in p['curves']['mem']]}")
    out = cfg.assert_writable_path(cfg.RESULTS_DIR / "aggregated" /
                                   f"memory_bakeoff_{args.variant}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
