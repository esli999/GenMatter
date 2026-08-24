"""
Pilot bake-off analysis -> full-run decision.

Compares configs x window variants on the pilot set and picks the full-run pair by:
  1. drop configs whose accuracy is clearly dominated (CI-separated) by another;
  2. among statistically tied configs, take the cheapest (fewest tracking sweeps,
     narrowest grid);
  3. pick the window variant with the best accuracy for the chosen config; prefer
     `tiledA`+`tiledB` (time-resolved, 2 windows/video) if their mean is within the
     CI of the single best variant, else the single best.
Emits results/pilot_decision.json with the choice, the evidence table, and the
projected full-run cost; flags AMBIGUOUS when the evidence is degenerate.

Usage: uv run python experiments/sfm/pilot_analysis.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.sfm import sfm_config as cfg
from genmatter.bootstrap_stats import bootstrap_mean_ci_95

CHANCE_BAND = 0.55          # below this mean probe accuracy everything is suspect
N_GPUS = 4


def load_rows(out_root):
    rows = []
    for rj in sorted(Path(out_root).glob("*/*/*/results.json")):
        d = json.loads(rj.read_text())
        if "error" in d:
            continue
        rows.append({"config": rj.parts[-4], "variant": rj.parts[-3],
                     "stim_id": d["stim_id"], "acc": d["mean_probe_accuracy"],
                     "jacc": d["mean_roi_jaccard"], "wall": d.get("wall_s", np.nan)})
    return rows


def table(rows):
    g = defaultdict(list)
    for r in rows:
        g[(r["config"], r["variant"])].append(r)
    out = {}
    for k, rs in g.items():
        accs = [r["acc"] for r in rs]
        m, lo, hi = bootstrap_mean_ci_95(accs)
        out[k] = {"n": len(rs), "acc": float(m), "acc_lo": float(lo), "acc_hi": float(hi),
                  "jacc": float(np.mean([r["jacc"] for r in rs])),
                  "wall_med": float(np.nanmedian([r["wall"] for r in rs]))}
    return out


def decide(tbl):
    configs = sorted({c for c, _ in tbl})
    variants = sorted({v for _, v in tbl})
    # per-config accuracy pooled over variants
    per_cfg = {c: {"acc": float(np.mean([tbl[(c, v)]["acc"] for v in variants if (c, v) in tbl])),
                   "lo": float(np.min([tbl[(c, v)]["acc_lo"] for v in variants if (c, v) in tbl])),
                   "hi": float(np.max([tbl[(c, v)]["acc_hi"] for v in variants if (c, v) in tbl])),
                   "wall": float(np.nanmedian([tbl[(c, v)]["wall_med"] for v in variants if (c, v) in tbl]))}
               for c in configs}
    best_acc = max(p["acc"] for p in per_cfg.values())
    if best_acc < CHANCE_BAND:
        return None, None, per_cfg, "AMBIGUOUS: best accuracy below chance band"
    # configs statistically tied with the best (CI overlap on the best's lower bound)
    best_cfg = max(per_cfg, key=lambda c: per_cfg[c]["acc"])
    tied = [c for c in configs
            if per_cfg[c]["hi"] >= per_cfg[best_cfg]["lo"] and
            per_cfg[c]["acc"] >= per_cfg[best_cfg]["acc"] - 0.03]
    all_cfgs = dict(cfg.CONFIGS)
    all_cfgs.update(cfg.pilot_config_grid())
    cost = {c: all_cfgs[c].track_sweeps * (all_cfgs[c].rot_angle_max_deg /
                                           all_cfgs[c].rot_angle_step_deg) ** 2
            for c in tied}
    chosen_cfg = min(tied, key=lambda c: (cost[c], -per_cfg[c]["acc"]))

    # variant choice for the chosen config
    v_acc = {v: tbl[(chosen_cfg, v)] for v in variants if (chosen_cfg, v) in tbl}
    best_v = max(v_acc, key=lambda v: v_acc[v]["acc"])
    tiled = [v for v in ("tiledA", "tiledB") if v in v_acc]
    if len(tiled) == 2:
        tiled_mean = np.mean([v_acc[v]["acc"] for v in tiled])
        if tiled_mean >= v_acc[best_v]["acc_lo"]:
            return chosen_cfg, tiled, per_cfg, "tiled pair within CI of best single variant"
    return chosen_cfg, [best_v], per_cfg, f"single best variant {best_v}"


def main():
    out_root = cfg.RESULTS_DIR / "windows"
    rows = load_rows(out_root)
    if not rows:
        print("no pilot results found")
        sys.exit(2)
    tbl = table(rows)
    chosen_cfg, chosen_variants, per_cfg, note = decide(tbl)

    print(f"{'config':14s} {'variant':9s} {'n':>3s} {'acc':>6s} {'CI':>15s} "
          f"{'jacc':>6s} {'t_w(s)':>7s}")
    for (c, v), s in sorted(tbl.items()):
        print(f"{c:14s} {v:9s} {s['n']:3d} {s['acc']:6.3f} "
              f"[{s['acc_lo']:.3f},{s['acc_hi']:.3f}] {s['jacc']:6.3f} {s['wall_med']:7.1f}")

    decision = {"chosen_config": chosen_cfg, "chosen_variants": chosen_variants,
                "note": note, "per_config": per_cfg,
                "table": {f"{c}|{v}": s for (c, v), s in tbl.items()}}
    if chosen_cfg:
        t_w = per_cfg[chosen_cfg]["wall"]
        n_windows = cfg.N_STIMULI * len(chosen_variants)
        gpu_h = n_windows * t_w / 3600
        decision["projection"] = {
            "t_w_s": t_w, "n_windows": n_windows, "gpu_hours": round(gpu_h, 1),
            "wall_hours_at_4gpus": round(gpu_h / N_GPUS, 1)}
        print(f"\nDECISION: config={chosen_cfg} variants={chosen_variants} ({note})")
        print(f"projection: {n_windows} windows x {t_w:.0f}s = {gpu_h:.1f} GPU-h "
              f"≈ {gpu_h/N_GPUS:.1f} h at {N_GPUS} GPUs")
    else:
        print(f"\nNO DECISION: {note}")
    out = cfg.assert_writable_path(cfg.RESULTS_DIR / "pilot_decision.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(decision, indent=1))
    print(f"wrote {out}")
    sys.exit(0 if chosen_cfg else 4)


if __name__ == "__main__":
    main()
