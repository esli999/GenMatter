"""Aggregate per-window results into CSV + condition summaries with bootstrap CIs.

Usage: uv run python experiments/sfm/aggregate.py [--out-root .../results/windows]
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.sfm import sfm_config as cfg
from genmatter.bootstrap_stats import bootstrap_mean_ci_95


def collect(out_root: Path):
    rows = []
    for rj in sorted(out_root.glob("*/*/*/results.json")):
        d = json.loads(rj.read_text())
        config, variant = rj.parts[-4], rj.parts[-3]
        s, o, t, v = cfg.condition_of(d["stim_id"])
        rows.append({
            "config": config, "variant": variant, "stim_id": d["stim_id"],
            "object": f"scene_{o:05d}", "texture": cfg.TEXTURES[t],
            "viewpoint": v, "size_deg": cfg.SIZES_DEG[s],
            "mean_probe_accuracy": d["mean_probe_accuracy"],
            "mean_roi_jaccard": d["mean_roi_jaccard"],
            "wall_s": d.get("wall_s"),
            **{f"acc_f{f['frame']}": f["probe_accuracy"] for f in d["frames"]},
            **{f"jacc_f{f['frame']}": f["roi_jaccard"] for f in d["frames"]},
        })
    return rows


def summarize(rows, keys):
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in ("config", "variant") + keys)].append(r)
    out = []
    for gk, rs in sorted(groups.items()):
        accs = [r["mean_probe_accuracy"] for r in rs]
        jacs = [r["mean_roi_jaccard"] for r in rs]
        am, alo, ahi = bootstrap_mean_ci_95(accs)
        jm, jlo, jhi = bootstrap_mean_ci_95(jacs)
        out.append({
            **dict(zip(("config", "variant") + keys, gk)), "n": len(rs),
            "accuracy": round(float(am), 4),
            "accuracy_ci": [round(float(alo), 4), round(float(ahi), 4)],
            "jaccard": round(float(jm), 4),
            "jaccard_ci": [round(float(jlo), 4), round(float(jhi), 4)],
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=str(cfg.RESULTS_DIR / "windows"))
    args = ap.parse_args()

    rows = collect(Path(args.out_root))
    if not rows:
        print("no results found")
        return
    agg_dir = cfg.assert_writable_path(cfg.RESULTS_DIR / "aggregated")
    agg_dir.mkdir(parents=True, exist_ok=True)

    fields = sorted({k for r in rows for k in r})
    with open(agg_dir / "windows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    summary = {
        "n_windows": len(rows),
        "overall": summarize(rows, ()),
        "by_texture": summarize(rows, ("texture",)),
        "by_size": summarize(rows, ("size_deg",)),
        "by_viewpoint": summarize(rows, ("viewpoint",)),
        "wall_s_median": float(np.median([r["wall_s"] for r in rows if r["wall_s"]])),
    }
    (agg_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({"n_windows": len(rows), "overall": summary["overall"],
                      "wall_s_median": summary["wall_s_median"]}, indent=1))
    print(f"wrote {agg_dir}/windows.csv and summary.json")


if __name__ == "__main__":
    main()
