#!/usr/bin/env python3
"""Human vs model correlation plot from psychophysics ``*_benchmark.json``.

Default input: ``<GENMATTER_RESULTS_DIR>/psychophysics/full_benchmark.json`` (see ``config.PSYCHOPHYSICS_OUTPUT_DIR``).
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from scipy.stats import pearsonr

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
import config


def main():
    parser = argparse.ArgumentParser(description="Psychophysics human–model correlation plot")
    default_json = config.PSYCHOPHYSICS_OUTPUT_DIR / "full_benchmark.json"
    parser.add_argument(
        "--input",
        type=str,
        default=str(default_json),
        help=f"Benchmark JSON (default: {default_json})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output PNG path (default: results/postprocessing/psychophysics_correlation_<stem>.png)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolution for PNG output (matches psychophysics_benchmark.ipynb savefig)",
    )
    args = parser.parse_args()

    inp = Path(args.input)
    if not inp.is_file():
        print(f"Error: not found: {inp}", file=sys.stderr)
        sys.exit(1)

    with open(inp, "r") as f:
        data = json.load(f)

    stimuli = data["stimuli"]
    human_results = data.get("human_results", {})

    human_values = []
    model_values = []
    gt_flags = []

    for stim_id in sorted(stimuli.keys()):
        if stim_id not in human_results:
            continue
        human_values.append(human_results[stim_id])
        model_values.append(stimuli[stim_id]["model_percent"])
        gt = stimuli[stim_id]["ground_truth"]
        gt_flags.append(bool(gt) if isinstance(gt, bool) else bool(int(gt)))

    if len(human_values) < 2:
        print("Error: need at least two stimuli with human results for correlation plot.", file=sys.stderr)
        sys.exit(1)

    human_values = np.array(human_values, dtype=np.float64)
    model_values = np.array(model_values, dtype=np.float64)
    correlation, _p_value = pearsonr(human_values, model_values)
    r_squared = float(correlation**2)

    plt.style.use("seaborn-v0_8-whitegrid")
    rcParams["axes.facecolor"] = "white"
    rcParams["figure.facecolor"] = "white"
    rcParams["grid.color"] = "#e0e0e0"
    rcParams["grid.linewidth"] = 0.8
    # Bundled with matplotlib; avoids DM Sans / system-font lookups.
    rcParams["font.family"] = "DejaVu Sans"

    fig, ax = plt.subplots(figsize=(12, 12), dpi=100)
    for i in range(len(human_values)):
        if gt_flags[i]:
            ax.scatter(
                human_values[i],
                model_values[i],
                s=250,
                c="#1e88e5",
                marker="o",
                edgecolors="black",
                linewidths=1.5,
                alpha=0.9,
                zorder=5,
            )
        else:
            ax.scatter(
                human_values[i],
                model_values[i],
                s=250,
                c="#ff5722",
                marker="^",
                edgecolors="black",
                linewidths=1.5,
                alpha=0.9,
                zorder=5,
            )

    min_val = min(float(human_values.min()), float(model_values.min()))
    max_val = max(float(human_values.max()), float(model_values.max()))
    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        "--",
        color="#2c3e50",
        alpha=0.7,
        linewidth=2.5,
        zorder=1,
    )

    ax.set_xlim(min_val - 5, max_val + 5)
    ax.set_ylim(min_val - 5, max_val + 5)
    ax.grid(True, linestyle="--", alpha=0.5, linewidth=1.0)

    ax.set_xlabel(
        "Participant Prediction (%)",
        fontsize=22,
        fontweight="bold",
        labelpad=20,
    )
    ax.set_ylabel(
        "GenMatter Prediction (%)",
        fontsize=22,
        fontweight="bold",
        labelpad=20,
    )
    ax.set_title(
        "Are the Red and Green Dots on the Same Object?",
        fontsize=28,
        fontweight="bold",
        pad=30,
    )

    r2_box = dict(
        boxstyle="round,pad=0.6",
        facecolor="white",
        alpha=0.9,
        edgecolor="#1e88e5",
        linewidth=2,
    )
    ax.text(
        0.05,
        0.90,
        f"$R^2 = {r_squared:.2f}$",
        transform=ax.transAxes,
        fontsize=28,
        fontweight="bold",
        bbox=r2_box,
    )

    ax.scatter(
        [],
        [],
        s=150,
        c="#1e88e5",
        marker="o",
        edgecolors="white",
        linewidths=1.5,
        label="Ground Truth: Same Object",
    )
    ax.scatter(
        [],
        [],
        s=150,
        c="#ff5722",
        marker="^",
        edgecolors="white",
        linewidths=1.5,
        label="Ground Truth: Different Objects",
    )
    legend = ax.legend(
        loc="lower right",
        fontsize=18,
        frameon=True,
        framealpha=0.95,
        edgecolor="#cccccc",
        borderpad=1.1,
    )
    legend.get_frame().set_linewidth(2)

    ax.tick_params(axis="both", which="major", labelsize=18, width=2, length=10, pad=10)
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)
        spine.set_color("#2c3e50")

    ax.set_axisbelow(True)
    ax.patch.set_alpha(1.0)

    out_post = config.POSTPROCESSING_OUTPUT_DIR
    out_post.mkdir(parents=True, exist_ok=True)
    if args.output:
        out_png = Path(args.output)
    else:
        stem = inp.stem.replace("_benchmark", "")
        out_png = out_post / f"psychophysics_correlation_{stem}.png"

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
