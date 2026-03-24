#!/usr/bin/env python3
"""Human vs model correlation plot from psychophysics ``*_benchmark.json``.

By default, processes every ``*_benchmark.json`` under ``results/psychophysics/`` (full run and
ablations). Use ``--input`` for a single file.
"""

from __future__ import annotations

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


def _discover_benchmark_jsons(psych_dir: Path) -> list[Path]:
    if not psych_dir.is_dir():
        return []
    return sorted(psych_dir.glob("*_benchmark.json"))


def _render_correlation_plot(
    inp: Path,
    out_png: Path | None,
    dpi: int,
) -> Path:
    """Load benchmark JSON, write one PNG. Raises ``ValueError`` if too few paired points."""
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
        raise ValueError(
            f"need at least two stimuli with human results for correlation plot (got {len(human_values)})"
        )

    human_values = np.array(human_values, dtype=np.float64)
    model_values = np.array(model_values, dtype=np.float64)
    correlation, _p_value = pearsonr(human_values, model_values)
    r_squared = float(correlation**2)

    plt.style.use("seaborn-v0_8-whitegrid")
    rcParams["axes.facecolor"] = "white"
    rcParams["figure.facecolor"] = "white"
    rcParams["grid.color"] = "#e0e0e0"
    rcParams["grid.linewidth"] = 0.8
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
    if out_png is not None:
        final_png = Path(out_png)
    else:
        stem = inp.stem.replace("_benchmark", "")
        final_png = out_post / f"psychophysics_correlation_{stem}.png"

    final_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(final_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return final_png


def main() -> int:
    psych_default = config.PSYCHOPHYSICS_OUTPUT_DIR
    parser = argparse.ArgumentParser(
        description="Psychophysics human–model correlation plots from benchmark JSON(s)"
    )
    parser.add_argument(
        "--psychophysics-dir",
        type=Path,
        default=psych_default,
        help=f"Directory to scan for *_benchmark.json when --input is omitted (default: {psych_default})",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Single benchmark JSON (if omitted, all *_benchmark.json under --psychophysics-dir)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output PNG path (only with --input; default: postprocessing/psychophysics_correlation_<stem>.png)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolution for PNG output (matches psychophysics_benchmark.ipynb savefig)",
    )
    args = parser.parse_args()

    if args.input is None and args.output is not None:
        print("Error: --output requires --input", file=sys.stderr)
        return 1

    if args.input is not None:
        inp = Path(args.input).resolve()
        if not inp.is_file():
            print(f"Error: not found: {inp}", file=sys.stderr)
            return 1
        out_arg = Path(args.output).resolve() if args.output else None
        try:
            out = _render_correlation_plot(inp, out_arg, args.dpi)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        print(f"Wrote {out}")
        return 0

    psych_dir = args.psychophysics_dir.resolve()
    paths = _discover_benchmark_jsons(psych_dir)
    if not paths:
        print(
            f"No *_benchmark.json files under {psych_dir}",
            file=sys.stderr,
        )
        return 1

    ok = 0
    for inp in paths:
        try:
            out = _render_correlation_plot(inp, None, args.dpi)
            print(f"Wrote {out}")
            ok += 1
        except ValueError as e:
            print(f"Skip {inp.name}: {e}", file=sys.stderr)

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
