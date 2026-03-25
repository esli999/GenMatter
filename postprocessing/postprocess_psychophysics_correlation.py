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


def _term_styles() -> dict[str, str]:
    if not sys.stdout.isatty():
        return {
            k: ""
            for k in (
                "reset",
                "bold",
                "dim",
                "cyan",
                "green",
                "magenta",
                "yellow",
            )
        }
    return {
        "reset": "\033[0m",
        "bold": "\033[1m",
        "dim": "\033[2m",
        "cyan": "\033[96m",
        "green": "\033[92m",
        "magenta": "\033[95m",
        "yellow": "\033[93m",
    }


def _load_paired_benchmark(inp: Path) -> tuple[np.ndarray, np.ndarray, list[bool]] | None:
    """Return (human %, model %, gt_same_object flags) or None if <2 paired stimuli."""
    with open(inp, "r") as f:
        data = json.load(f)

    stimuli = data["stimuli"]
    human_results = data.get("human_results", {})

    human_values: list[float] = []
    model_values: list[float] = []
    gt_flags: list[bool] = []

    for stim_id in sorted(stimuli.keys()):
        if stim_id not in human_results:
            continue
        human_values.append(human_results[stim_id])
        model_values.append(stimuli[stim_id]["model_percent"])
        gt = stimuli[stim_id]["ground_truth"]
        gt_flags.append(bool(gt) if isinstance(gt, bool) else bool(int(gt)))

    if len(human_values) < 2:
        return None

    return (
        np.array(human_values, dtype=np.float64),
        np.array(model_values, dtype=np.float64),
        gt_flags,
    )


def _benchmark_label(inp: Path) -> str:
    return inp.stem.replace("_benchmark", "")


def _print_r2_summary_table(
    rows: list[tuple[str, int, float, float]],
) -> None:
    """Pretty-print R / R² table (same metrics as plot annotation). *rows*: (label, n, r, r²)."""
    if not rows:
        return
    t = _term_styles()
    sep = "=" * 78
    print(f"\n{sep}")
    print(
        f"{t['cyan']}{t['bold']}Psychophysics — human vs model correlation{t['reset']}"
    )
    print(
        f"{t['dim']}(Pearson r and R²; R² matches the value shown on each plot){t['reset']}"
    )
    print(sep)
    w_name = max(28, max(len(r[0]) for r in rows))
    hdr = (
        f"{t['magenta']}{'Benchmark':<{w_name}} {'N':>4} "
        f"{'r':>9} {'R²':>9}{t['reset']}"
    )
    print(hdr)
    print(t["dim"] + "-" * (w_name + 4 + 9 + 9 + 3) + t["reset"])

    best_i = max(range(len(rows)), key=lambda i: rows[i][3])
    for i, (label, n, r_val, r2_val) in enumerate(rows):
        is_best = len(rows) > 1 and i == best_i
        r_s = f"{r_val:+.4f}"
        r2_s = f"{r2_val:.4f}"
        if is_best:
            line = (
                f"{label:<{w_name}} {n:>4} "
                f"{t['bold']}{t['green']}{r_s:>9}{t['reset']} "
                f"{t['bold']}{t['green']}{r2_s:>9}{t['reset']}"
            )
        else:
            line = f"{label:<{w_name}} {n:>4} {r_s:>9} {r2_s:>9}"
        print(line)
    print()


def _discover_benchmark_jsons(psych_dir: Path) -> list[Path]:
    if not psych_dir.is_dir():
        return []
    return sorted(psych_dir.glob("*_benchmark.json"))


def _render_correlation_plot(
    inp: Path,
    out_png: Path | None,
    dpi: int,
) -> tuple[Path, float, float, int]:
    """Load benchmark JSON, write one PNG.

    Returns ``(png_path, pearson_r, r_squared, n_points)``.
    Raises ``ValueError`` if too few paired points.
    """
    loaded = _load_paired_benchmark(inp)
    if loaded is None:
        raise ValueError(
            "need at least two stimuli with human results for correlation plot"
        )
    human_values, model_values, gt_flags = loaded
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
    return final_png, float(correlation), r_squared, len(human_values)


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
        print("=" * 80)
        print("Psychophysics correlation (single benchmark)")
        print(f"  Input: {inp}")
        print("=" * 80)
        out_arg = Path(args.output).resolve() if args.output else None
        try:
            out, r_val, r2_val, n_pts = _render_correlation_plot(inp, out_arg, args.dpi)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        tw = _term_styles()
        print(f"{tw['dim']}Wrote {out}{tw['reset']}")
        _print_r2_summary_table([(_benchmark_label(inp), n_pts, r_val, r2_val)])
        return 0

    psych_dir = args.psychophysics_dir.resolve()
    paths = _discover_benchmark_jsons(psych_dir)
    print("=" * 80)
    print("Psychophysics correlation (RDK benchmarks)")
    print(f"  Scanning: {psych_dir}")
    print(f"  Found {len(paths)} file(s): {[p.name for p in paths]}")
    print(f"  Output PNGs: {config.POSTPROCESSING_OUTPUT_DIR}/psychophysics_correlation_*.png")
    print("=" * 80)
    print()
    if not paths:
        print(
            f"No *_benchmark.json files under {psych_dir}",
            file=sys.stderr,
        )
        return 1

    ok = 0
    table_rows: list[tuple[str, int, float, float]] = []
    for inp in paths:
        try:
            out, r_val, r2_val, n_pts = _render_correlation_plot(inp, None, args.dpi)
            table_rows.append((_benchmark_label(inp), n_pts, r_val, r2_val))
            tw = _term_styles()
            print(f"{tw['dim']}Wrote {out}{tw['reset']}")
            ok += 1
        except ValueError as e:
            print(f"Skip {inp.name}: {e}", file=sys.stderr)

    _print_r2_summary_table(table_rows)

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
