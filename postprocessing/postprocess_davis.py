"""DAVIS tracking postprocessing: compare GenMatter, CoTracker, and subsampling ablations.

Loads per-video JSON results from DAVIS tracking, subsampling, ablation, and
CoTracker experiments.  Extracts matter-weighted Jaccard, precision, recall,
F1, FPS, FPR, FNR per video, aggregates across videos, and writes:

- results/postprocessing/davis_comparison.json   (all methods side-by-side)
- results/postprocessing/davis_subsampling_tradeoff.json  (subsample % vs perf)
- results/postprocessing/davis_results.csv
"""

import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DAVIS_VIDEOS = list(config.TAPVID_DAVIS_VIDEO_NAMES)

SUBSAMPLE_DIRS: dict[str, float] = {
    "subsample_12_5": 12.5,
    "subsample_6_25": 6.25,
    "subsample_3_125": 3.125,
    "subsample_1_5625": 1.5625,
    "subsample_0_78125": 0.78125,
    "subsample_0_390625": 0.390625,
    "subsample_0_1953125": 0.1953125,
    "subsample_0_09765625": 0.09765625,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _safe_std(values: list[float]) -> float:
    return float(np.std(values)) if values else float("nan")


def compute_particle_f1(
    fn_rate: float, fp_rate: float, n_object: int, n_background: int
) -> tuple[float, float, float]:
    """Compute recall, precision, F1 from particle-based FP/FN rates.

    FN/FP rates are fractions (0-1).  Returns (recall, precision, f1).
    """
    fn_count = fn_rate * n_object
    fp_count = fp_rate * n_background
    tp_count = n_object - fn_count

    recall = 1.0 - fn_rate if n_object > 0 else 0.0
    precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0

    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0
    return recall, precision, f1


# ---------------------------------------------------------------------------
# CoTracker loader
# ---------------------------------------------------------------------------


def load_cotracker_results(base_dir: Path) -> dict[str, dict[str, Any]]:
    """Load CoTracker per-video results from individual JSON files.

    Looks for ``{video}_results.json`` files inside *base_dir* (or a
    ``cotracker_TAPVID_even_init`` sub-directory if present).
    """
    candidates = [base_dir, base_dir / "cotracker_TAPVID_even_init"]
    results_dir = None
    for c in candidates:
        if c.is_dir() and any(c.glob("*_results.json")):
            results_dir = c
            break

    if results_dir is None:
        # Try loading all_videos_summary.json
        summary = base_dir / "all_videos_summary.json"
        for c in candidates:
            s = c / "all_videos_summary.json"
            if s.exists():
                summary = s
                break
        if summary.exists():
            with open(summary) as f:
                raw = json.load(f)
            out: dict[str, dict[str, Any]] = {}
            for video, vdata in raw.items():
                if not isinstance(vdata, dict):
                    continue
                out[video] = {
                    "mean_jaccard": vdata.get("mean_jaccard", float("nan")),
                    "mean_precision": vdata.get("mean_precision", float("nan")),
                    "mean_recall": vdata.get("mean_recall", float("nan")),
                    "mean_f1": vdata.get("mean_f1", float("nan")),
                    "mean_fn_rate": vdata.get("mean_fn_rate", float("nan")),
                    "mean_fp_rate": vdata.get("mean_fp_rate", float("nan")),
                    "n_object_particles": vdata.get("n_object_particles", 0),
                    "n_background_particles": vdata.get("n_background_particles", 0),
                }
            return out
        print(f"  [WARN] No CoTracker results found in {base_dir}")
        return {}

    out = {}
    for jf in sorted(results_dir.glob("*_results.json")):
        video = jf.stem.replace("_results", "")
        with open(jf) as f:
            data = json.load(f)
        out[video] = {
            "mean_jaccard": data.get("mean_jaccard", float("nan")),
            "mean_precision": data.get("mean_precision", float("nan")),
            "mean_recall": data.get("mean_recall", float("nan")),
            "mean_f1": data.get("mean_f1", float("nan")),
            "mean_fn_rate": data.get("mean_fn_rate", float("nan")),
            "mean_fp_rate": data.get("mean_fp_rate", float("nan")),
            "n_object_particles": data.get("n_object_particles", 0),
            "n_background_particles": data.get("n_background_particles", 0),
        }
    return out


# ---------------------------------------------------------------------------
# DINO tracking loader (main & ablation)
# ---------------------------------------------------------------------------


def load_dino_tracking_results(base_dir: Path) -> dict[str, dict[str, Any]]:
    """Load DINO tracking per-video results.

    Supports two formats:
    - Individual ``json_results/{video}_results.json`` (subsampling dense eval)
    - Aggregated ``all_videos_experiment_results.json``
    """
    json_results_dir = base_dir / "json_results"
    results: dict[str, dict[str, Any]] = {}

    if json_results_dir.is_dir():
        for jf in sorted(json_results_dir.glob("*_results.json")):
            video = jf.stem.replace("_results", "")
            with open(jf) as f:
                data = json.load(f)
            results[video] = _extract_dino_metrics(data)
    elif (base_dir / "all_videos_experiment_results.json").exists():
        with open(base_dir / "all_videos_experiment_results.json") as f:
            raw = json.load(f)
        for video, vdata in raw.items():
            if isinstance(vdata, dict):
                results[video] = _extract_dino_metrics(vdata)

    if not results:
        print(f"  [WARN] No DINO tracking results found in {base_dir}")
    return results


def _extract_dino_metrics(data: dict) -> dict[str, Any]:
    """Extract a uniform metrics dict from either per-video or subsampling JSON."""
    pm = data.get("pixel_metrics", {})

    matter_jaccard = (
        pm.get("avg_matter_weighted_jaccard_fixed")
        or data.get("particle_count_matter_fixed_jaccard")
    )
    matter_recall = (
        pm.get("avg_matter_weighted_recall_fixed")
        or data.get("matter_weighted_recall_fixed")
        or data.get("particle_count_matter_fixed_recall")
    )
    matter_precision = (
        pm.get("avg_matter_weighted_precision_fixed")
        or data.get("matter_weighted_precision_fixed")
        or data.get("particle_count_matter_fixed_precision")
    )
    matter_f1 = (
        pm.get("avg_matter_weighted_f1_fixed")
        or data.get("matter_weighted_f1_fixed")
        or data.get("particle_count_matter_fixed_f1")
    )

    fps = data.get("fps") or pm.get("fps_mean")
    fn_rate = data.get("mean_fn_rate") or data.get("particle_fn_rate")
    fp_rate = data.get("mean_fp_rate") or data.get("particle_fp_rate")

    n_obj = data.get("n_object_blobs") or data.get("n_object_particles", 0)
    n_bg = data.get("n_background_blobs") or data.get("n_background_particles", 0)

    return {
        "matter_weighted_jaccard": _to_float(matter_jaccard),
        "matter_weighted_recall": _to_float(matter_recall),
        "matter_weighted_precision": _to_float(matter_precision),
        "matter_weighted_f1": _to_float(matter_f1),
        "fps": _to_float(fps),
        "fn_rate": _to_float(fn_rate),
        "fp_rate": _to_float(fp_rate),
        "n_object_particles": int(n_obj) if n_obj else 0,
        "n_background_particles": int(n_bg) if n_bg else 0,
    }


def _to_float(v: Any) -> float:
    if v is None:
        return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


# ---------------------------------------------------------------------------
# Subsampling loader
# ---------------------------------------------------------------------------


def load_subsampling_results(
    base_dir: Path,
) -> dict[float, dict[str, dict[str, Any]]]:
    """Load subsampling results across all subsample percentages.

    Returns ``{percentage: {video: metrics_dict}}``.
    """
    sorted_dirs = sorted(SUBSAMPLE_DIRS.items(), key=lambda x: x[1], reverse=True)
    out: dict[float, dict[str, dict[str, Any]]] = {}

    for dir_name, percentage in sorted_dirs:
        json_dir = base_dir / dir_name / "json_results"
        if not json_dir.is_dir():
            continue
        per_video: dict[str, dict[str, Any]] = {}
        for jf in sorted(json_dir.glob("*_results.json")):
            video = jf.stem.replace("_results", "")
            try:
                with open(jf) as f:
                    data = json.load(f)
                per_video[video] = _extract_dino_metrics(data)
            except Exception:
                continue
        if per_video:
            out[percentage] = per_video

    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_per_video(
    results: dict[str, dict[str, Any]], metric_key: str
) -> tuple[float, float, list[float]]:
    """Return (mean, std, values) of *metric_key* across videos."""
    vals = [
        r[metric_key]
        for r in results.values()
        if not np.isnan(r.get(metric_key, float("nan")))
    ]
    return _safe_mean(vals), _safe_std(vals), vals


def build_comparison(
    dino_results: dict[str, dict[str, Any]],
    cotracker_results: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build per-video comparison between DINO tracking and CoTracker."""
    all_videos = sorted(set(dino_results) | set(cotracker_results))
    rows: list[dict[str, Any]] = []

    for video in all_videos:
        dino = dino_results.get(video, {})
        ct = cotracker_results.get(video, {})

        ct_fn_rate = ct.get("mean_fn_rate", float("nan"))
        ct_fp_rate = ct.get("mean_fp_rate", float("nan"))
        ct_n_obj = ct.get("n_object_particles", 0)
        ct_n_bg = ct.get("n_background_particles", 0)
        if not np.isnan(ct_fn_rate) and not np.isnan(ct_fp_rate):
            ct_fn_frac = ct_fn_rate / 100.0
            ct_fp_frac = ct_fp_rate / 100.0
            ct_recall, ct_precision, ct_f1 = compute_particle_f1(
                ct_fn_frac, ct_fp_frac, ct_n_obj, ct_n_bg
            )
        else:
            ct_recall = ct.get("mean_recall", float("nan"))
            ct_precision = ct.get("mean_precision", float("nan"))
            ct_f1 = ct.get("mean_f1", float("nan"))

        dino_fn_rate = dino.get("fn_rate", float("nan"))
        dino_fp_rate = dino.get("fp_rate", float("nan"))
        dino_n_obj = dino.get("n_object_particles", 0)
        dino_n_bg = dino.get("n_background_particles", 0)
        if not np.isnan(dino_fn_rate) and not np.isnan(dino_fp_rate):
            dino_fn_frac = dino_fn_rate / 100.0
            dino_fp_frac = dino_fp_rate / 100.0
            dino_p_recall, dino_p_precision, dino_p_f1 = compute_particle_f1(
                dino_fn_frac, dino_fp_frac, dino_n_obj, dino_n_bg
            )
        else:
            dino_p_recall = dino_p_precision = dino_p_f1 = float("nan")

        row: dict[str, Any] = {
            "video": video,
            "cotracker_jaccard": ct.get("mean_jaccard", float("nan")),
            "cotracker_precision": ct_precision,
            "cotracker_recall": ct_recall,
            "cotracker_f1": ct_f1,
            "cotracker_fn_rate": ct.get("mean_fn_rate", float("nan")),
            "cotracker_fp_rate": ct.get("mean_fp_rate", float("nan")),
            "dino_matter_jaccard": dino.get("matter_weighted_jaccard", float("nan")),
            "dino_matter_precision": dino.get(
                "matter_weighted_precision", float("nan")
            ),
            "dino_matter_recall": dino.get("matter_weighted_recall", float("nan")),
            "dino_matter_f1": dino.get("matter_weighted_f1", float("nan")),
            "dino_particle_recall": dino_p_recall,
            "dino_particle_precision": dino_p_precision,
            "dino_particle_f1": dino_p_f1,
            "dino_fps": dino.get("fps", float("nan")),
            "dino_fn_rate": dino.get("fn_rate", float("nan")),
            "dino_fp_rate": dino.get("fp_rate", float("nan")),
        }
        rows.append(row)

    return rows


def build_subsampling_tradeoff(
    subsampling: dict[float, dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Aggregate subsampling results per percentage level."""
    rows: list[dict[str, Any]] = []

    for pct in sorted(subsampling.keys(), reverse=True):
        per_video = subsampling[pct]
        jaccards = [
            m["matter_weighted_jaccard"]
            for m in per_video.values()
            if not np.isnan(m.get("matter_weighted_jaccard", float("nan")))
        ]
        fps_vals = [
            m["fps"]
            for m in per_video.values()
            if not np.isnan(m.get("fps", float("nan")))
        ]
        recalls = [
            m["matter_weighted_recall"]
            for m in per_video.values()
            if not np.isnan(m.get("matter_weighted_recall", float("nan")))
        ]
        precisions = [
            m["matter_weighted_precision"]
            for m in per_video.values()
            if not np.isnan(m.get("matter_weighted_precision", float("nan")))
        ]
        f1s = [
            m["matter_weighted_f1"]
            for m in per_video.values()
            if not np.isnan(m.get("matter_weighted_f1", float("nan")))
        ]

        rows.append(
            {
                "subsample_pct": pct,
                "n_videos": len(per_video),
                "jaccard_mean": _safe_mean(jaccards),
                "jaccard_std": _safe_std(jaccards),
                "fps_mean": _safe_mean(fps_vals),
                "fps_std": _safe_std(fps_vals),
                "recall_mean": _safe_mean(recalls),
                "recall_std": _safe_std(recalls),
                "precision_mean": _safe_mean(precisions),
                "precision_std": _safe_std(precisions),
                "f1_mean": _safe_mean(f1s),
                "f1_std": _safe_std(f1s),
                "per_video": {
                    video: {
                        "jaccard": m["matter_weighted_jaccard"],
                        "fps": m["fps"],
                        "recall": m["matter_weighted_recall"],
                        "precision": m["matter_weighted_precision"],
                        "f1": m["matter_weighted_f1"],
                    }
                    for video, m in per_video.items()
                },
            }
        )

    return rows


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def _default(o: Any) -> Any:
        if isinstance(o, (np.floating, np.float64, np.float32)):
            return float(o)
        if isinstance(o, (np.integer, np.int64, np.int32)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"Object of type {type(o)} is not JSON serializable")

    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_default)
    print(f"  Wrote {path}")


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        print(f"  [SKIP] No data for {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {path}")


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------


def print_summary(
    comparison: list[dict[str, Any]],
    subsampling_tradeoff: list[dict[str, Any]],
) -> None:
    """Print human-readable summary tables to stdout."""

    sep = "=" * 100

    # --- Method comparison ---
    print(f"\n{sep}")
    print("DAVIS TRACKING: METHOD COMPARISON")
    print(sep)

    ct_jaccards = [
        r["cotracker_jaccard"]
        for r in comparison
        if not np.isnan(r.get("cotracker_jaccard", float("nan")))
    ]
    dino_jaccards = [
        r["dino_matter_jaccard"]
        for r in comparison
        if not np.isnan(r.get("dino_matter_jaccard", float("nan")))
    ]
    dino_fps_vals = [
        r["dino_fps"]
        for r in comparison
        if not np.isnan(r.get("dino_fps", float("nan")))
    ]

    print(f"\n{'Method':<25} {'Jaccard':>12} {'Std':>10} {'#Videos':>10}")
    print("-" * 60)
    if ct_jaccards:
        print(
            f"{'CoTracker':<25} {_safe_mean(ct_jaccards):>12.4f} "
            f"{_safe_std(ct_jaccards):>10.4f} {len(ct_jaccards):>10}"
        )
    if dino_jaccards:
        print(
            f"{'DINO Tracking':<25} {_safe_mean(dino_jaccards):>12.4f} "
            f"{_safe_std(dino_jaccards):>10.4f} {len(dino_jaccards):>10}"
        )

    # Per-video detail
    print(f"\n{'Video':<22} {'CT Jaccard':>12} {'DINO Jaccard':>14} {'DINO FPS':>10}")
    print("-" * 62)
    for r in comparison:
        ct_j = r.get("cotracker_jaccard", float("nan"))
        d_j = r.get("dino_matter_jaccard", float("nan"))
        d_fps = r.get("dino_fps", float("nan"))
        ct_str = f"{ct_j:.4f}" if not np.isnan(ct_j) else "N/A"
        dj_str = f"{d_j:.4f}" if not np.isnan(d_j) else "N/A"
        fps_str = f"{d_fps:.2f}" if not np.isnan(d_fps) else "N/A"
        print(f"{r['video']:<22} {ct_str:>12} {dj_str:>14} {fps_str:>10}")

    # --- Subsampling tradeoff ---
    if subsampling_tradeoff:
        print(f"\n{sep}")
        print("DAVIS TRACKING: SUBSAMPLING TRADEOFF")
        print(sep)
        print(
            f"\n{'Subsample %':>14} {'Jaccard':>10} {'Std':>8} "
            f"{'FPS':>8} {'Std':>8} {'#Videos':>8}"
        )
        print("-" * 62)
        for row in subsampling_tradeoff:
            print(
                f"{row['subsample_pct']:>14.4f} "
                f"{row['jaccard_mean']:>10.4f} {row['jaccard_std']:>8.4f} "
                f"{row['fps_mean']:>8.2f} {row['fps_std']:>8.2f} "
                f"{row['n_videos']:>8}"
            )

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    out_dir = config.POSTPROCESSING_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load data --------------------------------------------------------
    print("Loading results...")

    print("  CoTracker baseline:")
    ct_results = load_cotracker_results(config.COTRACKER_OUTPUT_DIR)

    print("  DINO tracking (main):")
    dino_results = load_dino_tracking_results(config.DAVIS_TRACKING_OUTPUT_DIR)

    print("  DINO subsampling (SAM):")
    sub_sam = load_subsampling_results(config.DAVIS_SUBSAMPLING_OUTPUT_DIR)

    print("  DINO subsampling (no SAM / ablation):")
    sub_no_sam = load_subsampling_results(config.DAVIS_ABLATION_OUTPUT_DIR)

    # ---- Build comparison -------------------------------------------------
    print("\nBuilding comparison...")
    comparison = build_comparison(dino_results, ct_results)

    # ---- Build subsampling tradeoff (prefer SAM, fallback to no-SAM) ------
    subsampling_source = sub_sam if sub_sam else sub_no_sam
    subsampling_tradeoff = build_subsampling_tradeoff(subsampling_source)

    # If both SAM and no-SAM available, include both in the tradeoff JSON
    combined_tradeoff: dict[str, Any] = {}
    if sub_sam:
        combined_tradeoff["sam"] = build_subsampling_tradeoff(sub_sam)
    if sub_no_sam:
        combined_tradeoff["no_sam"] = build_subsampling_tradeoff(sub_no_sam)
    if not combined_tradeoff:
        combined_tradeoff["default"] = subsampling_tradeoff

    # ---- Print summary to stdout ------------------------------------------
    print_summary(comparison, subsampling_tradeoff)

    # ---- Write outputs ----------------------------------------------------
    print("Writing outputs...")

    comparison_out = {
        "per_video": comparison,
        "summary": {
            "cotracker": {
                "jaccard_mean": _safe_mean(
                    [
                        r["cotracker_jaccard"]
                        for r in comparison
                        if not np.isnan(r.get("cotracker_jaccard", float("nan")))
                    ]
                ),
                "jaccard_std": _safe_std(
                    [
                        r["cotracker_jaccard"]
                        for r in comparison
                        if not np.isnan(r.get("cotracker_jaccard", float("nan")))
                    ]
                ),
            },
            "dino_tracking": {
                "jaccard_mean": _safe_mean(
                    [
                        r["dino_matter_jaccard"]
                        for r in comparison
                        if not np.isnan(r.get("dino_matter_jaccard", float("nan")))
                    ]
                ),
                "jaccard_std": _safe_std(
                    [
                        r["dino_matter_jaccard"]
                        for r in comparison
                        if not np.isnan(r.get("dino_matter_jaccard", float("nan")))
                    ]
                ),
            },
        },
        "davis_videos": DAVIS_VIDEOS,
    }
    write_json(comparison_out, out_dir / "davis_comparison.json")
    write_json(combined_tradeoff, out_dir / "davis_subsampling_tradeoff.json")

    csv_rows = []
    for r in comparison:
        csv_row = {k: v for k, v in r.items()}
        for k, v in csv_row.items():
            if isinstance(v, float) and np.isnan(v):
                csv_row[k] = ""
        csv_rows.append(csv_row)
    write_csv(csv_rows, out_dir / "davis_results.csv")

    print("\nDone.")


if __name__ == "__main__":
    main()
