"""DAVIS tracking postprocessing: compare GenMatter, CoTracker, subsampling, and ablations.

Loads per-video JSON results from DAVIS full-grid tracking (SAM), hyperblob **ablation**
and **ablation-2** (SAM and no-SAM — including ``subsample_*`` subfolders when a run is
not at 100% grid), SAM and no-SAM subsampling runs, and CoTracker.  Extracts
**matter-weighted recall, precision, and Jaccard (fixed frame-0 weights)** plus FPS as
the GenMatter metrics, aggregates across videos, and writes:

- results/postprocessing/davis_comparison.json   (all methods side-by-side, incl. ablations)
- results/postprocessing/davis_subsampling_tradeoff.json  (subsample % vs perf)
- results/postprocessing/davis_results.csv
"""

import csv
import json
import re
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

# Must match folder names from ``run_davis_subsampling.py`` (``subsample_{pct}``).
# Also used as preferred order; unknown ``subsample_*`` dirs are discovered at load time.
SUBSAMPLE_DIRS: dict[str, float] = {
    "subsample_12_5": 12.5,
    "subsample_3_125": 3.125,
    "subsample_0_78125": 0.78125,
    "subsample_0_1953125": 0.1953125,
}

# Grid fraction labels (subsampling % of points kept ≈ 100 × 1/N).
PCT_TO_FRACTION_LABEL: dict[float, str] = {
    12.5: "1/8",
    3.125: "1/32",
    0.78125: "1/128",
    0.1953125: "1/512",
}


def _fraction_label_for_pct(pct: float) -> str:
    for k, lab in PCT_TO_FRACTION_LABEL.items():
        if abs(float(pct) - k) < 1e-6:
            return lab
    return f"{pct:g}%"


def _ablation_column_label(meta: dict[str, Any]) -> str:
    if not meta.get("layout"):
        return "ablation"
    if meta.get("layout") == "flat":
        return "ablation (full)"
    pct = meta.get("subsample_pct")
    if pct is None:
        return "ablation"
    return f"ablation ({_fraction_label_for_pct(float(pct))})"


def _ablation2_column_label(meta: dict[str, Any]) -> str:
    """Same layout rules as :func:`_ablation_column_label`, prefixed for ablation-2 runs."""
    s = _ablation_column_label(meta)
    if s == "ablation":
        return "Ablation-2"
    return "Ablation-2" + s[len("ablation") :]


def _ablation3_column_label(meta: dict[str, Any]) -> str:
    """Same layout rules as :func:`_ablation_column_label`, prefixed for ablation-3 runs."""
    s = _ablation_column_label(meta)
    if s == "ablation":
        return "Ablation-3"
    return "Ablation-3" + s[len("ablation") :]


def _ablation4_column_label(meta: dict[str, Any]) -> str:
    """Same layout rules as :func:`_ablation_column_label`, prefixed for ablation-4 runs."""
    s = _ablation_column_label(meta)
    if s == "ablation":
        return "Ablation-4"
    return "Ablation-4" + s[len("ablation") :]


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _visible_len(s: str) -> int:
    return len(ANSI_RE.sub("", s))


def _term_styles() -> dict[str, str]:
    if not sys.stdout.isatty():
        return {k: "" for k in ("reset", "bold", "dim", "cyan", "green", "yellow", "magenta")}
    return {
        "reset": "\033[0m",
        "bold": "\033[1m",
        "dim": "\033[2m",
        "cyan": "\033[96m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "magenta": "\033[95m",
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


def load_cotracker_results(
    base_dir: Path, *, silent: bool = False
) -> dict[str, dict[str, Any]]:
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
        if not silent:
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


def load_dino_tracking_results(
    base_dir: Path, *, silent_empty: bool = False
) -> dict[str, dict[str, Any]]:
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

    if not results and not silent_empty:
        print(f"  [WARN] No DINO tracking results found in {base_dir}")
    return results


def _parse_subsample_percent_from_dirname(dirname: str) -> float | None:
    """Parse ``12_5`` / ``0_78125`` suffix from ``subsample_*`` (matches run_davis_* naming)."""
    if not dirname.startswith("subsample_"):
        return None
    suffix = dirname[len("subsample_") :]
    if not suffix:
        return None
    if "_" not in suffix:
        try:
            return float(suffix)
        except ValueError:
            return None
    a, b = suffix.split("_", 1)
    try:
        return float(f"{a}.{b}")
    except ValueError:
        return None


def load_dino_ablation_results(
    base_dir: Path | str, *, silent: bool = False
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Load DAVIS hyperblob-ablation metrics.

    **Flat layout (full grid):** ``json_results/`` or ``all_videos_experiment_results.json``
    directly under *base_dir* — same as :func:`load_dino_tracking_results`.

    **Nested subsampling layout:** ``subsample_<pct>/json_results/`` (ablation often runs
    at less than 100% grid). If no flat results exist, we discover every ``subsample_*``
    directory that contains per-video JSON and pick the run with the **largest**
    percentage (most pixels retained). If only one run exists (e.g. 1/128 ≈ 0.78125%),
    that run is used.
    """
    base_dir = Path(base_dir)
    meta: dict[str, Any] = {
        "base_dir": str(base_dir),
        "layout": None,
        "subsample_pct": None,
        "subsample_dir": None,
    }

    flat = load_dino_tracking_results(base_dir, silent_empty=True)
    if flat:
        meta["layout"] = "flat"
        return flat, meta

    candidates: list[tuple[float, Path, dict[str, dict[str, Any]]]] = []
    if not base_dir.is_dir():
        if not silent:
            print(f"  [WARN] Ablation directory missing: {base_dir}")
        return {}, meta

    for child in sorted(base_dir.iterdir()):
        if not child.is_dir() or not child.name.startswith("subsample_"):
            continue
        pct = _parse_subsample_percent_from_dirname(child.name)
        if pct is None:
            continue
        json_dir = child / "json_results"
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
            candidates.append((pct, child, per_video))

    if not candidates:
        if not silent:
            print(
                f"  [WARN] No DINO ablation results (flat or subsample_*) under {base_dir}"
            )
        return {}, meta

    candidates.sort(key=lambda x: x[0], reverse=True)
    best_pct, best_path, best_map = candidates[0]
    meta["layout"] = "subsample"
    meta["subsample_pct"] = float(best_pct)
    meta["subsample_dir"] = best_path.name
    if len(candidates) > 1:
        meta["subsample_dirs_available"] = [
            {"dir": c[1].name, "pct": float(c[0])} for c in sorted(candidates, key=lambda x: -x[0])
        ]
    if not silent:
        print(
            f"  [INFO] Ablation: using {best_path.name} ({best_pct}% of grid); "
            f"{len(candidates)} subsample run(s) with data."
        )
    return best_map, meta


def _extract_dino_metrics(data: dict) -> dict[str, Any]:
    """Extract a uniform metrics dict from either per-video or subsampling JSON."""
    pm = data.get("pixel_metrics", {})

    # Primary GenMatter DAVIS metrics: matter-weighted R/P/J with frame-0 blob weights
    matter_jaccard_fixed = (
        pm.get("avg_matter_weighted_jaccard_fixed")
        or data.get("particle_count_matter_fixed_jaccard")
    )
    matter_recall_fixed = pm.get("avg_matter_weighted_recall_fixed") or data.get(
        "particle_count_matter_fixed_recall"
    )
    matter_precision_fixed = pm.get("avg_matter_weighted_precision_fixed") or data.get(
        "particle_count_matter_fixed_precision"
    )

    fps = data.get("fps") or pm.get("fps_mean")

    return {
        "matter_weighted_jaccard_fixed": _to_float(matter_jaccard_fixed),
        "matter_weighted_recall_fixed": _to_float(matter_recall_fixed),
        "matter_weighted_precision_fixed": _to_float(matter_precision_fixed),
        "fps": _to_float(fps),
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
    """Load subsampling results across all ``subsample_*`` runs under *base_dir*.

    Discovers any ``subsample_*`` directory with ``json_results/`` (not only known
    percentages). Returns ``{percentage: {video: metrics_dict}}``.
    """
    out: dict[float, dict[str, dict[str, Any]]] = {}
    if not base_dir.is_dir():
        return out

    pct_to_dirname: dict[float, str] = {}
    for dn, pct in SUBSAMPLE_DIRS.items():
        pct_to_dirname[float(pct)] = dn
    for child in sorted(base_dir.glob("subsample_*")):
        if not child.is_dir():
            continue
        pct = _parse_subsample_percent_from_dirname(child.name)
        if pct is None:
            continue
        pct_f = float(pct)
        if pct_f not in pct_to_dirname:
            pct_to_dirname[pct_f] = child.name

    for pct_f in sorted(pct_to_dirname.keys(), reverse=True):
        dir_name = pct_to_dirname[pct_f]
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
            out[pct_f] = per_video

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
    ablation_sam: dict[str, dict[str, Any]] | None = None,
    ablation_no_sam: dict[str, dict[str, Any]] | None = None,
    ablation_2_sam: dict[str, dict[str, Any]] | None = None,
    ablation_2_no_sam: dict[str, dict[str, Any]] | None = None,
    ablation_3_sam: dict[str, dict[str, Any]] | None = None,
    ablation_3_no_sam: dict[str, dict[str, Any]] | None = None,
    ablation_4_sam: dict[str, dict[str, Any]] | None = None,
    ablation_4_no_sam: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build per-video comparison: CoTracker, full DINO tracking, optional ablation runs."""
    ablation_sam = ablation_sam or {}
    ablation_no_sam = ablation_no_sam or {}
    ablation_2_sam = ablation_2_sam or {}
    ablation_2_no_sam = ablation_2_no_sam or {}
    ablation_3_sam = ablation_3_sam or {}
    ablation_3_no_sam = ablation_3_no_sam or {}
    ablation_4_sam = ablation_4_sam or {}
    ablation_4_no_sam = ablation_4_no_sam or {}
    all_videos = sorted(
        set(dino_results)
        | set(cotracker_results)
        | set(ablation_sam)
        | set(ablation_no_sam)
        | set(ablation_2_sam)
        | set(ablation_2_no_sam)
        | set(ablation_3_sam)
        | set(ablation_3_no_sam)
        | set(ablation_4_sam)
        | set(ablation_4_no_sam)
    )
    rows: list[dict[str, Any]] = []

    for video in all_videos:
        dino = dino_results.get(video, {})
        ct = cotracker_results.get(video, {})
        ab_s = ablation_sam.get(video, {})
        ab_n = ablation_no_sam.get(video, {})
        ab2_s = ablation_2_sam.get(video, {})
        ab2_n = ablation_2_no_sam.get(video, {})
        ab3_s = ablation_3_sam.get(video, {})
        ab3_n = ablation_3_no_sam.get(video, {})
        ab4_s = ablation_4_sam.get(video, {})
        ab4_n = ablation_4_no_sam.get(video, {})

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

        row: dict[str, Any] = {
            "video": video,
            "cotracker_jaccard": ct.get("mean_jaccard", float("nan")),
            "cotracker_precision": ct_precision,
            "cotracker_recall": ct_recall,
            "cotracker_f1": ct_f1,
            "cotracker_fn_rate": ct.get("mean_fn_rate", float("nan")),
            "cotracker_fp_rate": ct.get("mean_fp_rate", float("nan")),
            "dino_matter_jaccard_fixed": dino.get(
                "matter_weighted_jaccard_fixed", float("nan")
            ),
            "dino_matter_recall_fixed": dino.get(
                "matter_weighted_recall_fixed", float("nan")
            ),
            "dino_matter_precision_fixed": dino.get(
                "matter_weighted_precision_fixed", float("nan")
            ),
            "dino_fps": dino.get("fps", float("nan")),
            "ablation_sam_matter_jaccard_fixed": ab_s.get(
                "matter_weighted_jaccard_fixed", float("nan")
            ),
            "ablation_sam_matter_recall_fixed": ab_s.get(
                "matter_weighted_recall_fixed", float("nan")
            ),
            "ablation_sam_matter_precision_fixed": ab_s.get(
                "matter_weighted_precision_fixed", float("nan")
            ),
            "ablation_sam_fps": ab_s.get("fps", float("nan")),
            "ablation_no_sam_matter_jaccard_fixed": ab_n.get(
                "matter_weighted_jaccard_fixed", float("nan")
            ),
            "ablation_no_sam_matter_recall_fixed": ab_n.get(
                "matter_weighted_recall_fixed", float("nan")
            ),
            "ablation_no_sam_matter_precision_fixed": ab_n.get(
                "matter_weighted_precision_fixed", float("nan")
            ),
            "ablation_no_sam_fps": ab_n.get("fps", float("nan")),
            "ablation_2_sam_matter_jaccard_fixed": ab2_s.get(
                "matter_weighted_jaccard_fixed", float("nan")
            ),
            "ablation_2_sam_matter_recall_fixed": ab2_s.get(
                "matter_weighted_recall_fixed", float("nan")
            ),
            "ablation_2_sam_matter_precision_fixed": ab2_s.get(
                "matter_weighted_precision_fixed", float("nan")
            ),
            "ablation_2_sam_fps": ab2_s.get("fps", float("nan")),
            "ablation_2_no_sam_matter_jaccard_fixed": ab2_n.get(
                "matter_weighted_jaccard_fixed", float("nan")
            ),
            "ablation_2_no_sam_matter_recall_fixed": ab2_n.get(
                "matter_weighted_recall_fixed", float("nan")
            ),
            "ablation_2_no_sam_matter_precision_fixed": ab2_n.get(
                "matter_weighted_precision_fixed", float("nan")
            ),
            "ablation_2_no_sam_fps": ab2_n.get("fps", float("nan")),
            "ablation_3_sam_matter_jaccard_fixed": ab3_s.get(
                "matter_weighted_jaccard_fixed", float("nan")
            ),
            "ablation_3_sam_matter_recall_fixed": ab3_s.get(
                "matter_weighted_recall_fixed", float("nan")
            ),
            "ablation_3_sam_matter_precision_fixed": ab3_s.get(
                "matter_weighted_precision_fixed", float("nan")
            ),
            "ablation_3_sam_fps": ab3_s.get("fps", float("nan")),
            "ablation_3_no_sam_matter_jaccard_fixed": ab3_n.get(
                "matter_weighted_jaccard_fixed", float("nan")
            ),
            "ablation_3_no_sam_matter_recall_fixed": ab3_n.get(
                "matter_weighted_recall_fixed", float("nan")
            ),
            "ablation_3_no_sam_matter_precision_fixed": ab3_n.get(
                "matter_weighted_precision_fixed", float("nan")
            ),
            "ablation_3_no_sam_fps": ab3_n.get("fps", float("nan")),
            "ablation_4_sam_matter_jaccard_fixed": ab4_s.get(
                "matter_weighted_jaccard_fixed", float("nan")
            ),
            "ablation_4_sam_matter_recall_fixed": ab4_s.get(
                "matter_weighted_recall_fixed", float("nan")
            ),
            "ablation_4_sam_matter_precision_fixed": ab4_s.get(
                "matter_weighted_precision_fixed", float("nan")
            ),
            "ablation_4_sam_fps": ab4_s.get("fps", float("nan")),
            "ablation_4_no_sam_matter_jaccard_fixed": ab4_n.get(
                "matter_weighted_jaccard_fixed", float("nan")
            ),
            "ablation_4_no_sam_matter_recall_fixed": ab4_n.get(
                "matter_weighted_recall_fixed", float("nan")
            ),
            "ablation_4_no_sam_matter_precision_fixed": ab4_n.get(
                "matter_weighted_precision_fixed", float("nan")
            ),
            "ablation_4_no_sam_fps": ab4_n.get("fps", float("nan")),
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
            m["matter_weighted_jaccard_fixed"]
            for m in per_video.values()
            if not np.isnan(
                m.get("matter_weighted_jaccard_fixed", float("nan"))
            )
        ]
        fps_vals = [
            m["fps"]
            for m in per_video.values()
            if not np.isnan(m.get("fps", float("nan")))
        ]

        rows.append(
            {
                "subsample_pct": pct,
                "n_videos": len(per_video),
                "jaccard_mean": _safe_mean(jaccards),
                "jaccard_std": _safe_std(jaccards),
                "fps_mean": _safe_mean(fps_vals),
                "fps_std": _safe_std(fps_vals),
                "per_video": {
                    video: {
                        "matter_jaccard_fixed": m["matter_weighted_jaccard_fixed"],
                        "fps": m["fps"],
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


def _jaccard_from_metrics(m: dict[str, Any] | None) -> float:
    if not m:
        return float("nan")
    return float(m.get("matter_weighted_jaccard_fixed", float("nan")))


def _fps_from_metrics(m: dict[str, Any] | None) -> float:
    if not m:
        return float("nan")
    return float(m.get("fps", float("nan")))


def _mean_jaccard_over_videos(per_video: dict[str, dict[str, Any]]) -> float:
    if not per_video:
        return float("nan")
    vals = [
        _jaccard_from_metrics(per_video.get(v))
        for v in sorted(per_video.keys())
    ]
    vals = [x for x in vals if not np.isnan(x)]
    return float(np.mean(vals)) if vals else float("nan")


def _best_non_ablated_run(
    dino_full: dict[str, dict[str, Any]],
    sub: dict[float, dict[str, dict[str, Any]]],
) -> tuple[str, float | None, float]:
    """Return (label, subsample grid % or None if *full* won, mean Jaccard)."""
    candidates: list[tuple[str, float | None, float]] = []
    if dino_full:
        m = _mean_jaccard_over_videos(dino_full)
        candidates.append(("full", None, m))
    for pct in sorted(sub.keys(), reverse=True):
        m = _mean_jaccard_over_videos(sub[pct])
        candidates.append((_fraction_label_for_pct(pct), float(pct), m))
    valid = [(a, b, c) for a, b, c in candidates if not np.isnan(c)]
    if not valid:
        return "—", None, float("nan")
    return max(valid, key=lambda x: x[2])


def _ablation_load_note(
    meta: dict[str, Any],
    dino_full: dict[str, dict[str, Any]],
    sub: dict[float, dict[str, dict[str, Any]]],
    init_short: str,
) -> str:
    """Short note: ablation grid rate vs best non-ablated run (for Inputs loaded)."""
    if not meta.get("layout"):
        return ""
    if meta.get("layout") == "flat":
        return "full-grid ablation"
    pct_ab = meta.get("subsample_pct")
    if pct_ab is None:
        return ""
    blab, bpct, bj = _best_non_ablated_run(dino_full, sub)
    r = float(pct_ab)
    frac = _fraction_label_for_pct(r)
    if np.isnan(bj):
        return f"{frac} ({r:g}% of grid)"
    if bpct is not None and abs(float(pct_ab) - bpct) < 1e-5:
        return (
            f"same subsampling as best non-ablated {init_short} "
            f"({blab}, {r:g}% of grid)"
        )
    if bpct is None and blab == "full":
        return f"{frac} ({r:g}% of grid); best non-ablated {init_short} was full (J={bj:.4f})"
    return f"{frac} ({r:g}% of grid); best non-ablated {init_short}: {blab} (J={bj:.4f})"


def _video_union(
    *sources: dict[str, dict[str, Any]],
) -> list[str]:
    keys: set[str] = set()
    for s in sources:
        keys |= set(s.keys())
    return sorted(keys)


def print_davis_analysis_report(
    *,
    dino_sam: dict[str, dict[str, Any]],
    dino_no_sam: dict[str, dict[str, Any]],
    sub_sam: dict[float, dict[str, dict[str, Any]]],
    sub_no_sam: dict[float, dict[str, dict[str, Any]]],
    ablation_sam: dict[str, dict[str, Any]],
    ablation_no_sam: dict[str, dict[str, Any]],
    meta_ab_sam: dict[str, Any],
    meta_ab_no: dict[str, Any],
    ablation_2_sam: dict[str, dict[str, Any]] | None = None,
    ablation_2_no_sam: dict[str, dict[str, Any]] | None = None,
    meta_ab2_sam: dict[str, Any] | None = None,
    meta_ab2_no: dict[str, Any] | None = None,
    ablation_3_sam: dict[str, dict[str, Any]] | None = None,
    ablation_3_no_sam: dict[str, dict[str, Any]] | None = None,
    meta_ab3_sam: dict[str, Any] | None = None,
    meta_ab3_no: dict[str, Any] | None = None,
    ablation_4_sam: dict[str, dict[str, Any]] | None = None,
    ablation_4_no_sam: dict[str, dict[str, Any]] | None = None,
    meta_ab4_sam: dict[str, Any] | None = None,
    meta_ab4_no: dict[str, Any] | None = None,
    ct_results: dict[str, dict[str, Any]],
) -> None:
    """Pretty-print video-by-video (SAM vs GT init) and aggregate mean±std tables."""
    t = _term_styles()
    sep = "=" * 100

    def ct_jaccard(video: str) -> float:
        c = ct_results.get(video, {})
        return float(c.get("mean_jaccard", float("nan")))

    # Column specs: (header, getter(video) -> jaccard)
    ablation_2_sam = ablation_2_sam or {}
    ablation_2_no_sam = ablation_2_no_sam or {}
    meta_ab2_sam = meta_ab2_sam or {}
    meta_ab2_no = meta_ab2_no or {}
    ablation_3_sam = ablation_3_sam or {}
    ablation_3_no_sam = ablation_3_no_sam or {}
    meta_ab3_sam = meta_ab3_sam or {}
    meta_ab3_no = meta_ab3_no or {}
    ablation_4_sam = ablation_4_sam or {}
    ablation_4_no_sam = ablation_4_no_sam or {}
    meta_ab4_sam = meta_ab4_sam or {}
    meta_ab4_no = meta_ab4_no or {}

    def build_column_specs(
        *,
        dino_full: dict[str, dict[str, Any]],
        sub: dict[float, dict[str, dict[str, Any]]],
        ablation: dict[str, dict[str, Any]],
        ablation_2: dict[str, dict[str, Any]],
        ablation_3: dict[str, dict[str, Any]],
        ablation_4: dict[str, dict[str, Any]],
        include_full: bool,
    ) -> list[tuple[str, Any]]:
        cols: list[tuple[str, Any]] = []
        cols.append(("CoTracker", lambda v: ct_jaccard(v)))
        if include_full:
            cols.append(("full", lambda v: _jaccard_from_metrics(dino_full.get(v))))
        for pct in sorted(sub.keys(), reverse=True):
            lab = _fraction_label_for_pct(pct)
            pv = sub[pct]

            def _make_get(p: dict[str, dict[str, Any]]):
                return lambda v, _p=p: _jaccard_from_metrics(_p.get(v))

            cols.append((lab, _make_get(pv)))
        cols.append(("Ablation", lambda v: _jaccard_from_metrics(ablation.get(v))))
        if ablation_2:
            cols.append(("Ablation-2", lambda v: _jaccard_from_metrics(ablation_2.get(v))))
        if ablation_3:
            cols.append(("Ablation-3", lambda v: _jaccard_from_metrics(ablation_3.get(v))))
        if ablation_4:
            cols.append(("Ablation-4", lambda v: _jaccard_from_metrics(ablation_4.get(v))))
        return cols

    col_sam = build_column_specs(
        dino_full=dino_sam,
        sub=sub_sam,
        ablation=ablation_sam,
        ablation_2=ablation_2_sam,
        ablation_3=ablation_3_sam,
        ablation_4=ablation_4_sam,
        include_full=bool(dino_sam),
    )
    col_no = build_column_specs(
        dino_full=dino_no_sam,
        sub=sub_no_sam,
        ablation=ablation_no_sam,
        ablation_2=ablation_2_no_sam,
        ablation_3=ablation_3_no_sam,
        ablation_4=ablation_4_no_sam,
        include_full=bool(dino_no_sam),
    )

    videos_sam = _video_union(
        ct_results,
        dino_sam,
        ablation_sam,
        ablation_2_sam,
        ablation_3_sam,
        ablation_4_sam,
        *list(sub_sam.values()),
    )
    videos_no = _video_union(
        ct_results,
        dino_no_sam,
        ablation_no_sam,
        ablation_2_no_sam,
        ablation_3_no_sam,
        ablation_4_no_sam,
        *list(sub_no_sam.values()),
    )

    def _fit_header(text: str, width: int) -> str:
        if len(text) <= width:
            return text
        if width <= 1:
            return text[:width]
        return text[: width - 1] + "…"

    # One column width for both video tables: match the tighter GT-init grid (SAM often
    # has longer ablation labels that would otherwise widen every column).
    headers_gt = [c[0] for c in col_no]
    w_col_shared = max(12, max((len(h) for h in headers_gt), default=12) + 1)

    def print_video_table(title: str, videos: list[str], cols: list[tuple[str, Any]]) -> None:
        print(f"\n{sep}")
        print(f"{t['cyan']}{t['bold']}{title}{t['reset']}")
        print(sep)
        headers = [c[0] for c in cols]
        w_vid = 22
        w_col = w_col_shared
        header_line = f"{t['magenta']}{'Video':<{w_vid}}{t['reset']}"
        for h in headers:
            hh = _fit_header(h, w_col)
            header_line += f"{t['magenta']}{hh:>{w_col}}{t['reset']}"
        print(header_line)
        print(t["dim"] + "-" * (w_vid + w_col * len(headers)) + t["reset"])

        for video in videos:
            row_vals: list[float] = []
            for _, getter in cols:
                try:
                    j = float(getter(video))
                except Exception:
                    j = float("nan")
                row_vals.append(j)
            finite = [x for x in row_vals if not np.isnan(x)]
            best_j = max(finite) if finite else float("nan")

            line = f"{video:<{w_vid}}"
            for j in row_vals:
                txt = f"{j:.4f}" if not np.isnan(j) else "N/A"
                is_best = not np.isnan(j) and not np.isnan(best_j) and abs(j - best_j) < 1e-9
                if is_best and finite:
                    cell = f"{t['bold']}{t['green']}{txt:>{w_col}}{t['reset']}"
                else:
                    cell = f"{txt:>{w_col}}"
                line += cell
            print(line)

    print_video_table(
        "Video-by-video analysis — SAM init (frame-0 SAM)",
        videos_sam,
        col_sam,
    )
    print_video_table(
        "Video-by-video analysis — GT init",
        videos_no,
        col_no,
    )

    # --- Aggregate: all models, Jaccard and FPS as mean ± std ---
    print(f"\n{sep}")
    print(
        f"{t['cyan']}{t['bold']}Aggregate results (mean over all videos){t['reset']}"
    )
    print(sep)

    ab_lab_s = _ablation_column_label(meta_ab_sam)
    ab_lab_n = _ablation_column_label(meta_ab_no)
    ab2_lab_s = _ablation2_column_label(meta_ab2_sam)
    ab2_lab_n = _ablation2_column_label(meta_ab2_no)
    ab3_lab_s = _ablation3_column_label(meta_ab3_sam)
    ab3_lab_n = _ablation3_column_label(meta_ab3_no)
    ab4_lab_s = _ablation4_column_label(meta_ab4_sam)
    ab4_lab_n = _ablation4_column_label(meta_ab4_no)

    agg_rows: list[tuple[str, str, list[float], list[float]]] = []

    def collect_series(
        label: str,
        init: str,
        per_video: dict[str, dict[str, Any]],
    ) -> None:
        if not per_video:
            return
        js = [_jaccard_from_metrics(per_video.get(v)) for v in sorted(per_video.keys())]
        fs = [_fps_from_metrics(per_video.get(v)) for v in sorted(per_video.keys())]
        agg_rows.append((label, init, js, fs))

    if ct_results:
        js = [ct_jaccard(v) for v in sorted(ct_results.keys())]
        agg_rows.append(("CoTracker", "—", js, [float("nan")] * len(js)))
    if dino_sam:
        collect_series("full", "SAM", dino_sam)
    if dino_no_sam:
        collect_series("full", "GT", dino_no_sam)
    all_pcts = sorted(set(sub_sam.keys()) | set(sub_no_sam.keys()), reverse=True)
    for pct in all_pcts:
        if pct in sub_sam:
            collect_series(_fraction_label_for_pct(pct), "SAM", sub_sam[pct])
        if pct in sub_no_sam:
            collect_series(_fraction_label_for_pct(pct), "GT", sub_no_sam[pct])
    collect_series(ab_lab_s, "SAM", ablation_sam)
    collect_series(ab_lab_n, "GT", ablation_no_sam)
    if ablation_2_sam:
        collect_series(ab2_lab_s, "SAM", ablation_2_sam)
    if ablation_2_no_sam:
        collect_series(ab2_lab_n, "GT", ablation_2_no_sam)
    if ablation_3_sam:
        collect_series(ab3_lab_s, "SAM", ablation_3_sam)
    if ablation_3_no_sam:
        collect_series(ab3_lab_n, "GT", ablation_3_no_sam)
    if ablation_4_sam:
        collect_series(ab4_lab_s, "SAM", ablation_4_sam)
    if ablation_4_no_sam:
        collect_series(ab4_lab_n, "GT", ablation_4_no_sam)

    _FRACS_IN_ORDER: tuple[str, ...] = tuple(PCT_TO_FRACTION_LABEL.values())

    def _aggregate_row_sort_key(
        row: tuple[str, str, list[float], list[float]],
    ) -> tuple[int, int, str]:
        model, init, _, _ = row
        init_rank = {"—": 0, "SAM": 1, "GT": 2}.get(init, 9)
        if model == "full":
            return (init_rank, 0, "")
        if model in _FRACS_IN_ORDER:
            return (init_rank, 1, f"{_FRACS_IN_ORDER.index(model):04d}")
        return (init_rank, 2, model)

    agg_rows.sort(key=_aggregate_row_sort_key)

    w_model = max(28, max((len(r[0]) for r in agg_rows), default=28))
    w_init = 6
    w_n = 7
    w_j = 24
    w_f = 24

    hdr = (
        f"{t['magenta']}{'Model':<{w_model}} {'Init':<{w_init}} {'#Vids':>{w_n}} "
        f"{'Jaccard (mean ± std)':>{w_j}} {'FPS (mean ± std)':>{w_f}}{t['reset']}"
    )
    print(hdr)
    print(
        t["dim"]
        + "-" * (w_model + w_init + w_n + w_j + w_f + 4)
        + t["reset"]
    )

    table_rows: list[tuple[str, str, str, str, str, float]] = []
    for model, init, js, fs in agg_rows:
        jv = [x for x in js if not np.isnan(x)]
        fv = [x for x in fs if not np.isnan(x)]
        jm = _safe_mean(jv)
        jsd = _safe_std(jv)
        fm = _safe_mean(fv)
        fsd = _safe_std(fv)
        n_str = str(len(jv)) if jv else "—"
        j_str = f"{jm:.4f} ± {jsd:.4f}" if jv else "N/A"
        f_str = f"{fm:.2f} ± {fsd:.2f}" if fv else "N/A"
        table_rows.append((model, init, n_str, j_str, f_str, jm))

    best_mean = max(
        (m for _, _, _, _, _, m in table_rows if not np.isnan(m)),
        default=float("nan"),
    )

    for model, init, n_str, j_str, f_str, jm in table_rows:
        is_best = not np.isnan(jm) and not np.isnan(best_mean) and abs(jm - best_mean) < 1e-9
        if is_best:
            line = (
                f"{model:<{w_model}} {init:<{w_init}} {n_str:>{w_n}} "
                f"{t['bold']}{t['green']}{j_str:>{w_j}}{t['reset']} "
                f"{f_str:>{w_f}}"
            )
        else:
            line = (
                f"{model:<{w_model}} {init:<{w_init}} {n_str:>{w_n}} "
                f"{j_str:>{w_j}} {f_str:>{w_f}}"
            )
        print(line)

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _print_davis_inputs_loaded(
    out_dir: Path,
    *,
    ct_results: dict[str, dict[str, Any]],
    dino_sam: dict[str, dict[str, Any]],
    dino_no_sam: dict[str, dict[str, Any]],
    sub_sam: dict[float, dict[str, dict[str, Any]]],
    sub_no_sam: dict[float, dict[str, dict[str, Any]]],
    ablation_sam: dict[str, dict[str, Any]],
    ablation_no_sam: dict[str, dict[str, Any]],
    meta_ab_sam: dict[str, Any],
    meta_ab_no: dict[str, Any],
    ablation_2_sam: dict[str, dict[str, Any]] | None = None,
    ablation_2_no_sam: dict[str, dict[str, Any]] | None = None,
    meta_ab2_sam: dict[str, Any] | None = None,
    meta_ab2_no: dict[str, Any] | None = None,
    ablation_3_sam: dict[str, dict[str, Any]] | None = None,
    ablation_3_no_sam: dict[str, dict[str, Any]] | None = None,
    meta_ab3_sam: dict[str, Any] | None = None,
    meta_ab3_no: dict[str, Any] | None = None,
    ablation_4_sam: dict[str, dict[str, Any]] | None = None,
    ablation_4_no_sam: dict[str, dict[str, Any]] | None = None,
    meta_ab4_sam: dict[str, Any] | None = None,
    meta_ab4_no: dict[str, Any] | None = None,
) -> None:
    """Single compact, colored summary of which result trees were found (omit missing)."""
    t = _term_styles()
    print(f"\n{t['cyan']}{t['bold']}DAVIS postprocessing{t['reset']}")
    print(f"{t['dim']}{out_dir}{t['reset']}")
    print(
        f"{t['magenta']}writes{t['reset']} {t['dim']}"
        f"davis_comparison.json · davis_subsampling_tradeoff.json · davis_results.csv{t['reset']}\n"
    )
    print(f"{t['magenta']}{t['bold']}Inputs loaded{t['reset']}")

    def line(label: str, path: Path | str, ok: bool, extra: str = "") -> None:
        status = (
            f"{t['green']}yes{t['reset']}" if ok else f"{t['dim']}no{t['reset']}"
        )
        x = f"  {t['dim']}{extra}{t['reset']}" if extra else ""
        print(f"  {label:<30} {status}  {t['dim']}{path}{t['reset']}{x}")

    line("CoTracker", config.COTRACKER_OUTPUT_DIR, bool(ct_results),
         f"({len(ct_results)} videos)" if ct_results else "")
    if dino_sam:
        line(
            "DINO full-grid (SAM)",
            config.DAVIS_TRACKING_OUTPUT_DIR_SAM,
            True,
            f"({len(dino_sam)} videos)",
        )
    if dino_no_sam:
        line(
            "DINO full-grid (GT init)",
            config.DAVIS_TRACKING_OUTPUT_DIR_NO_SAM,
            True,
            f"({len(dino_no_sam)} videos)",
        )
    if sub_sam:
        line(
            "Subsampling (SAM)",
            config.DAVIS_SUBSAMPLING_OUTPUT_DIR_SAM,
            True,
            f"({len(sub_sam)} level(s))",
        )
    if sub_no_sam:
        line(
            "Subsampling (GT init)",
            config.DAVIS_SUBSAMPLING_OUTPUT_DIR_NO_SAM,
            True,
            f"({len(sub_no_sam)} level(s))",
        )
    if ablation_sam:
        note = _ablation_load_note(meta_ab_sam, dino_sam, sub_sam, "SAM")
        line("Ablation (SAM)", config.DAVIS_ABLATION_OUTPUT_DIR_SAM, True, note)
    if ablation_no_sam:
        note = _ablation_load_note(meta_ab_no, dino_no_sam, sub_no_sam, "GT")
        line("Ablation (GT init)", config.DAVIS_ABLATION_OUTPUT_DIR_NO_SAM, True, note)
    ablation_2_sam = ablation_2_sam or {}
    ablation_2_no_sam = ablation_2_no_sam or {}
    meta_ab2_sam = meta_ab2_sam or {}
    meta_ab2_no = meta_ab2_no or {}
    if ablation_2_sam:
        note2 = _ablation_load_note(meta_ab2_sam, dino_sam, sub_sam, "SAM")
        line("Ablation-2 (SAM)", config.DAVIS_ABLATION_2_OUTPUT_DIR_SAM, True, note2)
    if ablation_2_no_sam:
        note2n = _ablation_load_note(meta_ab2_no, dino_no_sam, sub_no_sam, "GT")
        line("Ablation-2 (GT init)", config.DAVIS_ABLATION_2_OUTPUT_DIR_NO_SAM, True, note2n)
    ablation_3_sam = ablation_3_sam or {}
    ablation_3_no_sam = ablation_3_no_sam or {}
    meta_ab3_sam = meta_ab3_sam or {}
    meta_ab3_no = meta_ab3_no or {}
    if ablation_3_sam:
        note3 = _ablation_load_note(meta_ab3_sam, dino_sam, sub_sam, "SAM")
        line("Ablation-3 (SAM)", config.DAVIS_ABLATION_3_OUTPUT_DIR_SAM, True, note3)
    if ablation_3_no_sam:
        note3n = _ablation_load_note(meta_ab3_no, dino_no_sam, sub_no_sam, "GT")
        line("Ablation-3 (GT init)", config.DAVIS_ABLATION_3_OUTPUT_DIR_NO_SAM, True, note3n)
    ablation_4_sam = ablation_4_sam or {}
    ablation_4_no_sam = ablation_4_no_sam or {}
    meta_ab4_sam = meta_ab4_sam or {}
    meta_ab4_no = meta_ab4_no or {}
    if ablation_4_sam:
        note4 = _ablation_load_note(meta_ab4_sam, dino_sam, sub_sam, "SAM")
        line("Ablation-4 (SAM)", config.DAVIS_ABLATION_4_OUTPUT_DIR_SAM, True, note4)
    if ablation_4_no_sam:
        note4n = _ablation_load_note(meta_ab4_no, dino_no_sam, sub_no_sam, "GT")
        line("Ablation-4 (GT init)", config.DAVIS_ABLATION_4_OUTPUT_DIR_NO_SAM, True, note4n)
    print()


def main() -> None:
    out_dir = config.POSTPROCESSING_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    t = _term_styles()

    # ---- Load data --------------------------------------------------------
    ct_results = load_cotracker_results(config.COTRACKER_OUTPUT_DIR, silent=True)
    dino_results = load_dino_tracking_results(
        config.DAVIS_TRACKING_OUTPUT_DIR_SAM, silent_empty=True
    )
    dino_no_sam = load_dino_tracking_results(
        config.DAVIS_TRACKING_OUTPUT_DIR_NO_SAM, silent_empty=True
    )
    sub_sam = load_subsampling_results(config.DAVIS_SUBSAMPLING_OUTPUT_DIR_SAM)
    sub_no_sam = load_subsampling_results(config.DAVIS_SUBSAMPLING_OUTPUT_DIR_NO_SAM)
    ablation_sam, meta_ab_sam = load_dino_ablation_results(
        config.DAVIS_ABLATION_OUTPUT_DIR_SAM, silent=True
    )
    ablation_no_sam, meta_ab_no = load_dino_ablation_results(
        config.DAVIS_ABLATION_OUTPUT_DIR_NO_SAM, silent=True
    )
    ablation_2_sam, meta_ab2_sam = load_dino_ablation_results(
        config.DAVIS_ABLATION_2_OUTPUT_DIR_SAM, silent=True
    )
    ablation_2_no_sam, meta_ab2_no = load_dino_ablation_results(
        config.DAVIS_ABLATION_2_OUTPUT_DIR_NO_SAM, silent=True
    )
    ablation_3_sam, meta_ab3_sam = load_dino_ablation_results(
        config.DAVIS_ABLATION_3_OUTPUT_DIR_SAM, silent=True
    )
    ablation_3_no_sam, meta_ab3_no = load_dino_ablation_results(
        config.DAVIS_ABLATION_3_OUTPUT_DIR_NO_SAM, silent=True
    )
    ablation_4_sam, meta_ab4_sam = load_dino_ablation_results(
        config.DAVIS_ABLATION_4_OUTPUT_DIR_SAM, silent=True
    )
    ablation_4_no_sam, meta_ab4_no = load_dino_ablation_results(
        config.DAVIS_ABLATION_4_OUTPUT_DIR_NO_SAM, silent=True
    )

    _print_davis_inputs_loaded(
        out_dir,
        ct_results=ct_results,
        dino_sam=dino_results,
        dino_no_sam=dino_no_sam,
        sub_sam=sub_sam,
        sub_no_sam=sub_no_sam,
        ablation_sam=ablation_sam,
        ablation_no_sam=ablation_no_sam,
        meta_ab_sam=meta_ab_sam,
        meta_ab_no=meta_ab_no,
        ablation_2_sam=ablation_2_sam,
        ablation_2_no_sam=ablation_2_no_sam,
        meta_ab2_sam=meta_ab2_sam,
        meta_ab2_no=meta_ab2_no,
        ablation_3_sam=ablation_3_sam,
        ablation_3_no_sam=ablation_3_no_sam,
        meta_ab3_sam=meta_ab3_sam,
        meta_ab3_no=meta_ab3_no,
        ablation_4_sam=ablation_4_sam,
        ablation_4_no_sam=ablation_4_no_sam,
        meta_ab4_sam=meta_ab4_sam,
        meta_ab4_no=meta_ab4_no,
    )

    print(f"{t['dim']}Building comparison & printing tables…{t['reset']}")

    # ---- Build comparison -------------------------------------------------
    comparison = build_comparison(
        dino_results,
        ct_results,
        ablation_sam=ablation_sam,
        ablation_no_sam=ablation_no_sam,
        ablation_2_sam=ablation_2_sam,
        ablation_2_no_sam=ablation_2_no_sam,
        ablation_3_sam=ablation_3_sam,
        ablation_3_no_sam=ablation_3_no_sam,
        ablation_4_sam=ablation_4_sam,
        ablation_4_no_sam=ablation_4_no_sam,
    )

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

    # ---- Print analysis to stdout -----------------------------------------
    print_davis_analysis_report(
        dino_sam=dino_results,
        dino_no_sam=dino_no_sam,
        sub_sam=sub_sam,
        sub_no_sam=sub_no_sam,
        ablation_sam=ablation_sam,
        ablation_no_sam=ablation_no_sam,
        meta_ab_sam=meta_ab_sam,
        meta_ab_no=meta_ab_no,
        ablation_2_sam=ablation_2_sam,
        ablation_2_no_sam=ablation_2_no_sam,
        meta_ab2_sam=meta_ab2_sam,
        meta_ab2_no=meta_ab2_no,
        ablation_3_sam=ablation_3_sam,
        ablation_3_no_sam=ablation_3_no_sam,
        meta_ab3_sam=meta_ab3_sam,
        meta_ab3_no=meta_ab3_no,
        ablation_4_sam=ablation_4_sam,
        ablation_4_no_sam=ablation_4_no_sam,
        meta_ab4_sam=meta_ab4_sam,
        meta_ab4_no=meta_ab4_no,
        ct_results=ct_results,
    )

    # ---- Write outputs ----------------------------------------------------
    print(f"{t['dim']}Writing JSON / CSV…{t['reset']}")

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
                "matter_jaccard_fixed_mean": _safe_mean(
                    [
                        r["dino_matter_jaccard_fixed"]
                        for r in comparison
                        if not np.isnan(
                            r.get("dino_matter_jaccard_fixed", float("nan"))
                        )
                    ]
                ),
                "matter_jaccard_fixed_std": _safe_std(
                    [
                        r["dino_matter_jaccard_fixed"]
                        for r in comparison
                        if not np.isnan(
                            r.get("dino_matter_jaccard_fixed", float("nan"))
                        )
                    ]
                ),
            },
            "dino_ablation_sam": {
                "matter_jaccard_fixed_mean": _safe_mean(
                    [
                        r["ablation_sam_matter_jaccard_fixed"]
                        for r in comparison
                        if not np.isnan(
                            r.get("ablation_sam_matter_jaccard_fixed", float("nan"))
                        )
                    ]
                ),
                "matter_jaccard_fixed_std": _safe_std(
                    [
                        r["ablation_sam_matter_jaccard_fixed"]
                        for r in comparison
                        if not np.isnan(
                            r.get("ablation_sam_matter_jaccard_fixed", float("nan"))
                        )
                    ]
                ),
            },
            "dino_ablation_no_sam": {
                "matter_jaccard_fixed_mean": _safe_mean(
                    [
                        r["ablation_no_sam_matter_jaccard_fixed"]
                        for r in comparison
                        if not np.isnan(
                            r.get("ablation_no_sam_matter_jaccard_fixed", float("nan"))
                        )
                    ]
                ),
                "matter_jaccard_fixed_std": _safe_std(
                    [
                        r["ablation_no_sam_matter_jaccard_fixed"]
                        for r in comparison
                        if not np.isnan(
                            r.get("ablation_no_sam_matter_jaccard_fixed", float("nan"))
                        )
                    ]
                ),
            },
            "dino_ablation_2_sam": {
                "matter_jaccard_fixed_mean": _safe_mean(
                    [
                        r["ablation_2_sam_matter_jaccard_fixed"]
                        for r in comparison
                        if not np.isnan(
                            r.get("ablation_2_sam_matter_jaccard_fixed", float("nan"))
                        )
                    ]
                ),
                "matter_jaccard_fixed_std": _safe_std(
                    [
                        r["ablation_2_sam_matter_jaccard_fixed"]
                        for r in comparison
                        if not np.isnan(
                            r.get("ablation_2_sam_matter_jaccard_fixed", float("nan"))
                        )
                    ]
                ),
            },
            "dino_ablation_2_no_sam": {
                "matter_jaccard_fixed_mean": _safe_mean(
                    [
                        r["ablation_2_no_sam_matter_jaccard_fixed"]
                        for r in comparison
                        if not np.isnan(
                            r.get("ablation_2_no_sam_matter_jaccard_fixed", float("nan"))
                        )
                    ]
                ),
                "matter_jaccard_fixed_std": _safe_std(
                    [
                        r["ablation_2_no_sam_matter_jaccard_fixed"]
                        for r in comparison
                        if not np.isnan(
                            r.get("ablation_2_no_sam_matter_jaccard_fixed", float("nan"))
                        )
                    ]
                ),
            },
            "dino_ablation_3_sam": {
                "matter_jaccard_fixed_mean": _safe_mean(
                    [
                        r["ablation_3_sam_matter_jaccard_fixed"]
                        for r in comparison
                        if not np.isnan(
                            r.get("ablation_3_sam_matter_jaccard_fixed", float("nan"))
                        )
                    ]
                ),
                "matter_jaccard_fixed_std": _safe_std(
                    [
                        r["ablation_3_sam_matter_jaccard_fixed"]
                        for r in comparison
                        if not np.isnan(
                            r.get("ablation_3_sam_matter_jaccard_fixed", float("nan"))
                        )
                    ]
                ),
            },
            "dino_ablation_3_no_sam": {
                "matter_jaccard_fixed_mean": _safe_mean(
                    [
                        r["ablation_3_no_sam_matter_jaccard_fixed"]
                        for r in comparison
                        if not np.isnan(
                            r.get("ablation_3_no_sam_matter_jaccard_fixed", float("nan"))
                        )
                    ]
                ),
                "matter_jaccard_fixed_std": _safe_std(
                    [
                        r["ablation_3_no_sam_matter_jaccard_fixed"]
                        for r in comparison
                        if not np.isnan(
                            r.get("ablation_3_no_sam_matter_jaccard_fixed", float("nan"))
                        )
                    ]
                ),
            },
            "dino_ablation_4_sam": {
                "matter_jaccard_fixed_mean": _safe_mean(
                    [
                        r["ablation_4_sam_matter_jaccard_fixed"]
                        for r in comparison
                        if not np.isnan(
                            r.get("ablation_4_sam_matter_jaccard_fixed", float("nan"))
                        )
                    ]
                ),
                "matter_jaccard_fixed_std": _safe_std(
                    [
                        r["ablation_4_sam_matter_jaccard_fixed"]
                        for r in comparison
                        if not np.isnan(
                            r.get("ablation_4_sam_matter_jaccard_fixed", float("nan"))
                        )
                    ]
                ),
            },
            "dino_ablation_4_no_sam": {
                "matter_jaccard_fixed_mean": _safe_mean(
                    [
                        r["ablation_4_no_sam_matter_jaccard_fixed"]
                        for r in comparison
                        if not np.isnan(
                            r.get("ablation_4_no_sam_matter_jaccard_fixed", float("nan"))
                        )
                    ]
                ),
                "matter_jaccard_fixed_std": _safe_std(
                    [
                        r["ablation_4_no_sam_matter_jaccard_fixed"]
                        for r in comparison
                        if not np.isnan(
                            r.get("ablation_4_no_sam_matter_jaccard_fixed", float("nan"))
                        )
                    ]
                ),
            },
        },
        "ablation_loading": {
            "sam": meta_ab_sam,
            "no_sam": meta_ab_no,
        },
        "ablation_2_loading": {
            "sam": meta_ab2_sam,
            "no_sam": meta_ab2_no,
        },
        "ablation_3_loading": {
            "sam": meta_ab3_sam,
            "no_sam": meta_ab3_no,
        },
        "ablation_4_loading": {
            "sam": meta_ab4_sam,
            "no_sam": meta_ab4_no,
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

    print(f"{t['green']}Done.{t['reset']}\n")


if __name__ == "__main__":
    main()
