"""Gestalt postprocessing: main comparison (GenMatter vs SegAnyMo vs FlowSAM) and
optional depth-ablation comparison (baseline vs depth-ablation runs).

Writes gestalt_*.json/csv and gestalt_ablation_comparison.* under the postprocessing output dir.
"""

import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import stats
from scipy.ndimage import zoom

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

from postprocess_gestalt_ablation import run_gestalt_ablation_postprocess  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCENES = list(config.GESTALT_SCENES)
TEXTURES = list(config.GESTALT_TEXTURES)
NUM_FRAMES = 5
NUM_PROBES = 100
RANDOM_SEED = 42
NUM_RUNS = 5
TARGET_SIZE = 96

# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------


def load_ground_truth_mask(scene: str, frame_idx: int) -> np.ndarray | None:
    """Load ground truth binary mask (1000x1000)."""
    mask_file = os.path.join(
        config.GESTALT_BASE_PATH,
        scene,
        "render_passes",
        "masks",
        f"Image{frame_idx + 1:04d}.png",
    )
    if not os.path.exists(mask_file):
        return None
    mask_img = np.array(Image.open(mask_file))
    if mask_img.ndim == 3:
        mask_img = mask_img[:, :, 0]
    return mask_img > 127


def load_seganymo_mask(scene: str, texture: str, frame_idx: int) -> np.ndarray | None:
    """Load SegAnyMo mask (1000x1000)."""
    if config.SEGANYMO_BASE_PATH is None:
        return None
    mask_file = os.path.join(
        config.SEGANYMO_BASE_PATH,
        f"{scene}_{texture}",
        "sam2",
        "initial_preds",
        "output_six_frame",
        f"{frame_idx:05d}.png",
    )
    if not os.path.exists(mask_file):
        return None
    mask_img = np.array(Image.open(mask_file))
    return mask_img[:, :, 0] if mask_img.ndim == 3 else mask_img


def load_flowsam_mask(scene: str, texture: str, frame_idx: int) -> np.ndarray | None:
    """Load FlowSAM mask (1000x1000)."""
    mask_file = os.path.join(
        config.GESTALT_BASE_PATH,
        scene,
        texture,
        config.FLOWSAM_MASKS_SUBPATH,
        f"frame_{frame_idx:05d}_matched.png",
    )
    if not os.path.exists(mask_file):
        return None
    mask_img = np.array(Image.open(mask_file))
    return mask_img[:, :, 0] if mask_img.ndim == 3 else mask_img


def load_genmatter_mask_single_run(
    scene: str, texture: str, frame_idx: int, run_idx: int = 0
) -> np.ndarray | None:
    """Load GenMatter assignment mask from a single run (96x96)."""
    path = os.path.join(
        config.GESTALT_OUTPUT_DIR, scene, texture, f"run_{run_idx}", "assignments.npz"
    )
    if not os.path.exists(path):
        return None
    data = np.load(path, allow_pickle=True)
    key = "pixel_hyperblob_assignments"
    if key not in data or frame_idx >= len(data[key]):
        return None
    return data[key][frame_idx].astype(np.int32)


def load_genmatter_mask(
    scene: str, texture: str, frame_idx: int, num_runs: int = NUM_RUNS
) -> np.ndarray | None:
    """Load GenMatter mask averaged across runs using mode (96x96)."""
    masks = [
        load_genmatter_mask_single_run(scene, texture, frame_idx, i)
        for i in range(num_runs)
    ]
    masks = [m for m in masks if m is not None]
    if len(masks) == 0:
        return None
    if len(masks) == 1:
        return masks[0]
    masks_stack = np.stack(masks, axis=0)
    mode_mask, _ = stats.mode(masks_stack, axis=0, keepdims=False)
    return mode_mask.astype(np.int32)


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def evaluate_segmentation(
    true_mask: np.ndarray,
    predicted_mask: np.ndarray,
    num_probes: int = NUM_PROBES,
    random_seed: int | None = None,
) -> tuple[float | None, list[float]]:
    """Probe-point accuracy: sample probes from the GT object region,
    check whether the predicted segment that each probe falls into matches GT."""
    if true_mask is None or predicted_mask is None:
        return None, []

    if true_mask.shape != predicted_mask.shape:
        zf = (
            true_mask.shape[0] / predicted_mask.shape[0],
            true_mask.shape[1] / predicted_mask.shape[1],
        )
        predicted_mask = zoom(predicted_mask.astype(float), zf, order=0).astype(int)

    true_flat = true_mask.flatten()
    pred_flat = predicted_mask.flatten()

    object_indices = np.where(true_flat)[0]
    if len(object_indices) == 0:
        return 0.0, []

    if random_seed is not None:
        np.random.seed(random_seed)
    num_samples = min(num_probes, len(object_indices))
    probe_indices = np.random.choice(object_indices, size=num_samples, replace=False)

    probe_accuracies: list[float] = []
    for probe_idx in probe_indices:
        segment_id = pred_flat[probe_idx]
        segment_mask = pred_flat == segment_id
        accuracy = float(np.mean(true_flat == segment_mask))
        probe_accuracies.append(accuracy)

    return float(np.mean(probe_accuracies)), probe_accuracies


def compute_binary_metrics(
    true_mask: np.ndarray,
    predicted_mask: np.ndarray,
    num_probes: int = NUM_PROBES,
    random_seed: int | None = None,
) -> dict[str, float] | None:
    """Per-probe precision / recall / F1 / Jaccard / FPR / FNR, averaged."""
    if true_mask is None or predicted_mask is None:
        return None

    if true_mask.shape != predicted_mask.shape:
        zf = (
            true_mask.shape[0] / predicted_mask.shape[0],
            true_mask.shape[1] / predicted_mask.shape[1],
        )
        predicted_mask = zoom(predicted_mask.astype(float), zf, order=0).astype(int)

    true_flat = true_mask.flatten()
    pred_flat = predicted_mask.flatten()

    object_indices = np.where(true_flat)[0]
    if len(object_indices) == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "fpr": 0.0,
            "fnr": 1.0,
            "jaccard": 0.0,
        }

    if random_seed is not None:
        np.random.seed(random_seed)
    num_samples = min(num_probes, len(object_indices))
    probe_indices = np.random.choice(object_indices, size=num_samples, replace=False)

    precisions, recalls, f1s, fprs, fnrs, jaccards = [], [], [], [], [], []

    for probe_idx in probe_indices:
        segment_id = pred_flat[probe_idx]
        pred_seg = pred_flat == segment_id

        tp = np.sum(pred_seg & true_flat)
        fp = np.sum(pred_seg & ~true_flat)
        fn = np.sum(~pred_seg & true_flat)
        tn = np.sum(~pred_seg & ~true_flat)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        jaccard = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

        precisions.append(float(precision))
        recalls.append(float(recall))
        f1s.append(float(f1))
        fprs.append(float(fpr))
        fnrs.append(float(fnr))
        jaccards.append(float(jaccard))

    return {
        "precision": float(np.mean(precisions)),
        "recall": float(np.mean(recalls)),
        "f1": float(np.mean(f1s)),
        "fpr": float(np.mean(fprs)),
        "fnr": float(np.mean(fnrs)),
        "jaccard": float(np.mean(jaccards)),
    }


# ---------------------------------------------------------------------------
# Per-frame evaluation (all three methods)
# ---------------------------------------------------------------------------


def evaluate_all_methods(
    scene: str,
    texture: str,
    frame_idx: int,
    num_probes: int = NUM_PROBES,
    random_seed: int = RANDOM_SEED,
) -> dict | None:
    """Evaluate SegAnyMo, FlowSAM, and GenMatter on a single frame.

    All methods are compared at the GenMatter native resolution (96x96).
    A deterministic seed per (scene, texture, frame) ensures reproducibility.
    """
    true_mask = load_ground_truth_mask(scene, frame_idx)
    if true_mask is None:
        return None

    scene_num = int(scene.split("_")[1])
    texture_num = int(texture.split("_")[1])
    eval_seed = random_seed + scene_num * 1000 + texture_num * 10 + frame_idx

    genmatter_mask = load_genmatter_mask(scene, texture, frame_idx)
    if genmatter_mask is not None:
        target_size = genmatter_mask.shape[0]
    else:
        target_size = TARGET_SIZE

    zf_gt = (target_size / true_mask.shape[0], target_size / true_mask.shape[1])
    true_mask_resized = zoom(true_mask.astype(float), zf_gt, order=0) > 0.5

    def _eval_method(raw_mask, seed_offset):
        if raw_mask is None:
            return None, [], None
        if raw_mask.shape[0] != target_size or raw_mask.shape[1] != target_size:
            zf = (target_size / raw_mask.shape[0], target_size / raw_mask.shape[1])
            resized = zoom(raw_mask.astype(float), zf, order=0).astype(int)
        else:
            resized = raw_mask
        acc, probes = evaluate_segmentation(
            true_mask_resized, resized, num_probes, eval_seed + seed_offset
        )
        metrics = compute_binary_metrics(
            true_mask_resized, resized, num_probes, eval_seed + seed_offset + 1
        )
        return acc, probes, metrics

    seg_acc, seg_probes, seg_metrics = _eval_method(
        load_seganymo_mask(scene, texture, frame_idx), 0
    )
    fs_acc, fs_probes, fs_metrics = _eval_method(
        load_flowsam_mask(scene, texture, frame_idx), 2
    )
    gm_acc, gm_probes, gm_metrics = _eval_method(genmatter_mask, 4)

    return {
        "seganymo_acc": seg_acc,
        "flowsam_acc": fs_acc,
        "genmatter_acc": gm_acc,
        "seganymo_probes": seg_probes,
        "flowsam_probes": fs_probes,
        "genmatter_probes": gm_probes,
        "seganymo_metrics": seg_metrics,
        "flowsam_metrics": fs_metrics,
        "genmatter_metrics": gm_metrics,
    }


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def _average_metrics(metrics_list: list[dict | None]) -> dict | None:
    valid = [m for m in metrics_list if m is not None]
    if not valid:
        return None
    return {
        k: float(np.mean([m[k] for m in valid]))
        for k in ("precision", "recall", "f1", "fpr", "fnr", "jaccard")
    }


def _metrics_summary(all_results: list[dict], key: str) -> dict | None:
    metrics_list = [r[key] for r in all_results if r[key] is not None]
    if not metrics_list:
        return None
    out = {}
    for k in ("precision", "recall", "f1", "jaccard", "fpr", "fnr"):
        vals = [m[k] for m in metrics_list]
        out[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return out


def _numpy_safe(obj):
    """JSON serialiser for numpy scalars."""
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    np.random.seed(RANDOM_SEED)

    output_dir = Path(config.POSTPROCESSING_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Gestalt postprocessing")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print()
    print("[1] Main comparison (GenMatter vs SegAnyMo vs FlowSAM)")
    print(f"    GenMatter runs: {config.GESTALT_OUTPUT_DIR}")
    print(f"    GT / FlowSAM:   {config.GESTALT_BASE_PATH}")
    if config.SEGANYMO_BASE_PATH is not None:
        print(f"    SegAnyMo:       {config.SEGANYMO_BASE_PATH}")
    else:
        print("    SegAnyMo:       (SEGANYMO_BASE_PATH not set)")
    print()
    print("[2] Depth ablation (full-depth vs depth-ablation GenMatter)")
    print(f"    Baseline runs:  {config.GESTALT_OUTPUT_DIR}")
    print(f"    Ablation runs:  {config.GESTALT_DEPTH_ABLATION_OUTPUT_DIR}")
    print("=" * 80)
    print()

    # Discover which scene/texture combos have data
    existing_combinations: list[tuple[str, str]] = []
    for scene in SCENES:
        for texture in TEXTURES:
            seg_dir = (
                os.path.join(
                    config.SEGANYMO_BASE_PATH,
                    f"{scene}_{texture}",
                    "sam2",
                    "initial_preds",
                    "output_six_frame",
                )
                if config.SEGANYMO_BASE_PATH is not None
                else None
            )
            flowsam_dir = os.path.join(
                config.GESTALT_BASE_PATH,
                scene,
                texture,
                "masks",
                "flowsam_matched",
            )
            gm_dir = os.path.join(config.GESTALT_OUTPUT_DIR, scene, texture)
            has_gm = any(
                os.path.exists(os.path.join(gm_dir, f"run_{i}"))
                for i in range(NUM_RUNS)
            )
            if (
                (seg_dir is not None and os.path.exists(seg_dir))
                or os.path.exists(flowsam_dir)
                or has_gm
            ):
                existing_combinations.append((scene, texture))

    if not existing_combinations:
        print(
            "Skipping [1] main comparison: no SegAnyMo / FlowSAM / GenMatter run data found for any scene–texture.\n"
        )

    all_results: list[dict] = []
    if existing_combinations:
        print(f"[1] Processing {len(existing_combinations)} scene–texture combinations\n")

        # ---- per-combo evaluation ----
        for scene, texture in existing_combinations:
            print(f"  {scene} / {texture}")

            seg_frames: list[float | None] = []
            fs_frames: list[float | None] = []
            gm_frames: list[float | None] = []

            seg_m_frames: list[dict | None] = []
            fs_m_frames: list[dict | None] = []
            gm_m_frames: list[dict | None] = []

            for frame_idx in range(NUM_FRAMES):
                result = evaluate_all_methods(scene, texture, frame_idx, NUM_PROBES)
                if result is None:
                    seg_frames.append(None)
                    fs_frames.append(None)
                    gm_frames.append(None)
                    seg_m_frames.append(None)
                    fs_m_frames.append(None)
                    gm_m_frames.append(None)
                else:
                    seg_frames.append(result["seganymo_acc"])
                    fs_frames.append(result["flowsam_acc"])
                    gm_frames.append(result["genmatter_acc"])
                    seg_m_frames.append(result["seganymo_metrics"])
                    fs_m_frames.append(result["flowsam_metrics"])
                    gm_m_frames.append(result["genmatter_metrics"])

            def _mean_valid(vals):
                v = [x for x in vals if x is not None]
                return (float(np.mean(v)), float(np.std(v))) if v else (None, None)

            seg_mean, seg_std = _mean_valid(seg_frames)
            fs_mean, fs_std = _mean_valid(fs_frames)
            gm_mean, gm_std = _mean_valid(gm_frames)

            all_results.append(
                {
                    "scene": scene,
                    "texture": texture,
                    "seganymo_mean": seg_mean,
                    "seganymo_std": seg_std,
                    "flowsam_mean": fs_mean,
                    "flowsam_std": fs_std,
                    "genmatter_mean": gm_mean,
                    "genmatter_std": gm_std,
                    "seganymo_frames": seg_frames,
                    "flowsam_frames": fs_frames,
                    "genmatter_frames": gm_frames,
                    "seganymo_metrics": _average_metrics(seg_m_frames),
                    "flowsam_metrics": _average_metrics(fs_m_frames),
                    "genmatter_metrics": _average_metrics(gm_m_frames),
                }
            )

        # ---- save all_results.json ----
        all_results_path = output_dir / "gestalt_all_results.json"
        with open(all_results_path, "w") as f:
            json.dump(all_results, f, indent=2, default=_numpy_safe)

        # ---- summary ----
        seganymo_all = [r["seganymo_mean"] for r in all_results if r["seganymo_mean"] is not None]
        flowsam_all = [r["flowsam_mean"] for r in all_results if r["flowsam_mean"] is not None]
        genmatter_all = [r["genmatter_mean"] for r in all_results if r["genmatter_mean"] is not None]

        summary: dict = {}
        for name, vals, mkey in [
            ("seganymo", seganymo_all, "seganymo_metrics"),
            ("flowsam", flowsam_all, "flowsam_metrics"),
            ("genmatter", genmatter_all, "genmatter_metrics"),
        ]:
            summary[name] = {
                "n": len(vals),
                "accuracy": {
                    "mean": float(np.mean(vals)) if vals else None,
                    "std": float(np.std(vals)) if vals else None,
                },
                "metrics": _metrics_summary(all_results, mkey),
            }

        # Paired comparisons (only when all three methods have data)
        paired = [
            (r["seganymo_mean"], r["flowsam_mean"], r["genmatter_mean"])
            for r in all_results
            if r["seganymo_mean"] is not None
            and r["flowsam_mean"] is not None
            and r["genmatter_mean"] is not None
        ]
        if paired:
            p_seg, p_fs, p_gm = zip(*paired)
            p_seg, p_fs, p_gm = list(p_seg), list(p_fs), list(p_gm)

            def _comparison(a, b, name_a, name_b):
                diff = np.array(a) - np.array(b)
                t, p = stats.ttest_rel(a, b)
                return {
                    "n": len(a),
                    "mean_diff": float(np.mean(diff)),
                    "std_diff": float(np.std(diff)),
                    f"{name_a}_wins": int(np.sum(diff > 0)),
                    f"{name_b}_wins": int(np.sum(diff < 0)),
                    "t_stat": float(t),
                    "p_value": float(p),
                }

            summary["comparisons"] = {
                "seganymo_vs_flowsam": _comparison(p_seg, p_fs, "seganymo", "flowsam"),
                "seganymo_vs_genmatter": _comparison(p_seg, p_gm, "seganymo", "genmatter"),
                "flowsam_vs_genmatter": _comparison(p_fs, p_gm, "flowsam", "genmatter"),
            }

        summary_path = output_dir / "gestalt_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=_numpy_safe)

        # ---- CSV (one row per scene/texture) ----
        csv_path = output_dir / "gestalt_results.csv"
        fieldnames = [
            "scene",
            "texture",
            "seganymo_accuracy",
            "flowsam_accuracy",
            "genmatter_accuracy",
            "seganymo_jaccard",
            "flowsam_jaccard",
            "genmatter_jaccard",
            "seganymo_precision",
            "flowsam_precision",
            "genmatter_precision",
            "seganymo_recall",
            "flowsam_recall",
            "genmatter_recall",
            "seganymo_f1",
            "flowsam_f1",
            "genmatter_f1",
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in all_results:
                row: dict = {"scene": r["scene"], "texture": r["texture"]}
                for method in ("seganymo", "flowsam", "genmatter"):
                    row[f"{method}_accuracy"] = r[f"{method}_mean"]
                    m = r[f"{method}_metrics"]
                    for metric in ("jaccard", "precision", "recall", "f1"):
                        row[f"{method}_{metric}"] = m[metric] if m else None
                writer.writerow(row)

        # ---- stdout summary table ----
        _print_summary(all_results, summary)

        print(f"\nOutputs written to {output_dir}/")
        print(f"  - gestalt_all_results.json  ({len(all_results)} combos)")
        print("  - gestalt_summary.json")
        print("  - gestalt_results.csv")

    run_gestalt_ablation_postprocess()


# ---------------------------------------------------------------------------
# Pretty-print summary
# ---------------------------------------------------------------------------


def _print_summary(all_results: list[dict], summary: dict) -> None:
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS - ACCURACY")
    print("=" * 80)
    for name in ("seganymo", "flowsam", "genmatter"):
        s = summary[name]
        acc = s["accuracy"]
        if acc["mean"] is not None:
            print(
                f"\n  {name:12s}  (N={s['n']:>3d})  "
                f"Mean: {acc['mean']:.4f} +/- {acc['std']:.4f}"
            )

    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS - PRECISION, RECALL, F1, JACCARD")
    print("=" * 80)
    for name in ("seganymo", "flowsam", "genmatter"):
        m = summary[name].get("metrics")
        if m is None:
            continue
        print(f"\n  {name} (N={summary[name]['n']}):")
        for k in ("precision", "recall", "f1", "jaccard", "fpr", "fnr"):
            print(f"    {k:12s}: {m[k]['mean']:.4f} +/- {m[k]['std']:.4f}")

    if "comparisons" in summary:
        print("\n" + "=" * 80)
        print("PAIRWISE COMPARISONS - ACCURACY")
        print("=" * 80)
        for label, comp in summary["comparisons"].items():
            print(
                f"\n  {label}: mean_diff={comp['mean_diff']:+.4f}  "
                f"t={comp['t_stat']:.4f}  p={comp['p_value']:.6f}"
            )

    # Per-scene breakdown
    by_scene: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"seganymo": [], "flowsam": [], "genmatter": []}
    )
    for r in all_results:
        for method in ("seganymo", "flowsam", "genmatter"):
            val = r[f"{method}_mean"]
            if val is not None:
                by_scene[r["scene"]][method].append(val)

    print("\n" + "=" * 80)
    print("PER-SCENE BREAKDOWN")
    print("=" * 80)
    header = f"{'Scene':<15} {'N':>3}  {'SegAnyMo':>15}  {'FlowSAM':>15}  {'GenMatter':>15}"
    print(header)
    print("-" * len(header))
    for scene in sorted(by_scene):
        d = by_scene[scene]
        if not (d["seganymo"] and d["flowsam"] and d["genmatter"]):
            continue
        n = len(d["seganymo"])
        parts = []
        for method in ("seganymo", "flowsam", "genmatter"):
            vals = d[method]
            parts.append(f"{np.mean(vals):.3f}+/-{np.std(vals):.3f}")
        print(f"{scene:<15} {n:>3}  {'  '.join(f'{p:>15}' for p in parts)}")

    # Per-texture breakdown
    by_texture: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"seganymo": [], "flowsam": [], "genmatter": []}
    )
    for r in all_results:
        for method in ("seganymo", "flowsam", "genmatter"):
            val = r[f"{method}_mean"]
            if val is not None:
                by_texture[r["texture"]][method].append(val)

    print("\n" + "=" * 80)
    print("PER-TEXTURE BREAKDOWN")
    print("=" * 80)
    header = f"{'Texture':<15} {'N':>3}  {'SegAnyMo':>15}  {'FlowSAM':>15}  {'GenMatter':>15}"
    print(header)
    print("-" * len(header))
    for texture in sorted(by_texture):
        d = by_texture[texture]
        if not (d["seganymo"] and d["flowsam"] and d["genmatter"]):
            continue
        n = len(d["seganymo"])
        parts = []
        for method in ("seganymo", "flowsam", "genmatter"):
            vals = d[method]
            parts.append(f"{np.mean(vals):.3f}+/-{np.std(vals):.3f}")
        print(f"{texture:<15} {n:>3}  {'  '.join(f'{p:>15}' for p in parts)}")


if __name__ == "__main__":
    main()
