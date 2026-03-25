"""Postprocess Gestalt ablation results.

Compares baseline GenMatter (full depth) vs depth-ablation GenMatter results
across all Gestalt scenes and textures. Computes probe-point segmentation
metrics (accuracy, Jaccard, precision, recall, F1) and outputs summary
statistics as JSON and CSV.
"""

import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm
from PIL import Image
from scipy import stats
from scipy.ndimage import zoom

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

GESTALT_SCENES = list(config.GESTALT_SCENES)
GESTALT_TEXTURES = list(config.GESTALT_TEXTURES)
NUM_FRAMES = 6
NUM_PROBES = 100
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_ground_truth_mask(scene: str, frame_idx: int) -> np.ndarray | None:
    mask_file = os.path.join(
        config.GESTALT_BASE_PATH, scene, "render_passes", "masks",
        f"Image{frame_idx + 1:04d}.png",
    )
    if not os.path.exists(mask_file):
        return None
    mask_img = np.array(Image.open(mask_file))
    if mask_img.ndim == 3:
        mask_img = mask_img[:, :, 0]
    return mask_img > 127


def load_genmatter_mask(
    scene: str,
    texture: str,
    frame_idx: int,
    base_path: str | Path,
    run_idx: int = 0,
) -> np.ndarray | None:
    path = os.path.join(base_path, scene, texture, f"run_{run_idx}", "assignments.npz")
    if not os.path.exists(path):
        return None
    data = np.load(path, allow_pickle=True)
    if "pixel_hyperblob_assignments" not in data:
        return None
    assignments = data["pixel_hyperblob_assignments"]
    if frame_idx >= len(assignments):
        return None
    return assignments[frame_idx].astype(np.int32)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def evaluate_segmentation_accuracy(
    true_mask: np.ndarray,
    predicted_mask: np.ndarray,
    num_probes: int = NUM_PROBES,
    random_seed: int | None = None,
) -> float | None:
    """Probe-point accuracy (NO SKIPPING): sample probes from the GT object
    region, look up predicted segment ID at each probe, measure agreement."""
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
        return 0.0

    rng = np.random.RandomState(random_seed)
    n = min(num_probes, len(object_indices))
    probe_indices = rng.choice(object_indices, size=n, replace=False)

    accs = []
    for idx in probe_indices:
        seg_id = pred_flat[idx]
        seg_mask = pred_flat == seg_id
        accs.append(np.mean(true_flat == seg_mask))
    return float(np.mean(accs))


def compute_binary_metrics(
    true_mask: np.ndarray,
    predicted_mask: np.ndarray,
    num_probes: int = NUM_PROBES,
    random_seed: int | None = None,
) -> dict | None:
    """Probe-point binary metrics: precision, recall, F1, FPR, FNR, Jaccard."""
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
        return {k: 0.0 for k in ("precision", "recall", "f1", "fpr", "fnr", "jaccard")}

    rng = np.random.RandomState(random_seed)
    n = min(num_probes, len(object_indices))
    probe_indices = rng.choice(object_indices, size=n, replace=False)

    precisions, recalls, f1s, fprs, fnrs, jaccards = [], [], [], [], [], []

    for idx in probe_indices:
        seg_mask = pred_flat == pred_flat[idx]

        tp = np.sum(seg_mask & true_flat)
        fp = np.sum(seg_mask & ~true_flat)
        fn = np.sum(~seg_mask & true_flat)
        tn = np.sum(~seg_mask & ~true_flat)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        jaccard = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        fprs.append(fpr)
        fnrs.append(fnr)
        jaccards.append(jaccard)

    return {
        "precision": float(np.mean(precisions)),
        "recall": float(np.mean(recalls)),
        "f1": float(np.mean(f1s)),
        "fpr": float(np.mean(fprs)),
        "fnr": float(np.mean(fnrs)),
        "jaccard": float(np.mean(jaccards)),
    }


# ---------------------------------------------------------------------------
# Per-frame evaluation
# ---------------------------------------------------------------------------

def evaluate_frame(
    scene: str,
    texture: str,
    frame_idx: int,
    num_probes: int = NUM_PROBES,
    random_seed: int = RANDOM_SEED,
) -> dict | None:
    """Evaluate baseline and depth-ablation masks for a single frame."""
    true_mask = load_ground_truth_mask(scene, frame_idx)
    if true_mask is None:
        return None

    scene_num = int(scene.split("_")[1])
    texture_num = int(texture.split("_")[1])
    eval_seed = random_seed + scene_num * 1000 + texture_num * 10 + frame_idx

    baseline_mask = load_genmatter_mask(scene, texture, frame_idx, config.GESTALT_OUTPUT_DIR)
    ablation_mask = load_genmatter_mask(scene, texture, frame_idx, config.GESTALT_DEPTH_ABLATION_OUTPUT_DIR)

    target_size = 96
    if baseline_mask is not None:
        target_size = baseline_mask.shape[0]
    elif ablation_mask is not None:
        target_size = ablation_mask.shape[0]

    zf = (target_size / true_mask.shape[0], target_size / true_mask.shape[1])
    true_resized = zoom(true_mask.astype(float), zf, order=0) > 0.5

    result: dict = {}
    for tag, mask, seed_offset in [
        ("baseline", baseline_mask, 4),
        ("ablation", ablation_mask, 6),
    ]:
        if mask is not None:
            acc = evaluate_segmentation_accuracy(true_resized, mask, num_probes, eval_seed + seed_offset)
            metrics = compute_binary_metrics(true_resized, mask, num_probes, eval_seed + seed_offset + 1)
        else:
            acc = None
            metrics = None
        result[f"{tag}_acc"] = acc
        result[f"{tag}_metrics"] = metrics

    return result


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _average_metrics(metrics_list: list[dict | None]) -> dict | None:
    valid = [m for m in metrics_list if m is not None]
    if not valid:
        return None
    return {k: float(np.mean([m[k] for m in valid])) for k in valid[0]}


def _safe_float(x):
    if isinstance(x, (np.floating, np.integer)):
        return float(x)
    return x


def _term_styles() -> dict[str, str]:
    if not sys.stdout.isatty():
        return {k: "" for k in ("reset", "bold", "dim", "cyan", "green", "magenta", "yellow")}
    return {
        "reset": "\033[0m",
        "bold": "\033[1m",
        "dim": "\033[2m",
        "cyan": "\033[96m",
        "green": "\033[92m",
        "magenta": "\033[95m",
        "yellow": "\033[93m",
    }


def _fmt_pm(mean: float | None, std: float | None) -> str:
    if mean is None or std is None:
        return "N/A"
    return f"{mean:.4f} ± {std:.4f}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_gestalt_ablation_postprocess() -> bool:
    """Compare baseline vs depth-ablation GenMatter. Returns True if analysis ran."""
    np.random.seed(RANDOM_SEED)

    baseline_dir = str(config.GESTALT_OUTPUT_DIR)
    ablation_dir = str(config.GESTALT_DEPTH_ABLATION_OUTPUT_DIR)
    output_dir = config.POSTPROCESSING_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    any_combo = False
    for scene in GESTALT_SCENES:
        for texture in GESTALT_TEXTURES:
            if os.path.exists(os.path.join(baseline_dir, scene, texture, "run_0")) or os.path.exists(
                os.path.join(ablation_dir, scene, texture, "run_0")
            ):
                any_combo = True
                break
        if any_combo:
            break

    if not any_combo:
        print("\n" + "=" * 80)
        print("Skipping Gestalt depth-ablation comparison (no run_0 under baseline or ablation).")
        print(f"  Baseline: {baseline_dir}")
        print(f"  Ablation: {ablation_dir}")
        print("=" * 80)
        return False

    t0 = _term_styles()
    print(
        f"\n{t0['cyan']}{t0['bold']}Gestalt depth-ablation{t0['reset']}  "
        f"{t0['dim']}{output_dir}{t0['reset']}"
    )
    print(
        f"{t0['dim']}baseline: {baseline_dir}  ·  ablation: {ablation_dir}  ·  "
        f"GT: {config.GESTALT_BASE_PATH}{t0['reset']}\n"
    )

    # ------------------------------------------------------------------
    # Collect per-combination results
    # ------------------------------------------------------------------
    combos: list[tuple[str, str]] = []
    for scene in GESTALT_SCENES:
        for texture in GESTALT_TEXTURES:
            has_baseline = os.path.exists(os.path.join(baseline_dir, scene, texture, "run_0"))
            has_ablation = os.path.exists(os.path.join(ablation_dir, scene, texture, "run_0"))
            if has_baseline or has_ablation:
                combos.append((scene, texture))

    all_results: list[dict] = []
    for scene, texture in tqdm(combos, desc="Gestalt depth ablation", unit="combo", file=sys.stdout):
        baseline_accs: list[float | None] = []
        ablation_accs: list[float | None] = []
        baseline_metrics_frames: list[dict | None] = []
        ablation_metrics_frames: list[dict | None] = []

        for fi in range(NUM_FRAMES):
            res = evaluate_frame(scene, texture, fi, NUM_PROBES)
            if res is None:
                baseline_accs.append(None)
                ablation_accs.append(None)
                baseline_metrics_frames.append(None)
                ablation_metrics_frames.append(None)
            else:
                baseline_accs.append(res["baseline_acc"])
                ablation_accs.append(res["ablation_acc"])
                baseline_metrics_frames.append(res["baseline_metrics"])
                ablation_metrics_frames.append(res["ablation_metrics"])

        bl_valid = [a for a in baseline_accs if a is not None]
        ab_valid = [a for a in ablation_accs if a is not None]

        all_results.append({
            "scene": scene,
            "texture": texture,
            "baseline_mean_acc": float(np.mean(bl_valid)) if bl_valid else None,
            "baseline_std_acc": float(np.std(bl_valid)) if bl_valid else None,
            "ablation_mean_acc": float(np.mean(ab_valid)) if ab_valid else None,
            "ablation_std_acc": float(np.std(ab_valid)) if ab_valid else None,
            "baseline_metrics": _average_metrics(baseline_metrics_frames),
            "ablation_metrics": _average_metrics(ablation_metrics_frames),
        })

    # ------------------------------------------------------------------
    # Overall summary
    # ------------------------------------------------------------------
    bl_all = [r["baseline_mean_acc"] for r in all_results if r["baseline_mean_acc"] is not None]
    ab_all = [r["ablation_mean_acc"] for r in all_results if r["ablation_mean_acc"] is not None]

    summary: dict = {
        "baseline": {
            "n": len(bl_all),
            "accuracy": {"mean": float(np.mean(bl_all)), "std": float(np.std(bl_all))} if bl_all else None,
            "metrics": _metrics_summary(all_results, "baseline_metrics"),
        },
        "ablation": {
            "n": len(ab_all),
            "accuracy": {"mean": float(np.mean(ab_all)), "std": float(np.std(ab_all))} if ab_all else None,
            "metrics": _metrics_summary(all_results, "ablation_metrics"),
        },
    }

    # Paired comparison
    paired_bl = [r["baseline_mean_acc"] for r in all_results
                 if r["baseline_mean_acc"] is not None and r["ablation_mean_acc"] is not None]
    paired_ab = [r["ablation_mean_acc"] for r in all_results
                 if r["baseline_mean_acc"] is not None and r["ablation_mean_acc"] is not None]

    if paired_bl and paired_ab:
        diff = np.array(paired_bl) - np.array(paired_ab)
        t_stat, p_value = stats.ttest_rel(paired_bl, paired_ab)
        summary["paired_comparison"] = {
            "n": len(paired_bl),
            "mean_diff": float(np.mean(diff)),
            "std_diff": float(np.std(diff)),
            "baseline_wins": int(np.sum(diff > 0)),
            "ablation_wins": int(np.sum(diff < 0)),
            "t_stat": float(t_stat),
            "p_value": float(p_value),
            "pct_drop": float(np.mean(diff) / np.mean(paired_bl) * 100) if np.mean(paired_bl) else None,
        }

    # Per-scene breakdown
    by_scene: dict[str, dict[str, list]] = defaultdict(lambda: {"baseline": [], "ablation": []})
    for r in all_results:
        if r["baseline_mean_acc"] is not None:
            by_scene[r["scene"]]["baseline"].append(r["baseline_mean_acc"])
        if r["ablation_mean_acc"] is not None:
            by_scene[r["scene"]]["ablation"].append(r["ablation_mean_acc"])

    scene_breakdown = []
    for scene in sorted(by_scene):
        bl = by_scene[scene]["baseline"]
        ab = by_scene[scene]["ablation"]
        bl_mean = float(np.mean(bl)) if bl else None
        ab_mean = float(np.mean(ab)) if ab else None
        delta = (bl_mean - ab_mean) if (bl_mean is not None and ab_mean is not None) else None
        scene_breakdown.append({
            "scene": scene, "n": len(bl),
            "baseline_mean": bl_mean,
            "ablation_mean": ab_mean,
            "delta": delta,
        })
    summary["by_scene"] = scene_breakdown

    # Per-texture breakdown (accuracy + mean Jaccard per combo, averaged over scenes)
    by_texture: dict[str, dict[str, list]] = defaultdict(
        lambda: {"baseline": [], "ablation": [], "baseline_j": [], "ablation_j": []}
    )
    for r in all_results:
        tex = r["texture"]
        if r["baseline_mean_acc"] is not None:
            by_texture[tex]["baseline"].append(r["baseline_mean_acc"])
        if r["ablation_mean_acc"] is not None:
            by_texture[tex]["ablation"].append(r["ablation_mean_acc"])
        bm = r.get("baseline_metrics")
        if bm and bm.get("jaccard") is not None:
            by_texture[tex]["baseline_j"].append(float(bm["jaccard"]))
        am = r.get("ablation_metrics")
        if am and am.get("jaccard") is not None:
            by_texture[tex]["ablation_j"].append(float(am["jaccard"]))

    texture_breakdown = []
    for tex in sorted(by_texture):
        d = by_texture[tex]
        bl = d["baseline"]
        ab = d["ablation"]
        blj = d["baseline_j"]
        abj = d["ablation_j"]
        bl_mean = float(np.mean(bl)) if bl else None
        ab_mean = float(np.mean(ab)) if ab else None
        bl_j_mean = float(np.mean(blj)) if blj else None
        ab_j_mean = float(np.mean(abj)) if abj else None
        delta = (bl_mean - ab_mean) if (bl_mean is not None and ab_mean is not None) else None
        texture_breakdown.append({
            "texture": tex,
            "n": len(bl) or len(ab),
            "baseline_mean": bl_mean,
            "ablation_mean": ab_mean,
            "baseline_jaccard_mean": bl_j_mean,
            "ablation_jaccard_mean": ab_j_mean,
            "delta": delta,
        })
    summary["by_texture"] = texture_breakdown

    # Per-combination detail
    summary["per_combination"] = all_results

    # ------------------------------------------------------------------
    # Write JSON
    # ------------------------------------------------------------------
    json_path = os.path.join(output_dir, "gestalt_ablation_comparison.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=_safe_float)

    # ------------------------------------------------------------------
    # Write CSV (one row per scene/texture combination)
    # ------------------------------------------------------------------
    csv_path = os.path.join(output_dir, "gestalt_ablation_comparison.csv")
    metric_keys = ["precision", "recall", "f1", "fpr", "fnr", "jaccard"]
    fieldnames = ["scene", "texture", "baseline_acc", "ablation_acc", "delta_acc"]
    for prefix in ("baseline", "ablation"):
        for mk in metric_keys:
            fieldnames.append(f"{prefix}_{mk}")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            row = {
                "scene": r["scene"],
                "texture": r["texture"],
                "baseline_acc": r["baseline_mean_acc"],
                "ablation_acc": r["ablation_mean_acc"],
                "delta_acc": (
                    r["baseline_mean_acc"] - r["ablation_mean_acc"]
                    if r["baseline_mean_acc"] is not None and r["ablation_mean_acc"] is not None
                    else None
                ),
            }
            for prefix in ("baseline", "ablation"):
                m = r[f"{prefix}_metrics"]
                for mk in metric_keys:
                    row[f"{prefix}_{mk}"] = m[mk] if m else None
            writer.writerow(row)

    # ------------------------------------------------------------------
    # Print summary table to stdout
    # ------------------------------------------------------------------
    _print_summary(summary)
    tw = _term_styles()
    print(
        f"{tw['dim']}Wrote {json_path}  ·  {csv_path}{tw['reset']}\n"
    )
    return True


def _metrics_summary(all_results: list[dict], key: str) -> dict | None:
    metrics_list = [r[key] for r in all_results if r[key] is not None]
    if not metrics_list:
        return None
    out = {}
    for mk in metrics_list[0]:
        vals = [m[mk] for m in metrics_list]
        out[mk] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return out


def _print_summary(summary: dict) -> None:
    t = _term_styles()
    sep = "=" * 72
    w = 18

    bl_m = summary["baseline"].get("metrics") or {}
    ab_m = summary["ablation"].get("metrics") or {}
    bl_j = bl_m.get("jaccard", {})
    ab_j = ab_m.get("jaccard", {})

    bl_acc = summary["baseline"].get("accuracy")
    ab_acc = summary["ablation"].get("accuracy")

    bl_j_mean = bl_j.get("mean") if bl_j else None
    ab_j_mean = ab_j.get("mean") if ab_j else None
    if bl_j_mean is not None and ab_j_mean is not None:
        best_jacc = "baseline" if bl_j_mean >= ab_j_mean else "ablation"
    elif bl_j_mean is not None:
        best_jacc = "baseline"
    elif ab_j_mean is not None:
        best_jacc = "ablation"
    else:
        best_jacc = None

    print()
    print(sep)
    print(f"{t['cyan']}{t['bold']}Depth ablation — final results{t['reset']}")
    print(sep)

    for tag, label in [("baseline", "Baseline (full depth)"), ("ablation", "Ablation (no depth)")]:
        s = summary[tag]
        acc = s["accuracy"]
        m = s.get("metrics") or {}
        jm = m.get("jaccard", {})
        if acc is None:
            print(f"\n{t['dim']}{label}: no data{t['reset']}")
            continue
        hdr = f"{label} (N={s['n']})"
        is_best = tag == best_jacc and best_jacc is not None
        if is_best and jm:
            hdr = f"{t['bold']}{t['green']}{hdr}{t['reset']}"
        print(f"\n{hdr}")
        acc_s = _fmt_pm(acc["mean"], acc["std"])
        jac_s = _fmt_pm(jm.get("mean"), jm.get("std")) if jm else "N/A"
        if is_best and jm:
            print(f"  Accuracy: {t['bold']}{t['green']}{acc_s}{t['reset']}")
            print(f"  Jaccard:  {t['bold']}{t['green']}{jac_s}{t['reset']}")
        else:
            print(f"  Accuracy: {acc_s}")
            print(f"  Jaccard:  {jac_s}")

    pc = summary.get("paired_comparison")
    if pc:
        print(f"\n{sep}")
        print(f"{t['yellow']}{t['bold']}Paired comparison (accuracy){t['reset']}")
        print(t["dim"] + sep + t["reset"])
        print(
            f"  Δmean={pc['mean_diff']:+.4f} ± {pc['std_diff']:.4f}  ·  "
            f"wins: baseline {pc['baseline_wins']} / ablation {pc['ablation_wins']}  ·  "
            f"t={pc['t_stat']:.4f}, p={pc['p_value']:.6f}"
        )
        if pc.get("pct_drop") is not None:
            print(f"  {t['dim']}Rel. Δ accuracy: {pc['pct_drop']:.2f}%{t['reset']}")

    # Per-texture only (accuracy + Jaccard); no per-scene table
    by_tex = summary.get("by_texture", [])
    if by_tex:
        print(f"\n{sep}")
        print(f"{t['cyan']}{t['bold']}Per-texture breakdown{t['reset']}")
        print(sep)
        hdr = (
            f"{t['magenta']}{'Texture':<14} {'n':>3}  "
            f"{'B.Acc':>{w}} {'B.Jac':>{w}} {'A.Acc':>{w}} {'A.Jac':>{w}}{t['reset']}"
        )
        print(hdr)
        print(t["dim"] + "-" * 72 + t["reset"])
        for row in by_tex:
            tex = row["texture"]
            n = row["n"]
            bl_acc_m = row.get("baseline_mean")
            ab_acc_m = row.get("ablation_mean")
            b_jac = row.get("baseline_jaccard_mean")
            a_jac = row.get("ablation_jaccard_mean")
            b_acc_s = f"{bl_acc_m:.4f}" if bl_acc_m is not None else "N/A"
            a_acc_s = f"{ab_acc_m:.4f}" if ab_acc_m is not None else "N/A"
            b_j_s = f"{b_jac:.4f}" if b_jac is not None else "N/A"
            a_j_s = f"{a_jac:.4f}" if a_jac is not None else "N/A"
            bj = float(b_jac) if b_jac is not None else float("nan")
            aj = float(a_jac) if a_jac is not None else float("nan")
            if not np.isnan(bj) and not np.isnan(aj):
                jac_best = 0 if bj >= aj else 1
            elif not np.isnan(bj):
                jac_best = 0
            elif not np.isnan(aj):
                jac_best = 1
            else:
                jac_best = -1

            def _fmt_cell(val: str, *, highlight: bool) -> str:
                if val == "N/A" or not highlight:
                    return f"{val:>{w}}"
                return f"{t['bold']}{t['green']}{val:>{w}}{t['reset']}"

            print(
                f"{tex:<14} {n:>3}  "
                f"{b_acc_s:>{w}} {_fmt_cell(b_j_s, highlight=jac_best == 0)} "
                f"{a_acc_s:>{w}} {_fmt_cell(a_j_s, highlight=jac_best == 1)}"
            )

    print()


def main() -> None:
    run_gestalt_ablation_postprocess()


if __name__ == "__main__":
    main()
