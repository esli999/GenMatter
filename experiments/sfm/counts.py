"""Vectorized sample-count maps and evaluation metrics (replaces the pure-Python
nested loops in run_gestalt.compute_hyperblob_sample_counts — same semantics)."""
import numpy as np


def pixel_hyperblobs(blob_assignments, hyperblob_assignments, n_blobs):
    """(N,) datapoint blob ids + (L,) blob hyperblob ids -> (N,) pixel hyperblob ids
    (-1 = outlier, i.e. blob id >= n_blobs)."""
    ba = np.asarray(blob_assignments)
    ha = np.asarray(hyperblob_assignments)
    return np.where(ba < n_blobs, ha[np.minimum(ba, n_blobs - 1)], -1)


def hyperblob_counts(ba_hist, ha_hist, n_blobs, n_hyperblobs, grid_hw):
    """(S, N) + (S, L) assignment histories -> (grid, grid, K+1) per-pixel counts,
    channel K = outliers. Exact vectorization of the original nested-loop function."""
    ba = np.asarray(ba_hist)
    ha = np.asarray(ha_hist)
    S, N = ba.shape
    K = n_hyperblobs
    ph = np.where(ba < n_blobs, np.take_along_axis(ha, np.minimum(ba, n_blobs - 1), axis=1), K)
    flat = (np.arange(N)[None, :] * (K + 1) + ph).ravel()
    counts = np.bincount(flat, minlength=N * (K + 1)).reshape(N, K + 1)
    return counts.reshape(grid_hw, grid_hw, K + 1)


def hyperblob_counts_reference(ba_hist, ha_hist, n_blobs, n_hyperblobs, grid_hw):
    """The original pure-Python formulation (run_gestalt.py), kept for the parity test."""
    ba = np.asarray(ba_hist)
    ha = np.asarray(ha_hist)
    S, N = ba.shape
    counts = np.zeros((N, n_hyperblobs + 1), dtype=int)
    for s in range(S):
        for p in range(N):
            b = ba[s, p]
            k = ha[s, b] if b < n_blobs else -1
            counts[p, n_hyperblobs if k == -1 else k] += 1
    return counts.reshape(grid_hw, grid_hw, n_hyperblobs + 1)


def uncertainty_ratio(counts):
    """1 - max_count/total per pixel (0 = certain)."""
    total = counts.sum(axis=2)
    return 1.0 - counts.max(axis=2) / np.maximum(total, 1)


def probe_accuracy(gt_mask_flat, pixel_hb, num_probes=100, rng=None):
    """run_gestalt.evaluate_frame_accuracy with an explicit seeded RNG."""
    rng = rng or np.random.default_rng(0)
    gt = np.asarray(gt_mask_flat).ravel()
    roi = np.flatnonzero(gt)
    if roi.size == 0:
        return 0.0, []
    probes = rng.choice(roi, size=min(num_probes, roi.size), replace=False)
    accs = []
    for p in probes:
        hb = pixel_hb[p]
        if hb < 0:
            continue
        est = (pixel_hb == hb)
        accs.append(float((gt == est).mean()))
    return (float(np.mean(accs)) if accs else 0.0), accs


def jaccard(pred_mask, gt_mask):
    inter = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return float(inter) / float(union) if union else 1.0


def pick_roi_hyperblob(pixel_hb, reference_mask_flat, n_hyperblobs, fallback=0,
                       by_iou=True):
    """Hyperblob matching a reference mask (GT-free motion heuristic at frame 0,
    previous-frame ROI afterwards). IoU-based by default: raw overlap (run_gestalt's
    rule) degenerates to the giant background hyperblob when the object is a small
    fraction of the frame."""
    ref = np.asarray(reference_mask_flat).ravel()
    if not ref.any():
        return int(fallback)
    scores = []
    for k in range(n_hyperblobs):
        m = (pixel_hb == k)
        inter = np.logical_and(m, ref).sum()
        scores.append(inter / max(np.logical_or(m, ref).sum(), 1) if by_iou else inter)
    return int(np.argmax(scores)) if max(scores) > 0 else int(fallback)
