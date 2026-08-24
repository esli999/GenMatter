"""
Compile-once GenMatter inference for the SFM windows.

The Gibbs chain is bit-identical to the stock path: each scan step calls the repo's
own `f_gibbs_sweep` with the same carry structure `genmatter_full_gibbs` uses, so a
given key produces the same sweep sequence. What changes is consumption: per-sweep
outputs are only assignment vectors + a jitted assess score (running argmax carried in
the scan), instead of stacking full states and rescoring in Python.

Programs are cached per (num_sweeps, keep_last) — dials and inner-loop counts are
traced, exactly as in `genmatter_full_gibbs`, so one compilation serves every dial
configuration of the same length. The persistent XLA cache makes compilations
machine-wide one-offs.
"""
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.random import key as jkey

from genmatter.inference import f_gibbs_sweep, genmatter_full_gibbs  # noqa: F401 (parity)
from genmatter.model_3d import GenMatter_model_3d, GenMatter_Hyperparams
from genmatter.core_types import f_
from genmatter.datatypes import snp
from genmatter.utils import make_hierarchical_kmeans_chm_with_mask_fixed_hyperblob
from experiments.gestalt.algorithm import extract_gestalt_segmentation

from . import sfm_config as cfg
from . import counts as C
from .propagate import propagate_state
from .scoring import joint_logprob

model_jimportance = jax.jit(GenMatter_model_3d.importance)


class DegenerateWindowError(RuntimeError):
    """Window whose init structures cannot match the compiled static shapes."""


def sfm_flow_roi(flow_sq, floor):
    """Tight moving-object mask from 2D flow magnitude at frame 0: backgrounds are
    static (median magnitude = RAFT noise floor), the rotating object's surface
    slides by ~1-4 px. Flat (N,) bool over the grid, morphologically closed."""
    import cv2
    from scipy.ndimage import binary_fill_holes
    mag = np.linalg.norm(np.asarray(flow_sq[0], np.float32), axis=-1)
    thr = max(floor, 6.0 * float(np.median(mag)))
    m = (mag > thr).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return binary_fill_holes(m.astype(bool)).ravel()

# Dial settings copied verbatim from experiments/gestalt/run_gestalt.py
GIBBS_DIALS = {
    "blob_weights": True, "hyperblob_weights": False,
    "blob_assignments": True, "hyperblob_assignments": False,
    "hyperblob_covs": False, "blob_covs": True,
    "blob_vel_covs": True, "blob_vel_means": True,
    "hyperblob_means": False, "blob_means": True,
    "hyperblob_rot_vels": True, "hyperblob_trans_vels": True,
}
VELOCITY_UPDATE_DIALS = {
    "blob_weights": False, "hyperblob_weights": False,
    "blob_assignments": True, "hyperblob_assignments": False,
    "hyperblob_covs": False, "blob_covs": False,
    "blob_vel_covs": True, "blob_vel_means": True,
    "hyperblob_means": True, "blob_means": True,
    "hyperblob_rot_vels": True, "hyperblob_trans_vels": True,
}

_PROGRAM_CACHE = {}


def get_phase_program(num_sweeps: int, keep_last: int, rescore_stride: int):
    """Jitted: (key, state, dials, inner, weighted) ->
    (last_state, best_state, best_score, ba_hist, ha_hist, scores).

    Candidate set for the argmax matches run_gestalt's strided pick over the trace
    wrapper: {initial state} ∪ {post-sweep states at multiples of the stride}."""
    keep = min(keep_last, num_sweeps)
    key_ = (num_sweeps, keep, rescore_stride)
    if key_ in _PROGRAM_CACHE:
        return _PROGRAM_CACHE[key_]

    @jax.jit
    def run(key, state, dials, inner, weighted):
        init_score = joint_logprob(state)

        def body(carry, i):
            core, best_score, best_state = carry
            core, _ = f_gibbs_sweep(core, i)
            st = core[1]
            score = joint_logprob(st)
            eligible = ((i + 1) % rescore_stride == 0) if rescore_stride > 1 else True
            cand = jnp.where(eligible, score, -jnp.inf)
            better = cand > best_score
            best_state = jax.tree_util.tree_map(
                lambda n, o: jnp.where(better, n, o), st, best_state)
            best_score = jnp.where(better, cand, best_score)
            ys = (st.datapoints_state.blob_assignments.astype(jnp.int32),
                  st.blobs_state.hyperblob_assignments.astype(jnp.int32),
                  score)
            return (core, best_score, best_state), ys

        core0 = (key, state, dials, inner, weighted)
        (core, best_score, best_state), (ba, ha, scores) = jax.lax.scan(
            body, (core0, init_score, state), jnp.arange(num_sweeps))
        return core[1], best_state, best_score, ba[-keep:], ha[-keep:], scores

    _PROGRAM_CACHE[key_] = run
    return run


def build_hypers(mcfg: cfg.SfmModelConfig, kmeans_chm, roi_blob_idx, roi_hyper_idx):
    """Empirical + configured hyperparameters, mirroring run_gestalt.py. All tunable
    scalars are jnp-wrapped (traced); only grid parameters and counts are static."""
    n_datapoints = kmeans_chm["datapoints", "datapoint_positions"].shape[0]
    n_blobs = kmeans_chm["blobs", "hyperblob_assignments"].shape[0]
    n_hyper = kmeans_chm["hyperblobs", "hyperblob_means"].shape[0]

    mu_H = jnp.median(kmeans_chm["datapoints", "datapoint_positions"], axis=0)
    Psi_B = jnp.median(kmeans_chm["blobs", "blob_covs"][roi_blob_idx], axis=0)
    Psi_H = jnp.median(kmeans_chm["hyperblobs", "hyperblob_covs"][roi_hyper_idx], axis=0)
    Psi_V = jnp.median(kmeans_chm["blobs", "blob_vel_covs"][roi_blob_idx], axis=0)
    mean_blobs = jnp.sum(jnp.isin(kmeans_chm["blobs", "hyperblob_assignments"],
                                  roi_hyper_idx)) / len(roi_hyper_idx)
    nu_H = f_(int(mean_blobs))
    mean_points = jnp.sum(jnp.isin(kmeans_chm["datapoints", "blob_assignments"],
                                   roi_blob_idx)) / len(roi_blob_idx)
    nu_B = nu_V = f_(int(mean_points))

    return GenMatter_Hyperparams.create(
        outlier_prob=f_(mcfg.outlier_prob),
        outlier_velocity_gamma_shape=f_(mcfg.outlier_gamma_shape),
        outlier_velocity_gamma_rate=f_(mcfg.outlier_gamma_rate),
        alpha=f_(1.0), beta=f_(1.0),
        mu_H=mu_H, sigma_H=f_(mcfg.sigma_H),
        nu_H=nu_H, Psi_H=Psi_H, nu_B=nu_B, Psi_B=Psi_B,
        sigma_V=f_(mcfg.sigma_V), nu_V=nu_V, Psi_V=Psi_V,
        translation_gaussian_scale=snp(f_(mcfg.trans_gaussian_scale)),
        translation_max_radius=snp(mcfg.trans_max_radius),
        translation_num_radii_cells=snp(mcfg.trans_num_radii_cells),
        translation_theta_step_deg=snp(mcfg.trans_theta_step_deg),
        rotation_vmf_kappa=snp(f_(mcfg.rot_vmf_kappa)),
        rotation_angle_max_deg=snp(mcfg.rot_angle_max_deg),
        rotation_angle_step_deg=snp(mcfg.rot_angle_step_deg),
        n_hyperblobs=n_hyper, n_blobs=n_blobs, n_datapoints=n_datapoints,
    )


def infer_window(arrays: dict, mcfg: cfg.SfmModelConfig, seed: int = None):
    """Run the full window schedule (init + 4 tracked frames) on one bundle."""
    t_start = time.time()
    seed = mcfg.seed if seed is None else seed
    points = np.asarray(arrays["points_3d"], np.float32)     # (5, N, 3)
    motion = np.asarray(arrays["motion_3d"], np.float32)
    valid = np.asarray(arrays["motion_valid"], bool)
    depth_sq = np.asarray(arrays["depth_sq"], np.float32)
    flow_sq = np.asarray(arrays["flow_sq"], np.float32)
    gt_masks = np.asarray(arrays["gt_masks"], bool)          # (6, G, G); eval uses [0:5]
    G = cfg.GRID_HW

    # --- init structures (CPU, once per window)
    if mcfg.roi_heuristic == "sfm_flow":
        combined = sfm_flow_roi(flow_sq, mcfg.roi_flow_floor)
    else:
        first_seg = extract_gestalt_segmentation(depth_sq, flow_sq, points)
        combined = first_seg & valid[0]
    # Guarantee a non-degenerate ROI: with too few confident moving points the
    # k-means init crashes / breaks the pinned blob count. Fall back to the top-K
    # pixels by 2D flow magnitude ("most-moving points"), recorded as roi_fallback.
    roi_fallback = False
    min_roi = mcfg.n_roi_blobs * 2 + 16
    if int(combined.sum()) < min_roi:
        mag = np.linalg.norm(flow_sq[0], axis=-1).ravel()
        k = max(min_roi, 96)
        fb = np.zeros_like(combined)
        fb[np.argpartition(mag, -k)[-k:]] = True
        combined = fb
        roi_fallback = True
    kmeans_chm, roi_b, roi_h = make_hierarchical_kmeans_chm_with_mask_fixed_hyperblob(
        points, mcfg.n_blobs, mcfg.n_hyperblobs,
        segmentation_mask=combined, motion_vectors=motion,
        num_roi_blobs=mcfg.n_roi_blobs)
    actual_blobs = kmeans_chm["blobs", "hyperblob_assignments"].shape[0]
    if actual_blobs != mcfg.n_blobs:
        # A degenerate window (e.g. <n_roi_blobs moving points) would silently key a
        # fresh multi-minute XLA compile — refuse instead and let the caller record it.
        raise DegenerateWindowError(
            f"k-means produced {actual_blobs} blobs != configured {mcfg.n_blobs} "
            f"(roi points={int(combined.sum())})")
    hypers = build_hypers(mcfg, kmeans_chm, roi_b, roi_h)

    key = jkey(seed)
    key, k_imp = jax.random.split(key)
    init_tr, _ = model_jimportance(k_imp, kmeans_chm, (hypers,))
    state = init_tr.get_retval()

    p_init = get_phase_program(mcfg.init_sweeps, mcfg.keep_last_samples, mcfg.rescore_stride)
    p_vel = get_phase_program(mcfg.vel_sweeps, min(mcfg.keep_last_samples, mcfg.vel_sweeps),
                              max(1, mcfg.vel_sweeps // 10) if mcfg.rescore_stride > 1 else 1)
    p_track = get_phase_program(mcfg.track_sweeps, mcfg.keep_last_samples,
                                max(1, mcfg.track_sweeps // 20) if mcfg.rescore_stride > 1 else 1)

    per_frame = []      # (pixel_hb, counts, best_score)
    key, k0 = jax.random.split(key)
    last, best, bscore, ba, ha, scores = p_init(k0, state, GIBBS_DIALS,
                                                mcfg.inner_loops, True)
    # run_gestalt continues from the LAST init sample
    state = last
    per_frame.append(_frame_summary(state, ba, ha, scores, mcfg, G))

    for f in range(1, points.shape[0]):
        state = propagate_state(state, points[f], motion[f])
        key, k1, k2 = jax.random.split(key, 3)
        _, best, _, _, _, _ = p_vel(k1, state, VELOCITY_UPDATE_DIALS, mcfg.inner_loops, True)
        state = best
        _, best, bscore, ba, ha, scores = p_track(k2, state, GIBBS_DIALS,
                                                  mcfg.inner_loops, True)
        state = best
        per_frame.append(_frame_summary(state, ba, ha, scores, mcfg, G))

    jax.block_until_ready(state.datapoints_state.blob_assignments)
    t_infer = time.time() - t_start

    results, arrays_out = _evaluate(per_frame, gt_masks, combined, mcfg, t_infer, seed)
    results["roi_fallback"] = roi_fallback
    return results, arrays_out


def _frame_summary(state, ba_hist, ha_hist, scores, mcfg, G):
    ba = np.asarray(state.datapoints_state.blob_assignments)
    ha = np.asarray(state.blobs_state.hyperblob_assignments)
    pixel_hb = C.pixel_hyperblobs(ba, ha, mcfg.n_blobs)
    cnt = C.hyperblob_counts(np.asarray(ba_hist), np.asarray(ha_hist),
                             mcfg.n_blobs, mcfg.n_hyperblobs, G)
    return {"pixel_hb": pixel_hb, "blob_assignments": ba, "hyper_per_blob": ha,
            "counts": cnt, "scores": np.asarray(scores)}


def _evaluate(per_frame, gt_masks, first_seg_combined, mcfg, t_infer, seed):
    G = cfg.GRID_HW
    rng = np.random.default_rng(seed)
    results = {"frames": [], "t_infer_s": t_infer, "config": mcfg.name,
               "config_hash": mcfg.content_hash(), "seed": seed}
    roi_id = None
    prev_roi_mask = None
    arrays_out = {"pixel_hyperblob_assignments": [], "blob_assignments": [],
                  "hyperblob_assignments_per_blob": [], "roi_hyperblob_ids": [],
                  "hyperblob_sample_counts": []}

    for fidx, fr in enumerate(per_frame):
        pixel_hb = fr["pixel_hb"]
        ref = first_seg_combined if fidx == 0 else prev_roi_mask
        roi_id = C.pick_roi_hyperblob(pixel_hb, ref, mcfg.n_hyperblobs,
                                      fallback=roi_id or 0)
        pred_mask = (pixel_hb == roi_id)
        prev_roi_mask = pred_mask

        gt = gt_masks[fidx].ravel()
        acc, _ = C.probe_accuracy(gt, pixel_hb, num_probes=100, rng=rng)
        jac = C.jaccard(pred_mask, gt)
        # GT-referenced DIAGNOSTIC only (never used for selection): best achievable
        # single-hyperblob IoU — separates "grouping failed" from "selection failed".
        oracle = max(C.jaccard(pixel_hb == k, gt) for k in range(mcfg.n_hyperblobs))
        unc = C.uncertainty_ratio(fr["counts"])
        results["frames"].append({
            "frame": fidx, "probe_accuracy": acc, "roi_jaccard": jac,
            "oracle_jaccard": oracle,
            "roi_hyperblob": roi_id, "uncertainty_mean": float(unc.mean()),
            "gt_area": int(gt.sum()), "pred_area": int(pred_mask.sum()),
            "final_score": float(fr["scores"][-1]),
        })
        arrays_out["pixel_hyperblob_assignments"].append(pixel_hb.reshape(G, G).astype(np.int8))
        arrays_out["blob_assignments"].append(fr["blob_assignments"].reshape(G, G).astype(np.int16))
        arrays_out["hyperblob_assignments_per_blob"].append(fr["hyper_per_blob"].astype(np.int8))
        arrays_out["roi_hyperblob_ids"].append(roi_id)
        arrays_out["hyperblob_sample_counts"].append(fr["counts"].astype(np.uint8))

    accs = [f["probe_accuracy"] for f in results["frames"]]
    jacs = [f["roi_jaccard"] for f in results["frames"]]
    results["mean_probe_accuracy"] = float(np.mean(accs))
    results["mean_roi_jaccard"] = float(np.mean(jacs))
    results["mean_oracle_jaccard"] = float(np.mean(
        [f["oracle_jaccard"] for f in results["frames"]]))
    arrays_out = {k: np.stack(v) if k != "roi_hyperblob_ids" else np.asarray(v)
                  for k, v in arrays_out.items()}
    return results, arrays_out
