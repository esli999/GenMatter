"""
Chain-level parity between the fast SFM path and the stock GenMatter path.

1. Gibbs parity: my scan program's last state must EXACTLY equal
   genmatter_full_gibbs(...)'s final trace state for the same key (both call
   f_gibbs_sweep with the identical carry, and scoring consumes no randomness).
2. Propagation parity: propagate_state must equal run_gestalt's choicemap+importance
   reconstruction for every consumed quantity (blob means, datapoints; hyperblob means
   for non-empty hyperblobs).

Small synthetic problem (N=1024 datapoints — one full logprob chunk). Run directly;
CPU is fine (set JAX_PLATFORMS=cpu to run on a login/CPU node).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import jax
import jax.numpy as jnp
from jax.random import key as jkey

from genmatter.inference import genmatter_full_gibbs
from genmatter.utils import make_hierarchical_kmeans_chm_with_mask_fixed_hyperblob
from experiments.sfm.algorithm import (GIBBS_DIALS, get_phase_program, build_hypers,
                                       model_jimportance)
from experiments.sfm.propagate import propagate_state
from experiments.sfm import sfm_config as cfg
import dataclasses

TINY = dataclasses.replace(
    cfg.SFM_BASE, name="tiny", n_blobs=16, n_roi_blobs=8, n_hyperblobs=3,
    init_sweeps=6, vel_sweeps=4, track_sweeps=6, inner_loops=2,
    rot_angle_max_deg=10.0, rot_angle_step_deg=2.5,
    trans_num_radii_cells=5, trans_theta_step_deg=45)


def synth_problem(N=1024, T=3, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 1.0, size=(N, 3)).astype(np.float32)
    obj = rng.random(N) < 0.3
    vel = np.where(obj[:, None], np.array([0.05, 0.0, 0.02]), 0.0).astype(np.float32)
    vel += rng.normal(0, 0.005, size=(N, 3)).astype(np.float32)
    points = np.stack([base + t * vel for t in range(T)])
    motion = np.stack([vel] * T)
    return points, motion, obj


def leaves_equal(a, b, atol=0.0):
    la = jax.tree_util.tree_leaves(a)
    lb = jax.tree_util.tree_leaves(b)
    assert len(la) == len(lb)
    for x, y in zip(la, lb):
        if not np.allclose(np.asarray(x), np.asarray(y), atol=atol, rtol=0.0):
            return False
    return True


def main():
    points, motion, obj = synth_problem()
    chm, roi_b, roi_h = make_hierarchical_kmeans_chm_with_mask_fixed_hyperblob(
        points, TINY.n_blobs, TINY.n_hyperblobs,
        segmentation_mask=obj, motion_vectors=motion,
        num_roi_blobs=TINY.n_roi_blobs)
    n_actual = chm["blobs", "hyperblob_assignments"].shape[0]
    assert n_actual == TINY.n_blobs, f"blob count not pinned: {n_actual}"
    hypers = build_hypers(TINY, chm, roi_b, roi_h)
    tr, _ = model_jimportance(jkey(1), chm, (hypers,))
    state = tr.get_retval()

    # --- 1. Gibbs chain parity
    key = jkey(123)
    wtrs = genmatter_full_gibbs(key, state, TINY.init_sweeps, GIBBS_DIALS,
                                use_weighted_blobs=True,
                                num_gibbs_inner_loops=TINY.inner_loops)
    ref_final = wtrs[-1].retval
    prog = get_phase_program(TINY.init_sweeps, keep_last=4, rescore_stride=1)
    last, best, best_score, ba, ha, scores, _ = prog(key, state, GIBBS_DIALS,
                                                     TINY.inner_loops, True)
    assert leaves_equal(last.datapoints_state, ref_final.datapoints_state), \
        "datapoints_state diverged"
    assert leaves_equal(last.blobs_state, ref_final.blobs_state), "blobs_state diverged"
    assert leaves_equal(last.hyperblobs_state, ref_final.hyperblobs_state), \
        "hyperblobs_state diverged"
    # assignment history tail must match the wrapper's tail states
    # (wrapper index i = state after sweep i; my ba[-k] = state after sweep N+1-k)
    for k in range(1, 4):
        ref_ba = np.asarray(
            wtrs[TINY.init_sweeps + 1 - k].retval.datapoints_state.blob_assignments)
        assert np.array_equal(np.asarray(ba[-k]), ref_ba), f"ba history mismatch at -{k}"
    assert np.isfinite(float(best_score)), "best_score not finite"
    print("gibbs chain parity: PASS")

    # --- 1b. Trace emission: identical chain (same key), latent-complete traces,
    # thinned rows exactly matching the per-sweep assignment/score histories.
    thin = 2
    prog_t = get_phase_program(TINY.init_sweeps, keep_last=4, rescore_stride=1,
                               emit_traces=True, trace_thin=thin)
    last_t, _, _, ba_t, _, scores_t, tr = prog_t(key, state, GIBBS_DIALS,
                                                 TINY.inner_loops, True)
    assert leaves_equal(last_t, last), "emit_traces changed the chain"
    expected = {"hyperblob_weights", "hyperblob_means", "hyperblob_covs",
                "hyperblob_trans_vels", "hyperblob_rot_vels",
                "blob_hyperblob_assignments", "blob_weights", "blob_means",
                "blob_covs", "blob_vel_means", "blob_vel_covs",
                "datapoint_assignments", "scores"}
    assert set(tr) == expected, f"trace keys mismatch: {set(tr) ^ expected}"
    n_kept = int(np.ceil(TINY.init_sweeps / thin))
    for k, v in tr.items():
        rows = TINY.init_sweeps if k == "scores" else n_kept
        assert v.shape[0] == rows, f"{k}: {v.shape} (want {rows} rows)"
    assert tr["datapoint_assignments"].dtype == np.int16
    assert tr["blob_means"].dtype == np.float16
    # thinned trace row i is sweep i*thin; ba holds the LAST 4 sweeps
    da = np.asarray(tr["datapoint_assignments"], np.int32)
    for s in range(0, TINY.init_sweeps, thin):
        back = TINY.init_sweeps - 1 - s      # position of sweep s from the end
        if back < 4:
            assert np.array_equal(da[s // thin], np.asarray(ba_t[-(back + 1)])), \
                f"datapoint trace row for sweep {s} != ba history"
    assert np.allclose(np.asarray(scores_t), np.asarray(tr["scores"])), "scores trace"
    print("trace emission: PASS")

    # --- 2. Propagation parity (vs run_gestalt's numpy loops)
    st = last
    hb_a = np.asarray(st.blobs_state.hyperblob_assignments)
    R = np.asarray(st.hyperblobs_state.hyperblob_rot_vels)
    tv = np.asarray(st.hyperblobs_state.hyperblob_trans_vels)
    muH = np.asarray(st.hyperblobs_state.hyperblob_means)
    muB = np.asarray(st.blobs_state.blob_means)
    ref_blob = np.zeros_like(muB)
    for k in range(TINY.n_hyperblobs):
        mask = hb_a == k
        if mask.sum() == 0:
            continue
        c_new = muH[k] + tv[k]
        for l in np.where(mask)[0]:
            ref_blob[l] = c_new + R[k] @ (muB[l] - muH[k])
    new_state = propagate_state(st, jnp.asarray(points[1]), jnp.asarray(motion[1]))
    got_blob = np.asarray(new_state.blobs_state.blob_means)
    assert np.allclose(got_blob, ref_blob, atol=1e-5), "propagated blob means mismatch"
    nonempty = np.isin(np.arange(TINY.n_hyperblobs), hb_a)
    got_muH = np.asarray(new_state.hyperblobs_state.hyperblob_means)
    assert np.allclose(got_muH[nonempty], (muH + tv)[nonempty], atol=1e-6), \
        "propagated hyperblob means mismatch"
    assert np.allclose(np.asarray(new_state.datapoints_state.datapoint_positions),
                       points[1], atol=0), "datapoint splice mismatch"
    print("propagation parity: PASS")
    print("test_parity: PASS")


if __name__ == "__main__":
    main()
