"""
SMC over the mem phase programs: P particles vmapped through the SAME machinery
that batches windows (particles are the stacked axis). Gibbs phases are the move
kernel; the incremental weight is the assignment-marginalized datapoint
predictive of the NEW frame's data evaluated on the PROPAGATED state
(p(y_t | x_{t-1}) under the mixture) — the audited marginal_data_loglik.
Systematic resampling on the host at ESS < P/2; particles keep their own key
streams after resampling so duplicates diverge. The reported trajectory is the
FINAL max-weight particle's whole lineage (per-frame max-weight would flip
particles mid-track).
"""
import time

import jax
import numpy as np
import jax.numpy as jnp

from . import sfm_config as cfg
from .algorithm import (GIBBS_DIALS, VELOCITY_UPDATE_DIALS,
                        init_window_state, _frame_summary, _evaluate)
from .batched import (get_batched_program, unify_static_hypers, _vpropagate,
                      _vsplit)
from .memory_algorithm import marginal_data_loglik
from .memory_moves import base_aux, frame_aux
from genmatter.trace_wrappers import pytree_stack

_vmarg = jax.jit(jax.vmap(marginal_data_loglik))


def _systematic_resample(logw, rng):
    w = np.exp(logw - logw.max())
    w /= w.sum()
    P = len(w)
    u = (rng.random() + np.arange(P)) / P
    return np.searchsorted(np.cumsum(w), u).clip(0, P - 1)


def infer_window_smc(arrays: dict, mcfg: cfg.SfmModelConfig, mem, seed=None):
    """Same contract as infer_window_mem minus handoff (no carry)."""
    t_start = time.time()
    seed = mcfg.seed if seed is None else seed
    P = mem.n_particles
    points = np.asarray(arrays["points_3d"], np.float32)
    motion = np.asarray(arrays["motion_3d"], np.float32)
    valid = np.asarray(arrays["motion_valid"], bool)
    depth_sq = np.asarray(arrays["depth_sq"], np.float32)
    flow_sq = np.asarray(arrays["flow_sq"], np.float32)
    gt_masks = np.asarray(arrays["gt_masks"], bool)
    G = cfg.GRID_HW
    T = points.shape[0]
    emit = bool(mcfg.log_traces)
    rng = np.random.default_rng(seed)

    # P inits with distinct seeds: init diversity is what makes resampling live
    inits = [init_window_state(points, motion, valid, depth_sq, flow_sq, mcfg,
                               seed + 1013 * p) for p in range(P)]
    states = unify_static_hypers([s for s, _, _, _ in inits])
    keys = jnp.stack([k for _, k, _, _ in inits])
    combined, roi_fallback = inits[0][2], inits[0][3]

    p_init = get_batched_program(mcfg.init_sweeps, mcfg.keep_last_samples,
                                 mcfg.rescore_stride, emit, mcfg.trace_thin)
    p_vel = get_batched_program(
        mcfg.vel_sweeps, min(mcfg.keep_last_samples, mcfg.vel_sweeps),
        max(1, mcfg.vel_sweeps // 10) if mcfg.rescore_stride > 1 else 1,
        emit, mcfg.trace_thin)
    p_track = get_batched_program(
        mcfg.track_sweeps, mcfg.keep_last_samples,
        max(1, mcfg.track_sweeps // 20) if mcfg.rescore_stride > 1 else 1,
        emit, mcfg.trace_thin)
    vbase0 = jax.jit(jax.vmap(base_aux))
    vbase = jax.jit(jax.vmap(
        lambda st: base_aux(st, kappa=mem.kappa,
                            kappa_sel=mem.kappa_sel_resolved())))
    vframe = jax.jit(jax.vmap(
        lambda st, b: frame_aux(st, mem, b, vel_evidence=1.0)))
    vframe_ev = jax.jit(jax.vmap(
        lambda st, b, ev: frame_aux(st, mem, b, vel_evidence=ev)))
    evidence = np.clip(valid.mean(axis=1) / mem.vel_evidence_floor, 0.0, 1.0)

    state = pytree_stack(states)
    keys, k0 = _vsplit(keys, 2)
    last, _, _, ba, ha, scores, tr = p_init(
        k0, state, GIBBS_DIALS, mcfg.inner_loops, True, vbase0(state))
    state = last

    logw = np.asarray(_vmarg(state), np.float64)      # weight the P inits
    summaries = [(ba, ha, scores, state)]
    ancestry = [np.arange(P)]
    dev_traces = {"f0_init": tr}
    ess_log, resampled_at = [float(1.0 / np.sum(_norm(logw) ** 2))], []

    base = vbase(state)
    for f in range(1, T):
        state = _vpropagate(state,
                            jnp.broadcast_to(points[f], (P,) + points[f].shape),
                            jnp.broadcast_to(motion[f], (P,) + motion[f].shape))
        # proper SMC weight: predictive of the new frame's data BEFORE the move
        logw = logw + np.asarray(_vmarg(state), np.float64)
        wn = _norm(logw)
        ess = float(1.0 / np.sum(wn ** 2))
        ess_log.append(ess)
        anc = np.arange(P)
        if ess < P / 2:
            anc = _systematic_resample(logw, rng)
            state = jax.tree_util.tree_map(lambda x: x[anc], state)
            base = jax.tree_util.tree_map(lambda x: x[anc], base)
            logw = np.zeros(P)
            resampled_at.append(f)
        ancestry.append(anc)

        ev = jnp.float32(evidence[f])
        aux = vframe_ev(state, base,
                        jnp.broadcast_to(ev, (P,)))
        keys, k1, k2 = _vsplit(keys, 3)
        _, best, _, _, _, _, tr_v = p_vel(k1, state, VELOCITY_UPDATE_DIALS,
                                          mcfg.inner_loops, True, aux)
        state = best
        _, best, _, ba, ha, scores, tr = p_track(k2, state, GIBBS_DIALS,
                                                 mcfg.inner_loops, True, aux)
        state = best
        dev_traces[f"f{f}_vel"] = tr_v
        dev_traces[f"f{f}_track"] = tr
        summaries.append((ba, ha, scores, state))

    jax.block_until_ready(state.datapoints_state.blob_assignments)
    t_infer = time.time() - t_start

    # lineage of the final max-weight particle, walked back through resamplings
    idx = int(np.argmax(logw))
    lineage = [0] * T
    for f in range(T - 1, -1, -1):
        lineage[f] = idx
        idx = int(ancestry[f][idx])

    per_frame = []
    for f, (ba, ha, scores, st) in enumerate(summaries):
        p = lineage[f]
        st_p = jax.tree_util.tree_map(lambda x: x[p], st)
        per_frame.append(_frame_summary(st_p, ba[p], ha[p], scores[p], mcfg, G))

    host_traces = {}
    if emit:
        got = jax.device_get(dev_traces)
        host_traces = {"f0_init": {k: v[lineage[0]]
                                   for k, v in got["f0_init"].items()}}
        for f in range(1, T):
            for ph in (f"f{f}_vel", f"f{f}_track"):
                host_traces[ph] = {k: v[lineage[f]] for k, v in got[ph].items()}

    results, arrays_out = _evaluate(per_frame, gt_masks, combined, mcfg,
                                    t_infer, seed)
    results["roi_fallback"] = roi_fallback
    results["memory"] = mem.name
    results["memory_hash"] = mem.content_hash()
    results["smc"] = {"n_particles": P, "ess": [round(e, 2) for e in ess_log],
                      "resampled_at": resampled_at,
                      "final_logw_spread": float(np.ptp(logw))}
    return results, arrays_out, host_traces


def _norm(logw):
    w = np.exp(logw - logw.max())
    return w / w.sum()
