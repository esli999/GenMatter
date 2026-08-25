"""
Memory-augmented inference: a unified mem Gibbs sweep (sticky x filtering via one
compiled program), mem phase programs with sticky-augmented selection, per-window
memory inference, and whole-video segment handoff.

The mem sweep replicates f_gibbs_sweep's exact move order and key-split structure
(genmatter/inference.py:1009), swapping in the aux-carrying moves:
- blob_assignments  -> sticky_blob_assignments_batched (kappa = 0 == stock, bit-exact)
- blob_covs         -> gibbs_blob_covs_aux (batched NIW prior; base tiled == stock)
- blob_vel_covs     -> gibbs_blob_vel_covs_aux (same)
- blob_vel_means    -> lax.cond(use_filter, filtered, stock)
- blob_means        -> lax.cond(use_filter, filtered, stock)
Everything else calls the core moves unchanged. tests/test_memory.py asserts the
OFF configuration reproduces the baseline program bit-identically.
"""
import time

import jax
import jax.numpy as jnp
import numpy as np

from genmatter.inference import (
    empty_gibbs, empty_gibbs_weighted,
    gibbs_blob_weights, gibbs_hyperblob_weights, gibbs_hyperblob_assignments,
    gibbs_hyperblob_covs, gibbs_hyperblob_means, gibbs_hyperblob_rot,
    gibbs_hyperblob_trans, gibbs_blob_vel_means, gibbs_blob_means,
)

from . import sfm_config as cfg
from .algorithm import (GIBBS_DIALS, VELOCITY_UPDATE_DIALS, latent_trace,
                        init_window_state, _frame_summary, _evaluate)
from .memory_config import MemoryConfig
from .memory_moves import (MemAux, base_aux, frame_aux,
                           sticky_blob_assignments_batched,
                           gibbs_blob_covs_aux, gibbs_blob_vel_covs_aux,
                           gibbs_blob_means_filtered, gibbs_blob_vel_means_filtered)
from .propagate import propagate_state
from .scoring import joint_logprob


def mem_gibbs_sweep(core_carry, aux: MemAux):
    """One Gibbs sweep with memory moves. Mirrors f_gibbs_sweep exactly: same
    inner-loop structure, same move order, same key splits per move."""
    key, genmatter_state, gibbs_dials, num_gibbs_inner_loops, use_weighted_blobs = core_carry

    def datapoint_update_loop(i, state_key_tuple):
        genmatter_state, key = state_key_tuple
        key, gibbs_key = jax.random.split(key)
        genmatter_state = jax.lax.cond(
            gibbs_dials["blob_assignments"],
            lambda k, s: sticky_blob_assignments_batched(k, s, aux),
            empty_gibbs, gibbs_key, genmatter_state)
        return (genmatter_state, key)

    def blob_update_loop(i, state_key_tuple):
        genmatter_state, key = state_key_tuple
        key, gibbs_key = jax.random.split(key)
        genmatter_state = jax.lax.cond(gibbs_dials["blob_weights"],
                                       gibbs_blob_weights, empty_gibbs,
                                       gibbs_key, genmatter_state)
        key, gibbs_key = jax.random.split(key)
        genmatter_state = jax.lax.cond(gibbs_dials["hyperblob_assignments"],
                                       gibbs_hyperblob_assignments, empty_gibbs,
                                       gibbs_key, genmatter_state)
        key, gibbs_key = jax.random.split(key)
        genmatter_state = jax.lax.cond(
            gibbs_dials["blob_covs"],
            lambda k, s: gibbs_blob_covs_aux(k, s, aux),
            empty_gibbs, gibbs_key, genmatter_state)
        key, gibbs_key = jax.random.split(key)
        genmatter_state = jax.lax.cond(
            gibbs_dials["blob_vel_covs"],
            lambda k, s: gibbs_blob_vel_covs_aux(k, s, aux),
            empty_gibbs, gibbs_key, genmatter_state)
        key, gibbs_key = jax.random.split(key)
        genmatter_state = jax.lax.cond(
            gibbs_dials["blob_vel_means"],
            lambda k, s: jax.lax.cond(
                aux.use_filter,
                lambda kk, ss: gibbs_blob_vel_means_filtered(kk, ss, aux),
                gibbs_blob_vel_means, k, s),
            empty_gibbs, gibbs_key, genmatter_state)
        key, gibbs_key = jax.random.split(key)
        genmatter_state = jax.lax.cond(
            gibbs_dials["blob_means"],
            lambda k, s: jax.lax.cond(
                aux.use_filter,
                lambda kk, ss: gibbs_blob_means_filtered(kk, ss, aux),
                gibbs_blob_means, k, s),
            empty_gibbs, gibbs_key, genmatter_state)
        return (genmatter_state, key)

    def hyperblob_update_loop(i, state_key_tuple):
        genmatter_state, key, use_weighted_blobs = state_key_tuple
        key, gibbs_key = jax.random.split(key)
        genmatter_state = jax.lax.cond(gibbs_dials["hyperblob_weights"],
                                       gibbs_hyperblob_weights, empty_gibbs_weighted,
                                       gibbs_key, genmatter_state, use_weighted_blobs)
        key, gibbs_key = jax.random.split(key)
        genmatter_state = jax.lax.cond(gibbs_dials["hyperblob_covs"],
                                       gibbs_hyperblob_covs, empty_gibbs_weighted,
                                       gibbs_key, genmatter_state, use_weighted_blobs)
        key, gibbs_key = jax.random.split(key)
        genmatter_state = jax.lax.cond(gibbs_dials["hyperblob_means"],
                                       gibbs_hyperblob_means, empty_gibbs_weighted,
                                       gibbs_key, genmatter_state, use_weighted_blobs)
        key, gibbs_key = jax.random.split(key)
        genmatter_state = jax.lax.cond(gibbs_dials["hyperblob_rot_vels"],
                                       gibbs_hyperblob_rot, empty_gibbs_weighted,
                                       gibbs_key, genmatter_state, use_weighted_blobs)
        key, gibbs_key = jax.random.split(key)
        genmatter_state = jax.lax.cond(gibbs_dials["hyperblob_trans_vels"],
                                       gibbs_hyperblob_trans, empty_gibbs_weighted,
                                       gibbs_key, genmatter_state, use_weighted_blobs)
        return (genmatter_state, key, use_weighted_blobs)

    genmatter_state, key = jax.lax.fori_loop(
        0, num_gibbs_inner_loops, datapoint_update_loop, (genmatter_state, key))
    genmatter_state, key = jax.lax.fori_loop(
        0, num_gibbs_inner_loops, blob_update_loop, (genmatter_state, key))
    genmatter_state, key, _ = jax.lax.fori_loop(
        0, num_gibbs_inner_loops, hyperblob_update_loop,
        (genmatter_state, key, use_weighted_blobs))

    return (key, genmatter_state, gibbs_dials, num_gibbs_inner_loops,
            use_weighted_blobs)


@jax.jit
def marginal_data_loglik(genmatter_state):
    """Assignment-marginalized datapoint log-likelihood: mean over datapoints of
    logsumexp over the L+1 mixture components (blob likelihood + mixing weight;
    outlier component included). Pure data fit — no structural priors, no chain
    history — so unlike the joint it can compare a handed-off state against a
    fresh one without rewarding prior-typicality or longer chains (the failure
    mode that sank score-guarded handoff). Chunked like the assignment move."""
    from genmatter.inference import blob_datapoint_likelihood_model_no_assignment
    from genjax import ChoiceMapBuilder as C

    hypers = genmatter_state.hypers
    num_blobs = hypers.n_blobs
    num_datapoints = hypers.n_datapoints
    datapoint_positions = genmatter_state.datapoints_state.datapoint_positions
    datapoint_vels = genmatter_state.datapoints_state.datapoint_vels
    blobs_state = genmatter_state.blobs_state

    extended_weights = jnp.concatenate([blobs_state.blob_weights,
                                        jnp.array([hypers.outlier_prob])])
    log_mixture_weights = jnp.log(extended_weights / jnp.sum(extended_weights))

    def point_marginal(point_idx):
        chm = (C["datapoint_position"].set(datapoint_positions[point_idx]) |
               C["datapoint_vel"].set(datapoint_vels[point_idx]))
        log_liks = jax.vmap(
            lambda i: blob_datapoint_likelihood_model_no_assignment.assess(
                chm, (blobs_state[i],))[0]
        )(jnp.arange(num_blobs))
        v = datapoint_vels[point_idx]
        speed = jnp.linalg.norm(v)
        alpha = hypers.outlier_velocity_gamma_shape
        beta = hypers.outlier_velocity_gamma_rate
        log_gamma_vel = ((alpha - 1) * jnp.log(speed + 1e-8) - beta * speed
                         - alpha * jnp.log(1. / beta) - jax.lax.lgamma(alpha))
        logits = jnp.concatenate([log_liks, jnp.array([log_gamma_vel])])
        return jax.scipy.special.logsumexp(logits + log_mixture_weights)

    batch_size = 1024 if num_datapoints % 1024 == 0 else num_datapoints

    def batch_marginals(carry, batch_idx):
        idx = jnp.arange(batch_size) + batch_idx * batch_size
        return carry, jax.vmap(point_marginal)(idx)

    _, per_point = jax.lax.scan(batch_marginals, None,
                                jnp.arange(num_datapoints // batch_size))
    return jnp.mean(per_point)


_MEM_PROGRAM_CACHE = {}


def get_mem_phase_program(num_sweeps: int, keep_last: int, rescore_stride: int,
                          emit_traces: bool = False, trace_thin: int = 5):
    """Jitted: (key, state, dials, inner, weighted, aux) ->
    (last_state, best_state, best_score, ba_hist, ha_hist, scores, traces).

    Same contract as algorithm.get_phase_program, plus: the in-scan argmax runs on
    the sticky-augmented score joint + kappa_sel * sum(z == prev_assign) (a proper
    tempered target; kappa_sel = 0 reduces to the plain joint). `scores` and
    best_score report the RAW joint. All memory knobs are traced: one compilation
    serves every kappa/lambda/on-off configuration of this phase length."""
    keep = min(keep_last, num_sweeps)
    key_ = (num_sweeps, keep, rescore_stride, emit_traces, trace_thin)
    if key_ in _MEM_PROGRAM_CACHE:
        return _MEM_PROGRAM_CACHE[key_]

    @jax.jit
    def run(key, state, dials, inner, weighted, aux):
        def sel_score(st):
            raw = joint_logprob(st)
            matches = jnp.sum(st.datapoints_state.blob_assignments
                              == aux.prev_assign).astype(jnp.float32)
            return raw, raw + aux.kappa_sel * matches

        init_raw, init_aug = sel_score(state)

        def body(carry, i):
            core, best_aug, best_raw, best_state = carry
            core = mem_gibbs_sweep(core, aux)
            st = core[1]
            raw, aug = sel_score(st)
            eligible = ((i + 1) % rescore_stride == 0) if rescore_stride > 1 else True
            cand = jnp.where(eligible, aug, -jnp.inf)
            better = cand > best_aug
            best_state = jax.tree_util.tree_map(
                lambda n, o: jnp.where(better, n, o), st, best_state)
            best_aug = jnp.where(better, cand, best_aug)
            best_raw = jnp.where(better, raw, best_raw)
            tr = latent_trace(st) if emit_traces else {}
            ys = (st.datapoints_state.blob_assignments.astype(jnp.int32),
                  st.blobs_state.hyperblob_assignments.astype(jnp.int32),
                  raw, tr)
            return (core, best_aug, best_raw, best_state), ys

        core0 = (key, state, dials, inner, weighted)
        (core, _, best_raw, best_state), (ba, ha, scores, traces) = jax.lax.scan(
            body, (core0, init_aug, init_raw, state), jnp.arange(num_sweeps))
        if emit_traces:
            traces = {k: v[::trace_thin] for k, v in traces.items()}
            traces["scores"] = scores
        return core[1], best_state, best_raw, ba[-keep:], ha[-keep:], scores, traces

    _MEM_PROGRAM_CACHE[key_] = run
    return run


def _mem_programs(mcfg):
    emit = bool(mcfg.log_traces)
    p_init = get_mem_phase_program(mcfg.init_sweeps, mcfg.keep_last_samples,
                                   mcfg.rescore_stride, emit, mcfg.trace_thin)
    p_vel = get_mem_phase_program(
        mcfg.vel_sweeps, min(mcfg.keep_last_samples, mcfg.vel_sweeps),
        max(1, mcfg.vel_sweeps // 10) if mcfg.rescore_stride > 1 else 1,
        emit, mcfg.trace_thin)
    p_track = get_mem_phase_program(
        mcfg.track_sweeps, mcfg.keep_last_samples,
        max(1, mcfg.track_sweeps // 20) if mcfg.rescore_stride > 1 else 1,
        emit, mcfg.trace_thin)
    return p_init, p_vel, p_track


def _run_frames(state, key, points, motion, mcfg, mem, base, per_frame,
                dev_traces, frame_offset, G, p_vel, p_track, frame_evidence):
    """Track frames 1..T-1 of one window from an accepted frame-0 state."""
    for f in range(1, points.shape[0]):
        state = propagate_state(state, points[f], motion[f])
        aux = frame_aux(state, mem, base, vel_evidence=frame_evidence[f])
        key, k1, k2 = jax.random.split(key, 3)
        _, best, _, _, _, _, tr_v = p_vel(k1, state, VELOCITY_UPDATE_DIALS,
                                          mcfg.inner_loops, True, aux)
        state = best
        # one anchor per frame: both phases stick to the frame-start assignments
        _, best, _, ba, ha, scores, tr = p_track(k2, state, GIBBS_DIALS,
                                                 mcfg.inner_loops, True, aux)
        state = best
        g = frame_offset + f
        dev_traces[f"f{g}_vel"] = tr_v
        dev_traces[f"f{g}_track"] = tr
        per_frame.append(_frame_summary(state, ba, ha, scores, mcfg, G))
    return state, key


def infer_window_mem(arrays: dict, mcfg: cfg.SfmModelConfig, mem: MemoryConfig,
                     seed: int = None, carry_in: dict = None):
    """Memory-augmented window inference. Same contract as algorithm.infer_window
    plus: `carry_in` (from a previous segment) enables cross-segment handoff, and
    the return gains `carry_out` = {"state", "key", "handoff_used", "scores"}.

    Handoff (mem.handoff != "none" and carry_in given): the carried state is
    propagated onto this window's frame-0 data and burned in with a vel+track
    phase pair, its empirical hypers grafted fresh is NOT needed — hypers stay in
    the state and score comparisons share the model target via assess. In
    "guarded" mode a fresh k-means init phase also runs and the handoff is
    accepted only if it outscores fresh outright (ties -> fresh)."""
    t_start = time.time()
    seed = mcfg.seed if seed is None else seed
    points = np.asarray(arrays["points_3d"], np.float32)
    motion = np.asarray(arrays["motion_3d"], np.float32)
    valid = np.asarray(arrays["motion_valid"], bool)
    depth_sq = np.asarray(arrays["depth_sq"], np.float32)
    flow_sq = np.asarray(arrays["flow_sq"], np.float32)
    gt_masks = np.asarray(arrays["gt_masks"], bool)
    G = cfg.GRID_HW

    p_init, p_vel, p_track = _mem_programs(mcfg)
    emit = bool(mcfg.log_traces)

    # --- frame 0: fresh init (always computed: it is the guard's comparator)
    state0, key, combined, roi_fallback = init_window_state(
        points, motion, valid, depth_sq, flow_sq, mcfg, seed)
    aux0 = base_aux(state0)                      # init frame runs kappa = 0
    key, k0 = jax.random.split(key)
    fresh_last, fresh_best, fresh_score, ba, ha, scores, tr = p_init(
        k0, state0, GIBBS_DIALS, mcfg.inner_loops, True, aux0)

    handoff_used = False
    hand_score = None
    pred_scores = None
    if carry_in is not None and mem.handoff != "none":
        # carried state -> this window's frame-0 data, then a burn-in phase pair
        hstate = propagate_state(carry_in["state"], points[0], motion[0])
        base_h = base_aux(hstate, kappa=mem.kappa,
                          kappa_sel=mem.kappa_sel_resolved())
        ev0 = float(np.clip(valid[0].mean() / mem.vel_evidence_floor, 0.0, 1.0))
        haux = frame_aux(hstate, mem, base_h, vel_evidence=ev0)
        hkey = carry_in["key"]
        hkey, hk1, hk2 = jax.random.split(hkey, 3)
        _, hbest, _, _, _, _, _ = p_vel(hk1, hstate, VELOCITY_UPDATE_DIALS,
                                        mcfg.inner_loops, True, haux)
        _, hbest, hscore, hba, hha, hscores, htr = p_track(
            hk2, hbest, GIBBS_DIALS, mcfg.inner_loops, True, haux)
        hand_score = float(hscore)
        if mem.handoff == "always":
            accept = True
        elif mem.handoff == "static_pred":
            # accept by pure data fit (assignment-marginalized likelihood), not
            # the joint — compares best-of-burn-in against best-of-fresh-init
            h_pred = float(marginal_data_loglik(hbest))
            f_pred = float(marginal_data_loglik(fresh_best))
            pred_scores = (h_pred, f_pred)
            accept = h_pred > f_pred
        else:
            accept = float(hscore) > float(fresh_score)
        if accept:
            handoff_used = True
            state, key = hbest, hkey
            ba, ha, scores, tr = hba, hha, hscores, htr
        else:
            state = fresh_last
    else:
        state = fresh_last                       # run_gestalt continues from LAST

    dev_traces = {"f0_init": tr}
    per_frame = [_frame_summary(state, ba, ha, scores, mcfg, G)]
    state0 = state          # frame-0 accepted state (backward/head handoffs)

    base = base_aux(state, kappa=mem.kappa, kappa_sel=mem.kappa_sel_resolved())
    frame_evidence = np.clip(valid.mean(axis=1) / mem.vel_evidence_floor, 0.0, 1.0)
    state, key = _run_frames(state, key, points, motion, mcfg, mem, base,
                             per_frame, dev_traces, 0, G, p_vel, p_track,
                             frame_evidence)

    jax.block_until_ready(state.datapoints_state.blob_assignments)
    t_infer = time.time() - t_start
    traces = jax.device_get(dev_traces) if emit else {}

    results, arrays_out = _evaluate(per_frame, gt_masks, combined, mcfg,
                                    t_infer, seed)
    results["roi_fallback"] = roi_fallback
    results["memory"] = mem.name
    results["memory_hash"] = mem.content_hash()
    results["handoff_used"] = handoff_used
    results["fresh_score"] = float(fresh_score)
    if hand_score is not None:
        results["handoff_score"] = hand_score
    if pred_scores is not None:
        results["handoff_pred"], results["fresh_pred"] = pred_scores
    carry_out = {"state": state, "key": key, "state0": state0}
    return results, arrays_out, traces, carry_out
