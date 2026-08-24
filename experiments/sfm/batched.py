"""
Window-batched vmapped inference (Workstream 6): the Gibbs sampler is latency-bound
(~49% SM steady-state on an A100 at 16,384x150), so B independent windows are run
through ONE vmapped mem phase program. Everything per-window (data, empirical
hypers, memory aux, PRNG keys) is a traced leaf stacked over B; static grid/count
fields are shared by grafting each window's traced hyperparameters onto window 0's
hyperparameter object (identical static objects -> identical treedefs).

The mem path with memory OFF is the baseline algorithm, so this serves both the
plain accuracy runs and the memory bake-offs; it is also the substrate SMC vmaps
over (particles are one more stacked axis folded into B).

Note: XLA may reassociate reductions under vmap, so batched chains are not
guaranteed bit-identical to unbatched ones — compare like against like (both
batched or both unbatched) in paired analyses.
"""
import time
from concurrent.futures import ThreadPoolExecutor

import jax
import jax.numpy as jnp
import numpy as np

from genmatter.trace_wrappers import pytree_stack

from . import sfm_config as cfg
from .algorithm import (GIBBS_DIALS, VELOCITY_UPDATE_DIALS, DegenerateWindowError,
                        init_window_state, _frame_summary, _evaluate)
from .memory_algorithm import get_mem_phase_program
from .memory_config import MemoryConfig
from .memory_moves import base_aux, frame_aux
from .propagate import propagate_state

TRACED_HYPER_FIELDS = (
    "outlier_prob", "outlier_velocity_gamma_shape", "outlier_velocity_gamma_rate",
    "alpha", "beta", "mu_H", "sigma_H", "nu_H", "Psi_H", "nu_B", "Psi_B",
    "sigma_V", "nu_V", "Psi_V")


def unify_static_hypers(states):
    """Graft every window's traced hyperparameters onto window 0's hypers object so
    all states share the SAME static grid/count objects (=> equal treedefs)."""
    h0 = states[0].hypers
    out = [states[0]]
    for st in states[1:]:
        grafted = h0.replace({k: getattr(st.hypers, k) for k in TRACED_HYPER_FIELDS})
        out.append(st.replace({"hypers": grafted}))
    return out


_VMAP_CACHE = {}


def get_batched_program(num_sweeps, keep_last, rescore_stride,
                        emit_traces=False, trace_thin=5):
    key_ = (num_sweeps, min(keep_last, num_sweeps), rescore_stride,
            emit_traces, trace_thin)
    if key_ in _VMAP_CACHE:
        return _VMAP_CACHE[key_]
    prog = get_mem_phase_program(num_sweeps, keep_last, rescore_stride,
                                 emit_traces, trace_thin)
    vprog = jax.jit(jax.vmap(prog, in_axes=(0, 0, None, None, None, 0)))
    _VMAP_CACHE[key_] = vprog
    return vprog


_vpropagate = jax.jit(jax.vmap(propagate_state))


def _vsplit(keys, n):
    parts = jax.vmap(lambda k: jax.random.split(k, n))(keys)
    return [parts[:, i] for i in range(n)]


def infer_windows_batched(batch, mcfg: cfg.SfmModelConfig, mem: MemoryConfig,
                          seed: int = None):
    """Run B windows through vmapped phases. `batch` = list of bundle array dicts.
    Returns a list of per-window entries: either ("ok", results, arrays_out,
    traces) or ("error", exception). Windows keep the exact single-window key
    stream (same seed), so per-window behavior matches unbatched runs up to
    vmap-level float reassociation."""
    t_start = time.time()
    seed = mcfg.seed if seed is None else seed
    emit = bool(mcfg.log_traces)

    arrs = [{k: np.asarray(a[k]) for k in
             ("points_3d", "motion_3d", "motion_valid", "depth_sq", "flow_sq",
              "gt_masks")} for a in batch]

    def init_one(a):
        try:
            state, key, combined, roi_fb = init_window_state(
                a["points_3d"].astype(np.float32), a["motion_3d"].astype(np.float32),
                a["motion_valid"].astype(bool), a["depth_sq"].astype(np.float32),
                a["flow_sq"].astype(np.float32), mcfg, seed)
            return ("ok", state, key, combined, roi_fb)
        except (DegenerateWindowError, Exception) as e:
            return ("error", e)

    with ThreadPoolExecutor(max_workers=min(8, len(arrs))) as ex:
        inits = list(ex.map(init_one, arrs))

    live = [i for i, r in enumerate(inits) if r[0] == "ok"]
    out = [None] * len(batch)
    for i, r in enumerate(inits):
        if r[0] == "error":
            out[i] = ("error", r[1])
    if not live:
        return out

    states = unify_static_hypers([inits[i][1] for i in live])
    keys = jnp.stack([inits[i][2] for i in live])
    B = len(live)
    points = jnp.stack([arrs[i]["points_3d"].astype(np.float32) for i in live])
    motion = jnp.stack([arrs[i]["motion_3d"].astype(np.float32) for i in live])
    T = points.shape[1]

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

    vbase_aux0 = jax.jit(jax.vmap(base_aux))
    vbase_aux = jax.jit(jax.vmap(
        lambda st: base_aux(st, kappa=mem.kappa,
                            kappa_sel=mem.kappa_sel_resolved())))
    vframe_aux = jax.jit(jax.vmap(lambda st, b: frame_aux(st, mem, b)))

    state = pytree_stack(states)
    aux0 = vbase_aux0(state)                            # init frame: kappa = 0
    keys, k0 = _vsplit(keys, 2)
    last, best, bscore, ba, ha, scores, tr = p_init(
        k0, state, GIBBS_DIALS, mcfg.inner_loops, True, aux0)
    state = last                                        # continue from LAST init

    dev_traces = {"f0_init": tr}
    summaries = [(ba, ha, scores, state)]               # device-side, unstack later

    base = vbase_aux(state)
    for f in range(1, T):
        state = _vpropagate(state, points[:, f], motion[:, f])
        aux = vframe_aux(state, base)
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
    t_batch = time.time() - t_start
    host_traces = jax.device_get(dev_traces) if emit else {}
    G = cfg.GRID_HW

    for b, i in enumerate(live):
        per_frame = []
        for (ba, ha, scores, st) in summaries:
            st_b = jax.tree_util.tree_map(lambda x: x[b], st)
            per_frame.append(_frame_summary(st_b, ba[b], ha[b], scores[b], mcfg, G))
        results, arrays_out = _evaluate(
            per_frame, arrs[i]["gt_masks"].astype(bool), inits[i][3], mcfg,
            t_batch / B, seed)
        results["roi_fallback"] = inits[i][4]
        results["memory"] = mem.name
        results["memory_hash"] = mem.content_hash()
        results["batch_size"] = B
        traces_i = {ph: {k: v[b] for k, v in d.items()}
                    for ph, d in host_traces.items()} if emit else {}
        out[i] = ("ok", results, arrays_out, traces_i)
    return out
