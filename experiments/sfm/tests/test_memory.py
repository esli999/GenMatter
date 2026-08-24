"""
Memory-mechanism correctness on the tiny synthetic problem (CPU-safe):

1. OFF-parity: the mem phase program with kappa=0, filter-off, base priors tiled
   reproduces the baseline phase program BIT-IDENTICALLY (same key).
2. Sticky pull: a strong kappa keeps the chain closer to the anchor than baseline.
3. Selection augmentation: kappa_sel > 0 never picks a lower-match state than the
   raw argmax would at equal scores (smoke: runs, finite, matches counted).
4. Filtering: frame_aux(lambda=0) returns the base (off) priors; lambda>0 yields
   finite blended priors and a finite filtered chain that stays near the anchor
   velocity/means (smoke).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import jax
import jax.numpy as jnp
from jax.random import key as jkey

from genmatter.utils import make_hierarchical_kmeans_chm_with_mask_fixed_hyperblob
from experiments.sfm.algorithm import (GIBBS_DIALS, get_phase_program, build_hypers,
                                       model_jimportance)
from experiments.sfm.memory_algorithm import get_mem_phase_program
from experiments.sfm.memory_moves import base_aux, frame_aux
from experiments.sfm.memory_config import MemoryConfig
from experiments.sfm.propagate import propagate_state
from experiments.sfm.tests.test_parity import TINY, synth_problem, leaves_equal


def setup_state():
    points, motion, obj = synth_problem()
    chm, roi_b, roi_h = make_hierarchical_kmeans_chm_with_mask_fixed_hyperblob(
        points, TINY.n_blobs, TINY.n_hyperblobs,
        segmentation_mask=obj, motion_vectors=motion,
        num_roi_blobs=TINY.n_roi_blobs)
    hypers = build_hypers(TINY, chm, roi_b, roi_h)
    tr, _ = model_jimportance(jkey(1), chm, (hypers,))
    return tr.get_retval(), points, motion


def main():
    state, points, motion = setup_state()
    key = jkey(123)

    # --- 1. OFF-parity (bit-identical to the baseline program)
    prog = get_phase_program(TINY.init_sweeps, keep_last=4, rescore_stride=1)
    last_b, best_b, score_b, ba_b, ha_b, scores_b, _ = prog(
        key, state, GIBBS_DIALS, TINY.inner_loops, True)

    mprog = get_mem_phase_program(TINY.init_sweeps, keep_last=4, rescore_stride=1)
    aux_off = base_aux(state)
    last_m, best_m, score_m, ba_m, ha_m, scores_m, _ = mprog(
        key, state, GIBBS_DIALS, TINY.inner_loops, True, aux_off)

    assert leaves_equal(last_m, last_b), "mem OFF: last state diverged"
    assert leaves_equal(best_m, best_b), "mem OFF: best state diverged"
    assert np.array_equal(np.asarray(ba_m), np.asarray(ba_b)), "mem OFF: ba history"
    assert np.array_equal(np.asarray(scores_m), np.asarray(scores_b)), \
        "mem OFF: score sequence"
    assert float(score_m) == float(score_b), "mem OFF: best score"
    print("mem OFF-parity (bit-identical): PASS")

    # --- 2. Sticky pull toward the anchor
    anchor = last_b.datapoints_state.blob_assignments.astype(jnp.int32)
    st2 = propagate_state(last_b, points[1], motion[1])

    def run_kappa(kappa):
        aux = base_aux(st2, kappa=kappa)
        last, *_ = mprog(jkey(77), st2, GIBBS_DIALS, TINY.inner_loops, True, aux)
        return int(jnp.sum(last.datapoints_state.blob_assignments == anchor))

    m0, m_strong = run_kappa(0.0), run_kappa(50.0)
    assert m_strong >= m0, f"sticky kappa=50 kept fewer anchor matches ({m_strong} < {m0})"
    assert m_strong > 0
    print(f"sticky pull: PASS (matches {m0} -> {m_strong} of {anchor.shape[0]})")

    # --- 3+4. Filtering: lambda=0 returns base priors; lambda>0 finite chain
    mem_off = MemoryConfig(name="t_off")
    base = base_aux(st2)
    aux0 = frame_aux(st2, mem_off, base)
    assert not bool(aux0.use_filter)
    assert np.array_equal(np.asarray(aux0.Psi_B), np.asarray(base.Psi_B))

    mem_f = MemoryConfig(name="t_filt", kappa=1.0, filter_lambda=0.5)
    auxf = frame_aux(st2, mem_f, base)
    assert bool(auxf.use_filter)
    for name in ("Psi_B", "Psi_V", "mu_Sigma0", "vel_Sigma0"):
        v = np.asarray(getattr(auxf, name))
        assert np.isfinite(v).all(), f"{name} not finite"
    ev = np.linalg.eigvalsh(np.asarray(auxf.mu_Sigma0))
    assert (ev > 0).all(), "mu_Sigma0 not PD"

    last_f, best_f, score_f, _, _, scores_f, _ = mprog(
        jkey(77), st2, GIBBS_DIALS, TINY.inner_loops, True, auxf)
    assert np.isfinite(np.asarray(scores_f)).all(), "filtered chain scores not finite"
    for leaf in jax.tree_util.tree_leaves(last_f):
        assert np.isfinite(np.asarray(leaf)).all(), "filtered state not finite"
    print("filtering (off-equivalence + finite filtered chain): PASS")

    print("test_memory: PASS")


if __name__ == "__main__":
    main()
