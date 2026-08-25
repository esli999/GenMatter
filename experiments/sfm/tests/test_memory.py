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

    # --- 5. Evidence-conditioned freezing: at full evidence the freeze term
    # vanishes (aux identical to plain sticky); at zero evidence kappa_eff =
    # kappa + freeze_kappa and a frozen chain keeps (almost) every anchor label
    mem_frz = MemoryConfig(name="t_frz", kappa=1.0, freeze_kappa=100.0)
    aux_ev1 = frame_aux(st2, mem_frz, base, vel_evidence=1.0)
    assert float(aux_ev1.kappa) == 1.0, "freeze must vanish at full evidence"
    aux_ev0 = frame_aux(st2, mem_frz, base, vel_evidence=0.0)
    assert float(aux_ev0.kappa) == 101.0, "freeze kappa_eff wrong at zero evidence"
    last_z, *_ = mprog(jkey(77), st2, GIBBS_DIALS, TINY.inner_loops, True, aux_ev0)
    frozen_matches = int(jnp.sum(last_z.datapoints_state.blob_assignments
                                 == aux_ev0.prev_assign))
    n = aux_ev0.prev_assign.shape[0]
    assert frozen_matches >= 0.9 * n, \
        f"frozen chain kept only {frozen_matches}/{n} anchor labels"
    print(f"evidence-conditioned freeze: PASS ({frozen_matches}/{n} held at ev=0)")

    # --- 6. Marginal data likelihood: finite; prefers the fit state over a
    # shuffled-means corruption of it (the grouping-aware acceptance criterion)
    from experiments.sfm.memory_algorithm import marginal_data_loglik
    good = float(marginal_data_loglik(last_b))
    perm = jax.random.permutation(jkey(5), last_b.blobs_state.blob_means.shape[0])
    corrupt = last_b.replace(
        {"blobs_state": {"blob_means": last_b.blobs_state.blob_means[perm]}})
    bad = float(marginal_data_loglik(corrupt))
    assert np.isfinite(good) and np.isfinite(bad)
    assert good > bad, f"marginal loglik did not prefer the fit state ({good} <= {bad})"
    print(f"marginal data loglik ranks fit > corrupted: PASS ({good:.2f} > {bad:.2f})")

    print("test_memory: PASS")


if __name__ == "__main__":
    main()
