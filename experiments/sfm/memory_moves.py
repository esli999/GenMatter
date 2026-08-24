"""
Memory-augmented Gibbs moves (experiments/sfm only; the core sampler is untouched).

Design contract (validated against genmatter/inference.py):
- Every move here is a verbatim copy of its core counterpart with the memory term
  added, preserving the exact key-split structure — so the OFF configuration
  (kappa == 0, use_filter == False, base priors tiled) reproduces the stock chain
  BIT-IDENTICALLY (asserted in tests/test_memory.py).
- All memory quantities live in a traced `MemAux` pytree: one compilation serves
  every kappa / lambda / on-off configuration of the same phase length.

Mechanisms:
- STICKY ASSIGNMENTS: kappa * one_hot(prev_assign) added to the datapoint
  assignment logits. prev_assign is anchored at frame start (post-propagate) — a
  tempered target with a temporal assignment prior. kappa = 0 is a no-op.
- SUFFICIENT-STAT FILTERING: per-blob priors carried between frames.
  Covariance moves take batched NIW priors (nu[L], Psi[L,3,3]) — the core
  posterior helper accepts these natively; with base values tiled this is the
  stock move. Mean moves gain an extra Gaussian prior (mu0, Sigma0) combined
  exactly (precision addition), switched by lax.cond(use_filter).
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import genjax

from genmatter.inference import (
    stable_segment_sum,
    normal_inverse_wishart_posterior_by_cluster,
    normal_normal_posterior_full_cov_batched_flexible_prior,
    blob_datapoint_likelihood_model_no_assignment,
)
from genmatter.core_types import truncate_eigenval_ratio, inverse_wishart
from genjax import ChoiceMapBuilder as C


class MemAux(NamedTuple):
    """Traced memory state threaded through the mem sweep (a pytree of arrays)."""
    prev_assign: jnp.ndarray    # i32[N]  frame-start datapoint assignments (anchor)
    kappa: jnp.ndarray          # f32     sticky strength in the move logits
    kappa_sel: jnp.ndarray      # f32     sticky strength in the selection score
    use_filter: jnp.ndarray     # bool    switches the filtered mean moves
    nu_B: jnp.ndarray           # f32[L]      NIW dof, blob position covs
    Psi_B: jnp.ndarray          # f32[L,3,3]  NIW scale, blob position covs
    nu_V: jnp.ndarray           # f32[L]      NIW dof, blob velocity covs
    Psi_V: jnp.ndarray          # f32[L,3,3]  NIW scale, blob velocity covs
    mu_mu0: jnp.ndarray         # f32[L,3]    Gaussian prior mean, blob means
    mu_Sigma0: jnp.ndarray      # f32[L,3,3]  Gaussian prior cov, blob means
    vel_mu0: jnp.ndarray        # f32[L,3]    Gaussian prior mean, blob vel means
    vel_Sigma0: jnp.ndarray     # f32[L,3,3]  Gaussian prior cov, blob vel means


def base_aux(state, kappa=0.0, kappa_sel=None, use_filter=False):
    """Memory-off MemAux: base NIW priors tiled per blob (bit-identical to stock),
    inert mean priors, current assignments as the anchor."""
    h = state.hypers
    L = h.n_blobs
    kappa_sel = kappa if kappa_sel is None else kappa_sel
    eye = jnp.broadcast_to(jnp.eye(3), (L, 3, 3))
    return MemAux(
        prev_assign=state.datapoints_state.blob_assignments.astype(jnp.int32),
        kappa=jnp.float32(kappa),
        kappa_sel=jnp.float32(kappa_sel),
        use_filter=jnp.asarray(bool(use_filter)),
        nu_B=jnp.broadcast_to(jnp.float32(h.nu_B), (L,)),
        Psi_B=jnp.broadcast_to(h.Psi_B, (L, 3, 3)),
        nu_V=jnp.broadcast_to(jnp.float32(h.nu_V), (L,)),
        Psi_V=jnp.broadcast_to(h.Psi_V, (L, 3, 3)),
        mu_mu0=state.blobs_state.blob_means,
        mu_Sigma0=1e6 * eye,
        vel_mu0=state.blobs_state.blob_vel_means,
        vel_Sigma0=1e6 * eye,
    )


def frame_aux(state, mem, base, vel_evidence=1.0):
    """MemAux for a tracked frame, built from the POST-PROPAGATE state (blob means
    have ridden the rigid transform; covs/vel means carry the previous frame's
    accepted samples; datapoint assignments carry the previous frame's grouping).

    Filtering (lam = mem.filter_lambda > 0):
    - NIW scale blends toward the tracked covariances at UNCHANGED prior strength
      (nu stays at the base ~one-frame pseudo-count; the prior's shape tracks,
      its weight does not grow): Psi = (1-lam)*Psi_base + lam*(nu+d+1)*cov_prev,
      eigenvalue-truncated after every blend.
    - Mean priors: mu0 = propagated means; Sigma0 = cov_prev/(lam*N_l) + q*I
      (per-blob confidence from occupancy, floored by process noise q).
    - With mem.adaptive_vel, the VELOCITY memory strength is scaled by
      `vel_evidence` (traced scalar in [0,1], from the current frame's
      valid-motion fraction): stale motion is never asserted against a frame
      with no motion evidence.
    """
    lam = jnp.float32(mem.filter_lambda)
    anchor = state.datapoints_state.blob_assignments.astype(jnp.int32)
    if mem.filter_lambda <= 0.0:
        return base._replace(prev_assign=anchor,
                             kappa=jnp.float32(mem.kappa),
                             kappa_sel=jnp.float32(mem.kappa_sel_resolved()),
                             mu_mu0=state.blobs_state.blob_means,
                             vel_mu0=state.blobs_state.blob_vel_means)

    L = state.hypers.n_blobs
    d = 3
    covs = state.blobs_state.blob_covs
    vcovs = state.blobs_state.blob_vel_covs
    N_l = jax.ops.segment_sum(jnp.ones_like(anchor, dtype=jnp.float32),
                              anchor, num_segments=L)
    occ = jnp.maximum(N_l, 1.0)[:, None, None]
    eye = jnp.broadcast_to(jnp.eye(d), (L, d, d))

    Psi_B = truncate_eigenval_ratio(
        (1.0 - lam) * base.Psi_B + lam * (base.nu_B[:, None, None] + d + 1) * covs,
        threshold=1e6)
    mu_Sigma0 = covs / (lam * occ) + mem.process_noise_q * eye

    if mem.filter_velocity:
        lam_v = lam * (jnp.clip(jnp.float32(vel_evidence), 0.0, 1.0)
                       if mem.adaptive_vel else 1.0)
        Psi_V = truncate_eigenval_ratio(
            (1.0 - lam_v) * base.Psi_V
            + lam_v * (base.nu_V[:, None, None] + d + 1) * vcovs,
            threshold=1e6)
        vel_Sigma0 = vcovs / (jnp.maximum(lam_v, 1e-6) * occ) \
            + mem.vel_process_noise_q * eye
        vel_mu0 = state.blobs_state.blob_vel_means
    else:
        # position/shape memory only: velocity priors stay effectively stock
        # (base NIW scale, near-flat Gaussian toward 0 — velocities re-adapt to
        # the data within a frame, avoiding stale-motion entrenchment at
        # cessation)
        Psi_V = base.Psi_V
        vel_Sigma0 = 1e6 * eye
        vel_mu0 = jnp.zeros_like(state.blobs_state.blob_vel_means)

    return MemAux(
        prev_assign=anchor,
        kappa=jnp.float32(mem.kappa),
        kappa_sel=jnp.float32(mem.kappa_sel_resolved()),
        use_filter=jnp.asarray(True),
        nu_B=base.nu_B, Psi_B=Psi_B, nu_V=base.nu_V, Psi_V=Psi_V,
        mu_mu0=state.blobs_state.blob_means, mu_Sigma0=mu_Sigma0,
        vel_mu0=vel_mu0, vel_Sigma0=vel_Sigma0,
    )


# --------------------------------------------------------------------- sticky move
def sticky_blob_assignments_batched(key, genmatter_state, aux: MemAux):
    """Core gibbs_blob_assignments_batched + kappa*one_hot(prev_assign) in the
    logits. Copied verbatim (same key split, batching, outlier handling); the only
    delta is the sticky bonus, which is exactly 0 when aux.kappa == 0."""
    posterior_key, _ = jax.random.split(key)

    batch_size = 1024

    hypers = genmatter_state.hypers
    num_blobs = hypers.n_blobs
    num_datapoints = hypers.n_datapoints
    datapoint_positions = genmatter_state.datapoints_state.datapoint_positions
    datapoint_vels = genmatter_state.datapoints_state.datapoint_vels
    blobs_state = genmatter_state.blobs_state

    blob_weights = blobs_state.blob_weights
    outlier_prob = hypers.outlier_prob

    extended_weights = jnp.concatenate([blob_weights, jnp.array([outlier_prob])])
    normalized_weights = extended_weights / jnp.sum(extended_weights)
    log_mixture_weights = jnp.log(normalized_weights)

    def compute_local_density(point_idx):
        chm = (C["datapoint_position"].set(datapoint_positions[point_idx]) |
               C["datapoint_vel"].set(datapoint_vels[point_idx]))

        log_liks = jax.vmap(
            lambda i: blob_datapoint_likelihood_model_no_assignment.assess(
                chm, (blobs_state[i],)
            )[0]
        )(jnp.arange(num_blobs))

        v = datapoint_vels[point_idx]
        speed = jnp.linalg.norm(v)
        alpha = hypers.outlier_velocity_gamma_shape
        beta = hypers.outlier_velocity_gamma_rate
        log_gamma_vel = (
            (alpha - 1) * jnp.log(speed + 1e-8)
            - beta * speed
            - alpha * jnp.log(1. / beta)
            - jax.lax.lgamma(alpha)
        )

        raw_log_liks = jnp.concatenate([log_liks, jnp.array([log_gamma_vel])])
        sticky = aux.kappa * jax.nn.one_hot(aux.prev_assign[point_idx],
                                            num_blobs + 1)
        return raw_log_liks + log_mixture_weights + sticky

    def batch_compute_logprobs(carry, batch_idx):
        batch_start = batch_idx * batch_size
        batch_indices = jnp.arange(batch_size) + batch_start
        batch_logprobs = jax.vmap(compute_local_density)(batch_indices)
        return carry, batch_logprobs

    num_full_batches = num_datapoints // batch_size
    _, batched_logprobs = jax.lax.scan(
        batch_compute_logprobs, None, jnp.arange(num_full_batches))
    all_logprobs = batched_logprobs.reshape(num_full_batches * batch_size, -1)

    updated_assignments = genjax.categorical.sample(posterior_key, logits=all_logprobs)

    return genmatter_state.replace({
        'datapoints_state': {'blob_assignments': updated_assignments}})


# ----------------------------------------------------------- filtered covariance moves
def gibbs_blob_covs_aux(key, genmatter_state, aux: MemAux):
    """Core gibbs_blob_covs with the NIW prior taken from aux (batched per blob).
    With base priors tiled this is the stock move bit-for-bit."""
    posterior_key, _ = jax.random.split(key)

    datapoint_positions = genmatter_state.datapoints_state.datapoint_positions
    blob_assignments = genmatter_state.datapoints_state.blob_assignments
    assumed_known_means = genmatter_state.blobs_state.blob_means
    n_blobs = genmatter_state.hypers.n_blobs

    datapoint_pseudocounts = jnp.ones(datapoint_positions.shape[0],
                                      dtype=datapoint_positions.dtype)
    N_l = jax.ops.segment_sum(datapoint_pseudocounts, blob_assignments,
                              num_segments=n_blobs)

    nu_posteriors, Psi_posteriors = normal_inverse_wishart_posterior_by_cluster(
        datapoint_positions, blob_assignments, assumed_known_means,
        aux.nu_B, aux.Psi_B, datapoint_pseudocounts)

    updated_blob_covs = inverse_wishart.sample(posterior_key, nu_posteriors,
                                               Psi_posteriors)
    current_blob_covs = genmatter_state.blobs_state.blob_covs
    mask = (N_l >= 3)[:, None, None]
    final_blob_covs = jnp.where(mask, updated_blob_covs, current_blob_covs)
    return genmatter_state.replace({'blobs_state': {'blob_covs': final_blob_covs}})


def gibbs_blob_vel_covs_aux(key, genmatter_state, aux: MemAux):
    """Core gibbs_blob_vel_covs with the NIW prior taken from aux (batched)."""
    posterior_key, _ = jax.random.split(key)

    datapoint_vels = genmatter_state.datapoints_state.datapoint_vels
    blob_assignments = genmatter_state.datapoints_state.blob_assignments
    assumed_known_vel_means = genmatter_state.blobs_state.blob_vel_means
    n_blobs = genmatter_state.hypers.n_blobs

    datapoint_pseudocounts = jnp.ones(datapoint_vels.shape[0],
                                      dtype=datapoint_vels.dtype)
    N_l = jax.ops.segment_sum(datapoint_pseudocounts, blob_assignments,
                              num_segments=n_blobs)

    nu_posteriors, Psi_posteriors = normal_inverse_wishart_posterior_by_cluster(
        datapoint_vels, blob_assignments, assumed_known_vel_means,
        aux.nu_V, aux.Psi_V, datapoint_pseudocounts)

    updated = inverse_wishart.sample(posterior_key, nu_posteriors, Psi_posteriors)
    current = genmatter_state.blobs_state.blob_vel_covs
    mask = (N_l >= 3)[:, None, None]
    final = jnp.where(mask, updated, current)
    return genmatter_state.replace({'blobs_state': {'blob_vel_covs': final}})


# ------------------------------------------------------------- filtered mean moves
def gibbs_blob_means_filtered(key, genmatter_state, aux: MemAux):
    """Core gibbs_blob_means + an extra Gaussian prior (aux.mu_mu0, aux.mu_Sigma0)
    combined exactly in precision form."""
    posterior_key, _ = jax.random.split(key)

    datapoint_positions = genmatter_state.datapoints_state.datapoint_positions
    blob_assignments = genmatter_state.datapoints_state.blob_assignments
    blob_vel_means = genmatter_state.blobs_state.blob_vel_means
    hyperblob_assignments = genmatter_state.blobs_state.hyperblob_assignments
    blob_covs = genmatter_state.blobs_state.blob_covs

    mu_H = genmatter_state.hyperblobs_state.hyperblob_means
    trans_vels = genmatter_state.hyperblobs_state.hyperblob_trans_vels
    rot_vels = genmatter_state.hyperblobs_state.hyperblob_rot_vels

    sigmaV = genmatter_state.hypers.sigma_V
    L = genmatter_state.hypers.n_blobs
    d = genmatter_state.blobs_state.blob_means.shape[-1]

    muH_l = mu_H[hyperblob_assignments]
    hyperblob_cov_inv = jnp.linalg.inv(
        genmatter_state.hyperblobs_state.hyperblob_covs[hyperblob_assignments])

    N_l = jax.ops.segment_sum(jnp.ones(datapoint_positions.shape[0]),
                              blob_assignments, num_segments=L)
    S_l = stable_segment_sum(datapoint_positions, blob_assignments, L)
    blob_cov_inv = jnp.linalg.inv(blob_covs)

    A_l = rot_vels[hyperblob_assignments] - jnp.eye(d)
    b_l = trans_vels[hyperblob_assignments] - jnp.einsum("lij,lj->li", A_l, muH_l)
    residuals = blob_vel_means - b_l
    lhs_affine = jnp.einsum("lij,lik->ljk", A_l, A_l)
    rhs_affine = jnp.einsum("lij,li->lj", A_l, residuals)

    Sigma0_inv = jnp.linalg.inv(aux.mu_Sigma0)                    # memory prior

    P_post = (hyperblob_cov_inv + blob_cov_inv * N_l[:, None, None]
              + lhs_affine / sigmaV + Sigma0_inv)
    weighted_mean = (jnp.einsum("lij,lj->li", hyperblob_cov_inv, muH_l)
                     + jnp.einsum("lij,lj->li", blob_cov_inv, S_l)
                     + rhs_affine / sigmaV
                     + jnp.einsum("lij,lj->li", Sigma0_inv, aux.mu_mu0))

    cov_post = jnp.linalg.inv(P_post)
    mean_post = jnp.einsum("lij,lj->li", cov_post, weighted_mean)
    new_blob_means = genjax.mv_normal.sample(posterior_key, mean_post, cov_post)
    return genmatter_state.replace({'blobs_state': {'blob_means': new_blob_means}})


def gibbs_blob_vel_means_filtered(key, genmatter_state, aux: MemAux):
    """Core gibbs_blob_vel_means with the memory prior (aux.vel_mu0, aux.vel_Sigma0)
    combined exactly with the model's affine prior (variance sigma_V) before the
    conjugate update. Empty blobs keep the memory mean instead of zero."""
    posterior_key, _ = jax.random.split(key)

    datapoint_vels = genmatter_state.datapoints_state.datapoint_vels
    blob_assignments = genmatter_state.datapoints_state.blob_assignments
    hyperblob_assignments = genmatter_state.blobs_state.hyperblob_assignments
    blob_means = genmatter_state.blobs_state.blob_means
    assigned_hyperblob_per_blob = genmatter_state.hyperblobs_state[hyperblob_assignments]
    likelihood_blob_vel_covs = genmatter_state.blobs_state.blob_vel_covs

    prior_blob_vel_means_ = (assigned_hyperblob_per_blob.hyperblob_trans_vels
        + jnp.einsum('nij,nj->ni',
                     assigned_hyperblob_per_blob.hyperblob_rot_vels
                     - jnp.repeat(jnp.eye(3)[None, ...],
                                  genmatter_state.hypers.n_blobs, axis=0),
                     blob_means - assigned_hyperblob_per_blob.hyperblob_means))
    sigma_V = genmatter_state.hypers.sigma_V
    n_blobs = genmatter_state.hypers.n_blobs

    N_l = jax.ops.segment_sum(jnp.ones(datapoint_vels.shape[0],
                                       dtype=datapoint_vels.dtype),
                              blob_assignments, num_segments=n_blobs)

    # exact combination of the two Gaussian priors (precision addition)
    vSigma0_inv = jnp.linalg.inv(aux.vel_Sigma0)                  # [L,3,3]
    model_prec = jnp.broadcast_to(jnp.eye(3) / sigma_V, vSigma0_inv.shape)
    comb_cov = jnp.linalg.inv(vSigma0_inv + model_prec)
    comb_mu = jnp.einsum("lij,lj->li", comb_cov,
                         jnp.einsum("lij,lj->li", vSigma0_inv, aux.vel_mu0)
                         + prior_blob_vel_means_ / sigma_V)

    posterior_mus, posterior_covs = normal_normal_posterior_full_cov_batched_flexible_prior(
        datapoint_vels, blob_assignments, comb_mu, comb_cov, likelihood_blob_vel_covs)

    has_points = N_l > 0
    sampled = genjax.mv_normal.sample(posterior_key, posterior_mus, posterior_covs)
    out = jnp.where(has_points[:, None], sampled, aux.vel_mu0)
    return genmatter_state.replace({'blobs_state': {'blob_vel_means': out}})
