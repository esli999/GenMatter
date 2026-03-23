# HDGMM Tracking with DINO Features - Batch Processing
# Clean implementation for running experiments on multiple videos

import os
import json
import time
import gc
import jax

# JAX compilation cache setup
cache_dir = os.path.join(os.getcwd(), ".jax_cache")
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
jax.config.update("jax_compilation_cache_dir", cache_dir)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.experimental.compilation_cache.compilation_cache.set_cache_dir(cache_dir)

import jax.numpy as jnp
from jax.random import key as jkey

import numpy as np
import pickle
from tqdm import tqdm
import cv2
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
import config

from genparticles.datatypes import *
from genparticles.model_3d import *
from genparticles.inference import *
from genparticles.dataloader import *
from genparticles.utils import *
from genparticles.evaluation import *

import genjax
from genjax import Const, gen, Pytree

# ============================================================================
# Configuration
# ============================================================================

# FPS measurement flag
MEASURE_FPS = True

VIDEO_NAMES = list(config.TAPVID_DAVIS_VIDEO_NAMES)

NUM_INIT_PARTICLES_ON_MASK = None
USE_SAM_FRAME0 = True

# Paths
DAVIS_3D_MOTION_PATH = str(config.DAVIS_3D_MOTION_PATH)
DAVIS_SEGMASKS_PATH = str(config.DAVIS_SEGMASKS_PATH)
DAVIS_RGB_PATH = str(config.DAVIS_RGB_PATH)
DINO_PATH_TEMPLATE = str(config.DAVIS_DINO_PATH / '{}_dino_pca_per_pixel.npz')
SAM_FRAME0_PATH_TEMPLATE = str(config.DAVIS_SAM_FRAME0_PATH / '{}_SAM_frame0.png')

# Output directory
EXPERIMENT_SAVE_DIR = str(config.DAVIS_ABLATION_OUTPUT_DIR)

# Model hyperparameters
NUM_BLOBS = 500
NUM_HYPERBLOBS_ORIGINAL = 1  # Ablation: 1 cluster (no clustering effect)
FOCAL_LENGTH = 520.0
BLOB_COUNTING_THRESHOLD = 0
RANDOM_SEED = 42

# ============================================================================
# Model Definition with DINO Features
# ============================================================================

@Pytree.dataclass
class HDGMM_Hyperparams_DINO(Super_Pytree):
    outlier_prob: jnp.float32 = Super_Pytree.field()
    outlier_velocity_gamma_shape: jnp.float32 = Super_Pytree.field()
    outlier_velocity_gamma_rate: jnp.float32 = Super_Pytree.field()
    alpha: jnp.float32 = Super_Pytree.field()
    beta: jnp.float32 = Super_Pytree.field()
    mu_H: jnp.ndarray = Super_Pytree.field()
    sigma_H: jnp.float32 = Super_Pytree.field()
    nu_H: jnp.float32 = Super_Pytree.field()
    Psi_H: jnp.ndarray = Super_Pytree.field()
    nu_B: jnp.float32 = Super_Pytree.field()
    Psi_B: jnp.ndarray = Super_Pytree.field()
    sigma_V: jnp.float32 = Super_Pytree.field()
    nu_V: jnp.float32 = Super_Pytree.field()
    Psi_V: jnp.ndarray = Super_Pytree.field()
    mu_F: jnp.ndarray = Super_Pytree.field()
    sigma_F_prior: jnp.ndarray = Super_Pytree.field()
    sigma_F: jnp.float32 = Super_Pytree.field()
    translation_max_radius: StaticJnp = Super_Pytree.static()
    translation_num_radii_cells: StaticJnp = Super_Pytree.static()
    translation_theta_step_deg: StaticJnp = Super_Pytree.static()
    translation_gaussian_scale: StaticJnp = Super_Pytree.static()
    rotation_vmf_kappa: StaticJnp = Super_Pytree.static()
    rotation_angle_max_deg: StaticJnp = Super_Pytree.static()
    rotation_angle_step_deg: StaticJnp = Super_Pytree.static()
    n_hyperblobs: jnp.int32 = Super_Pytree.static()
    n_blobs: jnp.int32 = Super_Pytree.static()
    n_datapoints: jnp.int32 = Super_Pytree.static()
    discrete_translation: Precomputed_DiscreteDistribution = Super_Pytree.static()
    discrete_rotation: Precomputed_DiscreteDistribution = Super_Pytree.static()

    def __eq__(self, other):
        if jax.tree_util.tree_structure(self) != jax.tree_util.tree_structure(other):
            return False
        leaves1 = jax.tree_util.tree_leaves(self)
        leaves2 = jax.tree_util.tree_leaves(other)
        bools = [jnp.all(l1 == l2) for l1, l2 in zip(leaves1, leaves2)]
        return jnp.all(jnp.array(bools))

    @classmethod
    def create(cls, **kwargs):
        return HDGMM_Hyperparams.create.__func__(cls, **kwargs)

@Pytree.dataclass
class HDGMM_Blobs_State_DINO(Super_Pytree):
    hyperblob_assignments: jnp.ndarray
    blob_weights: jnp.ndarray
    blob_means: jnp.ndarray
    blob_covs: jnp.ndarray
    blob_vel_means: jnp.ndarray
    blob_vel_covs: jnp.ndarray
    blob_features: jnp.ndarray

@Pytree.dataclass
class HDGMM_Datapoints_State_DINO(Super_Pytree):
    blob_assignments: jnp.ndarray
    datapoint_positions: jnp.ndarray
    datapoint_vels: jnp.ndarray
    datapoint_features: jnp.ndarray

@Pytree.dataclass
class HDGMM_State_DINO(Super_Pytree):
    hypers: HDGMM_Hyperparams_DINO
    hyperblobs_state: HDGMM_Hyperblobs_State
    blobs_state: HDGMM_Blobs_State_DINO
    datapoints_state: HDGMM_Datapoints_State_DINO

@gen
def HDGMM_model_dino(hypers: HDGMM_Hyperparams_DINO):
    hyperblobs_state = HDGMM_hyperblobs_model(hypers) @ 'hyperblobs'
    blobs_state = HDGMM_blobs_model_dino(hypers, hyperblobs_state) @ 'blobs'
    datapoints_state = HDGMM_datapoints_model_dino(hypers, blobs_state) @ 'datapoints'
    return HDGMM_State_DINO(
        hypers=hypers,
        hyperblobs_state=hyperblobs_state,
        blobs_state=blobs_state,
        datapoints_state=datapoints_state
    )

@gen
def HDGMM_blobs_model_dino(hypers: HDGMM_Hyperparams_DINO, hyperblobs_state: HDGMM_Hyperblobs_State):
    sample_shape = Const((hypers.n_blobs,))
    hyperblob_assignments = genjax.categorical(
        probs=hyperblobs_state.hyperblob_weights,
        sample_shape=sample_shape
    ) @ 'hyperblob_assignments'
    blob_weights = genjax.dirichlet(jnp.repeat(hypers.beta, hypers.n_blobs)) @ 'blob_weights'
    assigned_hyperblob_per_blob = hyperblobs_state[hyperblob_assignments]
    blob_covs = inverse_wishart(hypers.nu_B, hypers.Psi_B, sample_shape=sample_shape) @ 'blob_covs'
    blob_means = genjax.mv_normal(
        assigned_hyperblob_per_blob.hyperblob_means,
        assigned_hyperblob_per_blob.hyperblob_covs
    ) @ 'blob_means'
    blob_vel_means_ = assigned_hyperblob_per_blob.hyperblob_trans_vels + jnp.einsum(
        'nij,nj->ni',
        assigned_hyperblob_per_blob.hyperblob_rot_vels - jnp.repeat(jnp.eye(3)[None, ...], hypers.n_blobs, axis=0),
        blob_means - assigned_hyperblob_per_blob.hyperblob_means
    )
    blob_vel_means = genjax.normal(blob_vel_means_, jnp.sqrt(hypers.sigma_V)) @ 'blob_vel_means'
    blob_vel_covs = inverse_wishart(hypers.nu_V, hypers.Psi_V, sample_shape=sample_shape) @ 'blob_vel_covs'
    blob_features = genjax.normal(hypers.mu_F, jnp.sqrt(hypers.sigma_F_prior**2)) @ 'blob_features'
    return HDGMM_Blobs_State_DINO(
        hyperblob_assignments=hyperblob_assignments,
        blob_weights=blob_weights,
        blob_means=blob_means,
        blob_covs=blob_covs,
        blob_vel_means=blob_vel_means,
        blob_vel_covs=blob_vel_covs,
        blob_features=blob_features,
    )

@gen
def HDGMM_datapoints_model_dino(hypers: HDGMM_Hyperparams_DINO, blobs_state: HDGMM_Blobs_State_DINO):
    blob_assignments = genjax.categorical(
        probs=blobs_state.blob_weights,
        sample_shape=Const((hypers.n_datapoints,))
    ) @ 'blob_assignments'
    assigned_blob_per_datapoint = blobs_state[blob_assignments]
    datapoint_positions = genjax.mv_normal(
        assigned_blob_per_datapoint.blob_means,
        assigned_blob_per_datapoint.blob_covs
    ) @ 'datapoint_positions'
    datapoint_vels = genjax.mv_normal(
        assigned_blob_per_datapoint.blob_vel_means,
        assigned_blob_per_datapoint.blob_vel_covs
    ) @ 'datapoint_vels'
    datapoint_features = genjax.normal(
        assigned_blob_per_datapoint.blob_features,
        jnp.sqrt(hypers.sigma_F)
    ) @ 'datapoint_features'
    return HDGMM_Datapoints_State_DINO(
        blob_assignments=blob_assignments,
        datapoint_positions=datapoint_positions,
        datapoint_vels=datapoint_vels,
        datapoint_features=datapoint_features,
    )

@gen
def blob_datapoint_likelihood_model_dino(blob_state: HDGMM_Blobs_State_DINO, sigma_F: jnp.float32):
    datapoint_position = genjax.mv_normal(blob_state.blob_means, blob_state.blob_covs) @ 'datapoint_position'
    datapoint_vel = genjax.mv_normal(blob_state.blob_vel_means, blob_state.blob_vel_covs) @ 'datapoint_vel'
    datapoint_feature = genjax.normal(blob_state.blob_features, jnp.sqrt(sigma_F)) @ 'datapoint_feature'
    return None

# JIT compile
model_jsimulate = jax.jit(HDGMM_model_dino.simulate)
model_jimportance = jax.jit(HDGMM_model_dino.importance)

# ============================================================================
# Helper Functions
# ============================================================================

def extract_dino_features(stimulus, pca_features, img_dims, num_timesteps):
    """Extract DINO features reshaped to match tracking data."""
    T, h, w, num_features = pca_features.shape
    target_h, target_w = img_dims

    if (h, w) != (target_h, target_w):
        tracked_features = np.zeros((num_timesteps, target_h, target_w, num_features), dtype=pca_features.dtype)
        for t in range(num_timesteps):
            for c in range(num_features):
                tracked_features[t, :, :, c] = cv2.resize(
                    pca_features[t, :, :, c],
                    (target_w, target_h),
                    interpolation=cv2.INTER_LINEAR
                )
    else:
        tracked_features = pca_features[:num_timesteps]

    tracked_features = tracked_features.reshape(num_timesteps, -1, num_features)
    return tracked_features

def sample_datapoints_percentage(tracked_points, tracked_motion_vectors, percentage, seed=None, same_indices_all_timesteps=True):
    """
    Randomly sample a percentage of datapoints from tracked_points and tracked_motion_vectors
    at each timestep.

    Args:
        tracked_points: Array of shape (T, num_points, ...)
        tracked_motion_vectors: Array of shape (T, num_points, ...)
        percentage: Percentage of points to keep (0-100)
        seed: Random seed for reproducibility
        same_indices_all_timesteps: If True, sample once and reuse for all timesteps

    Returns:
        tuple: (sampled_tracked_points, sampled_tracked_motion_vectors, sampled_indices_info)
            - If same_indices_all_timesteps: sampled_indices_info is shape (num_points_to_keep,)
            - Else: sampled_indices_info is shape (T, num_points_to_keep)
    """
    if seed is not None:
        np.random.seed(seed)

    T, num_points = tracked_points.shape[:2]
    num_points_to_keep = max(1, int(num_points * percentage / 100.0))

    if same_indices_all_timesteps:
        sampled_indices = np.random.choice(num_points, num_points_to_keep, replace=False)
        sampled_indices = np.sort(sampled_indices)
        sampled_tracked_points = tracked_points[:, sampled_indices, ...]
        sampled_tracked_motion_vectors = tracked_motion_vectors[:, sampled_indices, ...]
        return sampled_tracked_points, sampled_tracked_motion_vectors, sampled_indices
    else:
        sampled_indices_batched = np.zeros((T, num_points_to_keep), dtype=int)
        sampled_tracked_points = np.zeros((T, num_points_to_keep) + tracked_points.shape[2:], dtype=tracked_points.dtype)
        sampled_tracked_motion_vectors = np.zeros((T, num_points_to_keep) + tracked_motion_vectors.shape[2:], dtype=tracked_motion_vectors.dtype)
        for t in range(T):
            idx = np.random.choice(num_points, num_points_to_keep, replace=False)
            idx = np.sort(idx)
            sampled_indices_batched[t] = idx
            sampled_tracked_points[t] = tracked_points[t, idx, ...]
            sampled_tracked_motion_vectors[t] = tracked_motion_vectors[t, idx, ...]
        return sampled_tracked_points, sampled_tracked_motion_vectors, sampled_indices_batched

def initialize_model_with_dino(tracked_points, num_blobs, num_hyperblobs,
                               segmentation_mask, motion_vectors, tracked_features, img_dims, video_name,
                               subsampled_indices=None):
    """Initialize model with DINO features."""
    from genjax import ChoiceMapBuilder as C

    if USE_SAM_FRAME0:
        segmentation_mask = cv2.imread(SAM_FRAME0_PATH_TEMPLATE.format(video_name), cv2.IMREAD_COLOR)
        segmentation_mask = cv2.cvtColor(segmentation_mask, cv2.COLOR_BGR2RGB)
        kmeans_chm, roi_blob_indices, roi_hyperblob_indices, num_hyperblobs_actual = make_hierarchical_kmeans_chm_with_SAM_segmentations(
            tracked_points, num_blobs, segmentation_mask, img_dims,
            motion_vectors=motion_vectors,
            subsampled_indices=subsampled_indices,
        )
    else:
        kmeans_chm, roi_blob_indices, roi_hyperblob_indices = make_hierarchical_kmeans_chm_with_mask_fixed_hyperblob(
            tracked_points, num_blobs, num_hyperblobs,
            segmentation_mask=segmentation_mask,
            motion_vectors=motion_vectors,
            num_roi_blobs = NUM_INIT_PARTICLES_ON_MASK,
            subsampled_indices=subsampled_indices,
        )
        num_hyperblobs_actual = num_hyperblobs

    frame_features = tracked_features[0]
    blob_assignments = np.array(kmeans_chm['datapoints', 'blob_assignments'])
    num_blobs_actual = len(np.unique(blob_assignments))
    num_features = frame_features.shape[1]
    blob_features = np.zeros((num_blobs_actual, num_features), dtype=np.float32)

    for i in range(num_blobs_actual):
        points_in_blob = np.where(blob_assignments == i)[0]
        if len(points_in_blob) > 0:
            blob_features[i] = np.mean(frame_features[points_in_blob], axis=0)

    kmeans_chm = kmeans_chm | C['datapoints', 'datapoint_features'].set(a_(frame_features))
    kmeans_chm = kmeans_chm | C['blobs', 'blob_features'].set(a_(blob_features))

    return kmeans_chm, roi_blob_indices, roi_hyperblob_indices, num_hyperblobs_actual

# ============================================================================
# Inference Functions
# ============================================================================

def gibbs_blob_features_dino(key, hdgmm_state):
    """Gibbs update for blob DINO features."""
    posterior_key, _ = jax.random.split(key)
    datapoint_features = hdgmm_state.datapoints_state.datapoint_features
    blob_assignments = hdgmm_state.datapoints_state.blob_assignments
    mu_F = hdgmm_state.hypers.mu_F
    sigma_F_prior = hdgmm_state.hypers.sigma_F_prior
    sigma_F = hdgmm_state.hypers.sigma_F
    L = hdgmm_state.hypers.n_blobs

    N_l = jax.ops.segment_sum(jnp.ones(datapoint_features.shape[0]), blob_assignments, num_segments=L)
    S_l = stable_segment_sum(datapoint_features, blob_assignments, L)
    prior_precision = 1.0 / (sigma_F_prior ** 2)
    likelihood_precision = 1.0 / sigma_F
    posterior_precision = prior_precision[None, :] + N_l[:, None] * likelihood_precision
    posterior_mean_numerator = (mu_F[None, :] * prior_precision[None, :] + S_l * likelihood_precision)
    posterior_mean = posterior_mean_numerator / posterior_precision
    posterior_std = 1.0 / jnp.sqrt(posterior_precision)
    has_points = N_l > 0
    posterior_mean = jnp.where(has_points[:, None], posterior_mean, mu_F[None, :])
    posterior_std = jnp.where(has_points[:, None], posterior_std, sigma_F_prior[None, :])
    new_blob_features = genjax.normal.sample(posterior_key, posterior_mean, posterior_std)
    return hdgmm_state.replace({'blobs_state': {'blob_features': new_blob_features}})

@jax.jit
def dense_eval_blob_assignments(key, hdgmm_state, dense_positions, dense_vels, dense_features,
                                disable_outlier_prob=False):
    """Dense evaluation of blob assignments."""
    from genjax import ChoiceMapBuilder as C

    posterior_key, _ = jax.random.split(key)
    batch_size = 975

    hypers = hdgmm_state.hypers
    num_blobs = hypers.n_blobs
    num_datapoints = dense_positions.shape[0]
    datapoint_positions = dense_positions
    datapoint_vels = dense_vels
    datapoint_features = dense_features
    blobs_state = hdgmm_state.blobs_state

    blob_weights = blobs_state.blob_weights
    outlier_prob = jnp.where(disable_outlier_prob, 0.0, hypers.outlier_prob)
    extended_weights = jnp.concatenate([blob_weights, jnp.array([outlier_prob])])
    normalized_weights = extended_weights / jnp.sum(extended_weights)
    log_mixture_weights = jnp.log(normalized_weights)
    sigma_F = hypers.sigma_F

    def compute_local_density(point_idx):
        chm = (
            C["datapoint_position"].set(datapoint_positions[point_idx]) |
            C["datapoint_vel"].set(datapoint_vels[point_idx]) |
            C["datapoint_feature"].set(datapoint_features[point_idx])
        )
        log_liks = jax.vmap(
            lambda i: blob_datapoint_likelihood_model_dino.assess(chm, (blobs_state[i], sigma_F))[0]
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
        log_gamma_vel = jnp.where(disable_outlier_prob, 0.0, log_gamma_vel)
        raw_log_liks = jnp.concatenate([log_liks, jnp.array([log_gamma_vel])])
        return raw_log_liks + log_mixture_weights

    def batch_compute_logprobs(carry, batch_idx):
        batch_start = batch_idx * batch_size
        batch_indices = jnp.arange(batch_size) + batch_start
        batch_logprobs = jax.vmap(compute_local_density)(batch_indices)
        return carry, batch_logprobs

    num_full_batches = num_datapoints // batch_size
    _, batched_logprobs = jax.lax.scan(
        batch_compute_logprobs,
        None,
        jnp.arange(num_full_batches)
    )

    all_logprobs = batched_logprobs.reshape(num_full_batches * batch_size, -1)
    dense_eval_assignments = genjax.categorical.sample(posterior_key, logits=all_logprobs)

    return dense_eval_assignments

@jax.jit
def dense_eval_blob_weights(key, hdgmm_state : HDGMM_State, dense_assignments : jnp.ndarray): 
    posterior_key, _ = jax.random.split(key)

    # Get the data and parameters from the trace
    num_blobs = hdgmm_state.hypers.n_blobs
    prior_beta = hdgmm_state.hypers.beta
    blob_idxs = dense_assignments  # [N]

    # Compute counts via segment_sum
    blob_counts = jax.ops.segment_sum(
        jnp.ones_like(blob_idxs),  # [N]
        blob_idxs,                 # [N] (each datapoint's assigned blob)
        num_segments=num_blobs     # total number of blobs
    )  # [L]

    # Posterior Dirichlet parameters
    new_betas = prior_beta + blob_counts  # [L]

    # Sample new blob weights
    new_blob_weights = genjax.dirichlet.sample(posterior_key, new_betas)

    return new_blob_weights

def gibbs_blob_assignments_dino(key, hdgmm_state, position_only=False,
                                velocity_only=False, disable_outlier_prob=False, feature_only=False):
    """Gibbs update for blob assignments with DINO feature likelihood."""
    from genjax import ChoiceMapBuilder as C

    posterior_key, _ = jax.random.split(key)
    batch_size = 975

    hypers = hdgmm_state.hypers
    num_blobs = hypers.n_blobs
    num_datapoints = hypers.n_datapoints
    datapoint_positions = hdgmm_state.datapoints_state.datapoint_positions
    datapoint_vels = hdgmm_state.datapoints_state.datapoint_vels
    datapoint_features = hdgmm_state.datapoints_state.datapoint_features
    blobs_state = hdgmm_state.blobs_state

    gibbs_blob_vel_covs = jnp.where(
        jnp.logical_or(position_only, feature_only),
        1e14 * blobs_state.blob_vel_covs,
        blobs_state.blob_vel_covs
    )
    gibbs_blob_covs = jnp.where(
        jnp.logical_or(velocity_only, feature_only),
        1e14 * blobs_state.blob_covs,
        blobs_state.blob_covs
    )
    blobs_state = blobs_state.replace({
        'blob_vel_covs': gibbs_blob_vel_covs,
        'blob_covs': gibbs_blob_covs
    })

    blob_weights = blobs_state.blob_weights
    outlier_prob = jnp.where(disable_outlier_prob, 0.0, hypers.outlier_prob)
    extended_weights = jnp.concatenate([blob_weights, jnp.array([outlier_prob])])
    normalized_weights = extended_weights / jnp.sum(extended_weights)
    log_mixture_weights = jnp.log(normalized_weights)
    sigma_F = hypers.sigma_F

    def compute_local_density(point_idx):
        chm = (
            C["datapoint_position"].set(datapoint_positions[point_idx]) |
            C["datapoint_vel"].set(datapoint_vels[point_idx]) |
            C["datapoint_feature"].set(datapoint_features[point_idx])
        )
        log_liks = jax.vmap(
            lambda i: blob_datapoint_likelihood_model_dino.assess(chm, (blobs_state[i], sigma_F))[0]
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
        log_gamma_vel = jnp.where(disable_outlier_prob, 0.0, log_gamma_vel)
        raw_log_liks = jnp.concatenate([log_liks, jnp.array([log_gamma_vel])])
        return raw_log_liks + log_mixture_weights

    def batch_compute_logprobs(carry, batch_idx):
        batch_start = batch_idx * batch_size
        batch_indices = jnp.arange(batch_size) + batch_start
        batch_logprobs = jax.vmap(compute_local_density)(batch_indices)
        return carry, batch_logprobs

    num_full_batches = num_datapoints // batch_size
    _, batched_logprobs = jax.lax.scan(
        batch_compute_logprobs,
        None,
        jnp.arange(num_full_batches)
    )

    all_logprobs = batched_logprobs.reshape(num_full_batches * batch_size, -1)
    updated_assignments = genjax.categorical.sample(posterior_key, logits=all_logprobs)

    return hdgmm_state.replace({
        'datapoints_state': {'blob_assignments': updated_assignments}
    })

def blob_tracking_gibbs_dino(key, hdgmm_state):
    """Single-frame Gibbs updates."""
    def update_blob_assignments_position_only(i, carry):
        key, hdgmm_state = carry
        key, gibbs_key = jax.random.split(key)
        hdgmm_state = gibbs_blob_assignments_dino(
            gibbs_key, hdgmm_state, position_only=True, disable_outlier_prob=True
        )
        key, gibbs_key = jax.random.split(key)
        hdgmm_state = gibbs_blob_weights(gibbs_key, hdgmm_state)
        return key, hdgmm_state

    def update_blob_assignments_with_outlier(i, carry):
        key, hdgmm_state = carry
        key, gibbs_key = jax.random.split(key)
        hdgmm_state = gibbs_blob_assignments_dino(
            gibbs_key, hdgmm_state, position_only=True, disable_outlier_prob=False
        )
        key, gibbs_key = jax.random.split(key)
        hdgmm_state = gibbs_blob_weights(gibbs_key, hdgmm_state)
        return key, hdgmm_state

    def update_blob_velocities(i, carry):
        key, hdgmm_state = carry
        key, gibbs_key = jax.random.split(key)
        hdgmm_state = gibbs_blob_vel_means(gibbs_key, hdgmm_state)
        return key, hdgmm_state

    def update_blob_velocity_covariances(i, carry):
        key, hdgmm_state = carry
        key, gibbs_key = jax.random.split(key)
        hdgmm_state = gibbs_blob_vel_covs(gibbs_key, hdgmm_state)
        return key, hdgmm_state

    def update_blob_means(i, carry):
        key, hdgmm_state = carry
        key, gibbs_key = jax.random.split(key)
        hdgmm_state = gibbs_blob_means(gibbs_key, hdgmm_state)
        return key, hdgmm_state

    def update_blob_features_dino(i, carry):
        key, hdgmm_state = carry
        key, gibbs_key = jax.random.split(key)
        hdgmm_state = gibbs_blob_features_dino(gibbs_key, hdgmm_state)
        return key, hdgmm_state

    def hyperblob_update_loop(i, carry):
        key, hdgmm_state = carry
        key, gibbs_key = jax.random.split(key)
        hdgmm_state = gibbs_hyperblob_means(gibbs_key, hdgmm_state)
        key, gibbs_key = jax.random.split(key)
        hdgmm_state = gibbs_hyperblob_covs(gibbs_key, hdgmm_state)
        key, gibbs_key = jax.random.split(key)
        hdgmm_state = gibbs_hyperblob_rot(gibbs_key, hdgmm_state)
        key, gibbs_key = jax.random.split(key)
        hdgmm_state = gibbs_hyperblob_trans(gibbs_key, hdgmm_state)
        return key, hdgmm_state

    key, hdgmm_state = jax.lax.fori_loop(0, 3, hyperblob_update_loop, (key, hdgmm_state))
    # # added this to see if it helps below
    # key, hdgmm_state = jax.lax.fori_loop(0, 3, update_blob_assignments_feature_only, (key, hdgmm_state))
    # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    key, hdgmm_state = jax.lax.fori_loop(0, 1, update_blob_assignments_position_only, (key, hdgmm_state))
    key, hdgmm_state = jax.lax.fori_loop(0, 15, update_blob_means, (key, hdgmm_state))
    key, hdgmm_state = jax.lax.fori_loop(0, 3, update_blob_features_dino, (key, hdgmm_state))
    key, hdgmm_state = jax.lax.fori_loop(0, 1, update_blob_assignments_with_outlier, (key, hdgmm_state))
    key, hdgmm_state = jax.lax.fori_loop(0, 15, update_blob_velocities, (key, hdgmm_state))
    key, hdgmm_state = jax.lax.fori_loop(0, 15, update_blob_velocity_covariances, (key, hdgmm_state))
    key, hdgmm_state = jax.lax.fori_loop(0, 3, update_blob_features_dino, (key, hdgmm_state))
    key, hdgmm_state = jax.lax.fori_loop(0, 3, hyperblob_update_loop, (key, hdgmm_state))

    
    return hdgmm_state

def hdgmm_tracking_gibbs_dino(key, init_hdgmm_state, tracked_points,
                              tracked_motion_vectors, tracked_features, outlier_prob):
    """Track over time with DINO features."""
    init_hdgmm_state = init_hdgmm_state.replace({'hypers': {'outlier_prob': f_(outlier_prob)}})

    @jax.jit
    def f_tracking_sweep(carry, timestep_idx):
        key, hdgmm_state, tracked_points, tracked_motion_vectors, tracked_features = carry
        next_blob_means = hdgmm_state.blobs_state.blob_vel_means + hdgmm_state.blobs_state.blob_means
        hdgmm_state = hdgmm_state.replace({'blobs_state': {'blob_means': next_blob_means}})
        hdgmm_state = hdgmm_state.replace({
            'datapoints_state': {
                'datapoint_positions': tracked_points[timestep_idx],
                'datapoint_vels': tracked_motion_vectors[timestep_idx],
                'datapoint_features': tracked_features[timestep_idx]
            }
        })
        key, gibbs_key = jax.random.split(key)
        hdgmm_state = blob_tracking_gibbs_dino(gibbs_key, hdgmm_state)
        return (key, hdgmm_state, tracked_points, tracked_motion_vectors, tracked_features), \
               hdgmm_TraceWrapper(force_retval=hdgmm_state)

    timestep_indices = jnp.arange(1, len(tracked_points))
    _, stacked_hdgmm_wtrs = jax.lax.scan(
        f_tracking_sweep,
        (key, init_hdgmm_state, tracked_points, tracked_motion_vectors, tracked_features),
        timestep_indices,
        unroll=1
    )

    return HDGMM_Gibbs_TraceWrapper(
        hdgmm_TraceWrapper(force_retval=init_hdgmm_state),
        stacked_hdgmm_wtrs
    )

def init_gibbs_sweep_dino(key, hdgmm_state, num_sweeps=30):
    """Custom Gibbs sweeps for DINO initialization."""
    def gibbs_iteration(carry, i):
        key, hdgmm_state = carry
        key, gibbs_key = jax.random.split(key)
        hdgmm_state = gibbs_blob_assignments_dino(gibbs_key, hdgmm_state, position_only=True, disable_outlier_prob=True)
        key, gibbs_key = jax.random.split(key)
        hdgmm_state = gibbs_blob_weights(gibbs_key, hdgmm_state)
        key, gibbs_key = jax.random.split(key)
        hdgmm_state = gibbs_blob_means(gibbs_key, hdgmm_state)
        key, gibbs_key = jax.random.split(key)
        hdgmm_state = gibbs_blob_covs(gibbs_key, hdgmm_state)
        key, gibbs_key = jax.random.split(key)
        hdgmm_state = gibbs_blob_vel_means(gibbs_key, hdgmm_state)
        key, gibbs_key = jax.random.split(key)
        hdgmm_state = gibbs_blob_vel_covs(gibbs_key, hdgmm_state)
        key, gibbs_key = jax.random.split(key)
        hdgmm_state = gibbs_blob_features_dino(gibbs_key, hdgmm_state)
        return (key, hdgmm_state), hdgmm_TraceWrapper(force_retval=hdgmm_state)

    (key, final_state), traces = jax.lax.scan(gibbs_iteration, (key, hdgmm_state), jnp.arange(num_sweeps))
    return HDGMM_Gibbs_TraceWrapper(hdgmm_TraceWrapper(force_retval=hdgmm_state), traces)

# ============================================================================
# Evaluation Functions
# ============================================================================
#
# IMPORTANT: Four different metrics are computed:
#
# 1. PARTICLE-BASED METRICS (compute_error_rates) - PRIMARY METRIC
#    - Tracks discrete particle/blob location changes over time
#    - Projects particle means to 2D and checks if they fall within GT mask
#    - Identifies object vs background particles in frame 0
#    - FN Rate: % of object particles that left the GT mask
#    - FP Rate: % of background particles that entered the GT mask
#    - Jaccard: Intersection over Union of particle sets
#    - Accuracy: (TN + TP) / (TP + TN + FP + FN)
#    - These are the metrics saved to JSON and reported in summaries
#
# 2. PARTICLE-COUNT UNWEIGHTED METRICS (evaluate_single_davis_video)
#    - Counts particles with equal weight (1.0 per particle)
#    - Particle membership determined by FRACTIONAL proportion of pixels in GT mask
#    - Each particle contributes fractionally: if 70% of pixels in GT, contributes 0.7 to object count
#    - Standard recall/precision/F1/Jaccard computed on fractional particle counts
#    - Used for comparison with traditional segmentation methods
#
# 3. PARTICLE-COUNT MATTER-WEIGHTED ADAPTIVE METRICS (evaluate_single_davis_video)
#    - Counts particles weighted by (pixel_count × blob_weight) per frame
#    - Particle membership determined by FRACTIONAL proportion of pixels in GT mask
#    - Each particle contributes: (proportion_in_GT × pixel_count × blob_weight) to object count
#    - Recall/Precision/F1/Jaccard calculated on weighted fractional particle counts with adaptive weights
#    - Shows benefit of probabilistic representation
#
# 4. PARTICLE-COUNT MATTER-WEIGHTED FIXED METRICS (evaluate_single_davis_video)
#    - Counts particles weighted by (pixel_count × blob_weight) from frame 0
#    - Particle membership determined by FRACTIONAL proportion of pixels in GT mask
#    - Each particle contributes: (proportion_in_GT × pixel_count × blob_weight) to object count
#    - Recall/Precision/F1/Jaccard calculated on weighted fractional particle counts with fixed weights
#    - Fair comparison with point trackers that don't adapt weights
#
# The particle-based metrics are what we care about for tracking evaluation.
# ============================================================================

def compute_error_rates(tracking_data, segmentation_masks, img_dims,
                       blob_counting_threshold=0, focal_length=520.0, force_below_count_thresh_as_outlier=False,
                       subsampled_indices=None, is_mask_subsampled=False):
    """
    Compute PARTICLE-BASED false positive, false negative rates, Jaccard score, and accuracy over time.

    This tracks discrete particle location changes by projecting particle means to 2D,
    NOT fractional particle contributions. See section comment above for the difference between 
    particle-based and particle-count metrics.
    
    Object particles are determined SOLELY by whether they project onto mask pixels in frame 0,
    NOT by hyperblob assignment.
    
    Args:
        force_below_count_thresh_as_outlier: If True, particles below counting threshold are 
                                           treated as outliers and excluded from all metrics.
                                           If False (default), they are treated as background particles.
    """
    fx = fy = focal_length
    cx = img_dims[1] / 2.0
    cy = img_dims[0] / 2.0

    # Frame 0 setup
    frame0 = tracking_data[0]
    blob_assignments_frame0 = frame0['blob_assignments']
    blob_means_frame0 = frame0['blob_means']
    n_blobs_frame0 = frame0['n_blobs']
    gt_mask_frame0 = segmentation_masks[0]

    blob_pixel_counts_frame0 = np.bincount(
        blob_assignments_frame0[blob_assignments_frame0 < n_blobs_frame0],
        minlength=n_blobs_frame0
    )
    significant_blobs = np.where(blob_pixel_counts_frame0 >= blob_counting_threshold)[0]

    if force_below_count_thresh_as_outlier:
        # Only consider significant blobs, ignore below-threshold particles entirely
        blobs_to_consider = significant_blobs
    else:
        # Original behavior: consider all blobs
        blobs_to_consider = np.arange(n_blobs_frame0)

    print(f"is_mask_subsampled? ~~~~~~~~~~~> : {is_mask_subsampled}")

    if is_mask_subsampled:
        H, W = img_dims
        xs_sub = np.array(subsampled_indices) % W
        ys_sub = np.array(subsampled_indices) // W
        # helper to get nearest sampled mask value from a projected (x, y)
        def nearest_sampled_mask_value(mask_vec, x, y):
            # compute index in the sliced mask (aligned with subsampled_indices order)
            dx = xs_sub - x
            dy = ys_sub - y
            idx = int(np.argmin(dx * dx + dy * dy))
            return bool(mask_vec[idx])

    # Project blob means to 2D and determine object vs background particles based on mask
    x_2d = (blob_means_frame0[:, 0] / (blob_means_frame0[:, 2] + 1e-8)) * fx + cx
    y_2d = (blob_means_frame0[:, 1] / (blob_means_frame0[:, 2] + 1e-8)) * fy + cy
    x_2d = np.clip(x_2d.astype(int), 0, img_dims[1] - 1)
    y_2d = np.clip(y_2d.astype(int), 0, img_dims[0] - 1)

    object_blobs_frame0 = []
    background_blobs_frame0 = []

    for blob_idx in blobs_to_consider:
        if not is_mask_subsampled:
            pixel_idx = y_2d[blob_idx] * img_dims[1] + x_2d[blob_idx]
            is_on_mask = pixel_idx < len(gt_mask_frame0) and gt_mask_frame0[pixel_idx]
        else:
            is_on_mask = nearest_sampled_mask_value(gt_mask_frame0, x_2d[blob_idx], y_2d[blob_idx])
        
        if is_on_mask:
            object_blobs_frame0.append(blob_idx)
        else:
            background_blobs_frame0.append(blob_idx)

    object_blobs_frame0 = np.array(object_blobs_frame0)
    background_blobs_frame0 = np.array(background_blobs_frame0)
    n_object_blobs = len(object_blobs_frame0)
    n_background_blobs = len(background_blobs_frame0)

    # Track over time
    false_negative_rates = []
    false_positive_rates = []
    jaccard_scores = []
    accuracies = []

    for frame_idx in range(len(tracking_data)):
        frame = tracking_data[frame_idx]
        blob_means = frame['blob_means']
        n_blobs = frame['n_blobs']
        gt_mask = segmentation_masks[frame_idx]

        if not is_mask_subsampled:
            x_2d = (blob_means[:, 0] / (blob_means[:, 2] + 1e-8)) * fx + cx
            y_2d = (blob_means[:, 1] / (blob_means[:, 2] + 1e-8)) * fy + cy
            x_2d = np.clip(x_2d.astype(int), 0, img_dims[1] - 1)
            y_2d = np.clip(y_2d.astype(int), 0, img_dims[0] - 1)
            pixel_indices = y_2d * img_dims[1] + x_2d

        tp_count = 0
        if not is_mask_subsampled:
            for blob_idx in object_blobs_frame0:
                if blob_idx < n_blobs:
                    pixel_idx = pixel_indices[blob_idx]
                    if pixel_idx < len(gt_mask) and gt_mask[pixel_idx]:
                        tp_count += 1
        else:
            for blob_idx in object_blobs_frame0:
                if blob_idx < n_blobs:
                    if nearest_sampled_mask_value(gt_mask, x_2d[blob_idx], y_2d[blob_idx]):
                        tp_count += 1

        fn_count = n_object_blobs - tp_count
        fn_rate = (fn_count / n_object_blobs) * 100 if n_object_blobs > 0 else 0.0

        fp_count = 0
        if not is_mask_subsampled:
            for blob_idx in background_blobs_frame0:
                if blob_idx < n_blobs:
                    pixel_idx = pixel_indices[blob_idx]
                    if pixel_idx < len(gt_mask) and gt_mask[pixel_idx]:
                        fp_count += 1
        else:
            for blob_idx in background_blobs_frame0:
                if blob_idx < n_blobs:
                    if nearest_sampled_mask_value(gt_mask, x_2d[blob_idx], y_2d[blob_idx]):
                        fp_count += 1

        fp_rate = (fp_count / n_background_blobs) * 100 if n_background_blobs > 0 else 0.0

        # True negatives: background particles that stayed off mask
        tn_count = n_background_blobs - fp_count

        # Compute accuracy: (TN + TP) / (TP + TN + FP + FN)
        total_particles = n_object_blobs + n_background_blobs
        accuracy = ((tn_count + tp_count) / total_particles) * 100 if total_particles > 0 else 100.0

        # Compute Jaccard score (IoU for particles)
        # True positives: object particles still on mask
        # False positives: background particles that entered mask
        # False negatives: object particles that left mask
        intersection = tp_count
        union = tp_count + fp_count + fn_count
        jaccard = (intersection / union) if union > 0 else 1.0  # Perfect score if no particles

        false_negative_rates.append(fn_rate)
        false_positive_rates.append(fp_rate)
        jaccard_scores.append(jaccard)
        accuracies.append(accuracy)

    return {
        'false_negative_rates': false_negative_rates,
        'false_positive_rates': false_positive_rates,
        'jaccard_scores': jaccard_scores,
        'accuracies': accuracies,
        'mean_fn_rate': np.mean(false_negative_rates),
        'mean_fp_rate': np.mean(false_positive_rates),
        'mean_jaccard': np.mean(jaccard_scores),
        'mean_accuracy': np.mean(accuracies),
        'n_object_blobs': n_object_blobs,
        'n_background_blobs': n_background_blobs
    }

def plot_error_rates(error_results, video_name, save_path):
    """Plot and save error rate visualization."""
    fn_rates = error_results['false_negative_rates']
    fp_rates = error_results['false_positive_rates']
    jaccard_scores = error_results['jaccard_scores']
    accuracies = error_results['accuracies']
    mean_fn = error_results['mean_fn_rate']
    mean_fp = error_results['mean_fp_rate']
    mean_jaccard = error_results['mean_jaccard']
    mean_accuracy = error_results['mean_accuracy']

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
    
    # Top plot: Error rates
    ax1.plot(range(len(fn_rates)), fn_rates, linewidth=2, color='red',
            label=f'False Negative Rate (mean: {mean_fn:.2f}%)', alpha=0.8)
    ax1.plot(range(len(fp_rates)), fp_rates, linewidth=2, color='blue',
            label=f'False Positive Rate (mean: {mean_fp:.2f}%)', alpha=0.8)
    ax1.set_xlabel('Frame', fontsize=12)
    ax1.set_ylabel('Rate (%)', fontsize=12)
    ax1.set_title(f'{video_name} - Blob Tracking Error Rates', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    ax1.set_ylim([0, 105])
    
    # Middle plot: Jaccard score
    ax2.plot(range(len(jaccard_scores)), jaccard_scores, linewidth=2, color='green',
            label=f'Jaccard Score (mean: {mean_jaccard:.3f})', alpha=0.8)
    ax2.set_xlabel('Frame', fontsize=12)
    ax2.set_ylabel('Jaccard Score', fontsize=12)
    ax2.set_title(f'{video_name} - Particle Jaccard Score (IoU)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    ax2.set_ylim([0, 1.05])
    
    # Bottom plot: Accuracy
    ax3.plot(range(len(accuracies)), accuracies, linewidth=2, color='purple',
            label=f'Accuracy (mean: {mean_accuracy:.2f}%)', alpha=0.8)
    ax3.set_xlabel('Frame', fontsize=12)
    ax3.set_ylabel('Accuracy (%)', fontsize=12)
    ax3.set_title(f'{video_name} - Particle Tracking Accuracy', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=11)
    ax3.set_ylim([0, 105])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def create_3wide_video(tracking_data, video_name, rgb_path, img_dims, segmentation_masks,
                       num_blobs, save_path, subsampled_indices=None, is_subsampled=False,
                       ):
    """Create 3-wide synchronized video."""
    from glob import glob

    rgb_dir = os.path.join(rgb_path, video_name)
    rgb_files = sorted(glob(os.path.join(rgb_dir, "*.jpg")))

    if len(rgb_files) == 0:
        print(f"Warning: No RGB frames found for {video_name}")
        return

    img_height, img_width = img_dims
    # Seed numpy RNG for consistent visualization colors
    np.random.seed(RANDOM_SEED)
    blob_colors = np.random.randint(0, 255, size=(num_blobs, 3), dtype=np.uint8)
    combined_frames = []

    num_frames = min(len(tracking_data), len(rgb_files))

    # Determine object particles from frame 0 using segmentation mask (like compute_error_rates)
    frame0 = tracking_data[0]
    blob_assignments_frame0 = frame0['blob_assignments']
    n_blobs_frame0 = NUM_BLOBS
    gt_mask_frame0 = segmentation_masks[0]

    print(f"number of blobs in frame 0: {n_blobs_frame0}")
    
    # Count pixels per blob in frame 0
    blob_pixel_counts_frame0 = np.bincount(
        blob_assignments_frame0[blob_assignments_frame0 < n_blobs_frame0],
        minlength=n_blobs_frame0
    )
    
    # Determine which blobs are object vs background based on mask overlap
    # Use the same logic as compute_error_rates: check which pixels assigned to each blob overlap with mask
    object_blobs_frame0 = set()
    background_blobs_frame0 = set()
    
    for blob_idx in range(n_blobs_frame0):
        # Find pixels assigned to this blob
        blob_pixel_mask = (blob_assignments_frame0 == blob_idx)
        
        # Check if any of these pixels are on the segmentation mask
        pixels_on_mask = np.sum(gt_mask_frame0[blob_pixel_mask])
        
        if pixels_on_mask > 0:
            object_blobs_frame0.add(blob_idx)
        else:
            background_blobs_frame0.add(blob_idx)

    for frame_idx in tqdm(range(num_frames), desc="Rendering 3-wide video frames"):
        rgb_frame = cv2.imread(rgb_files[frame_idx])
        rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)

        frame = tracking_data[frame_idx]
        blob_assignments = frame['blob_assignments']
        n_blobs = NUM_BLOBS

        valid_mask = blob_assignments < n_blobs
        
        # Create binary mask based on object particles identified from frame 0
        # Convert object_blobs_frame0 set to array for vectorized operations
        object_blobs_array = np.array(list(object_blobs_frame0))
        
        # Vectorized check: for each blob_assignment, check if it's in object_blobs_frame0
        # Only consider valid blob assignments (< n_blobs) to avoid outlier blobs
        object_blob_mask = valid_mask & np.isin(blob_assignments, object_blobs_array)
        binary_mask = np.where(object_blob_mask, 255, 0).astype(np.uint8)
        
        background_color = np.array([128, 128, 128], dtype=np.uint8)

        blob_assignment_frame = np.zeros((len(blob_assignments), 3), dtype=np.uint8)
        blob_assignment_frame[~object_blob_mask] = background_color
        
        # Only assign colors to valid object blobs (ignore outlier blobs >= n_blobs)
        valid_object_mask = object_blob_mask & valid_mask
        if np.any(valid_object_mask):
            valid_object_indices = blob_assignments[valid_object_mask]
            blob_assignment_frame[valid_object_mask] = blob_colors[valid_object_indices]
        
        # Get RGB dimensions first
        rgb_h, rgb_w = rgb_frame.shape[:2]
        
        # Handle reshaping based on subsampling
        if is_subsampled:
            # For subsampled data, reconstruct full image with correct spatial positions
            # Create full-size arrays with background values
            full_binary = np.zeros(img_height * img_width, dtype=np.uint8)
            full_blob_frame = np.full((img_height * img_width, 3), background_color, dtype=np.uint8)
            
            # Place subsampled data at correct positions
            full_binary[subsampled_indices] = binary_mask
            full_blob_frame[subsampled_indices] = blob_assignment_frame
            
            # Reshape to image dimensions
            binary_image = full_binary.reshape(img_height, img_width)
            blob_assignment_frame = full_blob_frame.reshape(img_height, img_width, 3)
            
            # Resize to match RGB if dimensions differ
            if (rgb_h, rgb_w) != (img_height, img_width):
                binary_image = cv2.resize(binary_image, (rgb_w, rgb_h), interpolation=cv2.INTER_NEAREST)
                blob_assignment_frame = cv2.resize(blob_assignment_frame, (rgb_w, rgb_h), interpolation=cv2.INTER_NEAREST)
        else:
            # No subsampling: reshape to image dimensions
            binary_image = binary_mask.reshape(img_height, img_width)
            blob_assignment_frame = blob_assignment_frame.reshape(img_height, img_width, 3)
            
            # Resize to match RGB if dimensions differ
            if (rgb_h, rgb_w) != (img_height, img_width):
                binary_image = cv2.resize(binary_image, (rgb_w, rgb_h), interpolation=cv2.INTER_NEAREST)
                blob_assignment_frame = cv2.resize(blob_assignment_frame, (rgb_w, rgb_h), interpolation=cv2.INTER_NEAREST)
        
        # Create segmentation frame from binary image
        segmentation_frame = np.stack([binary_image, binary_image, binary_image], axis=-1)

        SCALE_FACTOR = 0.5
        target_w = int(rgb_w * SCALE_FACTOR)
        target_h = int(rgb_h * SCALE_FACTOR)
        rgb_frame = cv2.resize(rgb_frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        segmentation_frame = cv2.resize(segmentation_frame, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        blob_assignment_frame = cv2.resize(blob_assignment_frame, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        combined_frame = np.concatenate([rgb_frame, segmentation_frame, blob_assignment_frame], axis=1)
        combined_frames.append(combined_frame)

    # Write frames to temporary location, then use ffmpeg subprocess to encode with H.264
    # This avoids JAX fork issues since we call ffmpeg AFTER all JAX computation is done
    if len(combined_frames) == 0:
        print(f"Warning: No frames to write for {save_path}")
        return
    else:
        print(f"Writing {len(combined_frames)} frames to temporary location")

    import tempfile
    import subprocess
    import shutil

    # Create temporary directory for frames
    temp_dir = tempfile.mkdtemp()
    try:
        # Write frames as PNGs
        for i, frame in tqdm(enumerate(combined_frames), desc="Writing frames to temporary location"):
            frame_path = os.path.join(temp_dir, f"frame_{i:05d}.png")
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(frame_path, frame_bgr)

        # Use ffmpeg to create H.264 video
        # This is safe to call after JAX initialization since all compute is done
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-framerate', '30',
            '-i', os.path.join(temp_dir, 'frame_%05d.png'),
            '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',  # Force even dimensions
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '23',
            save_path
        ]
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
        print(f"Saved 3-wide video: {save_path}")
    finally:
        # Clean up temp directory
        shutil.rmtree(temp_dir)

# ============================================================================
# Cleanup Function
# ============================================================================

def cleanup_between_runs():
    """Clear JAX state, environment variables, and memory between experimental runs."""
    # This function helps prevent memory leaks that can accumulate across runs
    
    # Clear matplotlib figure cache to prevent memory leaks
    plt.close('all')
    
    # Clear OpenCV cache if any
    try:
        cv2.destroyAllWindows()
    except:
        pass
    
    # Clear JAX device memory explicitly
    # Note: PJRT backend doesn't support explicit memory release operations
    # (device.clear_cache() and defragment() both cause fatal errors).
    # JAX's automatic memory management + Python GC will handle memory cleanup.
    # Explicit deletion of variables + multiple GC cycles is sufficient.
    
    # Clear JAX compilation cache if it exists
    try:
        # Clear in-memory compilation cache
        if hasattr(jax, 'clear_caches'):
            jax.clear_caches()
    except:
        pass
    
    # Try to clear JAX backend cache if available
    try:
        import jax._src.lib.xla_bridge as xla_bridge
        if hasattr(xla_bridge, 'get_backend'):
            backend = xla_bridge.get_backend()
            if hasattr(backend, 'defragment'):
                # This might fail on PJRT, but try it
                try:
                    backend.defragment()
                except:
                    pass
    except:
        pass
    
    # Force garbage collection multiple times to ensure cleanup
    # Run multiple times because some objects may have circular references
    for _ in range(10):  # Increased to 10 for more aggressive cleanup
        gc.collect()

# ============================================================================
# Main Processing Function
# ============================================================================

def process_video(video_name, subsampling_percentage=100.0, subsampled_indices=None, run_save_dir=None):
    """Process a single video and return results."""
    print(f"\n{'='*80}")
    print(f"Processing: {video_name} | Subsampling: {subsampling_percentage}%")
    print(f"{'='*80}")

    try:
        # Load data
        dino_path = DINO_PATH_TEMPLATE.format(video_name)
        if not os.path.exists(dino_path):
            print(f"DINO features not found: {dino_path}")
            return None

        pca_data = np.load(dino_path)
        pca_features_unnormalized = pca_data['pca_features_unnormalized']
        gaussian_means = pca_data['gaussian_means']
        gaussian_stds = pca_data['gaussian_stds']
        # Close the npz file to free memory
        pca_data.close()

        tracked_points_full, tracked_motion_vectors_full, num_data_tsteps, img_dims = \
            extract_3d_points_and_motion_vectors_data(DAVIS_3D_MOTION_PATH, video_name)

        num_datapoints_full = tracked_points_full.shape[1]

        # Limit to first 90 frames for jello-trim
        if video_name == "jello_trim":
            num_data_tsteps = min(num_data_tsteps, 90)
            tracked_points = tracked_points_full[:num_data_tsteps]
            tracked_motion_vectors = tracked_motion_vectors_full[:num_data_tsteps]
            print(f"Limited to first {num_data_tsteps} frames for jello_trim")

        tracked_features_full = extract_dino_features(video_name, pca_features_unnormalized, img_dims, num_data_tsteps)

        first_frame_seg = get_segmentation_mask(
            video_name, 0, DAVIS_SEGMASKS_PATH, img_dims=img_dims, flatten=True
        )

        # Subsample datapoints consistently across all timesteps
        if subsampled_indices is None and subsampling_percentage < 100.0:
            tracked_points, tracked_motion_vectors, subsampled_indices = sample_datapoints_percentage(
                tracked_points_full, tracked_motion_vectors_full, subsampling_percentage, seed=RANDOM_SEED, same_indices_all_timesteps=True
            )
            # Subsample DINO features with same indices
            tracked_features = tracked_features_full[:, subsampled_indices, :]
            # Subsample the GT mask for frame 0 to align to datapoints where needed
            first_frame_seg_sub = first_frame_seg[subsampled_indices]
        else:
            first_frame_seg_sub = first_frame_seg
            tracked_features = tracked_features_full
            tracked_points = tracked_points_full
            tracked_motion_vectors = tracked_motion_vectors_full

        # Initialize
        kmeans_chm, roi_blob_indices, roi_hyperblob_indices, num_hyperblobs = initialize_model_with_dino(
            tracked_points, NUM_BLOBS, NUM_HYPERBLOBS_ORIGINAL,
            first_frame_seg, tracked_motion_vectors, tracked_features, img_dims, video_name,
            subsampled_indices=subsampled_indices
        )

        # Ablation: Set all hyperblob covariances to huge values to disable cluster-level effects
        from genjax import ChoiceMapBuilder as C
        num_hyperblobs_actual = kmeans_chm['hyperblobs', 'hyperblob_covs'].shape[0]
        huge_hyperblob_covs = jnp.tile(1e6 * jnp.eye(3)[None, :, :], (num_hyperblobs_actual, 1, 1))
        kmeans_chm = kmeans_chm | C['hyperblobs', 'hyperblob_covs'].set(huge_hyperblob_covs)

        # Create hyperparameters
        num_datapoints = kmeans_chm['datapoints', 'datapoint_positions'].shape[0]
        num_blobs = kmeans_chm['blobs', 'hyperblob_assignments'].shape[0]
        num_hyperblobs = kmeans_chm['hyperblobs', 'hyperblob_means'].shape[0]

        empirical_mu_H = jnp.median(kmeans_chm['datapoints', 'datapoint_positions'], axis=0)
        empirical_sigma_H = (10 * 0.5) ** 2
        empirical_Psi_B = jnp.median(kmeans_chm['blobs', 'blob_covs'][roi_blob_indices], axis=0)
        empirical_Psi_H_original = jnp.median(kmeans_chm['hyperblobs', 'hyperblob_covs'][roi_hyperblob_indices], axis=0)
        # Ablation: Set hyperblob covariance to huge value to disable cluster-level effects
        # This makes the hyperblob prior essentially uninformative
        empirical_Psi_H = 1e6 * jnp.eye(3)  # Huge covariance = no cluster-level constraint
        empirical_Psi_V = jnp.median(kmeans_chm['blobs', 'blob_vel_covs'][roi_blob_indices], axis=0)
        mean_blobs_per_roi_hyperblob = jnp.sum(
            jnp.isin(kmeans_chm['blobs', 'hyperblob_assignments'], roi_hyperblob_indices)
        ) / len(roi_hyperblob_indices)
        empirical_nu_H = f_(int(mean_blobs_per_roi_hyperblob))
        mean_points_per_roi_blob = jnp.sum(
            jnp.isin(kmeans_chm['datapoints', 'blob_assignments'], roi_blob_indices)
        ) / len(roi_blob_indices)
        empirical_nu_B = empirical_nu_V = f_(int(mean_points_per_roi_blob))

        hypers = HDGMM_Hyperparams_DINO.create(
            mu_F=jnp.array(gaussian_means),
            sigma_F_prior=jnp.array(gaussian_stds),
            # sigma_F=f_(20.0),
            sigma_F=f_(0.2),
            outlier_prob=f_(5e-0), ## this triggers the outlier for the first frame 0
            # outlier_prob=f_(5e-3), ## this triggers the outlier for the first frame 0
            outlier_velocity_gamma_shape=f_(5.0),
            outlier_velocity_gamma_rate=f_(1.0),
            alpha=f_(1.0),
            beta=f_(1.0),
            mu_H=empirical_mu_H,
            sigma_H=empirical_sigma_H,
            nu_H=empirical_nu_H,
            Psi_H=empirical_Psi_H,
            nu_B=empirical_nu_B,
            Psi_B=empirical_Psi_B,
            sigma_V=f_(10e14),
            nu_V=empirical_nu_V,
            Psi_V=empirical_Psi_V,
            translation_gaussian_scale=snp(f_(0.2)),
            translation_max_radius=snp(0.35),
            translation_num_radii_cells=snp(15),
            translation_theta_step_deg=snp(15),
            rotation_vmf_kappa=snp(f_(100)),
            rotation_angle_max_deg=snp(25),
            rotation_angle_step_deg=snp(0.375),
            n_hyperblobs=num_hyperblobs,
            n_blobs=num_blobs,
            n_datapoints=num_datapoints,
        )

        # Initialize model with fixed random seed for reproducibility
        key = jkey(RANDOM_SEED)
        key, key_importance = jax.random.split(key)
        init_tr, _ = model_jimportance(key_importance, kmeans_chm, (hypers,))
        init_hdgmm_state = init_tr.get_retval()

        # Initial Gibbs sweeps
        key, init_gibbs_key = jax.random.split(key)
        print("Running initial Gibbs sweeps...")
        gibbs_wtrs = init_gibbs_sweep_dino(init_gibbs_key, init_hdgmm_state, num_sweeps=15)
        init_hdgmm_state = gibbs_wtrs[-1].retval

        # # Post-Gibbs filtering (gets ignored if USE_SAM_FRAME0 is True)
        # fx = fy = FOCAL_LENGTH
        # cx = img_dims[1] / 2.0
        # cy = img_dims[0] / 2.0

        # blob_means_after_gibbs = np.array(init_hdgmm_state.blobs_state.blob_means)
        # hyperblob_assignments_after_gibbs = np.array(init_hdgmm_state.blobs_state.hyperblob_assignments)

        # x_2d = (blob_means_after_gibbs[:, 0] / (blob_means_after_gibbs[:, 2] + 1e-8)) * fx + cx
        # y_2d = (blob_means_after_gibbs[:, 1] / (blob_means_after_gibbs[:, 2] + 1e-8)) * fy + cy
        # x_2d = np.clip(x_2d.astype(int), 0, img_dims[1] - 1)
        # y_2d = np.clip(y_2d.astype(int), 0, img_dims[0] - 1)
        # pixel_indices = y_2d * img_dims[1] + x_2d

        # blob_assignments_frame0 = np.array(init_hdgmm_state.datapoints_state.blob_assignments)
        # n_blobs_after_gibbs = len(blob_means_after_gibbs)
        # valid_mask = blob_assignments_frame0 < n_blobs_after_gibbs
        # datapoint_to_hyperblob = np.full(len(blob_assignments_frame0), -1, dtype=int)
        # datapoint_to_hyperblob[valid_mask] = hyperblob_assignments_after_gibbs[blob_assignments_frame0[valid_mask]]

        # hyperblob_overlaps = {}
        # for hb_idx in range(num_hyperblobs):
        #     hb_mask = datapoint_to_hyperblob == hb_idx
        #     # Use subsampled segmask if subsampling is active
        #     overlap = np.sum(hb_mask & ((first_frame_seg_sub if subsampled_indices is not None else first_frame_seg) == 1))
        #     hyperblob_overlaps[hb_idx] = overlap

        # object_hyperblob_idx = max(hyperblob_overlaps, key=hyperblob_overlaps.get)

        # if not USE_SAM_FRAME0:
        #     # get ignored if USE_SAM_FRAME0 is True

        #     blobs_in_object = np.where(hyperblob_assignments_after_gibbs == object_hyperblob_idx)[0]
        #     blobs_to_reassign = []
        #     for blob_idx in blobs_in_object:
        #         pixel_idx = pixel_indices[blob_idx]
        #         if pixel_idx >= len(first_frame_seg) or not first_frame_seg[pixel_idx]:
        #             blobs_to_reassign.append(blob_idx)

        #     if len(blobs_to_reassign) > 0:
        #         background_hyperblobs = [hb for hb in range(num_hyperblobs) if hb != object_hyperblob_idx]
        #         background_blob_counts = {hb: np.sum(hyperblob_assignments_after_gibbs == hb) for hb in background_hyperblobs}
        #         target_background_hb = max(background_blob_counts, key=background_blob_counts.get)
        #         updated_hyperblob_assignments = jnp.array(hyperblob_assignments_after_gibbs)
        #         for blob_idx in blobs_to_reassign:
        #             updated_hyperblob_assignments = updated_hyperblob_assignments.at[blob_idx].set(target_background_hb)
        #         init_hdgmm_state = init_hdgmm_state.replace({
        #             'blobs_state': {'hyperblob_assignments': updated_hyperblob_assignments}
        #         })

        # Track over time with FPS measurement
        key, tracking_key = jax.random.split(key)
        print("Running tracking...")
        
        # Measure FPS if enabled
        fps = None
        if MEASURE_FPS:
            # First run to trigger JIT compilation (not timed)
            print("JIT compiling tracking function...")
            _ = hdgmm_tracking_gibbs_dino(
                tracking_key, init_hdgmm_state,
                tracked_points, tracked_motion_vectors, tracked_features,
                outlier_prob=1e-28
            )
            
            # Second run for timing (after JIT compilation)
            print("Measuring FPS after JIT compilation...")
            start_time = time.time()
            tracking_wtrs = hdgmm_tracking_gibbs_dino(
                tracking_key, init_hdgmm_state,
                tracked_points, tracked_motion_vectors, tracked_features,
                outlier_prob=1e-28
            )
            end_time = time.time()
            
            # Calculate FPS
            total_time = end_time - start_time
            num_frames = len(tracked_points) - 1  # Subtract 1 because tracking starts from frame 1
            fps = num_frames / total_time if total_time > 0 else 0.0
            print(f"Tracking FPS: {fps:.2f} frames/second (total time: {total_time:.2f}s for {num_frames} frames)")
        else:
            tracking_wtrs = hdgmm_tracking_gibbs_dino(
                tracking_key, init_hdgmm_state,
                tracked_points, tracked_motion_vectors, tracked_features,
                outlier_prob=1e-28
            )

        

        # Extract results
        tracking_data = []
        # NOTE that tracking wtrs also contains the first frame, which is good because we need to evaluate the first frame too
        for frame_idx in tqdm(range(len(tracking_wtrs)), desc="Dense Evaluating Blob Assignments"):
            frame = tracking_wtrs[frame_idx]
            #NOTE: This is the dense evaluation of the blob assignments using the full DINO features and motion vectors and positions
            key, dense_eval_assignments_key = jax.random.split(key)
            dense_eval_assignments = dense_eval_blob_assignments(
                key=dense_eval_assignments_key, 
                hdgmm_state=frame.retval, 
                dense_positions=tracked_points_full[frame_idx], 
                dense_vels=tracked_motion_vectors_full[frame_idx], 
                dense_features=tracked_features_full[frame_idx], 
                disable_outlier_prob=False
            )

            # Then we get the blob weights from the dense eval assignments
            key, dense_eval_weights_key = jax.random.split(key)
            dense_eval_weights = dense_eval_blob_weights(
                key=dense_eval_weights_key, 
                hdgmm_state=frame.retval, 
                dense_assignments=dense_eval_assignments
            )
            frame_data = {
                'n_blobs': frame.retval.hypers.n_blobs,
                'n_hyperblobs': frame.retval.hypers.n_hyperblobs,
                'n_datapoints': num_datapoints_full,
                'blob_assignments': np.array(dense_eval_assignments),
                'datapoint_positions': np.array(tracked_points_full[frame_idx]),
                'datapoint_vels': np.array(tracked_motion_vectors_full[frame_idx]),
                'datapoint_features': np.array(tracked_features_full[frame_idx]),
                'blob_weights': np.array(dense_eval_weights),
                'blob_means': np.array(frame.retval.blobs_state.blob_means),
                'blob_covs': np.array(frame.retval.blobs_state.blob_covs),
                'blob_vel_means': np.array(frame.retval.blobs_state.blob_vel_means),
                'blob_vel_covs': np.array(frame.retval.blobs_state.blob_vel_covs),
                'blob_features': np.array(frame.retval.blobs_state.blob_features),
                'hyperblob_assignments': np.array(frame.retval.blobs_state.hyperblob_assignments),
                'hyperblob_weights': np.array(frame.retval.hyperblobs_state.hyperblob_weights),
                'hyperblob_means': np.array(frame.retval.hyperblobs_state.hyperblob_means),
                'hyperblob_trans_vels': np.array(frame.retval.hyperblobs_state.hyperblob_trans_vels),
                'hyperblob_rot_vels': np.array(frame.retval.hyperblobs_state.hyperblob_rot_vels),
            }
            tracking_data.append(frame_data)

        # Evaluate
        print("Computing error rates...")
        segmentation_masks = []
        for frame_idx in range(len(tracking_data)):
            seg_mask = get_segmentation_mask(
                video_name, frame_idx, DAVIS_SEGMASKS_PATH, img_dims=img_dims, flatten=True
            )
            # if subsampled_indices is not None:
            #     seg_mask = seg_mask[subsampled_indices]
            segmentation_masks.append(seg_mask)

        # is_mask_subsampled = subsampling_percentage < 100
        error_results = compute_error_rates(
            tracking_data, segmentation_masks, img_dims,
                BLOB_COUNTING_THRESHOLD, FOCAL_LENGTH, force_below_count_thresh_as_outlier=True,
                subsampled_indices=None, is_mask_subsampled= False
            )

        # Get particle-count accuracy metrics
        all_results = {video_name: [tracking_data]}
        experiment_metrics, best_visualization_data = evaluate_single_davis_video(
            davis_name=video_name,
            multiple_genparticles_list=all_results[video_name],
            annotations_path=DAVIS_SEGMASKS_PATH,
            counting_threshold=BLOB_COUNTING_THRESHOLD,
            img_dims=img_dims,
            fps_list=None,
            render_results_video=False,
            experiment_save_dir=None,
            force_below_count_thresh_as_outlier=True,
            subsampled_indices=None
        )

        result = {
            'video_name': video_name,
            'error_results': error_results,
            'pixel_metrics': experiment_metrics,
            'tracking_data': tracking_data,
            'segmentation_masks': segmentation_masks,
            'img_dims': img_dims,
            'subsampled_indices': None,
            'subsampling_percentage': subsampling_percentage,
            'fps': fps  # Add FPS to result
        }

        print(f"Completed: {video_name}")
        
        if MEASURE_FPS and fps is not None:
            print(f"  FPS: {fps:.2f} frames/second")
        
        print(f"\n  1. PARTICLE-BASED METRICS (projected particle means):")
        print(f"    Mean FN Rate: {error_results['mean_fn_rate']:.2f}%  (object particles leaving mask)")
        print(f"    Mean FP Rate: {error_results['mean_fp_rate']:.2f}%  (background particles entering mask)")
        print(f"    Mean Jaccard: {error_results['mean_jaccard']:.3f}  (particle IoU)")
        print(f"    Mean Accuracy: {error_results['mean_accuracy']:.2f}%  ((TN + TP) / (TP + TN + FP + FN))")
        print(f"    Note: Projects particle means to 2D, checks if they fall within GT mask")

        # Compute F1 from recall and precision (not returned by evaluate_single_davis_video)
        recall = experiment_metrics['avg_recall']
        precision = experiment_metrics['avg_precision']
        avg_f1 = 2 * (recall * precision) / (recall + precision) if (recall + precision) > 0 else 0.0

        print(f"\n  2. PARTICLE-COUNT UNWEIGHTED METRICS (fractional particle contributions):")
        print(f"    Recall:     {recall:.3f}  (fractional object particles correctly predicted)")
        print(f"    Precision:  {precision:.3f}  (predicted fractional particles that are correct)")
        print(f"    F1:         {avg_f1:.3f}  (harmonic mean)")
        print(f"    Jaccard:    {experiment_metrics['avg_jaccard']:.3f}  (fractional particle IoU)")
        print(f"    Accuracy:   {experiment_metrics['avg_accuracy']:.3f}  (fractional particle accuracy)")
        print(f"    Note: Each particle contributes fractionally based on proportion of pixels in GT mask")

        print(f"\n  3. PARTICLE-COUNT MATTER-WEIGHTED ADAPTIVE METRICS (weighted fractional contributions):")
        print(f"    Recall:     {experiment_metrics['avg_matter_weighted_recall']:.3f}  (weighted object matter staying in mask)")
        print(f"    Precision:  {experiment_metrics['avg_matter_weighted_precision']:.3f}  (predicted weighted matter correctness)")
        print(f"    F1:         {experiment_metrics['avg_matter_weighted_f1']:.3f}  (harmonic mean)")
        print(f"    Jaccard:    {experiment_metrics['avg_matter_weighted_jaccard']:.3f}  (weighted matter IoU)")
        print(f"    Accuracy:   {experiment_metrics['avg_matter_weighted_accuracy']:.3f}  (weighted matter accuracy)")
        print(f"    Note: Weights fractional contributions by (pixel_count × blob_weight), adaptive per frame")

        print(f"\n  4. PARTICLE-COUNT MATTER-WEIGHTED FIXED METRICS (weighted fractional contributions):")
        print(f"    Recall:     {experiment_metrics['avg_matter_weighted_recall_fixed']:.3f}  (weighted object matter staying in mask)")
        print(f"    Precision:  {experiment_metrics['avg_matter_weighted_precision_fixed']:.3f}  (predicted weighted matter correctness)")
        print(f"    F1:         {experiment_metrics['avg_matter_weighted_f1_fixed']:.3f}  (harmonic mean)")
        print(f"    Jaccard:    {experiment_metrics['avg_matter_weighted_jaccard_fixed']:.3f}  (weighted matter IoU)")
        print(f"    Accuracy:   {experiment_metrics['avg_matter_weighted_accuracy_fixed']:.3f}  (weighted matter accuracy)")
        print(f"    Note: Weights fractional contributions by (pixel_count × blob_weight), fixed from frame 0")

        # Diagnostic checks for adversarial cases
        print(f"\n  Diagnostic Checks (detecting potential metric gaming):")

        # 1. Adaptive-Fixed Gap
        adaptive_fixed_gap = experiment_metrics['avg_matter_weighted_f1_fixed'] - experiment_metrics['avg_matter_weighted_f1']
        print(f"    Adaptive-Fixed Gap:  {adaptive_fixed_gap:+.3f}  (if > 0.1, model may downweight difficult particles)")
        if adaptive_fixed_gap > 0.1:
            print(f"    ⚠️  WARNING: Large gap suggests adaptive weights are gaming the metric")
        elif adaptive_fixed_gap < -0.05:
            print(f"    ⚠️  WARNING: Negative gap suggests fixed weights may be suboptimal")

        # 2. Weight Entropy (frame 0)
        frame0_blob_weights = np.array(tracking_data[0]['blob_weights'])
        weight_entropy = -np.sum(frame0_blob_weights * np.log(frame0_blob_weights + 1e-10))
        max_entropy = np.log(len(frame0_blob_weights))
        normalized_entropy = weight_entropy / max_entropy
        print(f"    Weight Entropy:      {normalized_entropy:.3f}  (if < 0.5, weights concentrated; 1.0 = uniform)")
        if normalized_entropy < 0.5:
            print(f"    ⚠️  WARNING: Weights are concentrated on few particles (potential gaming)")

        # 3. Spatial Coverage (reference particles)
        frame0_blob_means = np.array(tracking_data[0]['blob_means'])
        frame0_hyperblob_assignments = np.array(tracking_data[0]['hyperblob_assignments'])
        frame0_blob_assignments = np.array(tracking_data[0]['blob_assignments'])
        
        # Handle reshaping based on whether subsampling was used
        if result.get('subsampled_indices') is None:
            # No subsampling: reshape to image dimensions if needed
            if frame0_blob_assignments.ndim == 1:
                frame0_blob_assignments = frame0_blob_assignments.reshape(result['img_dims'])
        else:
            # Subsampling was used: keep as 1D (already subsampled)
            if frame0_blob_assignments.ndim > 1:
                frame0_blob_assignments = frame0_blob_assignments.flatten()
        
        frame0_n_blobs = tracking_data[0]['n_blobs']

        # Get reference particles (same logic as evaluation)
        blob_pixel_counts = np.bincount(
            frame0_blob_assignments.flatten()[frame0_blob_assignments.flatten() < frame0_n_blobs],
            minlength=frame0_n_blobs
        )
        reference_particle_indices = np.where(blob_pixel_counts >= BLOB_COUNTING_THRESHOLD)[0]

        if len(reference_particle_indices) > 1:
            ref_positions = frame0_blob_means[reference_particle_indices]
            spatial_std = np.std(ref_positions, axis=0)
            avg_spatial_std = np.mean(spatial_std)
            print(f"    Spatial Std Dev:     {avg_spatial_std:.3f}  (particles spread; low values = clustered)")
            if avg_spatial_std < 0.1:
                print(f"    ⚠️  WARNING: Reference particles are highly clustered (potential gaming)")
        else:
            print(f"    Spatial Std Dev:     N/A  (only {len(reference_particle_indices)} reference particle)")

        # 4. Number of reference particles
        print(f"    Ref Particles:       {len(reference_particle_indices)}  (particles used for evaluation)")
        if len(reference_particle_indices) < 10:
            print(f"    ⚠️  WARNING: Very few reference particles (metric may be fragile)")

        # Clean up large intermediate variables before returning
        # These are no longer needed after creating the result
        del tracking_wtrs
        del gibbs_wtrs
        del init_hdgmm_state
        del all_results
        del segmentation_masks
        
        # Clean up large input data arrays
        # These are no longer needed after tracking is complete
        del tracked_points
        del tracked_motion_vectors
        del tracked_features
        del pca_features_unnormalized
        
        # Force garbage collection to free memory immediately
        gc.collect()

        return result

    except Exception as e:
        print(f"Error processing {video_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        # Clean up memory on error
        cleanup_between_runs()
        return None

# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    os.makedirs(EXPERIMENT_SAVE_DIR, exist_ok=True)

    all_results = []
    all_accuracies = {}

    print(f"{'='*80}")
    print(f"DINO TRACKING EXPERIMENT - Processing {len(VIDEO_NAMES)} videos")
    if MEASURE_FPS:
        print(f"FPS measurement: ENABLED")
    else:
        print(f"FPS measurement: DISABLED")
    print(f"{'='*80}\n")

    subsampling_percentages = [0.78125]
    # subsampling_percentages = [100.0, 50.0, 25.0, 12.5, 6.25, 3.125, 1.5625, 0.78125]
    # subsampling_percentages = [3.125, 1.5625, 0.78125]

    for video_name in tqdm(VIDEO_NAMES, desc="Overall Progress", position=0):
        for percentage in subsampling_percentages:
            run_dir_name = f"subsample_{str(percentage).replace('.', '_')}"
            run_save_dir = os.path.join(EXPERIMENT_SAVE_DIR, run_dir_name)
            os.makedirs(run_save_dir, exist_ok=True)

            result = process_video(video_name, subsampling_percentage=percentage, subsampled_indices=None, run_save_dir=run_save_dir)

            # Clean up memory after each subsampling percentage run
            cleanup_between_runs()

            if result is not None:
                all_results.append(result)

                # Create subdirectories for outputs
                base_dir = run_save_dir if ('subsampling_percentage' in result) else EXPERIMENT_SAVE_DIR
                plots_dir = os.path.join(base_dir, "error_rate_plots")
                videos_dir = os.path.join(base_dir, "3wide_videos")
                os.makedirs(plots_dir, exist_ok=True)
                os.makedirs(videos_dir, exist_ok=True)

                # Save error rate plot
                plot_path = os.path.join(plots_dir, f"{video_name}_error_rates.png")
                plot_error_rates(result['error_results'], video_name, plot_path)
                print(f"Saved error rate plot: {plot_path}")

                # Save 3-wide video (TEMPORARILY DISABLED)
                video_path = os.path.join(videos_dir, f"{video_name}_3wide_synchronized.mp4")
                # is_subsampled = result.get('subsampling_percentage') < 100
                create_3wide_video(
                    result['tracking_data'], video_name, DAVIS_RGB_PATH,
                    result['img_dims'], result['segmentation_masks'],
                    NUM_BLOBS, video_path, subsampled_indices=None, is_subsampled=False
                )
                print(f"Saved 3-wide video: {video_path}")

                # Compute diagnostics for JSON
                tracking_data = result['tracking_data']
                frame0_blob_weights = np.array(tracking_data[0]['blob_weights'])
                weight_entropy = -np.sum(frame0_blob_weights * np.log(frame0_blob_weights + 1e-10))
                max_entropy = np.log(len(frame0_blob_weights))
                normalized_entropy = float(weight_entropy / max_entropy)

                frame0_blob_means = np.array(tracking_data[0]['blob_means'])
                frame0_blob_assignments_vec = np.array(tracking_data[0]['blob_assignments'])
                frame0_n_blobs = tracking_data[0]['n_blobs']
                blob_pixel_counts = np.bincount(
                    frame0_blob_assignments_vec[frame0_blob_assignments_vec < frame0_n_blobs],
                    minlength=frame0_n_blobs
                )
                reference_particle_indices = np.where(blob_pixel_counts >= BLOB_COUNTING_THRESHOLD)[0]

                if len(reference_particle_indices) > 1:
                    ref_positions = frame0_blob_means[reference_particle_indices]
                    spatial_std = np.std(ref_positions, axis=0)
                    avg_spatial_std = float(np.mean(spatial_std))
                else:
                    avg_spatial_std = None

                adaptive_fixed_gap = float(
                    result['pixel_metrics']['avg_matter_weighted_f1_fixed'] -
                    result['pixel_metrics']['avg_matter_weighted_f1']
                )

                # Store accuracy with all four metric types
                all_accuracies[video_name] = {
                    # Full pixel metrics dict (contains all trial data)
                    'pixel_metrics': result['pixel_metrics'],

                    # 1. Particle-based error rates (projected particle means)
                    'particle_fn_rate': result['error_results']['mean_fn_rate'],
                    'particle_fp_rate': result['error_results']['mean_fp_rate'],
                    'particle_jaccard': result['error_results']['mean_jaccard'],
                    'particle_accuracy': result['error_results']['mean_accuracy'],
                    'n_object_particles': result['error_results']['n_object_blobs'],
                    'n_background_particles': result['error_results']['n_background_blobs'],

                    # 2. Particle-count unweighted metrics (fractional particle contributions)
                    'particle_count_unweighted_recall': result['pixel_metrics']['avg_recall'],
                    'particle_count_unweighted_precision': result['pixel_metrics']['avg_precision'],
                    'particle_count_unweighted_f1': 2 * (result['pixel_metrics']['avg_recall'] * result['pixel_metrics']['avg_precision']) / (result['pixel_metrics']['avg_recall'] + result['pixel_metrics']['avg_precision']) if (result['pixel_metrics']['avg_recall'] + result['pixel_metrics']['avg_precision']) > 0 else 0.0,
                    'particle_count_unweighted_jaccard': result['pixel_metrics']['avg_jaccard'],
                    'particle_count_unweighted_accuracy': result['pixel_metrics']['avg_accuracy'],

                    # 3. Particle-count matter-weighted adaptive metrics (weighted fractional contributions)
                    'particle_count_matter_adaptive_recall': result['pixel_metrics']['avg_matter_weighted_recall'],
                    'particle_count_matter_adaptive_precision': result['pixel_metrics']['avg_matter_weighted_precision'],
                    'particle_count_matter_adaptive_f1': result['pixel_metrics']['avg_matter_weighted_f1'],
                    'particle_count_matter_adaptive_jaccard': result['pixel_metrics']['avg_matter_weighted_jaccard'],
                    'particle_count_matter_adaptive_accuracy': result['pixel_metrics']['avg_matter_weighted_accuracy'],

                    # 4. Particle-count matter-weighted fixed metrics (weighted fractional contributions)
                    'particle_count_matter_fixed_recall': result['pixel_metrics']['avg_matter_weighted_recall_fixed'],
                    'particle_count_matter_fixed_precision': result['pixel_metrics']['avg_matter_weighted_precision_fixed'],
                    'particle_count_matter_fixed_f1': result['pixel_metrics']['avg_matter_weighted_f1_fixed'],
                    'particle_count_matter_fixed_jaccard': result['pixel_metrics']['avg_matter_weighted_jaccard_fixed'],
                    'particle_count_matter_fixed_accuracy': result['pixel_metrics']['avg_matter_weighted_accuracy_fixed'],

                    # FPS measurement
                    'fps': result.get('fps'),

                    # Diagnostic metrics (detecting potential gaming)
                    'diagnostics': {
                        'adaptive_fixed_gap': adaptive_fixed_gap,
                        'weight_entropy_normalized': normalized_entropy,
                        'spatial_std_dev': avg_spatial_std,
                        'n_reference_particles': int(len(reference_particle_indices)),
                        'warning_adaptive_fixed_gap': adaptive_fixed_gap > 0.1,
                        'warning_weight_concentration': normalized_entropy < 0.5,
                        'warning_spatial_clustering': avg_spatial_std is not None and avg_spatial_std < 0.1,
                        'warning_few_particles': len(reference_particle_indices) < 10
                    }
                    }

                # Save per-run JSON
                json_results_dir = os.path.join(base_dir, "json_results")
                os.makedirs(json_results_dir, exist_ok=True)
                per_run_json_path = os.path.join(json_results_dir, f"{video_name}_results.json")
                with open(per_run_json_path, 'w') as f:
                    json.dump(all_accuracies[video_name], f, indent=2)
                print(f"Saved per-run JSON: {per_run_json_path}")

                # Save aggregated JSON after each run completes (incremental updates)
                json_path = os.path.join(base_dir, "all_videos_experiment_results.json")
                if os.path.exists(json_path):
                    with open(json_path, 'r') as f:
                        existing_results = json.load(f)
                    existing_results.update(all_accuracies)
                    all_accuracies_to_save = existing_results
                else:
                    all_accuracies_to_save = all_accuracies

                with open(json_path, 'w') as f:
                    json.dump(all_accuracies_to_save, f, indent=2)
                print(f"Saved experiment results (updated): {json_path}")

                # Remove large tracking_data from result to free memory
                if 'tracking_data' in result:
                    del result['tracking_data']
                try:
                    del tracking_data
                except NameError:
                    pass
                
                # Clean up local variables used for diagnostics
                try:
                    del frame0_blob_weights, frame0_blob_means, frame0_blob_assignments_vec
                    del blob_pixel_counts, reference_particle_indices
                except NameError:
                    pass
                
                # Force garbage collection after processing result
                gc.collect()

        # Clean up environment variables and memory between runs
        # This prevents memory leaks from accumulating across experimental runs
        cleanup_between_runs()

    # Print summary
    print(f"\n{'='*80}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*80}")
    print(f"Total videos processed: {len(all_results)}/{len(VIDEO_NAMES)}")
    print(f"Failed videos: {len(VIDEO_NAMES) - len(all_results)}")
    print(f"Results saved to: {EXPERIMENT_SAVE_DIR}")

    # Load all results from JSON file to compute aggregate statistics
    json_path = os.path.join(EXPERIMENT_SAVE_DIR, "all_videos_experiment_results.json")
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            all_accuracies_from_json = json.load(f)
    else:
        all_accuracies_from_json = all_accuracies

    # Filter out incomplete results (videos that failed to process)
    valid_results = {k: v for k, v in all_accuracies_from_json.items() 
                    if 'particle_fn_rate' in v and 'particle_count_unweighted_recall' in v}

    print(f"\n{'='*80}")
    print(f"SUMMARY - {len(valid_results)} Videos")
    print(f"{'='*80}")
    print(f"\nMetric Interpretation Guide:")
    print(f"  • Particle-based: Projects particle means to 2D, checks if they fall within GT mask")
    print(f"  • Particle-count unweighted: Fractional particle contributions, equal weight per particle")
    print(f"  • Particle-count matter adaptive: Fractional contributions weighted by (pixel_count × blob_weight), adaptive per frame")
    print(f"  • Particle-count matter fixed: Fractional contributions weighted by (pixel_count × blob_weight), fixed from frame 0")
    print(f"\n")

    # Compute aggregate statistics for all four metric types
    
    # if len(valid_results) == 0:
    #     print("⚠️  No valid results found in JSON file. Cannot compute aggregate statistics.")
    #     return
    
    particle_fn_rates = [m['particle_fn_rate'] for m in valid_results.values()]
    particle_fp_rates = [m['particle_fp_rate'] for m in valid_results.values()]
    particle_jaccard = [m['particle_jaccard'] for m in valid_results.values()]
    particle_accuracy = [m['particle_accuracy'] for m in valid_results.values()]
    
    particle_count_unweighted_recall = [m['particle_count_unweighted_recall'] for m in valid_results.values()]
    particle_count_unweighted_precision = [m['particle_count_unweighted_precision'] for m in valid_results.values()]
    particle_count_unweighted_f1 = [m['particle_count_unweighted_f1'] for m in valid_results.values()]
    particle_count_unweighted_jaccard = [m['particle_count_unweighted_jaccard'] for m in valid_results.values()]
    particle_count_unweighted_accuracy = [m['particle_count_unweighted_accuracy'] for m in valid_results.values()]
    
    particle_count_matter_adaptive_recall = [m['particle_count_matter_adaptive_recall'] for m in valid_results.values()]
    particle_count_matter_adaptive_precision = [m['particle_count_matter_adaptive_precision'] for m in valid_results.values()]
    particle_count_matter_adaptive_f1 = [m['particle_count_matter_adaptive_f1'] for m in valid_results.values()]
    particle_count_matter_adaptive_jaccard = [m['particle_count_matter_adaptive_jaccard'] for m in valid_results.values()]
    particle_count_matter_adaptive_accuracy = [m['particle_count_matter_adaptive_accuracy'] for m in valid_results.values()]
    
    particle_count_matter_fixed_recall = [m['particle_count_matter_fixed_recall'] for m in valid_results.values()]
    particle_count_matter_fixed_precision = [m['particle_count_matter_fixed_precision'] for m in valid_results.values()]
    particle_count_matter_fixed_f1 = [m['particle_count_matter_fixed_f1'] for m in valid_results.values()]
    particle_count_matter_fixed_jaccard = [m['particle_count_matter_fixed_jaccard'] for m in valid_results.values()]
    particle_count_matter_fixed_accuracy = [m['particle_count_matter_fixed_accuracy'] for m in valid_results.values()]

    # Collect FPS values (handle None values)
    fps_values = [m.get('fps') for m in valid_results.values() if m.get('fps') is not None]

    print(f"AGGREGATE STATISTICS ACROSS ALL VIDEOS:")
    
    print(f"\n  1. PARTICLE-BASED METRICS (projected particle means):")
    print(f"    Mean FN Rate:            {np.mean(particle_fn_rates):.2f}% ± {np.std(particle_fn_rates):.2f}%")
    print(f"    Mean FP Rate:            {np.mean(particle_fp_rates):.2f}% ± {np.std(particle_fp_rates):.2f}%")
    print(f"    Mean Jaccard:            {np.mean(particle_jaccard):.3f} ± {np.std(particle_jaccard):.3f}")

    print(f"\n  2. PARTICLE-COUNT UNWEIGHTED METRICS (fractional particle contributions):")
    print(f"    Recall:                  {np.mean(particle_count_unweighted_recall):.3f} ± {np.std(particle_count_unweighted_recall):.3f}")
    print(f"    Precision:               {np.mean(particle_count_unweighted_precision):.3f} ± {np.std(particle_count_unweighted_precision):.3f}")
    print(f"    F1:                      {np.mean(particle_count_unweighted_f1):.3f} ± {np.std(particle_count_unweighted_f1):.3f}")
    print(f"    Jaccard:                 {np.mean(particle_count_unweighted_jaccard):.3f} ± {np.std(particle_count_unweighted_jaccard):.3f}")

    print(f"\n  3. PARTICLE-COUNT MATTER-WEIGHTED ADAPTIVE METRICS (weighted fractional contributions):")
    print(f"    Recall:                  {np.mean(particle_count_matter_adaptive_recall):.3f} ± {np.std(particle_count_matter_adaptive_recall):.3f}")
    print(f"    Precision:               {np.mean(particle_count_matter_adaptive_precision):.3f} ± {np.std(particle_count_matter_adaptive_precision):.3f}")
    print(f"    F1:                      {np.mean(particle_count_matter_adaptive_f1):.3f} ± {np.std(particle_count_matter_adaptive_f1):.3f}")
    print(f"    Jaccard:                 {np.mean(particle_count_matter_adaptive_jaccard):.3f} ± {np.std(particle_count_matter_adaptive_jaccard):.3f}")

    print(f"\n  4. PARTICLE-COUNT MATTER-WEIGHTED FIXED METRICS (weighted fractional contributions):")
    print(f"    Recall:                  {np.mean(particle_count_matter_fixed_recall):.3f} ± {np.std(particle_count_matter_fixed_recall):.3f}")
    print(f"    Precision:               {np.mean(particle_count_matter_fixed_precision):.3f} ± {np.std(particle_count_matter_fixed_precision):.3f}")
    print(f"    F1:                      {np.mean(particle_count_matter_fixed_f1):.3f} ± {np.std(particle_count_matter_fixed_f1):.3f}")
    print(f"    Jaccard:                 {np.mean(particle_count_matter_fixed_jaccard):.3f} ± {np.std(particle_count_matter_fixed_jaccard):.3f}")

    # Print FPS statistics if available
    if fps_values:
        print(f"\n  5. PERFORMANCE METRICS:")
        print(f"    FPS:                     {np.mean(fps_values):.2f} ± {np.std(fps_values):.2f} frames/second")
        print(f"    Videos with FPS data:    {len(fps_values)}/{len(valid_results)}")
    else:
        print(f"\n  5. PERFORMANCE METRICS:")
        print(f"    FPS:                     No FPS data available")

    print(f"\n\nPER-VIDEO RESULTS:")
    print(f"{'='*80}\n")

    for video_name, metrics in valid_results.items():
        print(f"  {video_name}:")
        
        print(f"    1. PARTICLE-BASED METRICS (projected particle means):")
        print(f"      FN Rate:                 {metrics['particle_fn_rate']:.2f}%  (object particles leaving mask)")
        print(f"      FP Rate:                 {metrics['particle_fp_rate']:.2f}%  (background particles entering mask)")
        print(f"      Jaccard:                 {metrics['particle_jaccard']:.3f}  (particle IoU)")

        print(f"\n    2. PARTICLE-COUNT UNWEIGHTED METRICS (fractional particle contributions):")
        print(f"      Recall:                  {metrics['particle_count_unweighted_recall']:.3f}  (fractional object particles correctly predicted)")
        print(f"      Precision:               {metrics['particle_count_unweighted_precision']:.3f}  (predicted fractional particles that are correct)")
        print(f"      F1:                      {metrics['particle_count_unweighted_f1']:.3f}  (harmonic mean)")
        print(f"      Jaccard:                 {metrics['particle_count_unweighted_jaccard']:.3f}  (fractional particle IoU)")

        print(f"\n    3. PARTICLE-COUNT MATTER-WEIGHTED ADAPTIVE METRICS (weighted fractional contributions):")
        print(f"      Recall:                  {metrics['particle_count_matter_adaptive_recall']:.3f}  (weighted object matter staying in mask)")
        print(f"      Precision:               {metrics['particle_count_matter_adaptive_precision']:.3f}  (predicted weighted matter correctness)")
        print(f"      F1:                      {metrics['particle_count_matter_adaptive_f1']:.3f}  (harmonic mean)")
        print(f"      Jaccard:                 {metrics['particle_count_matter_adaptive_jaccard']:.3f}  (weighted matter IoU)")

        print(f"\n    4. PARTICLE-COUNT MATTER-WEIGHTED FIXED METRICS (weighted fractional contributions):")
        print(f"      Recall:                  {metrics['particle_count_matter_fixed_recall']:.3f}  (weighted object matter staying in mask)")
        print(f"      Precision:               {metrics['particle_count_matter_fixed_precision']:.3f}  (predicted weighted matter correctness)")
        print(f"      F1:                      {metrics['particle_count_matter_fixed_f1']:.3f}  (harmonic mean)")
        print(f"      Jaccard:                 {metrics['particle_count_matter_fixed_jaccard']:.3f}  (weighted matter IoU)")

        # Print FPS for this video if available
        video_fps = metrics.get('fps')
        if video_fps is not None:
            print(f"\n    5. PERFORMANCE METRICS:")
            print(f"      FPS:                     {video_fps:.2f} frames/second")
        else:
            print(f"\n    5. PERFORMANCE METRICS:")
            print(f"      FPS:                     Not measured")
        
        print(f"")
