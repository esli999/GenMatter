# HDGMM Tracking with DINO Features - Batch Processing
# Clean implementation for running experiments on multiple videos

import os
import json
import time
import gc
import jax

# JAX compilation cache setup
# cache_dir = os.path.join(os.getcwd(), ".jax_cache")
# if not os.path.exists(cache_dir):
#     os.makedirs(cache_dir)
# jax.config.update("jax_compilation_cache_dir", cache_dir)
# jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
# jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
# jax.experimental.compilation_cache.compilation_cache.set_cache_dir(cache_dir)

import jax.numpy as jnp
from jax.random import key as jkey

import numpy as np
import pickle
from tqdm import tqdm
import cv2
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

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

# All videos to process
VIDEO_NAMES = [
    "jello_trim", "bear", "blackswan", "breakdance", "breakdance-flare", "camel", "cows",
    "dance-jump", "dance-twirl", "dog", "elephant", "flamingo", 
    "goat",
    "hike", "lucia", "mallard-fly", "parkour", "rollerblade",
    "cloth_bag", "gray_jacket", "manta_ray",
    "new_eagle_trim", "ostrich_trim", 
    "whiskey_swirl_2", "wine_swirl", 
    "purple_jacket", "snake_trim"
]

# All videos to process
# VIDEO_NAMES = [
#     "dance-jump"
# ]


# Paths
DAVIS_3D_MOTION_PATH = "/home/esli/GenParticles_neural_stimulus/assets/cvpr_deformable_experiment/npzs"
DAVIS_SEGMASKS_PATH = "/home/esli/GenParticles_neural_stimulus/assets/cvpr_deformable_experiment/segmasks"
DAVIS_RGB_PATH = "/home/esli/GenParticles_neural_stimulus/assets/cvpr_deformable_experiment/frames"
DINO_PATH_TEMPLATE = '/home/esli/GenParticles_neural_stimulus/assets/cvpr_deformable_experiment/dino_features/dino_features_pca_l/{}_dino_pca_per_pixel.npz'

# Output directory
EXPERIMENT_SAVE_DIR = "/home/esli/GenParticles_NeurIPS/dino_tracking_cvpr_experiment_joint"

# Model hyperparameters
NUM_BLOBS = 500
NUM_HYPERBLOBS = 9
FOCAL_LENGTH = 520.0
BLOB_COUNTING_THRESHOLD = 100
RANDOM_SEED = 42
NUM_PCA_COMPONENTS = 10  # Limit PCA components to reduce memory usage

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
    blob_means: jnp.ndarray  # JOINT: [n_blobs, 3+feature_dim] position+features
    blob_covs: jnp.ndarray   # JOINT: [n_blobs, 3+feature_dim, 3+feature_dim]
    blob_vel_means: jnp.ndarray
    blob_vel_covs: jnp.ndarray

@Pytree.dataclass
class HDGMM_Datapoints_State_DINO(Super_Pytree):
    blob_assignments: jnp.ndarray
    datapoint_positions: jnp.ndarray  # JOINT: [n_datapoints, 3+feature_dim] position+features
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

    # JOINT: Sample joint (position+features) means and covariances
    full_dim = 3 + hypers.mu_F.shape[0]

    # Extend hyperblob means with feature means
    hyperblob_means_extended = jnp.concatenate([
        assigned_hyperblob_per_blob.hyperblob_means,  # position [n_blobs, 3]
        jnp.tile(hypers.mu_F[None, :], (hypers.n_blobs, 1))  # features [n_blobs, feature_dim]
    ], axis=1)

    # Extend hyperblob covariances to full dimensionality for blob mean prior
    hyperblob_covs_extended = jnp.zeros((hypers.n_blobs, full_dim, full_dim))
    hyperblob_covs_extended = hyperblob_covs_extended.at[:, :3, :3].set(assigned_hyperblob_per_blob.hyperblob_covs)
    # Weak prior for features (large variance since hyperblobs don't model features)
    hyperblob_covs_extended = hyperblob_covs_extended.at[:, 3:, 3:].set(
        1e6 * jnp.eye(full_dim - 3)[None, :, :]
    )

    # Sample joint blob means from extended hyperblob prior
    blob_means = genjax.mv_normal(
        hyperblob_means_extended,
        hyperblob_covs_extended
    ) @ 'blob_means'

    # Sample joint blob covariances
    blob_covs = inverse_wishart(hypers.nu_B, hypers.Psi_B, sample_shape=sample_shape) @ 'blob_covs'

    # Velocity remains separate (only uses spatial part of blob_means)
    blob_vel_means_ = assigned_hyperblob_per_blob.hyperblob_trans_vels + jnp.einsum(
        'nij,nj->ni',
        assigned_hyperblob_per_blob.hyperblob_rot_vels - jnp.repeat(jnp.eye(3)[None, ...], hypers.n_blobs, axis=0),
        blob_means[:, :3] - assigned_hyperblob_per_blob.hyperblob_means  # Extract spatial part
    )
    blob_vel_means = genjax.normal(blob_vel_means_, jnp.sqrt(hypers.sigma_V)) @ 'blob_vel_means'
    blob_vel_covs = inverse_wishart(hypers.nu_V, hypers.Psi_V, sample_shape=sample_shape) @ 'blob_vel_covs'

    return HDGMM_Blobs_State_DINO(
        hyperblob_assignments=hyperblob_assignments,
        blob_weights=blob_weights,
        blob_means=blob_means,
        blob_covs=blob_covs,
        blob_vel_means=blob_vel_means,
        blob_vel_covs=blob_vel_covs,
    )

@gen
def HDGMM_datapoints_model_dino(hypers: HDGMM_Hyperparams_DINO, blobs_state: HDGMM_Blobs_State_DINO):
    blob_assignments = genjax.categorical(
        probs=blobs_state.blob_weights,
        sample_shape=Const((hypers.n_datapoints,))
    ) @ 'blob_assignments'
    assigned_blob_per_datapoint = blobs_state[blob_assignments]

    # JOINT: Sample joint (position+features) observations
    datapoint_positions = genjax.mv_normal(
        assigned_blob_per_datapoint.blob_means,  # Joint means [n_datapoints, 3+feature_dim]
        assigned_blob_per_datapoint.blob_covs    # Joint covs [n_datapoints, 3+feature_dim, 3+feature_dim]
    ) @ 'datapoint_positions'

    # Velocity remains separate
    datapoint_vels = genjax.mv_normal(
        assigned_blob_per_datapoint.blob_vel_means,
        assigned_blob_per_datapoint.blob_vel_covs
    ) @ 'datapoint_vels'

    # datapoint_features removed - now part of datapoint_positions
    return HDGMM_Datapoints_State_DINO(
        blob_assignments=blob_assignments,
        datapoint_positions=datapoint_positions,  # JOINT observations
        datapoint_vels=datapoint_vels,
        datapoint_features=datapoint_positions[:, 3:],  # Extract for compatibility
    )

@gen
def blob_datapoint_likelihood_model_dino(blob_state: HDGMM_Blobs_State_DINO):
    # JOINT likelihood: position+features together
    datapoint_position = genjax.mv_normal(blob_state.blob_means, blob_state.blob_covs) @ 'datapoint_position'
    # Velocity separate
    datapoint_vel = genjax.mv_normal(blob_state.blob_vel_means, blob_state.blob_vel_covs) @ 'datapoint_vel'
    return None

# JIT compile
model_jsimulate = jax.jit(HDGMM_model_dino.simulate)
model_jimportance = jax.jit(HDGMM_model_dino.importance)

# ============================================================================
# Helper Functions
# ============================================================================

def extract_dino_features(stimulus, pca_features, img_dims, num_timesteps, max_components=None):
    """Extract DINO features reshaped to match tracking data."""
    T, h, w, num_features = pca_features.shape
    if max_components is not None:
        num_features = min(num_features, max_components)
        pca_features = pca_features[:, :, :, :num_features]
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

def initialize_model_with_dino(tracked_points, num_blobs, num_hyperblobs,
                               segmentation_mask, motion_vectors, tracked_features):
    """Initialize model with JOINT position+feature observations."""
    from genjax import ChoiceMapBuilder as C

    # Step 1: Run spatial k-means to get blob assignments (hyperblobs must be spatial-only)
    kmeans_chm, roi_blob_indices, roi_hyperblob_indices = make_hierarchical_kmeans_chm_with_mask_fixed_hyperblob(
        tracked_points, num_blobs, num_hyperblobs,
        segmentation_mask=segmentation_mask,
        motion_vectors=motion_vectors
    )

    # Step 2: Compute JOINT statistics (position+features) using the spatial blob assignments
    frame0_positions = tracked_points[0]  # [n_datapoints, 3]
    frame0_features = tracked_features[0]  # [n_datapoints, feature_dim]
    frame0_joint = np.concatenate([frame0_positions, frame0_features], axis=1)  # [n_datapoints, 3+feature_dim]

    blob_assignments = np.array(kmeans_chm['datapoints', 'blob_assignments'])
    num_blobs_actual = len(np.unique(blob_assignments))
    full_dim = frame0_joint.shape[1]

    # Compute JOINT blob means and covariances from empirical data
    blob_means_joint = np.zeros((num_blobs_actual, full_dim), dtype=np.float32)
    blob_covs_joint = np.zeros((num_blobs_actual, full_dim, full_dim), dtype=np.float32)

    for i in range(num_blobs_actual):
        points_in_blob = np.where(blob_assignments == i)[0]
        if len(points_in_blob) > 0:
            blob_joint_obs = frame0_joint[points_in_blob]
            blob_means_joint[i] = np.mean(blob_joint_obs, axis=0)

            # Compute full empirical joint covariance (captures position-feature correlation!)
            if len(points_in_blob) > full_dim + 2:
                blob_covs_joint[i] = np.cov(blob_joint_obs, rowvar=False) + 1e-4 * np.eye(full_dim)
            else:
                # Fallback for small blobs
                blob_covs_joint[i] = 1.0 * np.eye(full_dim)

    # Step 3: Build new choice map with joint data
    joint_chm = C['datapoints', 'blob_assignments'].set(a_(blob_assignments))
    joint_chm = joint_chm | C['datapoints', 'datapoint_positions'].set(a_(frame0_joint))  # JOINT
    joint_chm = joint_chm | C['datapoints', 'datapoint_vels'].set(kmeans_chm['datapoints', 'datapoint_vels'])
    joint_chm = joint_chm | C['datapoints', 'datapoint_features'].set(a_(frame0_features))  # For compatibility
    joint_chm = joint_chm | C['blobs', 'hyperblob_assignments'].set(kmeans_chm['blobs', 'hyperblob_assignments'])
    joint_chm = joint_chm | C['blobs', 'blob_weights'].set(kmeans_chm['blobs', 'blob_weights'])
    joint_chm = joint_chm | C['blobs', 'blob_means'].set(a_(blob_means_joint))  # JOINT
    joint_chm = joint_chm | C['blobs', 'blob_covs'].set(a_(blob_covs_joint))  # JOINT
    joint_chm = joint_chm | C['blobs', 'blob_vel_means'].set(kmeans_chm['blobs', 'blob_vel_means'])
    joint_chm = joint_chm | C['blobs', 'blob_vel_covs'].set(kmeans_chm['blobs', 'blob_vel_covs'])
    joint_chm = joint_chm | C['hyperblobs', 'hyperblob_weights'].set(kmeans_chm['hyperblobs', 'hyperblob_weights'])
    joint_chm = joint_chm | C['hyperblobs', 'hyperblob_means'].set(kmeans_chm['hyperblobs', 'hyperblob_means'])
    joint_chm = joint_chm | C['hyperblobs', 'hyperblob_covs'].set(kmeans_chm['hyperblobs', 'hyperblob_covs'])
    joint_chm = joint_chm | C['hyperblobs', 'hyperblob_trans_vels'].set(kmeans_chm['hyperblobs', 'hyperblob_trans_vels'])
    joint_chm = joint_chm | C['hyperblobs', 'hyperblob_rot_vels'].set(kmeans_chm['hyperblobs', 'hyperblob_rot_vels'])

    return joint_chm, roi_blob_indices, roi_hyperblob_indices

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

def gibbs_blob_means_dino(key, hdgmm_state):
    """Gibbs update for blob means with joint position+feature representation.
    
    Handles the case where blob_means has shape [L, full_dim] where full_dim = 3 (spatial) + feature_dim.
    Rotation velocities only affect the spatial (first 3) dimensions.
    """
    posterior_key, _ = jax.random.split(key)

    # Data - joint representation: [N, full_dim] where full_dim = 3 + feature_dim
    datapoint_positions = hdgmm_state.datapoints_state.datapoint_positions  # [N, full_dim]
    blob_assignments = hdgmm_state.datapoints_state.blob_assignments        # [N]
    blob_vel_means = hdgmm_state.blobs_state.blob_vel_means                 # [L, 3] - only spatial
    hyperblob_assignments = hdgmm_state.blobs_state.hyperblob_assignments   # [L]
    blob_covs = hdgmm_state.blobs_state.blob_covs                           # [L, full_dim, full_dim]

    # Hyperblob info - spatial only
    mu_H = hdgmm_state.hyperblobs_state.hyperblob_means                     # [K, 3]
    trans_vels = hdgmm_state.hyperblobs_state.hyperblob_trans_vels         # [K, 3]
    rot_vels = hdgmm_state.hyperblobs_state.hyperblob_rot_vels             # [K, 3, 3]

    # Hyperparams
    sigmaV = hdgmm_state.hypers.sigma_V
    mu_F = hdgmm_state.hypers.mu_F  # [feature_dim] - feature prior mean
    L = hdgmm_state.hypers.n_blobs
    full_dim = hdgmm_state.blobs_state.blob_means.shape[-1]  # full_dim (e.g., 13)
    d_spatial = 3  # Spatial dimension (for rotation velocities)
    feature_dim = full_dim - d_spatial

    # Extract spatial parts from joint blob means
    current_blob_means = hdgmm_state.blobs_state.blob_means  # [L, full_dim]
    blob_means_spatial = current_blob_means[:, :d_spatial]   # [L, 3]

    # Assign hyperblob means to blobs (spatial only)
    muH_l = mu_H[hyperblob_assignments]  # [L, 3]

    # Prior precision - extend spatial hyperblob covs to full dimension
    hyperblob_covs_spatial = hdgmm_state.hyperblobs_state.hyperblob_covs[hyperblob_assignments]  # [L, 3, 3]
    # Extend to full dimension: spatial block + weak prior for features
    hyperblob_covs_full = jnp.zeros((L, full_dim, full_dim))
    hyperblob_covs_full = hyperblob_covs_full.at[:, :d_spatial, :d_spatial].set(hyperblob_covs_spatial)
    # Weak prior for features (large variance since hyperblobs don't model features)
    hyperblob_covs_full = hyperblob_covs_full.at[:, d_spatial:, d_spatial:].set(
        1e6 * jnp.eye(full_dim - d_spatial)[None, :, :]
    )
    hyperblob_cov_inv_full = jnp.linalg.inv(hyperblob_covs_full)  # [L, full_dim, full_dim]
    hyperblob_cov_inv_spatial = hyperblob_cov_inv_full[:, :d_spatial, :d_spatial]  # [L, 3, 3]
    hyperblob_cov_inv_features = hyperblob_cov_inv_full[:, d_spatial:, d_spatial:]  # [L, feature_dim, feature_dim]

    # Data terms (from datapoints) - full joint representation
    N_l = jax.ops.segment_sum(jnp.ones(datapoint_positions.shape[0]), blob_assignments, num_segments=L) # [L]
    S_l = stable_segment_sum(datapoint_positions, blob_assignments, L) # [L, full_dim]

    # Likelihood precision (from blob covariances) - full joint representation
    blob_cov_inv = jnp.linalg.inv(blob_covs)  # [L, full_dim, full_dim]

    # Velocity affine likelihood - spatial only
    A_l = rot_vels[hyperblob_assignments] - jnp.eye(d_spatial)  # [L, 3, 3]
    b_l = trans_vels[hyperblob_assignments] - jnp.einsum("lij,lj->li", A_l, muH_l)  # [L, 3]
    v_l = blob_vel_means # [L, 3]
    residuals = v_l - b_l  # [L, 3]

    lhs_affine = jnp.einsum("lij,lik->ljk", A_l, A_l)    # [L, 3, 3]
    rhs_affine = jnp.einsum("lij,li->lj", A_l, residuals)  # [L, 3]

    # Create full-dimension precision and weighted mean
    # Start with full-dimension zero matrices
    P_post_full = jnp.zeros((L, full_dim, full_dim))
    
    # Set spatial block from prior (spatial part only)
    P_post_full = P_post_full.at[:, :d_spatial, :d_spatial].set(hyperblob_cov_inv_spatial)
    
    # Set feature block from prior (feature part - weak prior)
    P_post_full = P_post_full.at[:, d_spatial:, d_spatial:].set(hyperblob_cov_inv_features)
    
    # Add datapoint likelihood (full dimension)
    P_post_full = P_post_full + blob_cov_inv * N_l[:, None, None]
    
    # Add velocity model (spatial block only)
    P_post_full = P_post_full.at[:, :d_spatial, :d_spatial].add(lhs_affine / sigmaV)

    # Create full-dimension weighted mean
    weighted_mean_full = jnp.zeros((L, full_dim))
    
    # Spatial part: prior + velocity affine
    weighted_mean_spatial = (
        jnp.einsum("lij,lj->li", hyperblob_cov_inv_spatial, muH_l) +  # prior with hyperblob covariance
        rhs_affine / sigmaV                                   # velocity affine
    )
    weighted_mean_full = weighted_mean_full.at[:, :d_spatial].set(weighted_mean_spatial)
    
    # Feature part: prior mean contribution (weak prior, but still include it)
    weighted_mean_features = jnp.einsum("lij,j->li", hyperblob_cov_inv_features, mu_F)  # [L, feature_dim]
    weighted_mean_full = weighted_mean_full.at[:, d_spatial:].add(weighted_mean_features)
    
    # Add datapoint contribution (full dimension)
    weighted_mean_full = weighted_mean_full + jnp.einsum("lij,lj->li", blob_cov_inv, S_l)

    # Solve for posterior mean (full dimension)
    cov_post_full = jnp.linalg.inv(P_post_full)  # [L, full_dim, full_dim]
    mean_post_full = jnp.einsum("lij,lj->li", cov_post_full, weighted_mean_full)  # [L, full_dim]

    # Sample new blob means (full joint representation)
    new_blob_means = genjax.mv_normal.sample(posterior_key, mean_post_full, cov_post_full)

    return hdgmm_state.replace({'blobs_state': {'blob_means': new_blob_means}})

def gibbs_blob_vel_means_dino(key, hdgmm_state):
    """Gibbs update for blob velocity means with joint position+feature representation.
    
    Extracts spatial part (first 3 dims) from joint blob_means for velocity calculations.
    """
    posterior_key, _ = jax.random.split(key)

    datapoint_vels = hdgmm_state.datapoints_state.datapoint_vels
    blob_assignments = hdgmm_state.datapoints_state.blob_assignments
    hyperblob_assignments = hdgmm_state.blobs_state.hyperblob_assignments
    blob_means = hdgmm_state.blobs_state.blob_means  # [L, full_dim] - joint representation
    assigned_hyperblob_per_blob = hdgmm_state.hyperblobs_state[hyperblob_assignments]
    likelihood_blob_vel_covs = hdgmm_state.blobs_state.blob_vel_covs
    current_blob_vel_means = hdgmm_state.blobs_state.blob_vel_means

    # Extract spatial part from joint blob_means for velocity calculations
    blob_means_spatial = blob_means[:, :3]  # [L, 3] - spatial part only

    prior_blob_vel_means_ = assigned_hyperblob_per_blob.hyperblob_trans_vels + jnp.einsum(
        'nij,nj->ni', 
        assigned_hyperblob_per_blob.hyperblob_rot_vels - jnp.repeat(jnp.eye(3)[None, ...], hdgmm_state.hypers.n_blobs, axis=0), 
        blob_means_spatial - assigned_hyperblob_per_blob.hyperblob_means  # Use spatial part only
    )

    prior_variance = hdgmm_state.hypers.sigma_V

    # Count datapoints per blob
    n_blobs = hdgmm_state.hypers.n_blobs
    N_l = jax.ops.segment_sum(
        jnp.ones(datapoint_vels.shape[0], dtype=datapoint_vels.dtype), 
        blob_assignments,
        num_segments=n_blobs
    )  # [L]
    
    # Get posterior parameters
    posterior_mus, posterior_covs = normal_normal_posterior_full_cov_batched_flexible_prior(datapoint_vels, blob_assignments, prior_blob_vel_means_, prior_variance, likelihood_blob_vel_covs)
    
    # For blobs with no datapoints, set velocity to zero
    has_points = N_l > 0
    
    # Sample new blob velocity means for blobs with datapoints
    sampled_vel_means = genjax.mv_normal.sample(posterior_key, posterior_mus, posterior_covs)
    
    # Set velocity to zero for empty blobs
    zero_velocities = jnp.zeros_like(posterior_mus)
    posterior_blob_vel_means = jnp.where(has_points[:, None], sampled_vel_means, zero_velocities)
    
    return hdgmm_state.replace({'blobs_state': {'blob_vel_means': posterior_blob_vel_means}})

def gibbs_hyperblob_covs_dino(key, hdgmm_state, use_weighted_blobs = False):
    """Gibbs update for hyperblob covariances with joint position+feature representation.
    
    Extracts spatial part (first 3 dims) from joint blob_means for covariance calculations.
    Hyperblob covariances remain spatial-only [K, 3, 3].
    """
    posterior_key, _ = jax.random.split(key)

    blob_means = hdgmm_state.blobs_state.blob_means  # [L, full_dim] - joint representation
    hyperblob_assignments = hdgmm_state.blobs_state.hyperblob_assignments
    assumed_known_means = hdgmm_state.hyperblobs_state.hyperblob_means  # [K, 3] - spatial only
    prior_nu_H = hdgmm_state.hypers.nu_H
    prior_Psi_H = hdgmm_state.hypers.Psi_H

    n_hyperblobs = hdgmm_state.hypers.n_hyperblobs

    # Extract spatial part from joint blob_means for covariance calculations
    blob_means_spatial = blob_means[:, :3]  # [L, 3] - spatial part only

    # for weighted sums (i.e. weighted by number of datapoints assigned to each blob)
    blob_weights = hdgmm_state.blobs_state.blob_weights
    num_blobs = hdgmm_state.hypers.n_blobs

    blob_pseudocounts = jnp.where(use_weighted_blobs, blob_weights * num_blobs, jnp.ones_like(hyperblob_assignments, dtype=jnp.float32))

    # --- 1. Get number of blobs assigned to each hyperblob ---
    N_k = jax.ops.segment_sum(
        blob_pseudocounts, 
        hyperblob_assignments,
        num_segments=n_hyperblobs
    )  # [K]

    # --- 2. Compute normal NIW posterior (using spatial part only) ---
    nu_posteriors, Psi_posteriors = normal_inverse_wishart_posterior_by_cluster(
        blob_means_spatial, hyperblob_assignments, assumed_known_means, prior_nu_H, prior_Psi_H, blob_pseudocounts
    )

    # --- 3. Sample new covariances from posterior ---
    updated_hyperblob_covs = inverse_wishart.sample(posterior_key, nu_posteriors, Psi_posteriors)

    # --- 4. Mask: retain previous covariances if N_k <= 3 (3 min points per hyperblob for scatter matrix to be valid in NIW --> full rank) ---
    current_hyperblob_covs = hdgmm_state.hyperblobs_state.hyperblob_covs  # [K, 3, 3]
    mask = (N_k >= 3)[:, None, None]  # [K,1,1] to broadcast over 3x3 matrices

    final_hyperblob_covs = jnp.where(mask, updated_hyperblob_covs, current_hyperblob_covs)

    return hdgmm_state.replace({'hyperblobs_state': {'hyperblob_covs': final_hyperblob_covs}})

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

    def compute_local_density(point_idx):
        # JOINT likelihood: use datapoint_positions which contains position+features
        chm = (
            C["datapoint_position"].set(datapoint_positions[point_idx]) |
            C["datapoint_vel"].set(datapoint_vels[point_idx])
        )
        log_liks = jax.vmap(
            lambda i: blob_datapoint_likelihood_model_dino.assess(chm, (blobs_state[i],))[0]
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

# JOINT model wrappers for hyperblob Gibbs samplers
# These extract spatial part (first 3 dims) of blob_means before calling standard samplers

def gibbs_hyperblob_means_joint(key, hdgmm_state):
    """Extract spatial part of joint blob_means for hyperblob sampling."""
    spatial_means = hdgmm_state.blobs_state.blob_means[:, :3]
    temp_blobs = hdgmm_state.blobs_state.replace({'blob_means': spatial_means})
    temp_state = hdgmm_state.replace({'blobs_state': temp_blobs})
    updated_state = gibbs_hyperblob_means(key, temp_state)
    # Restore original joint blob_means
    return updated_state.replace({'blobs_state': hdgmm_state.blobs_state})

def gibbs_hyperblob_rot_joint(key, hdgmm_state):
    """Extract spatial part of joint blob_means for rotation sampling."""
    spatial_means = hdgmm_state.blobs_state.blob_means[:, :3]
    temp_blobs = hdgmm_state.blobs_state.replace({'blob_means': spatial_means})
    temp_state = hdgmm_state.replace({'blobs_state': temp_blobs})
    updated_state = gibbs_hyperblob_rot(key, temp_state)
    return updated_state.replace({'blobs_state': hdgmm_state.blobs_state})

def gibbs_hyperblob_trans_joint(key, hdgmm_state):
    """Extract spatial part of joint blob_means for translation sampling."""
    spatial_means = hdgmm_state.blobs_state.blob_means[:, :3]
    temp_blobs = hdgmm_state.blobs_state.replace({'blob_means': spatial_means})
    temp_state = hdgmm_state.replace({'blobs_state': temp_blobs})
    updated_state = gibbs_hyperblob_trans(key, temp_state)
    return updated_state.replace({'blobs_state': hdgmm_state.blobs_state})

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
        hdgmm_state = gibbs_blob_vel_means_dino(gibbs_key, hdgmm_state)
        return key, hdgmm_state

    def update_blob_velocity_covariances(i, carry):
        key, hdgmm_state = carry
        key, gibbs_key = jax.random.split(key)
        hdgmm_state = gibbs_blob_vel_covs(gibbs_key, hdgmm_state)
        return key, hdgmm_state

    def update_blob_means(i, carry):
        key, hdgmm_state = carry
        key, gibbs_key = jax.random.split(key)
        hdgmm_state = gibbs_blob_means_dino(gibbs_key, hdgmm_state)
        return key, hdgmm_state

    def hyperblob_update_loop(i, carry):
        key, hdgmm_state = carry
        key, gibbs_key = jax.random.split(key)
        hdgmm_state = gibbs_hyperblob_means_joint(gibbs_key, hdgmm_state)
        key, gibbs_key = jax.random.split(key)
        hdgmm_state = gibbs_hyperblob_covs_dino(gibbs_key, hdgmm_state)  # Extracts spatial part from joint blob_means
        key, gibbs_key = jax.random.split(key)
        hdgmm_state = gibbs_hyperblob_rot_joint(gibbs_key, hdgmm_state)
        key, gibbs_key = jax.random.split(key)
        hdgmm_state = gibbs_hyperblob_trans_joint(gibbs_key, hdgmm_state)
        return key, hdgmm_state

    key, hdgmm_state = jax.lax.fori_loop(0, 3, hyperblob_update_loop, (key, hdgmm_state))
    key, hdgmm_state = jax.lax.fori_loop(0, 3, update_blob_assignments_position_only, (key, hdgmm_state))
    key, hdgmm_state = jax.lax.fori_loop(0, 15, update_blob_means, (key, hdgmm_state))
    key, hdgmm_state = jax.lax.fori_loop(0, 3, update_blob_assignments_with_outlier, (key, hdgmm_state))
    key, hdgmm_state = jax.lax.fori_loop(0, 15, update_blob_velocities, (key, hdgmm_state))
    key, hdgmm_state = jax.lax.fori_loop(0, 15, update_blob_velocity_covariances, (key, hdgmm_state))
    key, hdgmm_state = jax.lax.fori_loop(0, 3, hyperblob_update_loop, (key, hdgmm_state))

    return hdgmm_state

def hdgmm_tracking_gibbs_dino(key, init_hdgmm_state, tracked_points,
                              tracked_motion_vectors, tracked_features, outlier_prob):
    """Track over time with DINO features."""
    init_hdgmm_state = init_hdgmm_state.replace({'hypers': {'outlier_prob': f_(outlier_prob)}})

    @jax.jit
    def f_tracking_sweep(carry, timestep_idx):
        key, hdgmm_state, tracked_points, tracked_motion_vectors, tracked_features = carry

        # JOINT: Update only spatial part with velocity, features stay same
        current_blob_means = hdgmm_state.blobs_state.blob_means
        next_spatial = current_blob_means[:, :3] + hdgmm_state.blobs_state.blob_vel_means
        next_blob_means = jnp.concatenate([next_spatial, current_blob_means[:, 3:]], axis=1)
        hdgmm_state = hdgmm_state.replace({'blobs_state': {'blob_means': next_blob_means}})

        # JOINT: Concatenate position and features for observations
        joint_observations = jnp.concatenate([
            tracked_points[timestep_idx],
            tracked_features[timestep_idx]
        ], axis=1)

        hdgmm_state = hdgmm_state.replace({
            'datapoints_state': {
                'datapoint_positions': joint_observations,  # JOINT
                'datapoint_vels': tracked_motion_vectors[timestep_idx],
                'datapoint_features': tracked_features[timestep_idx]  # For compatibility
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
        hdgmm_state = gibbs_blob_means_dino(gibbs_key, hdgmm_state)
        key, gibbs_key = jax.random.split(key)
        hdgmm_state = gibbs_blob_covs(gibbs_key, hdgmm_state)
        key, gibbs_key = jax.random.split(key)
        hdgmm_state = gibbs_blob_vel_means_dino(gibbs_key, hdgmm_state)
        key, gibbs_key = jax.random.split(key)
        hdgmm_state = gibbs_blob_vel_covs(gibbs_key, hdgmm_state)
        # NO gibbs_blob_features_dino - features are now part of joint blob_means/blob_covs
        return (key, hdgmm_state), hdgmm_TraceWrapper(force_retval=hdgmm_state)

    (key, final_state), traces = jax.lax.scan(gibbs_iteration, (key, hdgmm_state), jnp.arange(num_sweeps))
    return HDGMM_Gibbs_TraceWrapper(hdgmm_TraceWrapper(force_retval=hdgmm_state), traces)

# ============================================================================
# Evaluation Functions
# ============================================================================
#
# IMPORTANT: Two different metrics are computed:
#
# 1. PARTICLE-BASED METRICS (compute_error_rates) - PRIMARY METRIC
#    - Tracks discrete particle/blob location changes over time
#    - Identifies object vs background particles in frame 0
#    - FN Rate: % of object particles that left the GT mask
#    - FP Rate: % of background particles that entered the GT mask
#    - These are the metrics saved to JSON and reported in summaries
#
# 2. PIXEL-BASED METRICS (evaluate_single_davis_video in evaluation.py)
#    - Counts pixels weighted by fractional particle membership
#    - Recall/Precision/FPR calculated at pixel level with fractional contributions
#    - Used for visualization but NOT the primary evaluation metric
#
# The particle-based metrics are what we care about for tracking evaluation.
# ============================================================================

def compute_error_rates(tracking_data, segmentation_masks, object_hyperblob_idx, img_dims,
                       blob_counting_threshold=100, focal_length=520.0):
    """
    Compute PARTICLE-BASED false positive and false negative rates over time.

    This tracks discrete particle location changes, NOT pixel-level accuracy.
    See section comment above for the difference between particle-based and pixel-based metrics.
    """
    fx = fy = focal_length
    cx = img_dims[1] / 2.0
    cy = img_dims[0] / 2.0

    # Frame 0 setup
    frame0 = tracking_data[0]
    blob_assignments_frame0 = frame0['blob_assignments']
    hyperblob_assignments_frame0 = frame0['hyperblob_assignments']
    blob_means_frame0 = frame0['blob_means']
    n_blobs_frame0 = frame0['n_blobs']
    gt_mask_frame0 = segmentation_masks[0]

    blob_pixel_counts_frame0 = np.bincount(
        blob_assignments_frame0[blob_assignments_frame0 < n_blobs_frame0],
        minlength=n_blobs_frame0
    )
    significant_blobs = np.where(blob_pixel_counts_frame0 >= blob_counting_threshold)[0]

    object_blobs_frame0 = []
    background_blobs_frame0 = []
    for blob_idx in significant_blobs:
        if hyperblob_assignments_frame0[blob_idx] == object_hyperblob_idx:
            object_blobs_frame0.append(blob_idx)
        else:
            background_blobs_frame0.append(blob_idx)

    object_blobs_frame0 = np.array(object_blobs_frame0)
    background_blobs_frame0 = np.array(background_blobs_frame0)

    # Project and verify
    x_2d = (blob_means_frame0[:, 0] / (blob_means_frame0[:, 2] + 1e-8)) * fx + cx
    y_2d = (blob_means_frame0[:, 1] / (blob_means_frame0[:, 2] + 1e-8)) * fy + cy
    x_2d = np.clip(x_2d.astype(int), 0, img_dims[1] - 1)
    y_2d = np.clip(y_2d.astype(int), 0, img_dims[0] - 1)
    pixel_indices_frame0 = y_2d * img_dims[1] + x_2d

    object_blobs_on_mask = []
    for blob_idx in object_blobs_frame0:
        pixel_idx = pixel_indices_frame0[blob_idx]
        if pixel_idx < len(gt_mask_frame0) and gt_mask_frame0[pixel_idx]:
            object_blobs_on_mask.append(blob_idx)

    background_blobs_off_mask = []
    for blob_idx in background_blobs_frame0:
        pixel_idx = pixel_indices_frame0[blob_idx]
        if pixel_idx >= len(gt_mask_frame0) or not gt_mask_frame0[pixel_idx]:
            background_blobs_off_mask.append(blob_idx)

    object_blobs_on_mask = np.array(object_blobs_on_mask)
    background_blobs_off_mask = np.array(background_blobs_off_mask)
    n_object_blobs = len(object_blobs_on_mask)
    n_background_blobs = len(background_blobs_off_mask)

    # Track over time
    false_negative_rates = []
    false_positive_rates = []

    for frame_idx in range(len(tracking_data)):
        frame = tracking_data[frame_idx]
        blob_means = frame['blob_means']
        n_blobs = frame['n_blobs']
        gt_mask = segmentation_masks[frame_idx]

        x_2d = (blob_means[:, 0] / (blob_means[:, 2] + 1e-8)) * fx + cx
        y_2d = (blob_means[:, 1] / (blob_means[:, 2] + 1e-8)) * fy + cy
        x_2d = np.clip(x_2d.astype(int), 0, img_dims[1] - 1)
        y_2d = np.clip(y_2d.astype(int), 0, img_dims[0] - 1)
        pixel_indices = y_2d * img_dims[1] + x_2d

        tp_count = 0
        for blob_idx in object_blobs_on_mask:
            if blob_idx < n_blobs:
                pixel_idx = pixel_indices[blob_idx]
                if pixel_idx < len(gt_mask) and gt_mask[pixel_idx]:
                    tp_count += 1

        fn_count = n_object_blobs - tp_count
        fn_rate = (fn_count / n_object_blobs) * 100 if n_object_blobs > 0 else 0.0

        fp_count = 0
        for blob_idx in background_blobs_off_mask:
            if blob_idx < n_blobs:
                pixel_idx = pixel_indices[blob_idx]
                if pixel_idx < len(gt_mask) and gt_mask[pixel_idx]:
                    fp_count += 1

        fp_rate = (fp_count / n_background_blobs) * 100 if n_background_blobs > 0 else 0.0

        false_negative_rates.append(fn_rate)
        false_positive_rates.append(fp_rate)

    return {
        'false_negative_rates': false_negative_rates,
        'false_positive_rates': false_positive_rates,
        'mean_fn_rate': np.mean(false_negative_rates),
        'mean_fp_rate': np.mean(false_positive_rates),
        'n_object_blobs': n_object_blobs,
        'n_background_blobs': n_background_blobs
    }

def plot_error_rates(error_results, video_name, save_path):
    """Plot and save error rate visualization."""
    fn_rates = error_results['false_negative_rates']
    fp_rates = error_results['false_positive_rates']
    mean_fn = error_results['mean_fn_rate']
    mean_fp = error_results['mean_fp_rate']

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(range(len(fn_rates)), fn_rates, linewidth=2, color='red',
            label=f'False Negative Rate (mean: {mean_fn:.2f}%)', alpha=0.8)
    ax.plot(range(len(fp_rates)), fp_rates, linewidth=2, color='blue',
            label=f'False Positive Rate (mean: {mean_fp:.2f}%)', alpha=0.8)
    ax.set_xlabel('Frame', fontsize=12)
    ax.set_ylabel('Rate (%)', fontsize=12)
    ax.set_title(f'{video_name} - Blob Tracking Error Rates', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_ylim([0, 105])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def create_3wide_video(tracking_data, video_name, rgb_path, img_dims, hyperblob_idx,
                       num_blobs, save_path):
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

    for frame_idx in range(num_frames):
        rgb_frame = cv2.imread(rgb_files[frame_idx])
        rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)

        frame = tracking_data[frame_idx]
        blob_assignments = frame['blob_assignments']
        hyperblob_assignments = frame['hyperblob_assignments']
        n_blobs = frame['n_blobs']

        valid_mask = blob_assignments < n_blobs
        datapoint_hyperblob_assignments = np.full(len(blob_assignments), -1, dtype=int)
        datapoint_hyperblob_assignments[valid_mask] = hyperblob_assignments[blob_assignments[valid_mask]]

        binary_mask = np.where(datapoint_hyperblob_assignments == hyperblob_idx, 255, 0).astype(np.uint8)
        binary_image = binary_mask.reshape(img_height, img_width)
        segmentation_frame = np.stack([binary_image, binary_image, binary_image], axis=-1)

        background_color = np.array([128, 128, 128], dtype=np.uint8)
        object_blob_mask = np.zeros(len(blob_assignments), dtype=bool)
        for i in range(len(blob_assignments)):
            if valid_mask[i]:
                blob_idx = blob_assignments[i]
                if hyperblob_assignments[blob_idx] == hyperblob_idx:
                    object_blob_mask[i] = True

        blob_assignment_frame = np.zeros((len(blob_assignments), 3), dtype=np.uint8)
        blob_assignment_frame[~object_blob_mask] = background_color
        blob_assignment_frame[object_blob_mask] = blob_colors[blob_assignments[object_blob_mask]]
        blob_assignment_frame = blob_assignment_frame.reshape(img_height, img_width, 3)

        rgb_h, rgb_w = rgb_frame.shape[:2]
        if (rgb_h, rgb_w) != (img_height, img_width):
            segmentation_frame = cv2.resize(segmentation_frame, (rgb_w, rgb_h), interpolation=cv2.INTER_NEAREST)
            blob_assignment_frame = cv2.resize(blob_assignment_frame, (rgb_w, rgb_h), interpolation=cv2.INTER_NEAREST)

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

    import tempfile
    import subprocess
    import shutil

    # Create temporary directory for frames
    temp_dir = tempfile.mkdtemp()
    try:
        # Write frames as PNGs
        for i, frame in enumerate(combined_frames):
            frame_path = os.path.join(temp_dir, f"frame_{i:05d}.png")
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(frame_path, frame_bgr)

        # Use ffmpeg to create H.264 video
        # This is safe to call after JAX initialization since all compute is done
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-framerate', '30',
            '-i', os.path.join(temp_dir, 'frame_%05d.png'),
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
    
    # Force garbage collection multiple times to ensure cleanup
    # Run multiple times because some objects may have circular references
    for _ in range(5):  # Increased from 3 to 5 for more thorough cleanup
        gc.collect()

# ============================================================================
# Main Processing Function
# ============================================================================

def process_video(video_name):
    """Process a single video and return results."""
    print(f"\n{'='*80}")
    print(f"Processing: {video_name}")
    print(f"{'='*80}")

    try:
        # Load data
        dino_path = DINO_PATH_TEMPLATE.format(video_name)
        if not os.path.exists(dino_path):
            print(f"DINO features not found: {dino_path}")
            return None

        pca_data = np.load(dino_path)
        pca_features_unnormalized = pca_data['pca_features_unnormalized']
        gaussian_means = pca_data['gaussian_means'][:NUM_PCA_COMPONENTS]
        gaussian_stds = pca_data['gaussian_stds'][:NUM_PCA_COMPONENTS]
        # Close the npz file to free memory
        pca_data.close()

        tracked_points, tracked_motion_vectors, num_data_tsteps, img_dims = \
            extract_3d_points_and_motion_vectors_data(DAVIS_3D_MOTION_PATH, video_name)

        # Limit to first 90 frames for jello-trim
        if video_name == "jello_trim":
            num_data_tsteps = min(num_data_tsteps, 90)
            tracked_points = tracked_points[:num_data_tsteps]
            tracked_motion_vectors = tracked_motion_vectors[:num_data_tsteps]
            print(f"Limited to first {num_data_tsteps} frames for jello_trim")

        tracked_features = extract_dino_features(video_name, pca_features_unnormalized, img_dims, num_data_tsteps, max_components=NUM_PCA_COMPONENTS)

        first_frame_seg = get_segmentation_mask(
            video_name, 0, DAVIS_SEGMASKS_PATH, img_dims=img_dims, flatten=True
        )

        # Initialize
        kmeans_chm, roi_blob_indices, roi_hyperblob_indices = initialize_model_with_dino(
            tracked_points, NUM_BLOBS, NUM_HYPERBLOBS,
            first_frame_seg, tracked_motion_vectors, tracked_features
        )

        # Create hyperparameters
        num_datapoints = kmeans_chm['datapoints', 'datapoint_positions'].shape[0]
        num_blobs = kmeans_chm['blobs', 'hyperblob_assignments'].shape[0]
        num_hyperblobs = kmeans_chm['hyperblobs', 'hyperblob_means'].shape[0]

        # empirical_mu_H should be spatial only (3D), not joint
        empirical_mu_H = jnp.median(kmeans_chm['datapoints', 'datapoint_positions'][:, :3], axis=0)
        empirical_sigma_H = (10 * 0.5) ** 2
        empirical_Psi_B = jnp.median(kmeans_chm['blobs', 'blob_covs'][roi_blob_indices], axis=0)
        empirical_Psi_H = jnp.median(kmeans_chm['hyperblobs', 'hyperblob_covs'][roi_hyperblob_indices], axis=0)
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
            sigma_F=f_(20.0),
            outlier_prob=f_(5e0),
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

        # Post-Gibbs filtering
        fx = fy = FOCAL_LENGTH
        cx = img_dims[1] / 2.0
        cy = img_dims[0] / 2.0

        blob_means_after_gibbs = np.array(init_hdgmm_state.blobs_state.blob_means)
        hyperblob_assignments_after_gibbs = np.array(init_hdgmm_state.blobs_state.hyperblob_assignments)

        # Extract spatial part (first 3 dims) from joint representation
        blob_means_spatial = blob_means_after_gibbs[:, :3]  # [L, 3]

        x_2d = (blob_means_spatial[:, 0] / (blob_means_spatial[:, 2] + 1e-8)) * fx + cx
        y_2d = (blob_means_spatial[:, 1] / (blob_means_spatial[:, 2] + 1e-8)) * fy + cy
        
        # Handle NaN/inf values before casting to avoid warnings
        x_2d = np.nan_to_num(x_2d, nan=0.0, posinf=img_dims[1]-1, neginf=0.0)
        y_2d = np.nan_to_num(y_2d, nan=0.0, posinf=img_dims[0]-1, neginf=0.0)
        
        x_2d = np.clip(x_2d.astype(int), 0, img_dims[1] - 1)
        y_2d = np.clip(y_2d.astype(int), 0, img_dims[0] - 1)
        pixel_indices = y_2d * img_dims[1] + x_2d

        blob_assignments_frame0 = np.array(init_hdgmm_state.datapoints_state.blob_assignments)
        n_blobs_after_gibbs = len(blob_means_after_gibbs)
        valid_mask = blob_assignments_frame0 < n_blobs_after_gibbs
        datapoint_to_hyperblob = np.full(len(blob_assignments_frame0), -1, dtype=int)
        datapoint_to_hyperblob[valid_mask] = hyperblob_assignments_after_gibbs[blob_assignments_frame0[valid_mask]]

        hyperblob_overlaps = {}
        for hb_idx in range(NUM_HYPERBLOBS):
            hb_mask = datapoint_to_hyperblob == hb_idx
            overlap = np.sum(hb_mask & (first_frame_seg == 1))
            hyperblob_overlaps[hb_idx] = overlap

        object_hyperblob_idx = max(hyperblob_overlaps, key=hyperblob_overlaps.get)

        blobs_in_object = np.where(hyperblob_assignments_after_gibbs == object_hyperblob_idx)[0]
        blobs_to_reassign = []
        for blob_idx in blobs_in_object:
            pixel_idx = pixel_indices[blob_idx]
            if pixel_idx >= len(first_frame_seg) or not first_frame_seg[pixel_idx]:
                blobs_to_reassign.append(blob_idx)

        if len(blobs_to_reassign) > 0:
            background_hyperblobs = [hb for hb in range(NUM_HYPERBLOBS) if hb != object_hyperblob_idx]
            background_blob_counts = {hb: np.sum(hyperblob_assignments_after_gibbs == hb) for hb in background_hyperblobs}
            target_background_hb = max(background_blob_counts, key=background_blob_counts.get)
            updated_hyperblob_assignments = jnp.array(hyperblob_assignments_after_gibbs)
            for blob_idx in blobs_to_reassign:
                updated_hyperblob_assignments = updated_hyperblob_assignments.at[blob_idx].set(target_background_hb)
            init_hdgmm_state = init_hdgmm_state.replace({
                'blobs_state': {'hyperblob_assignments': updated_hyperblob_assignments}
            })

        # Track over time
        key, tracking_key = jax.random.split(key)
        print("Running tracking...")
        tracking_wtrs = hdgmm_tracking_gibbs_dino(
            tracking_key, init_hdgmm_state,
            tracked_points, tracked_motion_vectors, tracked_features,
            outlier_prob=1e-40
        )

        # Extract results
        tracking_data = []
        for frame_idx in range(len(tracking_wtrs)):
            frame = tracking_wtrs[frame_idx]
            frame_data = {
                'n_blobs': frame.retval.hypers.n_blobs,
                'n_hyperblobs': frame.retval.hypers.n_hyperblobs,
                'n_datapoints': frame.retval.hypers.n_datapoints,
                'blob_assignments': np.array(frame.retval.datapoints_state.blob_assignments),
                'datapoint_positions': np.array(frame.retval.datapoints_state.datapoint_positions),
                'datapoint_vels': np.array(frame.retval.datapoints_state.datapoint_vels),
                'datapoint_features': np.array(frame.retval.datapoints_state.datapoint_features),
                'blob_weights': np.array(frame.retval.blobs_state.blob_weights),
                'blob_means': np.array(frame.retval.blobs_state.blob_means),
                'blob_covs': np.array(frame.retval.blobs_state.blob_covs),
                'blob_vel_means': np.array(frame.retval.blobs_state.blob_vel_means),
                'blob_vel_covs': np.array(frame.retval.blobs_state.blob_vel_covs),
                'blob_features': np.array(frame.retval.blobs_state.blob_means[:, 3:]),  # Extract features from joint
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
            segmentation_masks.append(seg_mask)

        error_results = compute_error_rates(
            tracking_data, segmentation_masks, object_hyperblob_idx, img_dims,
            BLOB_COUNTING_THRESHOLD, FOCAL_LENGTH
        )

        # Get particle accuracy
        all_results = {video_name: [tracking_data]}
        experiment_metrics, best_visualization_data = evaluate_single_davis_video(
            davis_name=video_name,
            multiple_genparticles_list=all_results[video_name],
            annotations_path=DAVIS_SEGMASKS_PATH,
            counting_threshold=100,
            img_dims=img_dims,
            fps_list=None,
            render_results_video=False,
            experiment_save_dir=None
        )

        result = {
            'video_name': video_name,
            'error_results': error_results,
            'particle_accuracy': experiment_metrics,
            'tracking_data': tracking_data,
            'object_hyperblob_idx': object_hyperblob_idx,
            'img_dims': img_dims
        }

        print(f"Completed: {video_name}")
        print(f"\n  Particle-Based Error Rates (unweighted, discrete particle tracking):")
        print(f"    Mean FN Rate: {error_results['mean_fn_rate']:.2f}%  (object particles leaving mask)")
        print(f"    Mean FP Rate: {error_results['mean_fp_rate']:.2f}%  (background particles entering mask)")
        print(f"    Note: These are NON-WEIGHTED - treat all particles equally regardless of size/importance")

        print(f"\n  Matter-Weighted Metrics (Adaptive - per-frame weights):")
        print(f"    Recall:     {experiment_metrics['avg_matter_weighted_recall']:.3f}  (object matter staying in mask)")
        print(f"    Precision:  {experiment_metrics['avg_matter_weighted_precision']:.3f}  (predicted matter correctness)")
        print(f"    F1:         {experiment_metrics['avg_matter_weighted_f1']:.3f}  (harmonic mean)")

        print(f"\n  Matter-Weighted Metrics (Fixed - frame 0 weights, for fair comparison):")
        print(f"    Recall:     {experiment_metrics['avg_matter_weighted_recall_fixed']:.3f}  (object matter staying in mask)")
        print(f"    Precision:  {experiment_metrics['avg_matter_weighted_precision_fixed']:.3f}  (predicted matter correctness)")
        print(f"    F1:         {experiment_metrics['avg_matter_weighted_f1_fixed']:.3f}  (harmonic mean)")

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
        frame0_blob_assignments = np.array(tracking_data[0]['blob_assignments']).reshape(result['img_dims'])
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
    print(f"{'='*80}\n")

    for video_name in tqdm(VIDEO_NAMES, desc="Overall Progress", position=0):
        result = process_video(video_name)

        if result is not None:
            all_results.append(result)

            # Create subdirectories for outputs
            plots_dir = os.path.join(EXPERIMENT_SAVE_DIR, "error_rate_plots")
            videos_dir = os.path.join(EXPERIMENT_SAVE_DIR, "3wide_videos")
            os.makedirs(plots_dir, exist_ok=True)
            os.makedirs(videos_dir, exist_ok=True)

            # Save error rate plot
            plot_path = os.path.join(plots_dir, f"{video_name}_error_rates.png")
            plot_error_rates(result['error_results'], video_name, plot_path)
            print(f"Saved error rate plot: {plot_path}")

            # Save 3-wide video
            # video_path = os.path.join(videos_dir, f"{video_name}_3wide_synchronized.mp4")
            # create_3wide_video(
            #     result['tracking_data'], video_name, DAVIS_RGB_PATH,
            #     result['img_dims'], result['object_hyperblob_idx'],
            #     NUM_BLOBS, video_path
            # )
            # print(f"Saved 3-wide video: {video_path}")

            # Compute diagnostics for JSON
            tracking_data = result['tracking_data']
            frame0_blob_weights = np.array(tracking_data[0]['blob_weights'])
            weight_entropy = -np.sum(frame0_blob_weights * np.log(frame0_blob_weights + 1e-10))
            max_entropy = np.log(len(frame0_blob_weights))
            normalized_entropy = float(weight_entropy / max_entropy)

            frame0_blob_means = np.array(tracking_data[0]['blob_means'])
            frame0_blob_assignments = np.array(tracking_data[0]['blob_assignments']).reshape(result['img_dims'])
            frame0_n_blobs = tracking_data[0]['n_blobs']
            blob_pixel_counts = np.bincount(
                frame0_blob_assignments.flatten()[frame0_blob_assignments.flatten() < frame0_n_blobs],
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
                result['particle_accuracy']['avg_matter_weighted_f1_fixed'] -
                result['particle_accuracy']['avg_matter_weighted_f1']
            )

            # Store accuracy with both adaptive and fixed metrics
            all_accuracies[video_name] = {
                # Full particle accuracy dict (contains all trial data)
                'particle_accuracy': result['particle_accuracy'],

                # Unweighted error rates
                'mean_fn_rate': result['error_results']['mean_fn_rate'],
                'mean_fp_rate': result['error_results']['mean_fp_rate'],
                'n_object_blobs': result['error_results']['n_object_blobs'],
                'n_background_blobs': result['error_results']['n_background_blobs'],

                # Matter-weighted metrics (Adaptive)
                'matter_weighted_recall_adaptive': result['particle_accuracy']['avg_matter_weighted_recall'],
                'matter_weighted_precision_adaptive': result['particle_accuracy']['avg_matter_weighted_precision'],
                'matter_weighted_f1_adaptive': result['particle_accuracy']['avg_matter_weighted_f1'],

                # Matter-weighted metrics (Fixed - for fair comparison)
                'matter_weighted_recall_fixed': result['particle_accuracy']['avg_matter_weighted_recall_fixed'],
                'matter_weighted_precision_fixed': result['particle_accuracy']['avg_matter_weighted_precision_fixed'],
                'matter_weighted_f1_fixed': result['particle_accuracy']['avg_matter_weighted_f1_fixed'],

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
            json_results_dir = os.path.join(EXPERIMENT_SAVE_DIR, "json_results")
            os.makedirs(json_results_dir, exist_ok=True)
            per_run_json_path = os.path.join(json_results_dir, f"{video_name}_results.json")
            with open(per_run_json_path, 'w') as f:
                json.dump(all_accuracies[video_name], f, indent=2)
            print(f"Saved per-run JSON: {per_run_json_path}")

            # Save aggregated JSON after each video completes (incremental updates)
            json_path = os.path.join(EXPERIMENT_SAVE_DIR, "all_videos_experiment_results.json")

            # Read existing results if file exists, then merge with new results
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
            # Since 3wide video is disabled, we don't need to keep it in memory
            # The metrics and error_results are already saved to JSON
            del result['tracking_data']
            del tracking_data
            
            # Clean up local variables used for diagnostics
            del frame0_blob_weights, frame0_blob_means, frame0_blob_assignments
            del blob_pixel_counts, reference_particle_indices

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

    print(f"\n{'='*80}")
    print(f"SUMMARY - {len(all_accuracies_from_json)} Videos")
    print(f"{'='*80}")
    print(f"\nMetric Interpretation Guide:")
    print(f"  • FN/FP Rates: UNWEIGHTED discrete particle counts (treat all particles equally)")
    print(f"  • Matter-Weighted: Weight by (pixel_count × blob_weight) to emphasize important particles")
    print(f"  • Fixed-weight: Use frame-0 weights for fair comparison with point trackers")
    print(f"  • Adaptive-weight: Use per-frame weights to show probabilistic representation benefit")
    print(f"\n")

    # Compute aggregate statistics
    mean_fn_rates = [m['mean_fn_rate'] for m in all_accuracies_from_json.values()]
    mean_fp_rates = [m['mean_fp_rate'] for m in all_accuracies_from_json.values()]
    recall_adaptive = [m['matter_weighted_recall_adaptive'] for m in all_accuracies_from_json.values()]
    precision_adaptive = [m['matter_weighted_precision_adaptive'] for m in all_accuracies_from_json.values()]
    f1_adaptive = [m['matter_weighted_f1_adaptive'] for m in all_accuracies_from_json.values()]
    recall_fixed = [m['matter_weighted_recall_fixed'] for m in all_accuracies_from_json.values()]
    precision_fixed = [m['matter_weighted_precision_fixed'] for m in all_accuracies_from_json.values()]
    f1_fixed = [m['matter_weighted_f1_fixed'] for m in all_accuracies_from_json.values()]

    print(f"AGGREGATE STATISTICS ACROSS ALL VIDEOS:")
    print(f"\n  Unweighted Error Rates:")
    print(f"    Mean FN Rate:            {np.mean(mean_fn_rates):.2f}% ± {np.std(mean_fn_rates):.2f}%")
    print(f"    Mean FP Rate:            {np.mean(mean_fp_rates):.2f}% ± {np.std(mean_fp_rates):.2f}%")

    print(f"\n  Matter-Weighted (Adaptive):")
    print(f"    Recall:                  {np.mean(recall_adaptive):.3f} ± {np.std(recall_adaptive):.3f}")
    print(f"    Precision:               {np.mean(precision_adaptive):.3f} ± {np.std(precision_adaptive):.3f}")
    print(f"    F1:                      {np.mean(f1_adaptive):.3f} ± {np.std(f1_adaptive):.3f}")

    print(f"\n  Matter-Weighted (Fixed - for fair comparison):")
    print(f"    Recall:                  {np.mean(recall_fixed):.3f} ± {np.std(recall_fixed):.3f}")
    print(f"    Precision:               {np.mean(precision_fixed):.3f} ± {np.std(precision_fixed):.3f}")
    print(f"    F1:                      {np.mean(f1_fixed):.3f} ± {np.std(f1_fixed):.3f}")

    print(f"\n\nPER-VIDEO RESULTS:")
    print(f"{'='*80}\n")

    for video_name, metrics in all_accuracies_from_json.items():
        print(f"  {video_name}:")
        print(f"    Unweighted Error Rates:")
        print(f"      Mean FN Rate:            {metrics['mean_fn_rate']:.2f}%  (particles leaving mask, unweighted)")
        print(f"      Mean FP Rate:            {metrics['mean_fp_rate']:.2f}%  (particles entering mask, unweighted)")

        print(f"\n    Matter-Weighted (Adaptive):")
        print(f"      Recall:                  {metrics['matter_weighted_recall_adaptive']:.3f}  (object matter staying in mask)")
        print(f"      Precision:               {metrics['matter_weighted_precision_adaptive']:.3f}  (predicted matter correctness)")
        print(f"      F1:                      {metrics['matter_weighted_f1_adaptive']:.3f}  (harmonic mean)")

        print(f"\n    Matter-Weighted (Fixed - for fair comparison):")
        print(f"      Recall:                  {metrics['matter_weighted_recall_fixed']:.3f}  (object matter staying in mask)")
        print(f"      Precision:               {metrics['matter_weighted_precision_fixed']:.3f}  (predicted matter correctness)")
        print(f"      F1:                      {metrics['matter_weighted_f1_fixed']:.3f}  (harmonic mean)")
        print(f"")
