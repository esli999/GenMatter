# DINO Tracking Visualization for Single Video (dog)
# Run tracking on 'dog' video and create visualization

import os
import json
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
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from genparticles.datatypes import *
from genparticles.model_3d import *
from genparticles.inference import *
from genparticles.dataloader import *
from genparticles.utils import *
from genparticles.evaluation import *

import genjax
from genjax import Const, gen, Pytree

# Import the model and inference functions from dino_tracking.py
import sys
sys.path.insert(0, '/home/esli/GenParticles_NeurIPS')
from dino_tracking import (
    HDGMM_Hyperparams_DINO,
    HDGMM_Blobs_State_DINO,
    HDGMM_Datapoints_State_DINO,
    HDGMM_State_DINO,
    HDGMM_model_dino,
    extract_dino_features,
    initialize_model_with_dino,
    init_gibbs_sweep_dino,
    hdgmm_tracking_gibbs_dino,
    compute_error_rates,
    plot_error_rates,
    create_3wide_video,
    model_jimportance
)

# ============================================================================
# Configuration
# ============================================================================

VIDEO_NAME = "dog"

DAVIS_3D_MOTION_PATH = "/home/esli/GenMatter/assets/cvpr_deformable_experiment/npzs"
DAVIS_SEGMASKS_PATH = "/home/esli/GenMatter/assets/cvpr_deformable_experiment/segmasks"
DAVIS_RGB_PATH = "/home/esli/GenMatter/assets/cvpr_deformable_experiment/frames"
DINO_PATH_TEMPLATE = '/home/esli/GenMatter/assets/cvpr_deformable_experiment/dino_features/dino_features_pca_l/{}_dino_pca_per_pixel.npz'

OUTPUT_DIR = "/home/esli/GenParticles_NeurIPS/dog_tracking_results"

NUM_BLOBS = 500
NUM_HYPERBLOBS = 9
FOCAL_LENGTH = 520.0
BLOB_COUNTING_THRESHOLD = 100
RANDOM_SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# Main Processing
# ============================================================================

print(f"{'='*80}")
print(f"Processing: {VIDEO_NAME}")
print(f"{'='*80}\n")

# Load data
dino_path = DINO_PATH_TEMPLATE.format(VIDEO_NAME)
if not os.path.exists(dino_path):
    print(f"ERROR: DINO features not found: {dino_path}")
    sys.exit(1)

pca_data = np.load(dino_path)
pca_features_unnormalized = pca_data['pca_features_unnormalized']
gaussian_means = pca_data['gaussian_means']
gaussian_stds = pca_data['gaussian_stds']

tracked_points, tracked_motion_vectors, num_data_tsteps, img_dims = \
    extract_3d_points_and_motion_vectors_data(DAVIS_3D_MOTION_PATH, VIDEO_NAME)

tracked_features = extract_dino_features(VIDEO_NAME, pca_features_unnormalized, img_dims, num_data_tsteps)

first_frame_seg = get_segmentation_mask(
    VIDEO_NAME, 0, DAVIS_SEGMASKS_PATH, img_dims=img_dims, flatten=True
)

print(f"Loaded data:")
print(f"  Frames: {num_data_tsteps}")
print(f"  Image dims: {img_dims}")
print(f"  DINO feature dims: {tracked_features.shape}")

# Initialize
print("\nInitializing model...")
kmeans_chm, roi_blob_indices, roi_hyperblob_indices = initialize_model_with_dino(
    tracked_points, NUM_BLOBS, NUM_HYPERBLOBS,
    first_frame_seg, tracked_motion_vectors, tracked_features
)

# Create hyperparameters
num_datapoints = kmeans_chm['datapoints', 'datapoint_positions'].shape[0]
num_blobs = kmeans_chm['blobs', 'hyperblob_assignments'].shape[0]
num_hyperblobs = kmeans_chm['hyperblobs', 'hyperblob_means'].shape[0]

empirical_mu_H = jnp.median(kmeans_chm['datapoints', 'datapoint_positions'], axis=0)
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

# Initialize model
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

x_2d = (blob_means_after_gibbs[:, 0] / (blob_means_after_gibbs[:, 2] + 1e-8)) * fx + cx
y_2d = (blob_means_after_gibbs[:, 1] / (blob_means_after_gibbs[:, 2] + 1e-8)) * fy + cy
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
print(f"Identified object hyperblob: {object_hyperblob_idx}")

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
print("Extracting tracking results...")
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
        VIDEO_NAME, frame_idx, DAVIS_SEGMASKS_PATH, img_dims=img_dims, flatten=True
    )
    segmentation_masks.append(seg_mask)

error_results = compute_error_rates(
    tracking_data, segmentation_masks, object_hyperblob_idx, img_dims,
    BLOB_COUNTING_THRESHOLD, FOCAL_LENGTH
)

# Get particle accuracy
all_results = {VIDEO_NAME: [tracking_data]}
experiment_metrics, best_visualization_data = evaluate_single_davis_video(
    davis_name=VIDEO_NAME,
    multiple_genparticles_list=all_results[VIDEO_NAME],
    annotations_path=DAVIS_SEGMASKS_PATH,
    counting_threshold=100,
    img_dims=img_dims,
    fps_list=None,
    render_results_video=False,
    experiment_save_dir=None
)

# ============================================================================
# Save Results and Visualizations
# ============================================================================

print("\n" + "="*80)
print("RESULTS")
print("="*80)

print(f"\nParticle-Based Error Rates:")
print(f"  Mean FN Rate: {error_results['mean_fn_rate']:.2f}%")
print(f"  Mean FP Rate: {error_results['mean_fp_rate']:.2f}%")

print(f"\nMatter-Weighted Metrics (Adaptive):")
print(f"  Recall:     {experiment_metrics['avg_matter_weighted_recall']:.3f}")
print(f"  Precision:  {experiment_metrics['avg_matter_weighted_precision']:.3f}")
print(f"  F1:         {experiment_metrics['avg_matter_weighted_f1']:.3f}")

print(f"\nMatter-Weighted Metrics (Fixed):")
print(f"  Recall:     {experiment_metrics['avg_matter_weighted_recall_fixed']:.3f}")
print(f"  Precision:  {experiment_metrics['avg_matter_weighted_precision_fixed']:.3f}")
print(f"  F1:         {experiment_metrics['avg_matter_weighted_f1_fixed']:.3f}")

# Save error rate plot
plot_path = os.path.join(OUTPUT_DIR, f"{VIDEO_NAME}_error_rates.png")
plot_error_rates(error_results, VIDEO_NAME, plot_path)
print(f"\nSaved error rate plot: {plot_path}")

# Save 3-wide video
video_path = os.path.join(OUTPUT_DIR, f"{VIDEO_NAME}_3wide_synchronized.mp4")
print(f"\nCreating 3-wide visualization video...")
create_3wide_video(
    tracking_data, VIDEO_NAME, DAVIS_RGB_PATH,
    img_dims, object_hyperblob_idx,
    NUM_BLOBS, video_path
)
print(f"Saved 3-wide video: {video_path}")

# Save JSON results
results_dict = {
    'video_name': VIDEO_NAME,
    'error_results': {
        'mean_fn_rate': float(error_results['mean_fn_rate']),
        'mean_fp_rate': float(error_results['mean_fp_rate']),
        'false_negative_rates': [float(x) for x in error_results['false_negative_rates']],
        'false_positive_rates': [float(x) for x in error_results['false_positive_rates']],
        'n_object_blobs': int(error_results['n_object_blobs']),
        'n_background_blobs': int(error_results['n_background_blobs'])
    },
    'matter_weighted_metrics_adaptive': {
        'recall': float(experiment_metrics['avg_matter_weighted_recall']),
        'precision': float(experiment_metrics['avg_matter_weighted_precision']),
        'f1': float(experiment_metrics['avg_matter_weighted_f1'])
    },
    'matter_weighted_metrics_fixed': {
        'recall': float(experiment_metrics['avg_matter_weighted_recall_fixed']),
        'precision': float(experiment_metrics['avg_matter_weighted_precision_fixed']),
        'f1': float(experiment_metrics['avg_matter_weighted_f1_fixed'])
    }
}

json_path = os.path.join(OUTPUT_DIR, f"{VIDEO_NAME}_results.json")
with open(json_path, 'w') as f:
    json.dump(results_dict, f, indent=2)
print(f"Saved JSON results: {json_path}")

print(f"\n{'='*80}")
print("COMPLETE")
print(f"{'='*80}")
print(f"All results saved to: {OUTPUT_DIR}")
