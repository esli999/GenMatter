"""
Re-run the top trials most affected by depth ablation with detailed logging.
This will help us understand WHY removing depth breaks the GenParticles model.
"""
import os
import sys
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import jax
import jax.numpy as jnp
from jax.random import key as jkey
from jax.scipy.ndimage import map_coordinates

# Import from gestalt_experiment_algorithm
from gestalt_experiment_algorithm import (
    HDGMM_model_3d, resize_to_square, load_gestalt_data,
    compute_3d_points_and_motion, extract_gestalt_segmentation
)

from genparticles.utils import make_hierarchical_kmeans_chm_with_mask_fixed_hyperblob

from genparticles.datatypes import *
from genparticles.model_3d import *
from genparticles.inference import *
from genparticles.dataloader import *
from genparticles.utils import *

from genjax import ChoiceMapBuilder as C

# JIT compile model functions
model_jsimulate = jax.jit(HDGMM_model_3d.simulate)
model_jimportance = jax.jit(HDGMM_model_3d.importance)

# Configuration
OUTPUT_DIR = "/home/esli/GenMatter/depth_ablation_detailed_analysis"
GESTALT_BASE_PATH = '/home/esli/GenMatter/assets/from_thomas'
RAFT_FLOWS_PATH = '/home/esli/GenMatter/raft_flows'

# Top 5 trials most affected by depth ablation
TOP_TRIALS = [
    ('scene_00003', 'texture_21'),
    ('scene_00013', 'texture_13'),
    ('scene_00004', 'texture_07'),
    ('scene_00014', 'texture_07'),
    ('scene_00017', 'texture_16'),
]

# Experiment parameters
RANDOM_SEED = 42
FOCAL_LENGTH_SCALE = 2.0
Z_SOFTENING_SCALE = 1.5  # Scale factor for Z variance when depth is ablated

# Hyperparameters
HYPERPARAMS = {
    "number_of_blobs": 100,
    "number_of_hyperblobs": 5,
}

# Gibbs sampling parameters
NUM_INITIAL_GIBBS_ITERATIONS = 50
NUM_GIBBS_INNER_LOOPS = 5
NUM_TRACKING_GIBBS_ITERATIONS = 500
NUM_VELOCITY_UPDATE_ITERATIONS = 20

# Gibbs dials
GIBBS_DIALS = {
    "blob_weights": True,
    "hyperblob_weights": False,
    "blob_assignments": True,
    "hyperblob_assignments": False,
    "hyperblob_covs": False,
    "blob_covs": True,
    "blob_vel_covs": True,
    "blob_vel_means": True,
    "hyperblob_means": False,
    "blob_means": True,
    'hyperblob_rot_vels': True,
    'hyperblob_trans_vels': True
}

VELOCITY_UPDATE_DIALS = {
    "blob_weights": False,
    "hyperblob_weights": False,
    "blob_assignments": True,
    "hyperblob_assignments": False,
    "hyperblob_covs": False,
    "blob_covs": False,
    "blob_vel_covs": True,
    "blob_vel_means": True,
    "hyperblob_means": True,
    "blob_means": True,
    'hyperblob_rot_vels': True,
    'hyperblob_trans_vels': True
}


def load_six_frame_gestalt_data(scene, texture, use_depth=True):
    """Load depth and flow data for first 6 frames

    Args:
        use_depth: If True, use real depth. If False, use constant depth (ablation)
    """
    depth_file = os.path.join(GESTALT_BASE_PATH, scene, texture, 'output_six_frame_depths.npz')
    if not os.path.exists(depth_file):
        raise FileNotFoundError(f"Depth file not found: {depth_file}")

    depth_data = np.load(depth_file)
    if 'depths' in depth_data:
        inv_rel_depth = depth_data['depths'][:6]
    else:
        inv_rel_depth = next(iter(depth_data.values()))[:6]

    if not use_depth:
        # ABLATION: Replace with constant depth
        print(f"  [ABLATION] Replacing depth with constant value")
        depth_shape = inv_rel_depth.shape
        inv_rel_depth = np.ones(depth_shape, dtype=np.float32)

    # Load flow data
    flow_file = os.path.join(RAFT_FLOWS_PATH, f'raft_flows_{scene}_{texture}.npz')
    if not os.path.exists(flow_file):
        raise FileNotFoundError(f"Flow file not found: {flow_file}")

    flow_data = np.load(flow_file)
    if 'flow' in flow_data:
        flow = flow_data['flow'][:5]
    else:
        flow = next(iter(flow_data.values()))[:5]

    return inv_rel_depth, flow


def add_z_variance_for_depth_ablation(points_3d, motion_3d, scale_factor=1.0, random_state=None):
    """
    Add variance to the Z channel when depth is ablated (constant depth).
    
    The problem: with constant depth, all Z coordinates are identical, causing the 
    model to learn a very tight "pancake" distribution in Z. This causes tracking 
    to fail because the model can't accommodate natural depth variation.
    
    Solution: Add Gaussian noise to Z coordinates proportional to the observed
    XY motion magnitude. This simulates uncertainty about depth without assuming
    access to the actual depth estimate.
    
    Args:
        points_3d: List/array of 3D points, shape (num_frames, num_points, 3)
        motion_3d: List/array of 3D motion vectors, shape (num_frames, num_points, 3)
        scale_factor: Multiplier for the Z variance (default 1.0)
        random_state: Random state for reproducibility
    
    Returns:
        points_3d_softened: Points with added Z variance
        motion_3d_softened: Motion vectors with added Z variance
        z_std: The standard deviation used for Z noise
    """
    if random_state is not None:
        rng = np.random.RandomState(random_state)
    else:
        rng = np.random.RandomState()
    
    # Convert to numpy arrays if needed
    points_3d = [np.array(p) for p in points_3d]
    motion_3d = [np.array(m) for m in motion_3d]
    
    # Estimate motion scale from XY components of motion vectors
    # (This is what we CAN observe from flow, without knowing depth)
    all_motion = np.concatenate(motion_3d, axis=0)
    xy_motion_magnitude = np.sqrt(all_motion[:, 0]**2 + all_motion[:, 1]**2)
    
    # Use median of non-trivial motion as scale (robust to outliers and static pixels)
    moving_mask = xy_motion_magnitude > 0.001
    if np.sum(moving_mask) > 10:
        median_xy_motion = np.median(xy_motion_magnitude[moving_mask])
    else:
        # Fallback: use overall XY spread of points
        all_points = np.concatenate(points_3d, axis=0)
        median_xy_motion = np.std(all_points[:, :2])
    
    # Z variance should be on similar scale to XY motion
    # This represents our uncertainty about depth given only XY observations
    z_std = median_xy_motion * scale_factor
    
    print(f"  [Z-SOFTENING] Median XY motion: {median_xy_motion:.4f}")
    print(f"  [Z-SOFTENING] Adding Z noise with std: {z_std:.4f} (scale_factor={scale_factor})")
    
    # Add Gaussian noise to Z channel of points
    points_3d_softened = []
    for frame_idx, points in enumerate(points_3d):
        points_copy = points.copy()
        z_noise = rng.randn(len(points)) * z_std
        points_copy[:, 2] += z_noise
        points_3d_softened.append(points_copy)
    
    # Add Gaussian noise to Z channel of motion vectors
    # (Since with constant depth, the Z component of motion would be ~0, which is wrong)
    motion_3d_softened = []
    for frame_idx, motion in enumerate(motion_3d):
        motion_copy = motion.copy()
        # Motion Z uncertainty should be smaller (motion is relative)
        z_motion_noise = rng.randn(len(motion)) * (z_std * 0.5)
        motion_copy[:, 2] += z_motion_noise
        motion_3d_softened.append(motion_copy)
    
    return points_3d_softened, motion_3d_softened, z_std


def load_ground_truth_masks(scene, texture, num_frames=6):
    """Load ground truth segmentation masks"""
    from PIL import Image

    masks = []
    for frame_idx in range(num_frames):
        mask_file = os.path.join(GESTALT_BASE_PATH, scene, 'render_passes', 'masks', f'Image{frame_idx+1:04d}.png')

        if not os.path.exists(mask_file):
            print(f"  Warning: Mask not found: {mask_file}")
            return None

        mask_img = np.array(Image.open(mask_file))
        if mask_img.ndim == 3:
            mask = mask_img[:, :, 0] > 127
        else:
            mask = mask_img > 127

        masks.append(mask)

    return masks


def compute_segmentation_accuracy(true_mask, estimated_mask):
    """Compute pixel-wise accuracy"""
    if true_mask is None:
        return None
    correct = (true_mask == estimated_mask)
    accuracy = np.mean(correct)
    return accuracy


def extract_depth_free_segmentation(flow_sq, target_hw, min_flow_threshold=0.05):
    """
    Depth-free segmentation using simple flow thresholding.

    This is the CORRECTED version that applies threshold directly to flow,
    not using the motion_valid_mask from compute_3d_points_and_motion.

    Args:
        flow_sq: Flow array (already resized to square)
        target_hw: Target height/width
        min_flow_threshold: Minimum flow magnitude (default 0.05)

    Returns:
        segmentation_mask: Boolean array of shape (target_hw * target_hw,)
    """
    # Compute flow magnitude at each pixel
    flow_magnitude = np.sqrt(flow_sq[0, :, :, 0]**2 + flow_sq[0, :, :, 1]**2)

    # Simple rule: pixels with flow > threshold are part of the moving object
    flow_mask = flow_magnitude > min_flow_threshold

    # Flatten to 1D mask
    segmentation_mask = flow_mask.flatten()

    return segmentation_mask


def evaluate_frame_accuracy(true_mask, frame_state, number_of_blobs, num_probes=100):
    """Evaluate segmentation accuracy for a frame by probe point sampling"""
    if true_mask is None:
        return None, []

    frame_blob_assignments = np.array(frame_state.datapoints_state.blob_assignments)
    frame_hyperblob_assignments = np.array(frame_state.blobs_state.hyperblob_assignments)

    frame_pixel_hyperblobs = np.array([
        frame_hyperblob_assignments[blob_id] if blob_id < len(frame_hyperblob_assignments) else -1
        for blob_id in frame_blob_assignments
    ])

    true_roi_indices = np.where(true_mask.flatten())[0]

    if len(true_roi_indices) == 0:
        return 0.0, []

    num_samples = min(num_probes, len(true_roi_indices))
    probe_indices = np.random.choice(true_roi_indices, size=num_samples, replace=False)

    probe_accuracies = []

    for probe_idx in probe_indices:
        probe_hyperblob_id = frame_pixel_hyperblobs[probe_idx]

        if probe_hyperblob_id < 0:
            continue

        estimated_mask = (frame_pixel_hyperblobs == probe_hyperblob_id)
        accuracy = compute_segmentation_accuracy(true_mask.flatten(), estimated_mask)
        probe_accuracies.append(accuracy)

    if len(probe_accuracies) == 0:
        return 0.0, []

    mean_accuracy = np.mean(probe_accuracies)
    return mean_accuracy, probe_accuracies


def run_single_trial_with_logging(scene, texture, use_depth=True, random_seed=42):
    """Run a single trial and log detailed state information for visualization"""

    print(f"\n{'='*70}")
    print(f"Running: {scene} / {texture}")
    print(f"Mode: {'BASELINE (with depth)' if use_depth else 'ABLATION (no depth)'}")
    print(f"{'='*70}")

    # Create output directory
    mode_str = "baseline" if use_depth else "ablation"
    output_dir = os.path.join(OUTPUT_DIR, f"{scene}_{texture}_{mode_str}")
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print("Loading data...")
    depth, flow = load_six_frame_gestalt_data(scene, texture, use_depth=use_depth)
    print(f"  Depth shape: {depth.shape}, Flow shape: {flow.shape}")

    if use_depth:
        print(f"  Depth range: [{np.min(depth):.3f}, {np.max(depth):.3f}]")
    else:
        print(f"  Using constant depth = {depth[0,0,0]:.3f}")

    # Save depth for visualization
    np.savez(os.path.join(output_dir, 'depth_flow.npz'), depth=depth, flow=flow)

    # Load ground truth
    true_masks = load_ground_truth_masks(scene, texture, num_frames=6)
    if true_masks is None:
        print("  Warning: No ground truth masks found")

    # Compute 3D points and motion
    print("Computing 3D points and motion...")
    points_3d, motion_3d, motion_valid_mask = compute_3d_points_and_motion(
        depth, flow, downsample_factor=8, min_motion_magnitude=0.05,
        focal_length_scale=FOCAL_LENGTH_SCALE
    )

    # Apply Z softening when depth is ablated
    z_std_used = None
    if not use_depth:
        print("Applying Z-channel softening for depth ablation...")
        points_3d, motion_3d, z_std_used = add_z_variance_for_depth_ablation(
            points_3d, motion_3d, 
            scale_factor=Z_SOFTENING_SCALE, 
            random_state=random_seed
        )
        # Convert back to numpy array format (required by downstream functions)
        points_3d = np.array(points_3d)
        motion_3d = np.array(motion_3d)

    # Save 3D points for visualization
    np.savez(os.path.join(output_dir, 'points_3d_motion.npz'),
             points_3d=points_3d, motion_3d=motion_3d, motion_valid_mask=motion_valid_mask,
             z_std_used=z_std_used if z_std_used is not None else 0.0)

    # Process to square format
    orig_H, orig_W = depth.shape[1:3]
    target_hw = max(orig_H, orig_W) // 8
    target_hw = (target_hw // 32) * 32
    if target_hw < 32:
        target_hw = 32

    depth_sq = resize_to_square(depth, target_hw)
    flow_sq = resize_to_square(flow, target_hw)

    # Extract segmentation mask
    print("Extracting segmentation mask...")
    if use_depth:
        # BASELINE: Use depth-based segmentation (original method)
        first_frame_seg = extract_gestalt_segmentation(depth_sq, flow_sq, points_3d)
        combined_mask = first_frame_seg & motion_valid_mask[0]
        print(f"  Using depth-based segmentation")
    else:
        # ABLATION: Use flow-based segmentation (corrected method)
        combined_mask = extract_depth_free_segmentation(flow_sq, target_hw, min_flow_threshold=0.05)
        first_frame_seg = combined_mask.reshape(target_hw, target_hw)
        print(f"  Using depth-free flow-based segmentation (threshold=0.05)")

    print(f"  ROI pixels: {np.sum(combined_mask)} / {len(combined_mask)} ({100*np.sum(combined_mask)/len(combined_mask):.1f}%)")

    # Save segmentation mask
    np.savez(os.path.join(output_dir, 'segmentation_mask.npz'),
             first_frame_seg=first_frame_seg, combined_mask=combined_mask)

    # Create hierarchical k-means clustering
    print("Creating hierarchical k-means clustering...")
    tracked_points = points_3d
    tracked_motion_vectors = motion_3d

    kmeans_chm, roi_blob_indices, roi_hyperblob_indices = make_hierarchical_kmeans_chm_with_mask_fixed_hyperblob(
        tracked_points, HYPERPARAMS["number_of_blobs"], HYPERPARAMS["number_of_hyperblobs"],
        segmentation_mask=combined_mask, motion_vectors=tracked_motion_vectors
    )

    # Set up hyperparameters
    num_datapoints = kmeans_chm['datapoints', 'datapoint_positions'].shape[0]
    num_blobs = kmeans_chm['blobs', 'hyperblob_assignments'].shape[0]
    num_hyperblobs = kmeans_chm['hyperblobs', 'hyperblob_means'].shape[0]

    empirical_mu_H = jnp.median(kmeans_chm['datapoints', 'datapoint_positions'], axis=0)
    empirical_sigma_H = (10*0.5)**2
    empirical_Psi_B = jnp.median(kmeans_chm['blobs', 'blob_covs'][roi_blob_indices], axis=0)
    empirical_Psi_H = jnp.median(kmeans_chm['hyperblobs', 'hyperblob_covs'][roi_hyperblob_indices], axis=0)
    empirical_Psi_V = jnp.median(kmeans_chm['blobs', 'blob_vel_covs'][roi_blob_indices], axis=0)
    mean_blobs_per_roi_hyperblob = jnp.sum(jnp.isin(kmeans_chm['blobs', 'hyperblob_assignments'], roi_hyperblob_indices)) / len(roi_hyperblob_indices)
    empirical_nu_H = f_(int(mean_blobs_per_roi_hyperblob))
    mean_points_per_roi_blob = jnp.sum(jnp.isin(kmeans_chm['datapoints', 'blob_assignments'], roi_blob_indices)) / len(roi_blob_indices)
    empirical_nu_B = empirical_nu_V = f_(int(mean_points_per_roi_blob))

    hypers = HDGMM_Hyperparams.create(
        outlier_prob=f_(0.001),
        outlier_velocity_gamma_shape=f_(7.5),
        outlier_velocity_gamma_rate=f_(0.5),
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
        rotation_vmf_kappa=snp(f_(5.0)),
        rotation_angle_max_deg=snp(15),
        rotation_angle_step_deg=snp(1.0),
        n_hyperblobs=num_hyperblobs,
        n_blobs=num_blobs,
        n_datapoints=num_datapoints,
    )

    # Initialize model
    print("Initializing model...")
    key = jkey(random_seed)
    key, key_importance = jax.random.split(key)
    init_tr, _ = model_jimportance(key_importance, kmeans_chm, (hypers,))
    init_hdgmm_state = init_tr.get_retval()

    # Run initial Gibbs sweeps
    print(f"Running initial Gibbs sweeps ({NUM_INITIAL_GIBBS_ITERATIONS} iterations)...")
    key, init_gibbs_key = jax.random.split(key)
    gibbs_wtrs = hdgmm_full_gibbs(
        init_gibbs_key, init_hdgmm_state, NUM_INITIAL_GIBBS_ITERATIONS,
        GIBBS_DIALS, use_weighted_blobs=True, num_gibbs_inner_loops=NUM_GIBBS_INNER_LOOPS
    )

    converged_state = gibbs_wtrs[-1].retval

    # Store all frame information
    all_frame_data = []

    # Store initial frame (frame 0)
    initial_blob_assignments = np.array(converged_state.datapoints_state.blob_assignments)
    initial_hyperblob_assignments = np.array(converged_state.blobs_state.hyperblob_assignments)
    initial_pixel_hyperblobs = np.array([
        initial_hyperblob_assignments[blob_id] if blob_id < len(initial_hyperblob_assignments) else -1
        for blob_id in initial_blob_assignments
    ])

    # Determine initial ROI hyperblob
    # Ensure first_frame_seg is flattened for comparison
    first_frame_seg_flat = first_frame_seg.flatten() if hasattr(first_frame_seg, 'flatten') else first_frame_seg

    hyperblob_roi_overlap = np.zeros(num_hyperblobs)
    for hb_id in range(num_hyperblobs):
        hb_mask = (initial_pixel_hyperblobs == hb_id)
        overlap = np.sum(hb_mask & first_frame_seg_flat)
        hyperblob_roi_overlap[hb_id] = overlap
    initial_roi_hyperblob_id = int(np.argmax(hyperblob_roi_overlap)) if np.sum(hyperblob_roi_overlap) > 0 else 0

    # Evaluate initial frame
    if true_masks is not None:
        from scipy.ndimage import zoom
        true_mask = true_masks[0]
        zoom_factors = (target_hw / true_mask.shape[0], target_hw / true_mask.shape[1])
        true_mask_resized = zoom(true_mask.astype(float), zoom_factors, order=0) > 0.5
        mean_acc, probe_accs = evaluate_frame_accuracy(true_mask_resized, converged_state,
                                                       HYPERPARAMS["number_of_blobs"], num_probes=100)
        initial_accuracy = mean_acc
    else:
        initial_accuracy = None

    # Store initial frame data
    frame_0_data = {
        'frame_idx': 0,
        'blob_assignments': initial_blob_assignments.reshape(target_hw, target_hw),
        'hyperblob_assignments': initial_hyperblob_assignments,
        'pixel_hyperblobs': initial_pixel_hyperblobs.reshape(target_hw, target_hw),
        'roi_hyperblob_id': initial_roi_hyperblob_id,
        'blob_means': np.array(converged_state.blobs_state.blob_means),
        'hyperblob_means': np.array(converged_state.hyperblobs_state.hyperblob_means),
        'blob_vel_means': np.array(converged_state.blobs_state.blob_vel_means),
        'hyperblob_trans_vels': np.array(converged_state.hyperblobs_state.hyperblob_trans_vels),
        'hyperblob_rot_vels': np.array(converged_state.hyperblobs_state.hyperblob_rot_vels),
        'accuracy': initial_accuracy
    }
    all_frame_data.append(frame_0_data)

    if initial_accuracy is not None:
        print(f"  Frame 0: ROI = HB {initial_roi_hyperblob_id}, Accuracy = {initial_accuracy:.3f}")
    else:
        print(f"  Frame 0: ROI = HB {initial_roi_hyperblob_id}, Accuracy = N/A")

    # Track subsequent frames
    num_frames_available = len(points_3d)
    num_tracking_frames = num_frames_available - 1
    current_state = converged_state

    for frame_idx in range(1, num_tracking_frames + 1):
        print(f"\nProcessing frame {frame_idx}...")

        # Get frame data
        tracked_points_frame = tracked_points[frame_idx]
        tracked_motion_frame = tracked_motion_vectors[frame_idx]

        # Extract motion from current state
        blob_means_current = np.array(current_state.blobs_state.blob_means)
        blob_hyperblob_assignments = np.array(current_state.blobs_state.hyperblob_assignments)
        blob_vel_means_current = np.array(current_state.blobs_state.blob_vel_means)

        hyperblob_means_current = np.array(current_state.hyperblobs_state.hyperblob_means)
        hyperblob_trans_vels_current = np.array(current_state.hyperblobs_state.hyperblob_trans_vels)
        hyperblob_rot_vels_current = np.array(current_state.hyperblobs_state.hyperblob_rot_vels)

        # Propagate blob positions
        propagated_blob_means = np.zeros_like(blob_means_current)
        propagated_hyperblob_means = np.zeros_like(hyperblob_means_current)

        for hb_id in range(num_hyperblobs):
            hb_blob_mask = blob_hyperblob_assignments == hb_id
            n_blobs_in_hb = np.sum(hb_blob_mask)

            if n_blobs_in_hb > 0:
                hb_center_old = hyperblob_means_current[hb_id]
                hb_rotation = hyperblob_rot_vels_current[hb_id]
                hb_translation = hyperblob_trans_vels_current[hb_id]

                hb_center_new = hb_center_old + hb_translation
                propagated_hyperblob_means[hb_id] = hb_center_new

                for blob_id in np.where(hb_blob_mask)[0]:
                    blob_pos = blob_means_current[blob_id]
                    relative_pos = blob_pos - hb_center_old
                    rotated_relative_pos = hb_rotation @ relative_pos
                    propagated_blob_means[blob_id] = hb_center_new + rotated_relative_pos

        # Get current assignments
        current_blob_assignments = np.array(current_state.datapoints_state.blob_assignments)

        # Create choice map with propagated state
        propagated_chm = (
            C['hyperblobs', 'hyperblob_weights'].set(current_state.hyperblobs_state.hyperblob_weights) |
            C['hyperblobs', 'hyperblob_means'].set(propagated_hyperblob_means) |
            C['hyperblobs', 'hyperblob_covs'].set(current_state.hyperblobs_state.hyperblob_covs) |
            C['hyperblobs', 'hyperblob_trans_vels'].set(hyperblob_trans_vels_current) |
            C['hyperblobs', 'hyperblob_rot_vels'].set(hyperblob_rot_vels_current) |

            C['blobs', 'hyperblob_assignments'].set(blob_hyperblob_assignments) |
            C['blobs', 'blob_weights'].set(current_state.blobs_state.blob_weights) |
            C['blobs', 'blob_means'].set(propagated_blob_means) |
            C['blobs', 'blob_covs'].set(current_state.blobs_state.blob_covs) |
            C['blobs', 'blob_vel_means'].set(blob_vel_means_current) |
            C['blobs', 'blob_vel_covs'].set(current_state.blobs_state.blob_vel_covs) |

            C['datapoints', 'blob_assignments'].set(current_blob_assignments) |
            C['datapoints', 'datapoint_positions'].set(tracked_points_frame) |
            C['datapoints', 'datapoint_vels'].set(tracked_motion_frame)
        )

        # Initialize with propagated state
        key, frame_key_importance = jax.random.split(key)
        frame_init_tr, _ = model_jimportance(frame_key_importance, propagated_chm, (current_state.hypers,))
        frame_init_hdgmm_state = frame_init_tr.get_retval()

        # STEP 1: Update velocities
        print(f"  Updating velocities ({NUM_VELOCITY_UPDATE_ITERATIONS} iterations)...")
        key, velocity_gibbs_key = jax.random.split(key)
        velocity_update_wtrs = hdgmm_full_gibbs(
            velocity_gibbs_key,
            frame_init_hdgmm_state,
            NUM_VELOCITY_UPDATE_ITERATIONS,
            VELOCITY_UPDATE_DIALS,
            use_weighted_blobs=True,
            num_gibbs_inner_loops=NUM_GIBBS_INNER_LOOPS
        )

        # Pick best velocity sample
        velocity_sample_freq = max(1, NUM_VELOCITY_UPDATE_ITERATIONS // 10)
        velocity_log_probs = []

        for idx in range(0, len(velocity_update_wtrs), velocity_sample_freq):
            state = velocity_update_wtrs[idx].retval
            chm = (
                C['hyperblobs', 'hyperblob_weights'].set(state.hyperblobs_state.hyperblob_weights) |
                C['hyperblobs', 'hyperblob_means'].set(state.hyperblobs_state.hyperblob_means) |
                C['hyperblobs', 'hyperblob_covs'].set(state.hyperblobs_state.hyperblob_covs) |
                C['hyperblobs', 'hyperblob_trans_vels'].set(state.hyperblobs_state.hyperblob_trans_vels) |
                C['hyperblobs', 'hyperblob_rot_vels'].set(state.hyperblobs_state.hyperblob_rot_vels) |
                C['blobs', 'hyperblob_assignments'].set(state.blobs_state.hyperblob_assignments) |
                C['blobs', 'blob_weights'].set(state.blobs_state.blob_weights) |
                C['blobs', 'blob_means'].set(state.blobs_state.blob_means) |
                C['blobs', 'blob_covs'].set(state.blobs_state.blob_covs) |
                C['blobs', 'blob_vel_means'].set(state.blobs_state.blob_vel_means) |
                C['blobs', 'blob_vel_covs'].set(state.blobs_state.blob_vel_covs) |
                C['datapoints', 'blob_assignments'].set(state.datapoints_state.blob_assignments) |
                C['datapoints', 'datapoint_positions'].set(state.datapoints_state.datapoint_positions) |
                C['datapoints', 'datapoint_vels'].set(state.datapoints_state.datapoint_vels)
            )
            weight, _ = HDGMM_model_3d.assess(chm, (state.hypers,))
            velocity_log_probs.append(float(weight))

        best_velocity_idx = np.argmax(velocity_log_probs) * velocity_sample_freq
        velocity_updated_state = velocity_update_wtrs[best_velocity_idx].retval

        # STEP 2: Full Gibbs sampling
        # Use fewer iterations for ablation (100 vs 500 for baseline)
        num_tracking_iters = NUM_TRACKING_GIBBS_ITERATIONS if use_depth else 100
        print(f"  Running full Gibbs ({num_tracking_iters} iterations)...")
        key, frame_gibbs_key = jax.random.split(key)
        frame_gibbs_wtrs = hdgmm_full_gibbs(
            frame_gibbs_key,
            velocity_updated_state,
            num_tracking_iters,
            GIBBS_DIALS,
            use_weighted_blobs=True,
            num_gibbs_inner_loops=NUM_GIBBS_INNER_LOOPS
        )

        # Pick best sample
        full_sample_freq = max(1, num_tracking_iters // 20)
        full_log_probs = []

        for idx in range(0, len(frame_gibbs_wtrs), full_sample_freq):
            state = frame_gibbs_wtrs[idx].retval
            chm = (
                C['hyperblobs', 'hyperblob_weights'].set(state.hyperblobs_state.hyperblob_weights) |
                C['hyperblobs', 'hyperblob_means'].set(state.hyperblobs_state.hyperblob_means) |
                C['hyperblobs', 'hyperblob_covs'].set(state.hyperblobs_state.hyperblob_covs) |
                C['hyperblobs', 'hyperblob_trans_vels'].set(state.hyperblobs_state.hyperblob_trans_vels) |
                C['hyperblobs', 'hyperblob_rot_vels'].set(state.hyperblobs_state.hyperblob_rot_vels) |
                C['blobs', 'hyperblob_assignments'].set(state.blobs_state.hyperblob_assignments) |
                C['blobs', 'blob_weights'].set(state.blobs_state.blob_weights) |
                C['blobs', 'blob_means'].set(state.blobs_state.blob_means) |
                C['blobs', 'blob_covs'].set(state.blobs_state.blob_covs) |
                C['blobs', 'blob_vel_means'].set(state.blobs_state.blob_vel_means) |
                C['blobs', 'blob_vel_covs'].set(state.blobs_state.blob_vel_covs) |
                C['datapoints', 'blob_assignments'].set(state.datapoints_state.blob_assignments) |
                C['datapoints', 'datapoint_positions'].set(state.datapoints_state.datapoint_positions) |
                C['datapoints', 'datapoint_vels'].set(state.datapoints_state.datapoint_vels)
            )
            weight, _ = HDGMM_model_3d.assess(chm, (state.hypers,))
            full_log_probs.append(float(weight))

        best_full_idx = np.argmax(full_log_probs) * full_sample_freq
        current_state = frame_gibbs_wtrs[best_full_idx].retval

        # Extract assignments
        frame_blob_assignments = np.array(current_state.datapoints_state.blob_assignments)
        frame_hyperblob_assignments = np.array(current_state.blobs_state.hyperblob_assignments)

        frame_pixel_hyperblobs = np.array([
            frame_hyperblob_assignments[blob_id] if blob_id < HYPERPARAMS["number_of_blobs"] else -1
            for blob_id in frame_blob_assignments
        ])

        # Determine ROI hyperblob by tracking from previous frame
        prev_frame_data = all_frame_data[-1]
        prev_roi_hb_id = prev_frame_data['roi_hyperblob_id']
        prev_pixel_hyperblobs = prev_frame_data['pixel_hyperblobs']
        prev_roi_mask = (prev_pixel_hyperblobs == prev_roi_hb_id)

        hyperblob_roi_overlap = np.zeros(num_hyperblobs)
        for hb_id in range(num_hyperblobs):
            hb_mask = (frame_pixel_hyperblobs.reshape(target_hw, target_hw) == hb_id)
            overlap = np.sum(hb_mask & prev_roi_mask)
            hyperblob_roi_overlap[hb_id] = overlap

        frame_roi_hyperblob_id = int(np.argmax(hyperblob_roi_overlap)) if np.sum(hyperblob_roi_overlap) > 0 else prev_roi_hb_id

        # Evaluate accuracy
        if true_masks is not None and frame_idx < len(true_masks):
            true_mask = true_masks[frame_idx]
            zoom_factors = (target_hw / true_mask.shape[0], target_hw / true_mask.shape[1])
            true_mask_resized = zoom(true_mask.astype(float), zoom_factors, order=0) > 0.5
            mean_acc, probe_accs = evaluate_frame_accuracy(true_mask_resized, current_state,
                                                          HYPERPARAMS["number_of_blobs"], num_probes=100)
            frame_accuracy = mean_acc
        else:
            frame_accuracy = None

        # Store frame data
        frame_data = {
            'frame_idx': frame_idx,
            'blob_assignments': frame_blob_assignments.reshape(target_hw, target_hw),
            'hyperblob_assignments': frame_hyperblob_assignments,
            'pixel_hyperblobs': frame_pixel_hyperblobs.reshape(target_hw, target_hw),
            'roi_hyperblob_id': frame_roi_hyperblob_id,
            'blob_means': np.array(current_state.blobs_state.blob_means),
            'hyperblob_means': np.array(current_state.hyperblobs_state.hyperblob_means),
            'blob_vel_means': np.array(current_state.blobs_state.blob_vel_means),
            'hyperblob_trans_vels': np.array(current_state.hyperblobs_state.hyperblob_trans_vels),
            'hyperblob_rot_vels': np.array(current_state.hyperblobs_state.hyperblob_rot_vels),
            'accuracy': frame_accuracy
        }
        all_frame_data.append(frame_data)

        if frame_accuracy is not None:
            print(f"  Frame {frame_idx}: ROI = HB {frame_roi_hyperblob_id}, Accuracy = {frame_accuracy:.3f}")
        else:
            print(f"  Frame {frame_idx}: ROI = HB {frame_roi_hyperblob_id}, Accuracy = N/A")

    # Save all results
    print(f"\nSaving results to {output_dir}...")

    # Save frame-by-frame data
    with open(os.path.join(output_dir, 'frame_data.pkl'), 'wb') as f:
        pickle.dump(all_frame_data, f)

    # Compute overall accuracy
    valid_accuracies = [fd['accuracy'] for fd in all_frame_data if fd['accuracy'] is not None]
    if len(valid_accuracies) > 0:
        overall_accuracy = np.mean(valid_accuracies)
    else:
        overall_accuracy = None

    # Save summary
    summary = {
        'scene': scene,
        'texture': texture,
        'mode': mode_str,
        'use_depth': use_depth,
        'random_seed': random_seed,
        'overall_accuracy': overall_accuracy,
        'frame_accuracies': [fd['accuracy'] for fd in all_frame_data],
        'roi_hyperblob_ids': [fd['roi_hyperblob_id'] for fd in all_frame_data],
        'num_frames': len(all_frame_data),
        'target_hw': target_hw,
        'num_blobs': HYPERPARAMS["number_of_blobs"],
        'num_hyperblobs': HYPERPARAMS["number_of_hyperblobs"],
        'z_softening_scale': Z_SOFTENING_SCALE if not use_depth else None,
        'z_std_used': float(z_std_used) if z_std_used is not None else None
    }

    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else None)

    print(f"\n{'='*70}")
    print(f"RESULTS: {scene} / {texture} ({'BASELINE' if use_depth else 'ABLATION'})")
    print(f"{'='*70}")
    if overall_accuracy is not None:
        print(f"Overall Accuracy: {overall_accuracy:.3f}")
    else:
        print(f"Overall Accuracy: N/A")
    frame_accs_str = [f'{a:.3f}' if a is not None else 'N/A' for a in summary['frame_accuracies']]
    print(f"Frame accuracies: {frame_accs_str}")
    print(f"ROI hyperblob IDs: {summary['roi_hyperblob_ids']}")
    print(f"{'='*70}\n")

    return summary


def main():
    """Run both baseline and ablation for top affected trials"""

    print(f"Depth Ablation Analysis - Re-running Top Affected Trials")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Number of trials to re-run: {len(TOP_TRIALS)}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_results = []

    for scene, texture in TOP_TRIALS:
        print(f"\n\n{'#'*80}")
        print(f"# Processing: {scene} / {texture}")
        print(f"{'#'*80}\n")

        # Run ablation first (without depth) - easier to debug
        ablation_result = run_single_trial_with_logging(scene, texture, use_depth=False, random_seed=RANDOM_SEED)

        # Run baseline (with depth)
        baseline_result = run_single_trial_with_logging(scene, texture, use_depth=True, random_seed=RANDOM_SEED)

        # Compare results
        if baseline_result['overall_accuracy'] is not None and ablation_result['overall_accuracy'] is not None:
            drop = baseline_result['overall_accuracy'] - ablation_result['overall_accuracy']
            pct_drop = 100 * drop / baseline_result['overall_accuracy']

            print(f"\n{'='*80}")
            print(f"COMPARISON: {scene} / {texture}")
            print(f"{'='*80}")
            print(f"Baseline accuracy:  {baseline_result['overall_accuracy']:.4f}")
            print(f"Ablation accuracy:  {ablation_result['overall_accuracy']:.4f}")
            print(f"Performance drop:   {drop:.4f} ({pct_drop:.1f}%)")
            print(f"{'='*80}\n")

        all_results.append({
            'scene': scene,
            'texture': texture,
            'baseline': baseline_result,
            'ablation': ablation_result
        })

    # Save overall summary
    summary_file = os.path.join(OUTPUT_DIR, 'overall_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else None)

    print(f"\n{'#'*80}")
    print(f"# ALL TRIALS COMPLETE")
    print(f"{'#'*80}")
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"Overall summary: {summary_file}")

    # Print comparison table
    print(f"\n{'='*80}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*80}")
    print(f"{'Scene':<15} {'Texture':<12} {'Baseline':<10} {'Ablation':<10} {'Drop':<10} {'% Drop':<10}")
    print(f"{'-'*80}")

    for result in all_results:
        scene = result['scene']
        texture = result['texture']
        baseline_acc = result['baseline'].get('overall_accuracy')
        ablation_acc = result['ablation'].get('overall_accuracy')

        if baseline_acc is not None and ablation_acc is not None:
            drop = baseline_acc - ablation_acc
            pct_drop = 100 * drop / baseline_acc
            print(f"{scene:<15} {texture:<12} {baseline_acc:<10.4f} {ablation_acc:<10.4f} {drop:<10.4f} {pct_drop:<10.1f}%")
        else:
            print(f"{scene:<15} {texture:<12} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10}")

    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
