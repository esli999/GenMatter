"""
Run mask propagation gestalt experiment across all scenes and textures.
This is the constant depth ABLATION with FLOW-BASED INITIALIZATION (instead of depth-based).
Saves segmentation masks and accuracy statistics.

MODIFICATION: Uses extract_depth_free_segmentation() instead of extract_gestalt_segmentation()
for better initialization in the ablation condition.
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

# JIT compile model functions
model_jsimulate = jax.jit(HDGMM_model_3d.simulate)
model_jimportance = jax.jit(HDGMM_model_3d.importance)

# Configuration
EXPERIMENT_NAME = "Video_Gestalt_Mask_Propagation_No_Depth_Ablation_Flow_Init"
OUTPUT_DIR = "/home/esli/GenMatter/video_gestalt_masks_rebuttal_ablation_flow_init"
GESTALT_BASE_PATH = '/home/esli/GenMatter/assets/from_thomas'
RAFT_FLOWS_PATH = '/home/esli/GenMatter/raft_flows'

# Experiment parameters
SCENES = [f'scene_{i:05d}' for i in range(20)]  # scene_00000 to scene_00019
TEXTURES = ['texture_00', 'texture_07', 'texture_13', 'texture_16', 'texture_21', 'texture_22', 'texture_25']
NUM_RUNS = 1  # Number of independent runs per scene/texture
RANDOM_SEED_BASE = 42  # Base seed, will use 42, 43, 44, 45, 46 for 5 runs
FOCAL_LENGTH_SCALE = 2.0  # Scale focal length for better depth separation
Z_SOFTENING_SCALE = 1.5  # Scale factor for Z variance when depth is ablated

# Hyperparameters
HYPERPARAMS = {
    "number_of_blobs": 100,
    "number_of_hyperblobs": 5,
}

# Gibbs sampling parameters (matching notebook values)
NUM_INITIAL_GIBBS_ITERATIONS = 50  # Match notebook
NUM_GIBBS_INNER_LOOPS = 5           # Match notebook
NUM_TRACKING_GIBBS_ITERATIONS = 500 # Match notebook
NUM_VELOCITY_UPDATE_ITERATIONS = 20 # Match notebook
# NUM_TRACKING_FRAMES determined dynamically from available data

# Gibbs dials matching notebook: Update blob assignments but keep hyperblob assignments fixed
GIBBS_DIALS = {
    "blob_weights": True,
    "hyperblob_weights": False,
    "blob_assignments": True,  # UPDATE blob assignments to capture uncertainty
    "hyperblob_assignments": False,  # Keep hyperblob assignments FIXED
    "hyperblob_covs": False,
    "blob_covs": True,
    "blob_vel_covs": True,
    "blob_vel_means": True,
    "hyperblob_means": False,  # Keep hyperblob structure fixed
    "blob_means": True,  # Allow updates with soft prior
    'hyperblob_rot_vels': True,  # ENABLED for tracking
    'hyperblob_trans_vels': True  # ENABLED for tracking
}

# Velocity update dials (for first step during tracking)
VELOCITY_UPDATE_DIALS = {
    "blob_weights": False,
    "hyperblob_weights": False,
    "blob_assignments": True,  # UPDATE blob assignments (matching notebook)
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


def compute_hyperblob_sample_counts(blob_assignments, hyperblob_assignments,
                                    num_blobs, num_hyperblobs, target_hw):
    """
    Compute per-pixel empirical sample counts for each hyperblob category.

    Args:
        blob_assignments: (num_samples, num_pixels) array of blob assignments
        hyperblob_assignments: (num_samples, num_blobs) array of hyperblob assignments
        num_blobs: Number of blobs
        num_hyperblobs: Number of hyperblobs
        target_hw: Image size

    Returns:
        counts_map: (H, W, num_hyperblobs+1) array where counts_map[i,j,k] is the number of
                    samples that assigned pixel (i,j) to hyperblob k.
                    - Channels 0 to num_hyperblobs-1: counts for hyperblobs 0 to num_hyperblobs-1
                    - Channel num_hyperblobs (last): counts for outlier assignments (-1)
    """
    num_samples, num_pixels = blob_assignments.shape

    # Map pixels to hyperblobs for each sample
    pixel_hyperblobs = np.zeros((num_samples, num_pixels), dtype=int)

    for sample_idx in range(num_samples):
        for pixel_idx in range(num_pixels):
            blob_id = blob_assignments[sample_idx, pixel_idx]
            if blob_id < num_blobs:
                pixel_hyperblobs[sample_idx, pixel_idx] = hyperblob_assignments[sample_idx, blob_id]
            else:
                # Outlier blob
                pixel_hyperblobs[sample_idx, pixel_idx] = -1

    # Compute counts for each pixel and hyperblob category
    # Shape: (num_pixels, num_hyperblobs+1) where last channel is outliers
    counts_map = np.zeros((num_pixels, num_hyperblobs + 1), dtype=int)

    for pixel_idx in range(num_pixels):
        pixel_assignments = pixel_hyperblobs[:, pixel_idx]

        # Count frequency for each hyperblob
        for assignment in pixel_assignments:
            if assignment == -1:
                # Outliers go to last channel
                counts_map[pixel_idx, num_hyperblobs] += 1
            else:
                # Regular hyperblobs go to their corresponding channels
                counts_map[pixel_idx, assignment] += 1

    # Reshape to (H, W, num_hyperblobs+1)
    return counts_map.reshape((target_hw, target_hw, num_hyperblobs + 1))


def extract_frame_hyperblob_counts(gibbs_wtrs_frame, num_blobs, num_hyperblobs, target_hw, num_samples_to_use=50):
    """
    Extract per-pixel hyperblob sample counts from Gibbs samples for a single frame.
    Returns hyperblob count map.
    """
    # Extract assignments from Gibbs samples
    num_total_samples = len(gibbs_wtrs_frame)
    start_idx = max(0, num_total_samples - num_samples_to_use)

    blob_assignments_list = []
    hyperblob_assignments_list = []

    for sample_idx in range(start_idx, num_total_samples):
        sample = gibbs_wtrs_frame[sample_idx]
        blob_assigns = np.array(sample.retval.datapoints_state.blob_assignments)
        hyperblob_assigns = np.array(sample.retval.blobs_state.hyperblob_assignments)
        blob_assignments_list.append(blob_assigns)
        hyperblob_assignments_list.append(hyperblob_assigns)

    blob_assignments_array = np.array(blob_assignments_list)
    hyperblob_assignments_array = np.array(hyperblob_assignments_list)

    # Compute hyperblob sample counts
    hyperblob_counts = compute_hyperblob_sample_counts(
        blob_assignments_array, hyperblob_assignments_array,
        num_blobs, num_hyperblobs, target_hw
    )

    return hyperblob_counts


def load_six_frame_gestalt_data(scene, texture):
    """Load only first 6 frames of depth and flow data

    ABLATION: Returns dummy depth data since we'll only use 2D observations
    """
    depth_file = os.path.join(GESTALT_BASE_PATH, scene, texture, 'output_six_frame_depths.npz')
    if not os.path.exists(depth_file):
        raise FileNotFoundError(f"Depth file not found: {depth_file}")

    depth_data = np.load(depth_file)
    if 'depths' in depth_data:
        inv_rel_depth = depth_data['depths'][:6]  # First 6 frames
    else:
        inv_rel_depth = next(iter(depth_data.values()))[:6]

    # ABLATION: We only use depth to get the image shape, then ignore all depth information
    # All points will be projected to a canonical depth plane
    # This creates a "dummy" depth array with the right shape
    depth_shape = inv_rel_depth.shape

    # Load flow data (first 5 frames) - THIS is our actual observation
    flow_file = os.path.join(RAFT_FLOWS_PATH, f'raft_flows_{scene}_{texture}.npz')
    if not os.path.exists(flow_file):
        raise FileNotFoundError(f"Flow file not found: {flow_file}")

    flow_data = np.load(flow_file)
    if 'flow' in flow_data:
        flow = flow_data['flow'][:5]  # First 5 frames
    else:
        flow = next(iter(flow_data.values()))[:5]

    # Return dummy depth (will be replaced in compute_3d_points_and_motion_2d_only)
    # We need this to have the right shape for the pipeline
    dummy_depth = np.ones(depth_shape, dtype=np.float32)

    print(f"  [ABLATION - NO DEPTH] Using 2D observations only: pixel positions + optic flow")

    return dummy_depth, flow


def load_ground_truth_masks(scene, texture, num_frames=6):
    """
    Load ground truth segmentation masks for evaluation.
    Masks are stored in render_passes/masks/ with 1-indexed naming (Image0001.png, etc.)
    """
    from PIL import Image

    masks = []
    for frame_idx in range(num_frames):
        # Masks are 1-indexed: frame 0 -> Image0001.png
        mask_file = os.path.join(GESTALT_BASE_PATH, scene, 'render_passes', 'masks', f'Image{frame_idx+1:04d}.png')

        if not os.path.exists(mask_file):
            print(f"  Warning: Mask not found: {mask_file}")
            return None

        # Load mask (RGB image, but all channels are the same)
        mask_img = np.array(Image.open(mask_file))

        # Convert to binary (take first channel if RGB)
        if mask_img.ndim == 3:
            mask = mask_img[:, :, 0] > 127  # Binary threshold
        else:
            mask = mask_img > 127

        masks.append(mask)

    return masks


def compute_segmentation_accuracy(true_mask, estimated_mask):
    """Compute pixel-wise accuracy between true and estimated masks"""
    if true_mask is None:
        return None
    correct = (true_mask == estimated_mask)
    accuracy = np.mean(correct)
    return accuracy


def evaluate_frame_accuracy(true_mask, frame_state, number_of_blobs, num_probes=100):
    """
    Evaluate segmentation accuracy for a frame by probe point sampling.

    Returns:
        mean_accuracy: Average accuracy across probe points
        probe_accuracies: List of individual accuracies
    """
    if true_mask is None:
        return None, []

    # Get estimated assignments
    frame_blob_assignments = np.array(frame_state.datapoints_state.blob_assignments)
    frame_hyperblob_assignments = np.array(frame_state.blobs_state.hyperblob_assignments)

    # Map datapoints to hyperblobs
    frame_pixel_hyperblobs = np.array([
        frame_hyperblob_assignments[blob_id] if blob_id < len(frame_hyperblob_assignments) else -1
        for blob_id in frame_blob_assignments
    ])

    # Find probe points
    true_roi_indices = np.where(true_mask.flatten())[0]

    if len(true_roi_indices) == 0:
        return 0.0, []

    # Sample probe points
    num_samples = min(num_probes, len(true_roi_indices))
    probe_indices = np.random.choice(true_roi_indices, size=num_samples, replace=False)

    probe_accuracies = []

    for probe_idx in probe_indices:
        probe_hyperblob_id = frame_pixel_hyperblobs[probe_idx]

        if probe_hyperblob_id < 0:
            continue

        # Create estimated mask for this hyperblob
        estimated_mask = (frame_pixel_hyperblobs == probe_hyperblob_id)

        # Compute accuracy
        accuracy = compute_segmentation_accuracy(true_mask.flatten(), estimated_mask)
        probe_accuracies.append(accuracy)

    if len(probe_accuracies) == 0:
        return 0.0, []

    mean_accuracy = np.mean(probe_accuracies)
    return mean_accuracy, probe_accuracies


def run_single_experiment(scene, texture, random_seed, true_masks, points_3d, motion_3d, motion_valid_mask,
                         depth_sq, flow_sq, combined_mask, target_hw, hypers, kmeans_chm, num_hyperblobs,
                         first_frame_seg):
    """Run a single experiment with a given random seed using CORRECT tracking implementation"""
    from genjax import ChoiceMapBuilder as C

    tracked_points = points_3d
    tracked_motion_vectors = motion_3d

    # Calculate number of tracking frames (frames after initial frame 0)
    num_frames_available = len(points_3d)
    num_tracking_frames = num_frames_available - 1

    # Initialize model
    key = jkey(random_seed)
    key, key_importance = jax.random.split(key)
    init_tr, _ = model_jimportance(key_importance, kmeans_chm, (hypers,))
    init_hdgmm_state = init_tr.get_retval()

    # Run initial Gibbs sweeps on first frame
    key, init_gibbs_key = jax.random.split(key)
    gibbs_wtrs = hdgmm_full_gibbs(
        init_gibbs_key, init_hdgmm_state, NUM_INITIAL_GIBBS_ITERATIONS,
        GIBBS_DIALS, use_weighted_blobs=True, num_gibbs_inner_loops=NUM_GIBBS_INNER_LOOPS
    )

    converged_state = gibbs_wtrs[-1].retval

    # Store initial frame (frame 0) before tracking loop
    all_frame_states = []
    all_frame_masks = []
    all_roi_hyperblob_ids = []
    all_frame_hyperblob_counts = []

    number_of_blobs = HYPERPARAMS["number_of_blobs"]

    # Compute hyperblob sample counts for initial frame from Gibbs samples
    print("  Computing hyperblob sample counts for initial frame...")
    initial_hyperblob_counts = extract_frame_hyperblob_counts(
        gibbs_wtrs, number_of_blobs, num_hyperblobs, target_hw, num_samples_to_use=50
    )
    all_frame_hyperblob_counts.append(initial_hyperblob_counts)

    # Compute statistics from counts: max count and most uncertain pixels
    max_counts = np.max(initial_hyperblob_counts, axis=2)  # Max count across hyperblobs
    num_samples = np.sum(initial_hyperblob_counts[0, 0, :])  # Total samples per pixel
    uncertainty_ratio = 1.0 - (max_counts / num_samples)  # 0 = certain, high = uncertain
    print(f"  Initial frame uncertainty: mean ratio={np.mean(uncertainty_ratio):.3f}, median={np.median(uncertainty_ratio):.3f}, max={np.max(uncertainty_ratio):.3f}")
    print(f"  High uncertainty pixels (ratio > 0.5): {np.sum(uncertainty_ratio > 0.5)} / {target_hw**2}")

    # Store initial frame
    all_frame_states.append(converged_state)

    initial_blob_assignments = np.array(converged_state.datapoints_state.blob_assignments)
    initial_hyperblob_assignments = np.array(converged_state.blobs_state.hyperblob_assignments)
    initial_pixel_hyperblobs = np.array([
        initial_hyperblob_assignments[blob_id] if blob_id < len(initial_hyperblob_assignments) else -1
        for blob_id in initial_blob_assignments
    ])
    all_frame_masks.append(initial_pixel_hyperblobs)

    # Determine initial ROI hyperblob from heuristic segmentation
    # Ensure first_frame_seg is flattened for comparison
    first_frame_seg_flat = first_frame_seg.flatten() if hasattr(first_frame_seg, 'flatten') else first_frame_seg
    hyperblob_roi_overlap = np.zeros(num_hyperblobs)
    for hb_id in range(num_hyperblobs):
        hb_mask = (initial_pixel_hyperblobs == hb_id)
        overlap = np.sum(hb_mask & first_frame_seg_flat)
        hyperblob_roi_overlap[hb_id] = overlap
    initial_roi_hyperblob_id = int(np.argmax(hyperblob_roi_overlap)) if np.sum(hyperblob_roi_overlap) > 0 else 0
    all_roi_hyperblob_ids.append(initial_roi_hyperblob_id)

    current_state = converged_state

    # Now track subsequent frames (1, 2, 3, ...)
    for frame_idx in range(1, num_tracking_frames + 1):
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

        # Propagate blob positions using BOTH hyperblob rotation AND translation
        propagated_blob_means = np.zeros_like(blob_means_current)
        propagated_hyperblob_means = np.zeros_like(hyperblob_means_current)

        for hb_id in range(num_hyperblobs):
            hb_blob_mask = blob_hyperblob_assignments == hb_id
            n_blobs_in_hb = np.sum(hb_blob_mask)

            if n_blobs_in_hb > 0:
                # Get hyperblob center (old position)
                hb_center_old = hyperblob_means_current[hb_id]

                # Get inferred rotation and translation velocities
                hb_rotation = hyperblob_rot_vels_current[hb_id]
                hb_translation = hyperblob_trans_vels_current[hb_id]

                # Apply translation to hyperblob center
                hb_center_new = hb_center_old + hb_translation
                propagated_hyperblob_means[hb_id] = hb_center_new

                # Propagate each blob: rotate around old center, then translate
                for blob_id in np.where(hb_blob_mask)[0]:
                    blob_pos = blob_means_current[blob_id]
                    relative_pos = blob_pos - hb_center_old
                    rotated_relative_pos = hb_rotation @ relative_pos
                    propagated_blob_means[blob_id] = hb_center_new + rotated_relative_pos

        # Get current blob and datapoint assignments (FIXED, not updated)
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

        # STEP 1: Update velocities first
        key, velocity_gibbs_key = jax.random.split(key)
        velocity_update_wtrs = hdgmm_full_gibbs(
            velocity_gibbs_key,
            frame_init_hdgmm_state,
            NUM_VELOCITY_UPDATE_ITERATIONS,
            VELOCITY_UPDATE_DIALS,
            use_weighted_blobs=True,
            num_gibbs_inner_loops=NUM_GIBBS_INNER_LOOPS
        )

        # Pick best sample from velocity updates
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
        key, frame_gibbs_key = jax.random.split(key)
        frame_gibbs_wtrs = hdgmm_full_gibbs(
            frame_gibbs_key,
            velocity_updated_state,
            NUM_TRACKING_GIBBS_ITERATIONS,
            GIBBS_DIALS,
            use_weighted_blobs=True,
            num_gibbs_inner_loops=NUM_GIBBS_INNER_LOOPS
        )

        # Extract hyperblob sample counts from this frame's Gibbs samples
        print(f"    Computing hyperblob sample counts for frame {frame_idx}...")
        frame_hyperblob_counts = extract_frame_hyperblob_counts(
            frame_gibbs_wtrs, number_of_blobs, num_hyperblobs, target_hw,
            num_samples_to_use=50
        )
        all_frame_hyperblob_counts.append(frame_hyperblob_counts)

        # Pick best sample from full Gibbs
        full_sample_freq = max(1, NUM_TRACKING_GIBBS_ITERATIONS // 20)
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

        # Extract blob and hyperblob assignments
        frame_blob_assignments = np.array(current_state.datapoints_state.blob_assignments)
        frame_hyperblob_assignments = np.array(current_state.blobs_state.hyperblob_assignments)

        # Map pixels to hyperblobs
        frame_pixel_hyperblobs = np.array([
            frame_hyperblob_assignments[blob_id] if blob_id < number_of_blobs else -1
            for blob_id in frame_blob_assignments
        ])

        # Determine ROI hyperblob for this frame by tracking from previous frame
        prev_frame_roi_hb_id = all_roi_hyperblob_ids[-1]
        prev_frame_pixel_hyperblobs = all_frame_masks[-1]
        prev_roi_mask = (prev_frame_pixel_hyperblobs == prev_frame_roi_hb_id)

        # Find which current hyperblob has maximum overlap with previous ROI
        hyperblob_roi_overlap = np.zeros(num_hyperblobs)
        for hb_id in range(num_hyperblobs):
            hb_mask = (frame_pixel_hyperblobs == hb_id)
            overlap = np.sum(hb_mask & prev_roi_mask)
            hyperblob_roi_overlap[hb_id] = overlap

        frame_roi_hyperblob_id = int(np.argmax(hyperblob_roi_overlap)) if np.sum(hyperblob_roi_overlap) > 0 else prev_frame_roi_hb_id

        # Log uncertainty statistics from sample counts
        max_counts = np.max(frame_hyperblob_counts, axis=2)
        num_samples = np.sum(frame_hyperblob_counts[0, 0, :])
        uncertainty_ratio = 1.0 - (max_counts / num_samples)
        print(f"    Uncertainty ratio: mean={np.mean(uncertainty_ratio):.3f}, median={np.median(uncertainty_ratio):.3f}, max={np.max(uncertainty_ratio):.3f}")
        print(f"    High uncertainty pixels (ratio > 0.5): {np.sum(uncertainty_ratio > 0.5)} / {target_hw**2}")

        # Store results
        all_frame_states.append(current_state)
        all_roi_hyperblob_ids.append(frame_roi_hyperblob_id)
        all_frame_masks.append(frame_pixel_hyperblobs)

    # Evaluate accuracy
    total_frames = num_tracking_frames + 1  # Include initial frame
    frame_accuracies = []
    for frame_idx in range(total_frames):
        if true_masks is not None and frame_idx < len(true_masks):
            from scipy.ndimage import zoom
            true_mask = true_masks[frame_idx]
            zoom_factors = (target_hw / true_mask.shape[0], target_hw / true_mask.shape[1])
            true_mask_resized = zoom(true_mask.astype(float), zoom_factors, order=0) > 0.5

            mean_acc, probe_accs = evaluate_frame_accuracy(
                true_mask_resized, all_frame_states[frame_idx],
                number_of_blobs, num_probes=100
            )
            frame_accuracies.append(mean_acc)
        else:
            frame_accuracies.append(None)

    valid_accuracies = [acc for acc in frame_accuracies if acc is not None]
    if len(valid_accuracies) > 0:
        overall_accuracy = np.mean(valid_accuracies)
    else:
        overall_accuracy = None

    return {
        'frame_accuracies': frame_accuracies,
        'overall_accuracy': overall_accuracy,
        'all_frame_states': all_frame_states,
        'masks': all_frame_masks,
        'roi_hyperblob_ids': all_roi_hyperblob_ids,
        'num_tracking_frames': num_tracking_frames,
        'total_frames': total_frames,
        'hyperblob_sample_counts': all_frame_hyperblob_counts
    }


def run_experiment_for_scene_texture(scene, texture):
    """Run the complete experiment for a single scene/texture combination with multiple runs"""
    print(f"\n{'='*70}")
    print(f"Processing {scene} - {texture}")
    print(f"{'='*70}")

    try:
        # Load data (shared across all runs)
        print("Loading data...")
        depth, flow = load_six_frame_gestalt_data(scene, texture)
        print(f"  Depth shape: {depth.shape}, Flow shape: {flow.shape}")

        # Load ground truth masks
        true_masks = load_ground_truth_masks(scene, texture, num_frames=6)
        if true_masks is None:
            print("  Warning: No ground truth masks found")

        # Compute 3D points and motion (shared across all runs)
        print("Computing 3D points and motion...")
        points_3d, motion_3d, motion_valid_mask = compute_3d_points_and_motion(
            depth, flow, downsample_factor=8, min_motion_magnitude=0.05, focal_length_scale=FOCAL_LENGTH_SCALE
        )

        # Apply Z softening for depth ablation (constant depth causes tight Z distribution)
        print("Applying Z-channel softening for depth ablation...")
        points_3d, motion_3d, z_std_used = add_z_variance_for_depth_ablation(
            points_3d, motion_3d, 
            scale_factor=Z_SOFTENING_SCALE, 
            random_state=RANDOM_SEED_BASE
        )
        # Convert back to numpy array format (required by downstream functions)
        points_3d = np.array(points_3d)
        motion_3d = np.array(motion_3d)

        # Process depth and flow to square format
        orig_H, orig_W = depth.shape[1:3]
        target_hw = max(orig_H, orig_W) // 8
        target_hw = (target_hw // 32) * 32
        if target_hw < 32:
            target_hw = 32

        depth_sq = resize_to_square(depth, target_hw)
        flow_sq = resize_to_square(flow, target_hw)

        # Extract segmentation mask - USING FLOW-BASED METHOD
        print("Extracting segmentation mask using FLOW-BASED initialization...")
        combined_mask = extract_depth_free_segmentation(flow_sq, target_hw, min_flow_threshold=0.05)
        first_frame_seg = combined_mask  # Keep as 1D for consistency
        print(f"  Flow-based segmentation: {np.sum(combined_mask)} / {len(combined_mask)} pixels ({100*np.sum(combined_mask)/len(combined_mask):.1f}%)")

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
            # Rotation hyperparameters - broader proposals for tracking
            rotation_vmf_kappa=snp(f_(5.0)),      # Weak prior for broad proposals
            rotation_angle_max_deg=snp(15),       # Allow up to 15 degrees
            rotation_angle_step_deg=snp(1.0),     # 1 degree steps for good coverage
            n_hyperblobs=num_hyperblobs,
            n_blobs=num_blobs,
            n_datapoints=num_datapoints,
        )

        # Run multiple independent experiments
        print(f"\nRunning {NUM_RUNS} independent experiments...")
        all_run_results = []

        for run_idx in range(NUM_RUNS):
            random_seed = RANDOM_SEED_BASE + run_idx
            print(f"\n--- Run {run_idx + 1}/{NUM_RUNS} (seed={random_seed}) ---")

            run_result = run_single_experiment(
                scene, texture, random_seed, true_masks, points_3d, motion_3d, motion_valid_mask,
                depth_sq, flow_sq, combined_mask, target_hw, hypers, kmeans_chm, num_hyperblobs,
                first_frame_seg
            )

            all_run_results.append(run_result)

            if run_result['overall_accuracy'] is not None:
                print(f"Run {run_idx + 1} Overall Accuracy: {run_result['overall_accuracy']:.3f}")

        # Aggregate results across runs
        print(f"\n{'='*60}")
        print("AGGREGATING RESULTS ACROSS RUNS")
        print(f"{'='*60}")

        # Get total frames from first run
        total_frames = all_run_results[0]['total_frames'] if len(all_run_results) > 0 else 0
        num_tracking_frames = all_run_results[0]['num_tracking_frames'] if len(all_run_results) > 0 else 0

        # Collect accuracies from all runs
        run_overall_accuracies = [r['overall_accuracy'] for r in all_run_results if r['overall_accuracy'] is not None]

        # Collect uncertainty statistics from all runs (using sample counts)
        run_uncertainty_ratios = []
        for r in all_run_results:
            if 'hyperblob_sample_counts' in r and len(r['hyperblob_sample_counts']) > 0:
                # Compute mean uncertainty ratio across all frames
                frame_uncertainty_ratios = []
                for counts_map in r['hyperblob_sample_counts']:
                    max_counts = np.max(counts_map, axis=2)
                    num_samples = np.sum(counts_map[0, 0, :])
                    uncertainty_ratio = 1.0 - (max_counts / num_samples)
                    frame_uncertainty_ratios.append(np.mean(uncertainty_ratio))
                run_uncertainty_ratios.append(np.mean(frame_uncertainty_ratios))

        if len(run_overall_accuracies) > 0:
            mean_overall_accuracy = np.mean(run_overall_accuracies)
            std_overall_accuracy = np.std(run_overall_accuracies)
            print(f"Mean Overall Accuracy (across {len(run_overall_accuracies)} runs): {mean_overall_accuracy:.3f} ± {std_overall_accuracy:.3f}")
            print(f"Total frames: {total_frames} (1 initial + {num_tracking_frames} tracked)")

            # Print uncertainty statistics
            if len(run_uncertainty_ratios) > 0:
                print(f"\nUncertainty Statistics (averaged across frames and runs):")
                print(f"  Mean uncertainty ratio: {np.mean(run_uncertainty_ratios):.3f} ± {np.std(run_uncertainty_ratios):.3f}")

            # Per-frame aggregation
            all_frame_accuracies = []
            for frame_idx in range(total_frames):
                frame_accs = [r['frame_accuracies'][frame_idx] for r in all_run_results
                              if frame_idx < len(r['frame_accuracies']) and r['frame_accuracies'][frame_idx] is not None]
                if len(frame_accs) > 0:
                    mean_frame_acc = np.mean(frame_accs)
                    std_frame_acc = np.std(frame_accs)
                    all_frame_accuracies.append({
                        'mean': mean_frame_acc,
                        'std': std_frame_acc,
                        'values': frame_accs
                    })
                    frame_label = f"Frame 0 (initial)" if frame_idx == 0 else f"Frame {frame_idx}"
                    print(f"  {frame_label}: {mean_frame_acc:.3f} ± {std_frame_acc:.3f}")
                else:
                    all_frame_accuracies.append(None)
        else:
            mean_overall_accuracy = None
            std_overall_accuracy = None
            all_frame_accuracies = None
            total_frames = 0
            num_tracking_frames = 0
            print("\nNo accuracy data available")

        # Save results
        output_subdir = os.path.join(OUTPUT_DIR, scene, texture)
        os.makedirs(output_subdir, exist_ok=True)

        # Save assignments from all runs as NPZ files
        for run_idx, run_result in enumerate(all_run_results):
            run_dir = os.path.join(output_subdir, f'run_{run_idx}')
            os.makedirs(run_dir, exist_ok=True)

            all_frame_states = run_result['all_frame_states']
            masks = run_result['masks']
            roi_hyperblob_ids = run_result['roi_hyperblob_ids']
            hyperblob_sample_counts = run_result['hyperblob_sample_counts']

            # Save all frames in a single NPZ file
            blob_assignments_all_frames = []
            hyperblob_assignments_all_frames = []
            pixel_hyperblob_assignments_all_frames = []

            for frame_idx in range(len(all_frame_states)):
                state = all_frame_states[frame_idx]

                # Get blob assignments for each pixel
                blob_assignments = np.array(state.datapoints_state.blob_assignments)
                blob_assignments_all_frames.append(blob_assignments.reshape((target_hw, target_hw)))

                # Get hyperblob assignments for each blob
                hyperblob_assignments = np.array(state.blobs_state.hyperblob_assignments)
                hyperblob_assignments_all_frames.append(hyperblob_assignments)

                # Get hyperblob assignments for each pixel
                pixel_hyperblobs = masks[frame_idx].reshape((target_hw, target_hw))
                pixel_hyperblob_assignments_all_frames.append(pixel_hyperblobs)

            # Save to NPZ
            npz_path = os.path.join(run_dir, 'assignments.npz')
            np.savez(
                npz_path,
                blob_assignments=np.array(blob_assignments_all_frames),  # Shape: (total_frames, H, W)
                hyperblob_assignments_per_blob=np.array(hyperblob_assignments_all_frames),  # Shape: (total_frames, num_blobs)
                pixel_hyperblob_assignments=np.array(pixel_hyperblob_assignments_all_frames),  # Shape: (total_frames, H, W)
                roi_hyperblob_ids=np.array(roi_hyperblob_ids),  # Shape: (total_frames,)
                hyperblob_sample_counts=np.array(hyperblob_sample_counts),  # Shape: (total_frames, H, W, num_hyperblobs+1)
                target_hw=target_hw,
                num_blobs=HYPERPARAMS["number_of_blobs"],
                num_hyperblobs=HYPERPARAMS["number_of_hyperblobs"]
            )

        # Save aggregated accuracy statistics
        stats = {
            'scene': scene,
            'texture': texture,
            'num_runs': NUM_RUNS,
            'mean_overall_accuracy': mean_overall_accuracy,
            'std_overall_accuracy': std_overall_accuracy,
            'frame_accuracies_aggregated': all_frame_accuracies,
            'individual_runs': [
                {
                    'run_idx': i,
                    'seed': RANDOM_SEED_BASE + i,
                    'frame_accuracies': r['frame_accuracies'],
                    'overall_accuracy': r['overall_accuracy']
                }
                for i, r in enumerate(all_run_results)
            ],
            'total_frames': total_frames,
            'num_tracking_frames': num_tracking_frames,
            'hyperparams': HYPERPARAMS,
            'focal_length_scale': FOCAL_LENGTH_SCALE,
            'initialization_method': 'flow_based_threshold_0.05',
            'z_softening_scale': Z_SOFTENING_SCALE,
            'z_std_used': float(z_std_used)
        }
        stats_file = os.path.join(output_subdir, 'accuracy_stats.json')
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2, default=lambda x: float(x) if x is not None else None)

        print(f"\n✓ Results saved to {output_subdir}")

        return {
            'scene': scene,
            'texture': texture,
            'mean_overall_accuracy': mean_overall_accuracy,
            'std_overall_accuracy': std_overall_accuracy,
            'success': True
        }

    except Exception as e:
        print(f"\n✗ Error processing {scene}/{texture}: {e}")
        import traceback
        traceback.print_exc()
        return {
            'scene': scene,
            'texture': texture,
            'error': str(e),
            'success': False
        }


def main():
    """Run experiment for all scenes and textures"""
    print(f"Starting {EXPERIMENT_NAME}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Scenes: {len(SCENES)}")
    print(f"Textures: {len(TEXTURES)}")
    print(f"Total combinations: {len(SCENES) * len(TEXTURES)}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = []

    for scene in SCENES:
        for texture in TEXTURES:
            result = run_experiment_for_scene_texture(scene, texture)
            results.append(result)

    # Save summary
    summary = {
        'experiment_name': EXPERIMENT_NAME,
        'total_combinations': len(SCENES) * len(TEXTURES),
        'successful': sum(1 for r in results if r['success']),
        'failed': sum(1 for r in results if not r['success']),
        'results': results
    }

    summary_file = os.path.join(OUTPUT_DIR, 'experiment_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=lambda x: float(x) if x is not None else None)

    print(f"\n{'='*70}")
    print(f"EXPERIMENT COMPLETE")
    print(f"{'='*70}")
    print(f"Successful: {summary['successful']}/{summary['total_combinations']}")
    print(f"Failed: {summary['failed']}/{summary['total_combinations']}")
    print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
