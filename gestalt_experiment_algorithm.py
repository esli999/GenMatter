import os
import json
import time
import jax.numpy as jnp
from jax.random import key as jkey
import jax
from jax.scipy.ndimage import map_coordinates

import numpy as np
import pickle

from genparticles.datatypes import *
from genparticles.model_3d import *
from genparticles.inference import *
from genparticles.dataloader import *
from genparticles.utils import *
from genparticles.evaluation import *

model_jsimulate = jax.jit(HDGMM_model_3d.simulate)
model_jimportance = jax.jit(HDGMM_model_3d.importance)

######################################################
## CONFIGURATION START
######################################################

EXPERIMENT_NAME = "Gestalt_3D_Experiment_Algorithm_Only"
EXPERIMENT_SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

HYPERPARAMS = {
    "number_of_blobs": 64,  # Reduced for smaller point clouds
    "number_of_hyperblobs": 3,  # Reduced to 3 as requested
    "num_seeded_runs": 3,
}

SAVE_DATA = True

# Gestalt scene and texture configurations
GESTALT_SCENES = ['scene_00000', 'scene_00001']
GESTALT_TEXTURES = ['texture_00', 'texture_07', 'texture_13', 'texture_16']
GESTALT_BASE_PATH = '/home/esli/GenParticles_neural_stimulus/assets/from_thomas'
RAFT_FLOWS_PATH = '/home/esli/GenParticles_neural_stimulus/raft_flows'

######################################################
## CONFIGURATION END
######################################################

def resize_to_square(arr, target_hw):
    """Resize array to square aspect ratio using JAX interpolation"""
    N = arr.shape[0]
    if arr.ndim == 3:
        C = None
    elif arr.ndim == 4:
        C = arr.shape[-1]
    else:
        raise ValueError("arr must be 3D or 4D")
    H, W = arr.shape[1:3]
    ys = jnp.linspace(0, H - 1, target_hw)
    xs = jnp.linspace(0, W - 1, target_hw)
    grid_y, grid_x = jnp.meshgrid(ys, xs, indexing='ij')
    coords = jnp.stack([grid_y, grid_x], axis=0)

    def resize_single(d):
        if C is None:
            return map_coordinates(d, coords, order=1, mode='nearest')
        else:
            return jnp.stack([map_coordinates(d[..., c], coords, order=1, mode='nearest') for c in range(C)], axis=-1)
    arr_jax = jnp.array(arr)
    arr_resized = jax.vmap(resize_single)(arr_jax)
    return np.array(arr_resized)

def load_gestalt_data(scene, texture):
    """Load depth and flow data for a gestalt scene/texture combination"""
    # Load depth data
    depth_file = os.path.join(GESTALT_BASE_PATH, scene, texture, 'output_depths.npz')
    if not os.path.exists(depth_file):
        raise FileNotFoundError(f"Depth file not found: {depth_file}")
    
    depth_data = np.load(depth_file)
    if 'depths' in depth_data:
        inv_rel_depth = depth_data['depths']
    else:
        inv_rel_depth = next(iter(depth_data.values()))
    
    # Process depth data with better near-camera resolution
    # Clip inverse depth to wider range for better near-camera detail
    clipped_inverse_depth = np.clip(inv_rel_depth, 5, 1500)
    epsilon = 1e-6
    depth = 3000.0 / (clipped_inverse_depth + epsilon)
    # Wider depth range with better near-camera resolution
    depth = np.clip(depth, 0.25, 20.0)
    
    # Load flow data
    flow_file = os.path.join(RAFT_FLOWS_PATH, f'raft_flows_{scene}_{texture}.npz')
    if not os.path.exists(flow_file):
        raise FileNotFoundError(f"Flow file not found: {flow_file}")
    
    flow_data = np.load(flow_file)
    if 'flow' in flow_data:
        flow = flow_data['flow']
    else:
        flow = next(iter(flow_data.values()))
    
    return depth, flow

def compute_3d_points_and_motion(depth, flow, downsample_factor=8, min_motion_magnitude=0.05, focal_length_scale=1.0):
    """
    Compute 3D points and motion vectors from depth and flow with downsampling.
    Returns a validity mask indicating which points have significant motion.

    Args:
        depth: Depth maps
        flow: Optical flow
        downsample_factor: Downsampling factor for computational efficiency
        min_motion_magnitude: Minimum 3D motion magnitude to be considered "moving" (default: 0.05)
        focal_length_scale: Multiplier for focal length (higher = more depth separation, default: 1.0)

    Returns:
        points_3d: Array of 3D points
        motion_3d: Array of 3D motion vectors
        valid_mask: Boolean mask indicating which points have motion magnitude >= min_motion_magnitude
    """
    # Camera intrinsics
    orig_H, orig_W = depth.shape[1:3]
    target_hw = max(orig_H, orig_W)

    # Downsample target size to reduce computational load
    target_hw = target_hw // downsample_factor
    # Ensure target_hw is divisible by common block sizes (e.g., 32, 64)
    target_hw = (target_hw // 32) * 32
    if target_hw < 32:
        target_hw = 32

    print(f"Downsampling from {max(orig_H, orig_W)} to {target_hw} (factor: {downsample_factor})")

    # Resize to square with downsampling
    depth_sq = resize_to_square(depth, target_hw)
    flow_sq = resize_to_square(flow, target_hw)

    # Scale focal length - higher values push points further from camera, increasing depth separation
    fx = fy = 1000.0 * focal_length_scale * (target_hw / max(orig_H, orig_W))
    cx = cy = target_hw / 2

    all_points_3d = []
    all_motion_3d = []
    all_valid_masks = []
    total_valid = 0
    total_points = 0

    for frame_idx in range(flow_sq.shape[0]):
        flow_xy = flow_sq[frame_idx]
        depth_map = depth_sq[frame_idx]
        H, W = depth_map.shape

        # Generate pixel grid
        ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        xs = xs.astype(np.float32)
        ys = ys.astype(np.float32)

        # Backproject to 3D
        Z = depth_map
        X = (xs - cx) * Z / fx
        Y = (ys - cy) * Z / fy

        points_3d = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)

        # Compute 3D motion vectors
        flow_x = flow_xy[..., 0]
        flow_y = flow_xy[..., 1]

        xs2 = xs + flow_x
        ys2 = ys + flow_y
        xs2 = np.clip(xs2, 0, W - 1)
        ys2 = np.clip(ys2, 0, H - 1)

        # Use depth from next frame if available, otherwise use current frame
        if frame_idx + 1 < depth_sq.shape[0]:
            depth_map_next = depth_sq[frame_idx + 1]
            depth2 = depth_map_next[ys2.astype(int), xs2.astype(int)]
        else:
            depth2 = depth_map[ys2.astype(int), xs2.astype(int)]
        Z2 = depth2
        X2 = (xs2 - cx) * Z2 / fx
        Y2 = (ys2 - cy) * Z2 / fy

        points_3d_2 = np.stack([X2, Y2, Z2], axis=-1).reshape(-1, 3)
        motion_3d = points_3d_2 - points_3d

        # Create validity mask for motion vectors with SIGNIFICANT motion
        # For rotating objects: points with larger motion are the moving object
        # Points with near-zero motion are static background
        motion_magnitudes = np.linalg.norm(motion_3d, axis=-1)
        valid_mask = motion_magnitudes >= min_motion_magnitude

        num_valid = np.sum(valid_mask)
        total_valid += num_valid
        total_points += len(motion_3d)

        all_points_3d.append(points_3d)
        all_motion_3d.append(motion_3d)
        all_valid_masks.append(valid_mask)

    if total_points > 0:
        valid_pct = 100.0 * total_valid / total_points
        print(f"Valid motion vectors: {total_valid}/{total_points} ({valid_pct:.2f}%) with magnitude >= {min_motion_magnitude}")

    return np.array(all_points_3d), np.array(all_motion_3d), np.array(all_valid_masks)

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


def extract_gestalt_segmentation(depth_sq, flow_sq, points_3d):
    """
    Extract segmentation mask from first frame depth and flow using heuristics
    """
    # Use first frame - the points_3d is computed from flow frames, so use matching indices
    first_depth = depth_sq[0]  # Shape: (target_hw, target_hw)
    first_flow = flow_sq[0]    # Shape: (target_hw, target_hw, 2)
    first_points = points_3d[0]  # Shape: (H*W, 3)
    
    # Compute flow magnitude
    flow_magnitude = np.linalg.norm(first_flow, axis=-1).flatten()
    
    # Compute depth statistics from the same depth frame used for points_3d
    depth_flat = first_depth.flatten()
    depth_median = np.median(depth_flat)
    depth_std = np.std(depth_flat)
    
    # The points_3d should match the flow dimensions since they're computed together
    print(f"Debug: flow_magnitude shape: {len(flow_magnitude)}, depth_flat shape: {len(depth_flat)}, points shape: {len(first_points)}")
    
    # Since points_3d is computed from the flow loop, it should match flow dimensions
    # Use flow dimensions as the reference
    num_points = len(first_points)
    
    # Resize depth to match if needed
    if len(depth_flat) != num_points:
        # Interpolate depth to match point dimensions
        from scipy.interpolate import griddata
        target_hw = int(np.sqrt(num_points))
        h_indices, w_indices = np.meshgrid(np.arange(target_hw), np.arange(target_hw), indexing='ij')
        h_indices = h_indices.flatten()
        w_indices = w_indices.flatten()
        
        orig_h, orig_w = first_depth.shape
        orig_h_indices = np.linspace(0, orig_h-1, orig_h)
        orig_w_indices = np.linspace(0, orig_w-1, orig_w)
        orig_hh, orig_ww = np.meshgrid(orig_h_indices, orig_w_indices, indexing='ij')
        
        # Interpolate depth to match points grid
        points_grid = np.column_stack([h_indices, w_indices])
        orig_grid = np.column_stack([orig_hh.flatten(), orig_ww.flatten()])
        depth_flat = griddata(orig_grid, depth_flat, points_grid, method='linear', fill_value=depth_median)
    
    assert len(flow_magnitude) == len(depth_flat) == len(first_points), \
        f"Dimension mismatch: flow={len(flow_magnitude)}, depth={len(depth_flat)}, points={len(first_points)}"
    
    # Heuristic 1: Points with significant motion (above median flow magnitude)
    flow_threshold = np.median(flow_magnitude) + 0.5 * np.std(flow_magnitude)
    motion_mask = flow_magnitude > flow_threshold
    
    # Heuristic 2: Points at intermediate depths (avoid far background and very close points)
    depth_min_thresh = depth_median - 0.8 * depth_std
    depth_max_thresh = depth_median + 0.8 * depth_std
    depth_mask = (depth_flat > depth_min_thresh) & (depth_flat < depth_max_thresh)
    
    # Heuristic 3: Points with coherent motion (local flow consistency)
    flow_x = first_flow[..., 0]
    flow_y = first_flow[..., 1]
    
    # Compute local flow variance (coherence measure)
    from scipy.ndimage import uniform_filter
    kernel_size = 5
    flow_x_var = uniform_filter(flow_x**2, kernel_size) - uniform_filter(flow_x, kernel_size)**2
    flow_y_var = uniform_filter(flow_y**2, kernel_size) - uniform_filter(flow_y, kernel_size)**2
    flow_coherence = 1.0 / (1.0 + flow_x_var + flow_y_var)  # Higher values = more coherent
    
    coherence_threshold = np.median(flow_coherence.flatten()) + 0.3 * np.std(flow_coherence.flatten())
    coherence_mask = flow_coherence.flatten() > coherence_threshold
    
    # Combine heuristics: points must satisfy motion AND (depth OR coherence)
    final_mask = motion_mask & (depth_mask | coherence_mask)
    
    # Ensure we have a reasonable number of ROI points (at least 10% of total)
    if np.sum(final_mask) < len(final_mask) * 0.1:
        # Fallback: use top 20% by flow magnitude
        flow_percentile = np.percentile(flow_magnitude, 80)
        final_mask = flow_magnitude > flow_percentile
    
    print(f"Segmentation extracted: {np.sum(final_mask)}/{len(final_mask)} points ({100*np.sum(final_mask)/len(final_mask):.1f}%) in ROI")
    
    return final_mask

def run_gestalt_experiment(scene, texture, hyperparams, experiment_name):
    """Run HDGMM experiment on gestalt data - algorithm only, no visualization"""
    print(f"\nRunning experiment for {scene}_{texture}")
    
    # Load data
    depth, flow = load_gestalt_data(scene, texture)
    points_3d, motion_3d, motion_valid_mask = compute_3d_points_and_motion(depth, flow, downsample_factor=8, min_motion_magnitude=0.05)

    # Extract hyperparameters
    number_of_blobs = hyperparams["number_of_blobs"]
    number_of_hyperblobs = hyperparams["number_of_hyperblobs"]
    num_seeded_runs = hyperparams["num_seeded_runs"]

    # Process depth and flow to square format (same as in compute_3d_points_and_motion)
    orig_H, orig_W = depth.shape[1:3]
    target_hw = max(orig_H, orig_W) // 8  # Match the downsample_factor
    target_hw = (target_hw // 32) * 32
    if target_hw < 32:
        target_hw = 32

    depth_sq = resize_to_square(depth, target_hw)
    flow_sq = resize_to_square(flow, target_hw)

    # Extract segmentation mask from processed square data
    first_frame_seg = extract_gestalt_segmentation(depth_sq, flow_sq, points_3d)

    # Combine segmentation mask with motion validity mask (first frame)
    # Points must be in ROI AND have valid motion magnitude
    combined_mask = first_frame_seg & motion_valid_mask[0]
    print(f"Combined mask: {np.sum(combined_mask)}/{len(combined_mask)} points ({100*np.sum(combined_mask)/len(combined_mask):.1f}%) pass both ROI and motion magnitude filters")

    # Prepare data for HDGMM
    tracked_points = points_3d
    tracked_motion_vectors = motion_3d
    
    # Create hierarchical k-means clustering using combined mask
    print(f"Tracked points shape: {tracked_points.shape}")
    print(f"Tracked motion vectors shape: {tracked_motion_vectors.shape}")
    print(f"Combined mask shape: {combined_mask.shape}, valid points: {np.sum(combined_mask)}")

    kmeans_chm, roi_blob_indices, roi_hyperblob_indices = make_hierarchical_kmeans_chm_with_mask_fixed_hyperblob(
        tracked_points, number_of_blobs, number_of_hyperblobs,
        segmentation_mask=combined_mask, motion_vectors=tracked_motion_vectors
    )
    
    # Set up Gibbs sampling dials (same as DAVIS benchmark)
    GIBBS_DIALS = {
        "blob_weights": True,
        "hyperblob_weights": False,
        "blob_assignments": True,
        "hyperblob_assignments": True, # was false 
        "hyperblob_covs": False,
        "blob_covs": True,
        "blob_vel_covs": True,
        "blob_vel_means": True,
        "hyperblob_means": True, # was false
        "blob_means": True,
        'hyperblob_rot_vels': False,
        'hyperblob_trans_vels': False
    }

    # Set up hyperparameters scaled to the length scale of the data
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

    # Create hyperparameters
    # Outlier prob interpretation: with blob_weights summing to ~1.0 (from Dirichlet),
    # normalized outlier prob = outlier_prob / (1.0 + outlier_prob)
    # Original 5.0 → 83% outlier rate (too lenient)
    # New 0.1 → 9% outlier rate (stricter)
    #
    # Gamma(shape, rate) models outlier velocity magnitude ||v||
    # Enable model-based outlier detection with higher probability
    hypers = HDGMM_Hyperparams.create(
        outlier_prob=f_(0.001),
        outlier_velocity_gamma_shape=f_(7.5),  # Shape parameter for Gamma distribution over ||v||
        outlier_velocity_gamma_rate=f_(0.5),  # Rate: mean=1.0, favors velocities around 1.0
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
    
    # Initialize results storage
    all_tracking_data = []
    
    for seed_idx in range(num_seeded_runs):
        print(f"Running seed {seed_idx + 1}/{num_seeded_runs}")
        
        key = jkey(np.random.randint(1e9))

        # Initialize model
        key, key_importance = jax.random.split(key)
        init_tr, _ = model_jimportance(key_importance, kmeans_chm, (hypers,))
        init_hdgmm_state = init_tr.get_retval()

        # Run initial Gibbs sweeps (increased burn-in iterations for better convergence)
        key, init_gibbs_key = jax.random.split(key)
        gibbs_wtrs = hdgmm_full_gibbs(init_gibbs_key, init_hdgmm_state, 500, GIBBS_DIALS, use_weighted_blobs=True, num_gibbs_inner_loops=5)

        # Get final state
        init_hdgmm_state = gibbs_wtrs[-1].retval
        
        # Run tracking
        key, tracking_key = jax.random.split(key)
        tracking_wtrs = hdgmm_tracking_gibbs(tracking_key, init_hdgmm_state, tracked_points, tracked_motion_vectors)

        # Extract tracking data (hard threshold outlier rejection disabled)
        tracking_data = []
        for frame_idx in range(len(tracking_wtrs)):
            frame = tracking_wtrs[frame_idx]
            blob_assignments = np.array(frame.retval.datapoints_state.blob_assignments)
            n_blobs = frame.retval.hypers.n_blobs

            # DISABLED: Hard threshold outlier rejection
            # Let the model decide outliers based on statistical fit
            # if frame_idx < len(motion_valid_mask):
            #     invalid_motion_mask = ~motion_valid_mask[frame_idx]
            #     blob_assignments[invalid_motion_mask] = n_blobs  # Assign to outlier blob

            frame_data = {
                'n_blobs': n_blobs,
                'blob_assignments': blob_assignments
            }
            tracking_data.append(frame_data)
        
        all_tracking_data.append(tracking_data)
    
    return all_tracking_data, points_3d, motion_3d

def analyze_depth_quality(scene, texture):
    """Analyze depth map quality similar to gestalt_run.py"""
    depth_file = os.path.join(GESTALT_BASE_PATH, scene, texture, 'output_depths.npz')
    if not os.path.exists(depth_file):
        return None
    
    depths = np.load(depth_file)['depths']
    stats = {
        'scene_texture': f"{scene}_{texture}",
        'shape': depths.shape,
        'min': float(depths.min()),
        'max': float(depths.max()),
        'mean': float(depths.mean()),
        'std': float(depths.std()),
        'var': float(depths.var())
    }
    
    return stats

def main():
    """Main experiment execution - algorithm only"""
    print(f"Starting Gestalt 3D Experiment (Algorithm Only)")
    print(f"Experiment: {EXPERIMENT_NAME}")
    
    all_results = {}
    depth_stats = []
    experiment_start_time = time.time()
    
    total_combinations = len(GESTALT_SCENES) * len(GESTALT_TEXTURES)
    combination_idx = 0
    
    for scene in GESTALT_SCENES:
        for texture in GESTALT_TEXTURES:
            combination_idx += 1
            print(f"\nProcessing combination {combination_idx}/{total_combinations}: {scene}_{texture}")
            
            try:
                # Analyze depth quality
                depth_stat = analyze_depth_quality(scene, texture)
                if depth_stat:
                    depth_stats.append(depth_stat)
                    print(f"Depth stats for {scene}_{texture}: {depth_stat}")
                
                # Run experiment
                tracking_data, points_3d, motion_3d = run_gestalt_experiment(
                    scene, texture, HYPERPARAMS, EXPERIMENT_NAME
                )
                
                all_results[f"{scene}_{texture}"] = {
                    'tracking_data': tracking_data,
                    'points_3d': points_3d,
                    'motion_3d': motion_3d
                }
                
                print(f"Completed {scene}_{texture} - {len(tracking_data)} seeds, {len(tracking_data[0])} frames")
                
            except Exception as e:
                print(f"Error processing {scene}_{texture}: {e}")
                continue
    
    experiment_end_time = time.time()
    total_minutes = (experiment_end_time - experiment_start_time) / 60
    
    print(f"\nExperiment completed in {total_minutes:.2f} minutes")
    print(f"Processed {len(all_results)} scene/texture combinations")
    
    # Save results
    if SAVE_DATA:
        output_dir = os.path.join(EXPERIMENT_SAVE_DIR, EXPERIMENT_NAME)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save tracking results
        with open(os.path.join(output_dir, 'tracking_results.pkl'), 'wb') as f:
            pickle.dump(all_results, f)
        
        # Save depth statistics
        with open(os.path.join(output_dir, 'depth_statistics.json'), 'w') as f:
            json.dump(depth_stats, f, indent=2)
        
        print(f"Results saved to: {output_dir}")
    
    return all_results, depth_stats

if __name__ == "__main__":
    results, stats = main()