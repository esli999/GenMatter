#!/usr/bin/env python3
"""
Run one-frame gestalt segmentation on all scenes and textures using k-means initialization.
Uses the exact experiment procedure from one_frame_gestalt_experiment.ipynb
Saves segmentation masks to one_frame_gestalt_masks/scene_XXXXX/texture_XX/mask.png
"""

import sys
import os
import numpy as np
from PIL import Image
import jax
from jax.random import key as jkey
from scipy.ndimage import zoom

sys.path.append('/home/esli/GenMatter')

from gestalt_experiment_algorithm import (
    load_gestalt_data,
    compute_3d_points_and_motion,
    extract_gestalt_segmentation,
    resize_to_square,
    make_hierarchical_kmeans_chm_with_mask_fixed_hyperblob,
    HDGMM_Hyperparams,
    StaticJnp,
    model_jimportance,
    hdgmm_full_gibbs,
    HDGMM_model_3d
)
from genjax import ChoiceMapBuilder as C
import jax.numpy as jnp
from genparticles.inference import f_gibbs_sweep
from genparticles.trace_wrappers import hdgmm_TraceWrapper
from genparticles.datatypes import HDGMM_Gibbs_TraceWrapper


# Global configuration
DOWNSAMPLE_FACTOR = 8  # Match notebook settings


# Custom Gibbs function that computes log likelihoods during sampling
def hdgmm_full_gibbs_with_loglik(key, init_hdgmm_state, num_gibbs_sweeps, gibbs_dials,
                                  use_weighted_blobs=False, num_gibbs_inner_loops=1):
    """Modified version of hdgmm_full_gibbs that computes log likelihoods for each sample."""

    def compute_log_likelihood(hdgmm_state):
        """Compute log likelihood for a given state."""
        chm = (
            C['hyperblobs', 'hyperblob_weights'].set(hdgmm_state.hyperblobs_state.hyperblob_weights) |
            C['hyperblobs', 'hyperblob_means'].set(hdgmm_state.hyperblobs_state.hyperblob_means) |
            C['hyperblobs', 'hyperblob_covs'].set(hdgmm_state.hyperblobs_state.hyperblob_covs) |
            C['hyperblobs', 'hyperblob_trans_vels'].set(hdgmm_state.hyperblobs_state.hyperblob_trans_vels) |
            C['hyperblobs', 'hyperblob_rot_vels'].set(hdgmm_state.hyperblobs_state.hyperblob_rot_vels) |
            C['blobs', 'hyperblob_assignments'].set(hdgmm_state.blobs_state.hyperblob_assignments) |
            C['blobs', 'blob_weights'].set(hdgmm_state.blobs_state.blob_weights) |
            C['blobs', 'blob_means'].set(hdgmm_state.blobs_state.blob_means) |
            C['blobs', 'blob_covs'].set(hdgmm_state.blobs_state.blob_covs) |
            C['blobs', 'blob_vel_means'].set(hdgmm_state.blobs_state.blob_vel_means) |
            C['blobs', 'blob_vel_covs'].set(hdgmm_state.blobs_state.blob_vel_covs) |
            C['datapoints', 'blob_assignments'].set(hdgmm_state.datapoints_state.blob_assignments) |
            C['datapoints', 'datapoint_positions'].set(hdgmm_state.datapoints_state.datapoint_positions) |
            C['datapoints', 'datapoint_vels'].set(hdgmm_state.datapoints_state.datapoint_vels)
        )
        weight, _ = HDGMM_model_3d.assess(chm, (hdgmm_state.hypers,))
        return weight

    def f_gibbs_sweep_with_loglik(carry, gibbs_sweep_idx):
        """Gibbs sweep that also computes log likelihood."""
        # Run the normal Gibbs sweep
        new_carry, wtr = f_gibbs_sweep(carry, gibbs_sweep_idx)
        # Compute log likelihood for the resulting state
        log_lik = compute_log_likelihood(wtr.retval)
        # Create new wrapper with computed log likelihood
        wtr_with_loglik = hdgmm_TraceWrapper(force_retval=wtr.retval, force_log_likelihood=log_lik)
        return new_carry, wtr_with_loglik

    _, stacked_hdgmm_wtrs = jax.lax.scan(
        f_gibbs_sweep_with_loglik,
        (key, init_hdgmm_state, gibbs_dials, num_gibbs_inner_loops, use_weighted_blobs),
        jnp.arange(num_gibbs_sweeps)
    )

    # Compute log likelihood for initial state too
    init_log_lik = compute_log_likelihood(init_hdgmm_state)
    init_wtr = hdgmm_TraceWrapper(force_retval=init_hdgmm_state, force_log_likelihood=init_log_lik)

    gibbs_wtrs = HDGMM_Gibbs_TraceWrapper(init_wtr, stacked_hdgmm_wtrs)
    return gibbs_wtrs


def f_(x):
    """Convert to float32"""
    return jnp.float32(x)


def snp(x):
    """Convert to StaticJnp"""
    return StaticJnp(jnp.array(x))


def compute_hyperparameters(tracked_points_frame0, tracked_motion_vectors_frame0,
                            blob_assignments, hyperblob_assignments,
                            number_of_blobs, number_of_hyperblobs):
    """Compute empirical hyperparameters from k-means clustering"""

    # Blob means - handle empty blobs
    blob_means = []
    for i in range(number_of_blobs):
        blob_points = tracked_points_frame0[blob_assignments == i]
        if len(blob_points) > 0:
            blob_means.append(blob_points.mean(axis=0))
        else:
            # Empty blob - use global mean
            blob_means.append(tracked_points_frame0.mean(axis=0))
    blob_means = np.array(blob_means)

    # Hyperblob means - handle empty hyperblobs
    hyperblob_means = []
    for i in range(number_of_hyperblobs):
        hyperblob_blob_indices = hyperblob_assignments == i
        if np.any(hyperblob_blob_indices):
            hyperblob_means.append(blob_means[hyperblob_blob_indices].mean(axis=0))
        else:
            # Empty hyperblob - use global mean
            hyperblob_means.append(blob_means.mean(axis=0))
    hyperblob_means = np.array(hyperblob_means)

    empirical_mu_H = hyperblob_means.mean(axis=0)
    empirical_sigma_H = np.std(hyperblob_means, axis=0).mean()

    # Blob covariances
    blob_points = [tracked_points_frame0[blob_assignments == i] for i in range(number_of_blobs)]
    blob_covs = [np.cov(pts.T) if len(pts) > 3 else np.eye(3) * 0.01 for pts in blob_points]
    empirical_Psi_B = np.mean(blob_covs, axis=0)

    # Hyperblob covariances
    hyperblob_blob_means = [blob_means[hyperblob_assignments == i] for i in range(number_of_hyperblobs)]
    hyperblob_covs = [np.cov(means.T) if len(means) > 3 else np.eye(3) * 0.1
                      for means in hyperblob_blob_means]
    empirical_Psi_H = np.mean(hyperblob_covs, axis=0)

    # Velocity covariances
    blob_velocities = [tracked_motion_vectors_frame0[blob_assignments == i] for i in range(number_of_blobs)]
    blob_vel_covs = [np.cov(vels.T) if len(vels) > 3 else np.eye(3) * 0.01 for vels in blob_velocities]
    empirical_Psi_V = np.mean(blob_vel_covs, axis=0)

    # Degrees of freedom
    empirical_nu_H = 5.0
    empirical_nu_B = 5.0
    empirical_nu_V = 5.0

    return (empirical_mu_H, empirical_sigma_H, empirical_nu_H, empirical_Psi_H,
            empirical_nu_B, empirical_Psi_B, empirical_nu_V, empirical_Psi_V)


def process_scene_texture(scene_name, texture_name, custom_hyperparams, trial_num=0, random_seed=None):
    """Process a single scene/texture combination and save the segmentation mask

    Args:
        trial_num: Trial number (used for output directory naming)
        random_seed: Random seed for reproducibility (if None, uses random)

    Returns:
        dict with keys: 'accuracy', 'iou', 'precision', 'recall', 'mask_size', 'gt_size'
        or None if processing failed
    """

    print(f"\n{'='*70}")
    print(f"Processing {scene_name} / {texture_name} - Trial {trial_num}")
    print(f"{'='*70}")

    number_of_blobs = custom_hyperparams["number_of_blobs"]
    number_of_hyperblobs = custom_hyperparams["number_of_hyperblobs"]
    num_gibbs_iterations = custom_hyperparams.get("num_gibbs_iterations", 101)
    num_inner_loops = custom_hyperparams.get("num_inner_loops", 5)

    # STEP 1: Load data
    print("Loading data...")
    depth, flow = load_gestalt_data(scene_name, texture_name)

    # STEP 2: Compute 3D points and motion
    print("Computing 3D points and motion...")
    points_3d, motion_3d, motion_valid_mask = compute_3d_points_and_motion(
        depth, flow, downsample_factor=DOWNSAMPLE_FACTOR, min_motion_magnitude=0.05
    )

    tracked_points_frame0 = points_3d[0]
    tracked_motion_vectors_frame0 = motion_3d[0]

    # STEP 3: Extract gestalt segmentation
    print("Extracting gestalt segmentation...")
    orig_H, orig_W = depth.shape[1:3]
    target_hw = max(orig_H, orig_W) // DOWNSAMPLE_FACTOR
    target_hw = (target_hw // 32) * 32
    if target_hw < 32:
        target_hw = 32

    depth_sq = resize_to_square(depth, target_hw)
    flow_sq = resize_to_square(flow, target_hw)

    first_frame_seg = extract_gestalt_segmentation(depth_sq, flow_sq, points_3d)
    combined_mask = first_frame_seg & motion_valid_mask[0]

    print(f"  Valid points: {np.sum(combined_mask)} / {combined_mask.size}")

    # STEP 4: Initialize with hierarchical k-means (using exact notebook approach)
    print("Running hierarchical k-means initialization...")
    kmeans_chm, roi_blob_indices, roi_hyperblob_indices = make_hierarchical_kmeans_chm_with_mask_fixed_hyperblob(
        points_3d, number_of_blobs, number_of_hyperblobs,
        segmentation_mask=combined_mask, motion_vectors=motion_3d, frame_idx=0
    )

    blob_assignments = np.array(kmeans_chm['datapoints', 'blob_assignments'])
    hyperblob_assignments = np.array(kmeans_chm['blobs', 'hyperblob_assignments'])

    # STEP 5: Compute hyperparameters (exact notebook values)
    print("Computing hyperparameters...")
    (empirical_mu_H, empirical_sigma_H, empirical_nu_H, empirical_Psi_H,
     empirical_nu_B, empirical_Psi_B, empirical_nu_V, empirical_Psi_V) = compute_hyperparameters(
        tracked_points_frame0, tracked_motion_vectors_frame0,
        blob_assignments, hyperblob_assignments,
        number_of_blobs, number_of_hyperblobs
    )

    num_datapoints = kmeans_chm['datapoints', 'datapoint_positions'].shape[0]

    # Create hyperparameters with exact notebook settings
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
        rotation_vmf_kappa=snp(f_(100)),  # Notebook value
        rotation_angle_max_deg=snp(25),   # Notebook value
        rotation_angle_step_deg=snp(15),  # Notebook value
        n_hyperblobs=number_of_hyperblobs,
        n_blobs=number_of_blobs,
        n_datapoints=num_datapoints
    )

    # Gibbs dials (exact notebook settings)
    GIBBS_DIALS = {
        "blob_weights": True,
        "hyperblob_weights": False,
        "blob_assignments": True,
        "hyperblob_assignments": True,
        "hyperblob_covs": False,
        "blob_covs": True,
        "blob_vel_covs": True,
        "blob_vel_means": True,
        "hyperblob_means": True,
        "blob_means": True,
        'hyperblob_rot_vels': False,
        'hyperblob_trans_vels': False
    }

    # STEP 6: Run Gibbs sampling
    print(f"Running Gibbs sampling ({num_gibbs_iterations} iterations)...")
    # Use provided seed or generate random one
    if random_seed is None:
        random_seed = np.random.randint(1e9)
    print(f"  Random seed: {random_seed}")
    key = jkey(random_seed)

    key, key_importance = jax.random.split(key)
    init_tr, _ = model_jimportance(key_importance, kmeans_chm, (hypers,))
    init_hdgmm_state = init_tr.get_retval()

    key, init_gibbs_key = jax.random.split(key)
    gibbs_wtrs = hdgmm_full_gibbs_with_loglik(
        init_gibbs_key,
        init_hdgmm_state,
        num_gibbs_iterations,
        GIBBS_DIALS,
        use_weighted_blobs=True,
        num_gibbs_inner_loops=num_inner_loops
    )

    # STEP 7: Find max scoring trace
    print("Finding max scoring trace...")
    sample_freq = 1

    # Extract pre-computed log likelihoods (computed during Gibbs sampling)
    sampled_indices = list(range(0, len(gibbs_wtrs), sample_freq))
    log_probs = np.array([float(gibbs_wtrs[i].log_likelihood_) for i in sampled_indices])

    max_score_idx_in_sampled = np.argmax(log_probs)
    max_score_idx = sampled_indices[max_score_idx_in_sampled]
    max_score_state = gibbs_wtrs[max_score_idx].retval

    print(f"  Max log probability: {log_probs[max_score_idx_in_sampled]:.4f} at iteration {max_score_idx}")

    # STEP 8: Extract segmentation mask at origin
    print("Extracting segmentation mask...")
    max_blob_assignments = np.array(max_score_state.datapoints_state.blob_assignments)
    max_hyperblob_assignments = np.array(max_score_state.blobs_state.hyperblob_assignments)

    # Map pixels to hyperblobs
    max_pixel_hyperblobs = np.zeros_like(max_blob_assignments)
    for i in range(len(max_blob_assignments)):
        blob_id = max_blob_assignments[i]
        if blob_id < number_of_blobs:
            max_pixel_hyperblobs[i] = max_hyperblob_assignments[blob_id]
        else:
            max_pixel_hyperblobs[i] = -1

    # Get origin pixel
    origin_y, origin_x = target_hw // 2, target_hw // 2
    origin_pixel_idx = origin_y * target_hw + origin_x
    origin_blob = max_blob_assignments[origin_pixel_idx]

    if origin_blob < number_of_blobs:
        origin_hyperblob = max_hyperblob_assignments[origin_blob]
        origin_hyperblob_mask = max_pixel_hyperblobs == origin_hyperblob

        # Create binary mask
        mask_img = (origin_hyperblob_mask.reshape((target_hw, target_hw)) * 255).astype(np.uint8)

        # Save mask
        output_dir = f'one_frame_gestalt_masks/{scene_name}/{texture_name}/trial_{trial_num:03d}'
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'mask.png')

        Image.fromarray(mask_img).save(output_path)

        print(f"  Saved mask to: {output_path}")
        print(f"  Origin pixel belongs to Hyperblob {origin_hyperblob}")
        print(f"  Mask contains {np.sum(origin_hyperblob_mask)} pixels ({100*np.sum(origin_hyperblob_mask)/len(origin_hyperblob_mask):.1f}%)")
    else:
        print(f"  WARNING: Origin pixel is an outlier (blob_id={origin_blob})")
        # Save empty mask
        mask_img = np.zeros((target_hw, target_hw), dtype=np.uint8)
        output_dir = f'one_frame_gestalt_masks/{scene_name}/{texture_name}/trial_{trial_num:03d}'
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'mask.png')
        Image.fromarray(mask_img).save(output_path)
        print(f"  Saved empty mask to: {output_path}")

    # STEP 9: Load ground truth and compute accuracy
    print("Computing accuracy...")
    gt_path = f'/home/esli/GenMatter/assets/from_thomas/{scene_name}/render_passes/masks/Image0001.png'

    try:
        gt_img = np.array(Image.open(gt_path))

        # Handle different image formats (RGB, RGBA, grayscale)
        if len(gt_img.shape) == 3:
            gt_img = gt_img[:, :, 0]  # Take first channel

        gt_size = gt_img.shape[0]

        # Upsample predicted mask to match ground truth size
        # mask_img is target_hw x target_hw
        scale_factor = gt_size / target_hw
        mask_binary = (mask_img > 127).astype(float)
        mask_upsampled = zoom(mask_binary, scale_factor, order=1)  # bilinear interpolation
        mask_upsampled = (mask_upsampled > 0.5).astype(np.uint8)

        # Ensure same size (handle rounding issues)
        if mask_upsampled.shape[0] != gt_size:
            mask_upsampled = mask_upsampled[:gt_size, :gt_size]

        # Ground truth binary mask (assuming white=255 is the object)
        gt_binary = (gt_img > 127).astype(np.uint8)

        # Compute metrics
        true_positives = np.sum((mask_upsampled == 1) & (gt_binary == 1))
        true_negatives = np.sum((mask_upsampled == 0) & (gt_binary == 0))
        false_positives = np.sum((mask_upsampled == 1) & (gt_binary == 0))
        false_negatives = np.sum((mask_upsampled == 0) & (gt_binary == 1))

        accuracy = (true_positives + true_negatives) / (gt_size * gt_size)

        # IoU (Intersection over Union)
        intersection = true_positives
        union = true_positives + false_positives + false_negatives
        iou = intersection / union if union > 0 else 0.0

        # Precision and Recall
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0

        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  IoU: {iou:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")

        print(f"✓ Completed {scene_name} / {texture_name} - Trial {trial_num}")

        return {
            'accuracy': accuracy,
            'iou': iou,
            'precision': precision,
            'recall': recall,
            'mask_size': np.sum(mask_upsampled),
            'gt_size': np.sum(gt_binary),
            'trial': trial_num,
            'seed': random_seed,
            'max_log_prob': float(log_probs[max_score_idx_in_sampled])  # Log probability of max scoring trace
        }

    except Exception as e:
        print(f"  ERROR loading ground truth: {str(e)}")
        print(f"✓ Completed {scene_name} / {texture_name} (no accuracy computed)")
        return None


def main():
    """Main processing loop"""

    # Seed NumPy random for reproducibility
    np.random.seed(42)

    # Configuration
    num_trials = 2  # Number of trials to run for each scene/texture combination

    for n_blobs in [100]:  # Match notebook settings
        # Configuration
        custom_hyperparams = {
            "number_of_blobs": n_blobs,
            "number_of_hyperblobs": 5,  # Match notebook settings
            "num_gibbs_iterations": 101,  # Match notebook settings
            "num_inner_loops": 5,
        }

        # Scene and texture ranges
        scenes = [f'scene_{i:05d}' for i in range(20)]  # scene_00000 to scene_00019
        textures = ['texture_00']

        print("="*70)
        print("GESTALT SEGMENTATION BATCH PROCESSING (K-MEANS INITIALIZATION)")
        print("="*70)
        print(f"Scenes: {len(scenes)} ({scenes[0]} to {scenes[-1]})")
        print(f"Textures: {len(textures)} ({', '.join(textures)})")
        print(f"Trials per combination: {num_trials}")
        print(f"Total runs: {len(scenes) * len(textures) * num_trials}")
        print(f"Number of blobs: {custom_hyperparams['number_of_blobs']}")
        print(f"Number of hyperblobs: {custom_hyperparams['number_of_hyperblobs']}")
        print(f"Gibbs iterations: {custom_hyperparams['num_gibbs_iterations']}")
        print("="*70)

        # Process all combinations with multiple trials
        total = len(scenes) * len(textures) * num_trials
        completed = 0
        results = []

        for scene in scenes:
            for texture in textures:
                for trial in range(num_trials):
                    try:
                        # Generate a unique random seed for each trial
                        random_seed = np.random.randint(1e9)

                        metrics = process_scene_texture(scene, texture, custom_hyperparams,
                                                       trial_num=trial, random_seed=random_seed)
                        if metrics is not None:
                            metrics['scene'] = scene
                            metrics['texture'] = texture
                            results.append(metrics)
                        completed += 1
                        print(f"\nProgress: {completed}/{total} completed")
                    except Exception as e:
                        print(f"ERROR processing {scene}/{texture} trial {trial}: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        continue

        print("\n" + "="*70)
        print("BATCH PROCESSING COMPLETE")
        print("="*70)
        print(f"Successfully processed: {completed}/{total}")
        print(f"Output directory: one_frame_gestalt_masks/")

        # Aggregate statistics
        if results:
            print("\n" + "="*70)
            print("ACCURACY STATISTICS")
            print("="*70)

            accuracies = [r['accuracy'] for r in results]
            ious = [r['iou'] for r in results]
            precisions = [r['precision'] for r in results]
            recalls = [r['recall'] for r in results]

            print(f"\nTotal samples with ground truth: {len(results)}")
            print(f"Number of blobs: {custom_hyperparams['number_of_blobs']}")
            print(f"Number of hyperblobs: {custom_hyperparams['number_of_hyperblobs']}")

            print(f"\nAccuracy:")
            print(f"  Mean:   {np.mean(accuracies):.4f}")
            print(f"  Median: {np.median(accuracies):.4f}")
            print(f"  Std:    {np.std(accuracies):.4f}")
            print(f"  Min:    {np.min(accuracies):.4f}")
            print(f"  Max:    {np.max(accuracies):.4f}")

            print(f"\nIoU (Intersection over Union):")
            print(f"  Mean:   {np.mean(ious):.4f}")
            print(f"  Median: {np.median(ious):.4f}")
            print(f"  Std:    {np.std(ious):.4f}")
            print(f"  Min:    {np.min(ious):.4f}")
            print(f"  Max:    {np.max(ious):.4f}")

            print(f"\nPrecision:")
            print(f"  Mean:   {np.mean(precisions):.4f}")
            print(f"  Median: {np.median(precisions):.4f}")
            print(f"  Std:    {np.std(precisions):.4f}")
            print(f"  Min:    {np.min(precisions):.4f}")
            print(f"  Max:    {np.max(precisions):.4f}")

            print(f"\nRecall:")
            print(f"  Mean:   {np.mean(recalls):.4f}")
            print(f"  Median: {np.median(recalls):.4f}")
            print(f"  Std:    {np.std(recalls):.4f}")
            print(f"  Min:    {np.min(recalls):.4f}")
            print(f"  Max:    {np.max(recalls):.4f}")

            # Save detailed results to file
            n_blobs = custom_hyperparams['number_of_blobs']
            results_path = f'one_frame_gestalt_masks/accuracy_results_nblobs_{n_blobs}_kmeans_multitrials.txt'
            max_log_probs = [r['max_log_prob'] for r in results]

            with open(results_path, 'w') as f:
                f.write("Scene,Texture,Trial,Seed,Accuracy,IoU,Precision,Recall,MaxLogProb,MaskSize,GTSize\n")
                for r in results:
                    f.write(f"{r['scene']},{r['texture']},{r['trial']},{r['seed']},{r['accuracy']:.4f},{r['iou']:.4f},"
                        f"{r['precision']:.4f},{r['recall']:.4f},{r['max_log_prob']:.4f},{r['mask_size']},{r['gt_size']}\n")

                # Add mean/median/std statistics
                f.write(f"\nMEAN,,,,,{np.mean(accuracies):.4f},{np.mean(ious):.4f},"
                        f"{np.mean(precisions):.4f},{np.mean(recalls):.4f},{np.mean(max_log_probs):.4f},,\n")
                f.write(f"MEDIAN,,,,,{np.median(accuracies):.4f},{np.median(ious):.4f},"
                        f"{np.median(precisions):.4f},{np.median(recalls):.4f},{np.median(max_log_probs):.4f},,\n")
                f.write(f"STD,,,,,{np.std(accuracies):.4f},{np.std(ious):.4f},"
                        f"{np.std(precisions):.4f},{np.std(recalls):.4f},{np.std(max_log_probs):.4f},,\n")

            print(f"\nDetailed results saved to: {results_path}")
            print("="*70)


if __name__ == "__main__":
    main()
