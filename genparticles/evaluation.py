"""
Evaluation metrics for GenParticles tracking.

This module computes multiple metrics for evaluating particle-based tracking:

1. Particle-level metrics (recall, precision, FPR, IoU)
   - Treats particles as discrete units with fractional pixel contributions
   - Used for visualization and understanding per-particle behavior
   - LIMITATION: Treats all particles equally, sensitive to boundary particles

2. Matter-Weighted Recall/Precision/F1 metrics (TWO VARIANTS - RECOMMENDED):

   a) FIXED-WEIGHT Matter Metrics (avg_matter_weighted_f1_fixed):
      - Uses blob weights from frame 0 only, held constant across all frames
      - RECOMMENDED for fair comparison with point trackers (CoTracker, TAP-Net, TAPIR, etc.)
      - Point trackers have no concept of evolving importance/confidence
      - Both systems evaluated on same initial matter distribution
      - Use this metric in papers/benchmarks when comparing across methods

      To compare with point trackers:
      1. Initialize point tracker with blob_means from frame 0 after initial Gibbs
      2. Use frame 0 blob_weights to weight each point
      3. Compute same matter-weighted recall/precision/F1 for both systems
      4. This gives a fair apples-to-apples comparison

   b) ADAPTIVE-WEIGHT Matter Metrics (avg_matter_weighted_f1):
      - Uses per-frame blob weights that evolve during tracking
      - Shows advantage of probabilistic matter representation
      - Particles can be downweighted if they drift or become uncertain
      - Demonstrates that the model learns tracking confidence online
      - Use this to show benefit of your approach over fixed-point tracking

Both matter-weighted variants:
- Weight by (pixel_count × blob_weight) to prevent gaming with tiny particles
- Compute matter-weighted TP/FP/FN for balanced recall/precision
- F1 score provides single metric balancing both
- Naturally emphasizes spatially important particles over boundary particles
"""

import os
import numpy as np
from PIL import Image
import json
import os
import numpy as np

def get_segmentation_mask(davis_name, frame_idx, annotations_path, img_dims = None, flatten = True):
    seg_masks_path = os.path.join(annotations_path, davis_name)
    frame_path = os.path.join(seg_masks_path, f"{frame_idx:05d}.png")
    if img_dims is not None:
        new_height, new_width = img_dims
        frame_seg = np.array(Image.open(frame_path).resize((new_width, new_height), Image.NEAREST))
    else:
        frame_seg = np.array(Image.open(frame_path))
    
    # Handle both single-channel and multi-channel images
    if len(frame_seg.shape) == 3:  # Multi-channel image (H, W, C)
        # Consider a pixel as part of the mask if any channel has value >= 1
        frame_mask = np.any(frame_seg >= 1, axis=2)
    else:  # Single-channel image (H, W)
        frame_mask = frame_seg >= 1
        
    if flatten:
        frame_mask = frame_mask.reshape(-1)
    return frame_mask


def evaluate_tracking_results(
    davis_genparticles_dict, 
    annotations_path,
    counting_threshold=100, 
    img_dims=(520, 960), 
    save_data=False,
    output_dir=None,
    experiment_name="tracking_evaluation"
):
    """
    Evaluate tracking results for multiple GenParticles runs with different random trials across multiple DAVIS datasets.

    Args:
        davis_genparticles_dict: Dictionary where keys are davis_names and values are lists of lists.
            Each outer list is for a different random trial, each inner list is over all frames.
            Each element is a dict with 'n_blobs' and 'blob_assignments'.
        annotations_path: Path to the DAVIS annotations
        counting_threshold: Threshold for counting blobs
        img_dims: Image dimensions
        save_data: Whether to save results to a JSON file
        output_dir: Directory where to save results (required if save_data is True)
        experiment_name: Name of the experiment
    """


    # Validate inputs
    if (save_data) and output_dir is None:
        raise ValueError("output_dir must be provided if save_data is True")

    # Create output directories if needed
    if save_data:
        os.makedirs(output_dir, exist_ok=True)

    # Initialize results dictionary
    results_data = {
        "experiment_name": experiment_name,
        "datasets": {}
    }

    # Store all dataset average results for summary table
    all_datasets_results = {}
    
    # Process each dataset
    for davis_name, multiple_genparticles_list in davis_genparticles_dict.items():
        # Get first frame segmentation mask
        first_frame_mask = get_segmentation_mask(davis_name, 0, annotations_path, img_dims, flatten=False)
        
        # Initialize arrays to store metrics for each trial
        all_trials_data = []
        
        # Process each trial
        for trial_idx, per_trial_list in enumerate(multiple_genparticles_list):
            # per_trial_list: list over frames, each is a dict with 'n_blobs' and 'blob_assignments'
            num_frames = len(per_trial_list)
            
            # Get first frame assignments
            first_frame_assignments = np.array(per_trial_list[0]['blob_assignments']).reshape(img_dims)
            first_frame_n_blobs = per_trial_list[0]['n_blobs']
            
            # Get unique assignment values from the first frame (reference blobs)
            values, counts = np.unique(first_frame_assignments[first_frame_mask], return_counts=True)
            # Filter values that appear at least counting_threshold times and exclude the outlier blob
            reference_blobs = values[(counts >= counting_threshold) & (values != first_frame_n_blobs)]
            total_reference_blobs = len(reference_blobs)
            
            # Initialize arrays to store metrics for each frame
            percentage_preserved_scores = np.zeros(num_frames)
            
            # Process each frame to calculate metrics
            for frame_idx in range(num_frames):
                # Load the ground truth segmentation mask for this frame
                frame_mask = get_segmentation_mask(davis_name, frame_idx, annotations_path, img_dims, flatten=False)
                
                # Get the blob assignments for this frame
                if frame_idx < len(per_trial_list):
                    frame_assignments = np.array(per_trial_list[frame_idx]['blob_assignments']).reshape(img_dims)
                    frame_n_blobs = per_trial_list[frame_idx]['n_blobs']
                else:
                    # If we don't have assignments for all frames, use the last available one
                    frame_assignments = np.array(per_trial_list[-1]['blob_assignments']).reshape(img_dims)
                    frame_n_blobs = per_trial_list[-1]['n_blobs']
                
                # Calculate percentage of preserved blobs
                frame_values = np.unique(frame_assignments[frame_mask])
                frame_values = frame_values[frame_values != frame_n_blobs]  # Exclude outlier blob
                common_blobs = np.intersect1d(reference_blobs, frame_values)
                percentage_preserved_scores[frame_idx] = (len(common_blobs) / total_reference_blobs) * 100 if total_reference_blobs > 0 else 0
            
            # Calculate average for this trial
            avg_preserved = float(np.mean(percentage_preserved_scores))
            
            # Store data for this trial
            trial_data = {
                'avg_preserved': avg_preserved
            }
            
            all_trials_data.append(trial_data)

        # Calculate average metrics across all trials
        avg_preserved_values = [data['avg_preserved'] for data in all_trials_data]
        
        avg_preserved = float(np.mean(avg_preserved_values))
        std_preserved = float(np.std(avg_preserved_values))
        
        # Store average results for summary table
        all_datasets_results[davis_name] = {
            'avg_preserved': avg_preserved,
            'std_preserved': std_preserved
        }
                
        # Store dataset results
        dataset_results = {
            "all_trials_data": all_trials_data,
            "avg_metrics": {
                'avg_preserved': avg_preserved,
                'std_preserved': std_preserved
            },
        }
        
        # Add to results data
        if save_data:
            results_data["datasets"][davis_name] = dataset_results
        
    # Save results to JSON if requested
    if save_data:
        json_path = os.path.join(output_dir, f"{experiment_name}_results.json")
        with open(json_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        print(f"Results saved to: {json_path}")


def evaluate_single_davis_video(
    davis_name,
    multiple_genparticles_list,
    annotations_path,
    counting_threshold=0,
    img_dims=(520, 960),
    fps_list=None,
    render_results_video=True,
    experiment_save_dir=None,
    force_below_count_thresh_as_outlier=False,
    subsampled_indices=None
):
    """
    Evaluate tracking results for a single DAVIS video with multiple random trials.
    For visualization, shows the best trial in terms of recall for most plots,
    but overlays the mean tracking metrics across all trials as lines (by-frame),
    not static lines. All plots and legends are made explicit to clarify this
    distinction. Also displays particle counts before and after first frame
    Gibbs filtering, and improves legends, coloring, and layout.
    
    Args:
        force_below_count_thresh_as_outlier: If True, particles below counting_threshold
            in the first frame mask are treated as outliers and excluded from all
            TP/FP/TN/FN calculations. If False (default), they contribute to background
            classification (TN/FP).
        subsampled_indices: If provided, indices to subsample the flattened image arrays.
            Used for memory efficiency with large images.
    """
    # if render_results_video and experiment_save_dir is None:
    #     raise ValueError("experiment_save_dir must be provided if render_results_video is True")
    
    print(f"\n{'='*80}")
    print(f"📊 Processing dataset: {davis_name}")
    print(f"{'='*80}")
    
    # Get first frame segmentation mask (slice to subsampled indices if provided)
    first_frame_mask_full = get_segmentation_mask(davis_name, 0, annotations_path, img_dims, flatten=True)
    first_frame_mask = first_frame_mask_full if subsampled_indices is None else first_frame_mask_full[subsampled_indices]
    
    all_trials_data = []
    all_visualization_trial_data = []
    all_reference_particles_before_threshold = None
    first_frame_outlier_count = None

    # To record per-frame metrics for mean calculation across runs
    per_frame_recall_all_trials = []
    per_frame_precision_all_trials = []
    per_frame_fpr_all_trials = []
    per_frame_jaccard_all_trials = []
    per_frame_accuracy_all_trials = []
    per_frame_matter_weighted_f1_all_trials = []
    per_frame_matter_weighted_f1_fixed_all_trials = []
    per_frame_matter_weighted_accuracy_all_trials = []
    per_frame_matter_weighted_accuracy_fixed_all_trials = []

    for trial_idx, per_trial_list in enumerate(multiple_genparticles_list):
        num_frames = len(per_trial_list)
        first_frame_assignments = np.array(per_trial_list[0]['blob_assignments'])
        
        # Ensure assignments are always 1D (flattened) for consistent processing
        # Note: If subsampled_indices is provided, assignments are already subsampled
        # (they match the subsampled datapoints from tracking). subsampled_indices is only used for the mask.
        if first_frame_assignments.ndim > 1:
            first_frame_assignments = first_frame_assignments.flatten()
        
        first_frame_n_blobs = per_trial_list[0]['n_blobs']

        true_unique_particles = [bid for bid in np.unique(first_frame_assignments) if bid != first_frame_n_blobs]
        num_true_unique_particles = len(true_unique_particles)

        # Use first_frame_assignments directly (already 1D) for indexing with first_frame_mask (which is always 1D)
        values_all_particles, counts_all_particles = np.unique(first_frame_assignments[first_frame_mask], return_counts=True)
        all_reference_particles_before_threshold = values_all_particles[values_all_particles != first_frame_n_blobs]
        n_all_reference_particles_before_threshold = len(all_reference_particles_before_threshold)

        reference_particles = values_all_particles[(counts_all_particles >= counting_threshold) & (values_all_particles != first_frame_n_blobs)]
        total_reference_particles = len(reference_particles)
        n_total_particles_in_scene = len(np.unique(first_frame_assignments[first_frame_assignments != first_frame_n_blobs]))

        # Determine which particles should be treated as outliers
        if force_below_count_thresh_as_outlier:
            # Particles below threshold in first frame mask become outliers
            below_threshold_particles = values_all_particles[(counts_all_particles < counting_threshold) & (values_all_particles != first_frame_n_blobs)]
            outlier_particles = np.concatenate([np.array([first_frame_n_blobs]), below_threshold_particles])
        else:
            # Only the original outlier blob
            outlier_particles = np.array([first_frame_n_blobs])

        outlier_mask = (first_frame_assignments == first_frame_n_blobs)
        first_frame_outlier_count = np.unique(first_frame_assignments)[np.unique(first_frame_assignments) == first_frame_n_blobs]
        n_outlier_particles = int(len(first_frame_outlier_count > 0))
        actual_outlier_pixels = np.sum(outlier_mask)

        recall_scores = np.zeros(num_frames)
        precision_scores = np.zeros(num_frames)
        fpr_scores = np.zeros(num_frames)
        jaccard_scores = np.zeros(num_frames)
        accuracy_scores = np.zeros(num_frames)

        # Matter-weighted metrics (adaptive and fixed variants)
        matter_weighted_recall_scores = np.zeros(num_frames)
        matter_weighted_precision_scores = np.zeros(num_frames)
        matter_weighted_f1_scores = np.zeros(num_frames)
        matter_weighted_jaccard_scores = np.zeros(num_frames)
        matter_weighted_accuracy_scores = np.zeros(num_frames)
        matter_weighted_recall_fixed_scores = np.zeros(num_frames)
        matter_weighted_precision_fixed_scores = np.zeros(num_frames)
        matter_weighted_f1_fixed_scores = np.zeros(num_frames)
        matter_weighted_jaccard_fixed_scores = np.zeros(num_frames)
        matter_weighted_accuracy_fixed_scores = np.zeros(num_frames)

        tpr_values = []
        fpr_values = []
        frame0_blob_weights = None  # Cache for fixed-weight metric

        for frame_idx in range(num_frames):
            frame_mask_full = get_segmentation_mask(davis_name, frame_idx, annotations_path, img_dims, flatten=True)
            frame_mask = frame_mask_full if subsampled_indices is None else frame_mask_full[subsampled_indices]
            
            if frame_idx < len(per_trial_list):
                frame_assignments = np.array(per_trial_list[frame_idx]['blob_assignments'])
                
                # Ensure assignments are always 1D (flattened) for consistent processing
                # Note: If subsampled_indices is provided, assignments are already subsampled
                # (they match the subsampled datapoints from tracking). subsampled_indices is only used for the mask.
                if frame_assignments.ndim > 1:
                    frame_assignments = frame_assignments.flatten()
                
                frame_n_blobs = per_trial_list[frame_idx]['n_blobs']
                frame_blob_weights = np.array(per_trial_list[frame_idx]['blob_weights'])
            else:
                raise ValueError(f"No assignments for frame {frame_idx} in trial {trial_idx}")

            all_current_particles = np.unique(frame_assignments)
            all_current_particles = all_current_particles[all_current_particles != frame_n_blobs]

            # Create mask for valid particles (excluding outliers)
            valid_mask = ~np.isin(frame_assignments, outlier_particles)
            
            # Both assignments and masks are already 1D, so we can index directly
            flat_assignments = frame_assignments[valid_mask]
            flat_gt_mask = frame_mask[valid_mask]

            unique_particles, particle_inverse, particle_counts = np.unique(flat_assignments, return_inverse=True, return_counts=True)
            ref_particle_mask = np.isin(unique_particles, reference_particles)

            pixels_inside_gt_per_particle = np.bincount(particle_inverse, weights=flat_gt_mask, minlength=len(unique_particles))
            total_particle_pixels = particle_counts
            pixels_outside_gt_per_particle = total_particle_pixels - pixels_inside_gt_per_particle

            with np.errstate(divide='ignore', invalid='ignore'):
                fraction_inside = np.where(total_particle_pixels > 0, pixels_inside_gt_per_particle / total_particle_pixels, 0)
                fraction_outside = np.where(total_particle_pixels > 0, pixels_outside_gt_per_particle / total_particle_pixels, 0)

            tp_count = np.sum(fraction_inside[ref_particle_mask])
            fn_count = np.sum(fraction_outside[ref_particle_mask])
            fp_count = np.sum(fraction_inside[~ref_particle_mask])
            tn_count = np.sum(fraction_outside[~ref_particle_mask])
            
            recall_scores[frame_idx] = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
            precision_scores[frame_idx] = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
            fpr_scores[frame_idx] = fp_count / (fp_count + tn_count) if (fp_count + tn_count) > 0 else 0
            accuracy_scores[frame_idx] = (tp_count + tn_count) / (tp_count + tn_count + fp_count + fn_count) if (tp_count + tn_count + fp_count + fn_count) > 0 else 0
            
            intersection = tp_count
            union = tp_count + fp_count + fn_count
            jaccard_scores[frame_idx] = intersection / union if union > 0 else 0
            tpr_values.append(recall_scores[frame_idx])
            fpr_values.append(fpr_scores[frame_idx])

            # ========================================================================
            # Matter-Weighted Recall/Precision/F1/Jaccard/Accuracy Metrics (Two Variants)
            # ========================================================================
            # These metrics address the limitation that unweighted particle-level
            # accuracy treats all particles equally, making results too sensitive to
            # boundary particles with few pixels.
            #
            # KEY INSIGHT: Not all particles represent equal "matter" or importance.
            # A particle could have:
            #   - High weight + 1 pixel (trivially correct but not spatially important)
            #   - Low weight + 1000 pixels (spatially important but downweighted)
            #
            # SOLUTION: Weight each particle's TP/FP/FN/TN contribution by its "matter":
            #   matter_weight = pixel_count × blob_weight
            #
            # This prevents gaming (can't get perfect score with 1 tiny particle) and
            # naturally emphasizes spatially important particles.
            #
            # We compute TWO variants for different evaluation purposes:
            #
            # 1. FIXED-WEIGHT MATTER METRICS (frame 0 weights only)
            #    - Uses blob weights from frame 0, held constant across all frames
            #    - RECOMMENDED for fair comparison with point trackers
            #    - Point trackers (CoTracker, TAP-Net, TAPIR) have no evolving importance
            #    - Both systems evaluated on same initial matter distribution
            #    - Use this for cross-method comparisons in papers/benchmarks
            #
            #    To compare with point trackers:
            #    a) Initialize point tracker with blob_means from frame 0 (post-Gibbs)
            #    b) Weight each point by frame 0 blob_weight for that particle
            #    c) Compute same matter-weighted recall/precision/accuracy for both systems
            #    d) This gives a fair apples-to-apples comparison
            #
            # 2. ADAPTIVE-WEIGHT MATTER METRICS (per-frame weights)
            #    - Uses per-frame blob weights that evolve during tracking
            #    - Shows advantage of probabilistic matter representation
            #    - Particles can be downweighted if they drift or become uncertain
            #    - Demonstrates that the model learns tracking confidence online
            #    - Use this to show benefit of your approach over fixed-point tracking
            #
            # METRICS COMPUTED:
            # - Matter-weighted Recall: What fraction of object matter stays in GT mask?
            #   recall = sum(TP_matter) / sum(TP_matter + FN_matter)
            #
            # - Matter-weighted Precision: What fraction of predicted matter is correct?
            #   precision = sum(TP_matter) / sum(TP_matter + FP_matter)
            #
            # - Matter-weighted Accuracy: What fraction of all matter is correctly classified?
            #   accuracy = sum(TP_matter + TN_matter) / sum(TP_matter + TN_matter + FP_matter + FN_matter)
            #
            # - Matter-weighted F1: Harmonic mean balancing recall and precision
            #   f1 = 2 * (recall × precision) / (recall + precision)
            #
            # - Matter-weighted Jaccard (IoU): Intersection over Union for matter
            #   jaccard = sum(TP_matter) / sum(TP_matter + FP_matter + FN_matter)
            #
            # These metrics are balanced (penalize both FP and FN) and prevent exploits.
            # ========================================================================

            # Get blob weights for this frame (adaptive) and frame 0 (fixed)
            particle_blob_weights_adaptive = np.array([frame_blob_weights[p] if p < len(frame_blob_weights) else 0.0
                                                       for p in unique_particles])

            # On first frame, cache the frame 0 weights for fixed-weight metric
            if frame_idx == 0:
                frame0_blob_weights = frame_blob_weights.copy()

            particle_blob_weights_fixed = np.array([frame0_blob_weights[p] if p < len(frame0_blob_weights) else 0.0
                                                    for p in unique_particles])

            # Compute matter weights for both variants
            # Matter weight = pixel_count × blob_weight (prevents 1-pixel exploit)
            matter_weight_adaptive = total_particle_pixels * particle_blob_weights_adaptive
            matter_weight_fixed = total_particle_pixels * particle_blob_weights_fixed

            # Compute matter-weighted TP/FP/FN/TN counts
            # fraction_inside[i] = fraction of particle i's pixels inside GT mask
            # fraction_outside[i] = fraction of particle i's pixels outside GT mask
            # ref_particle_mask[i] = True if particle i is a reference (object) particle

            # TP matter: Reference particles' pixels that are inside GT mask
            # FN matter: Reference particles' pixels that are outside GT mask
            # FP matter: Non-reference particles' pixels that are inside GT mask
            # TN matter: Non-reference particles' pixels that are outside GT mask

            # ADAPTIVE-WEIGHT METRICS
            tp_matter_adaptive = np.sum(fraction_inside[ref_particle_mask] * matter_weight_adaptive[ref_particle_mask])
            fn_matter_adaptive = np.sum(fraction_outside[ref_particle_mask] * matter_weight_adaptive[ref_particle_mask])
            fp_matter_adaptive = np.sum(fraction_inside[~ref_particle_mask] * matter_weight_adaptive[~ref_particle_mask])
            tn_matter_adaptive = np.sum(fraction_outside[~ref_particle_mask] * matter_weight_adaptive[~ref_particle_mask])

            if (tp_matter_adaptive + fn_matter_adaptive) > 0:
                matter_weighted_recall_scores[frame_idx] = tp_matter_adaptive / (tp_matter_adaptive + fn_matter_adaptive)
            else:
                matter_weighted_recall_scores[frame_idx] = 0.0

            if (tp_matter_adaptive + fp_matter_adaptive) > 0:
                matter_weighted_precision_scores[frame_idx] = tp_matter_adaptive / (tp_matter_adaptive + fp_matter_adaptive)
            else:
                matter_weighted_precision_scores[frame_idx] = 0.0

            # Accuracy score
            total_matter_adaptive = tp_matter_adaptive + tn_matter_adaptive + fp_matter_adaptive + fn_matter_adaptive
            if total_matter_adaptive > 0:
                matter_weighted_accuracy_scores[frame_idx] = (tp_matter_adaptive + tn_matter_adaptive) / total_matter_adaptive
            else:
                matter_weighted_accuracy_scores[frame_idx] = 0.0

            # F1 score (harmonic mean of recall and precision)
            recall_val = matter_weighted_recall_scores[frame_idx]
            precision_val = matter_weighted_precision_scores[frame_idx]
            if (recall_val + precision_val) > 0:
                matter_weighted_f1_scores[frame_idx] = 2 * (recall_val * precision_val) / (recall_val + precision_val)
            else:
                matter_weighted_f1_scores[frame_idx] = 0.0

            # Jaccard (IoU) score
            union_matter_adaptive = tp_matter_adaptive + fp_matter_adaptive + fn_matter_adaptive
            if union_matter_adaptive > 0:
                matter_weighted_jaccard_scores[frame_idx] = tp_matter_adaptive / union_matter_adaptive
            else:
                matter_weighted_jaccard_scores[frame_idx] = 0.0

            # FIXED-WEIGHT METRICS (for fair comparison with point trackers)
            tp_matter_fixed = np.sum(fraction_inside[ref_particle_mask] * matter_weight_fixed[ref_particle_mask])
            fn_matter_fixed = np.sum(fraction_outside[ref_particle_mask] * matter_weight_fixed[ref_particle_mask])
            fp_matter_fixed = np.sum(fraction_inside[~ref_particle_mask] * matter_weight_fixed[~ref_particle_mask])
            tn_matter_fixed = np.sum(fraction_outside[~ref_particle_mask] * matter_weight_fixed[~ref_particle_mask])

            if (tp_matter_fixed + fn_matter_fixed) > 0:
                matter_weighted_recall_fixed_scores[frame_idx] = tp_matter_fixed / (tp_matter_fixed + fn_matter_fixed)
            else:
                matter_weighted_recall_fixed_scores[frame_idx] = 0.0

            if (tp_matter_fixed + fp_matter_fixed) > 0:
                matter_weighted_precision_fixed_scores[frame_idx] = tp_matter_fixed / (tp_matter_fixed + fp_matter_fixed)
            else:
                matter_weighted_precision_fixed_scores[frame_idx] = 0.0

            # Accuracy score
            total_matter_fixed = tp_matter_fixed + tn_matter_fixed + fp_matter_fixed + fn_matter_fixed
            if total_matter_fixed > 0:
                matter_weighted_accuracy_fixed_scores[frame_idx] = (tp_matter_fixed + tn_matter_fixed) / total_matter_fixed
            else:
                matter_weighted_accuracy_fixed_scores[frame_idx] = 0.0

            # F1 score
            recall_val_fixed = matter_weighted_recall_fixed_scores[frame_idx]
            precision_val_fixed = matter_weighted_precision_fixed_scores[frame_idx]
            if (recall_val_fixed + precision_val_fixed) > 0:
                matter_weighted_f1_fixed_scores[frame_idx] = 2 * (recall_val_fixed * precision_val_fixed) / (recall_val_fixed + precision_val_fixed)
            else:
                matter_weighted_f1_fixed_scores[frame_idx] = 0.0

            # Jaccard (IoU) score
            union_matter_fixed = tp_matter_fixed + fp_matter_fixed + fn_matter_fixed
            if union_matter_fixed > 0:
                matter_weighted_jaccard_fixed_scores[frame_idx] = tp_matter_fixed / union_matter_fixed
            else:
                matter_weighted_jaccard_fixed_scores[frame_idx] = 0.0
        
        avg_recall = float(np.mean(recall_scores))
        std_recall = float(np.std(recall_scores))

        if len(tpr_values) > 1 and len(set(fpr_values)) > 1:
            sorted_indices = np.argsort(fpr_values)
            sorted_fpr = np.array(fpr_values)[sorted_indices]
            sorted_tpr = np.array(tpr_values)[sorted_indices]
            auc_roc = np.trapz(sorted_tpr, sorted_fpr)
        else:
            auc_roc = 0.0

        trial_data = {
            'avg_recall': avg_recall,
            'std_recall': std_recall,
            'avg_precision': float(np.mean(precision_scores)),
            'avg_fpr': float(np.mean(fpr_scores)),
            'avg_jaccard': float(np.mean(jaccard_scores)),
            'avg_accuracy': float(np.mean(accuracy_scores)),
            'avg_matter_weighted_recall': float(np.mean(matter_weighted_recall_scores)),
            'avg_matter_weighted_precision': float(np.mean(matter_weighted_precision_scores)),
            'avg_matter_weighted_f1': float(np.mean(matter_weighted_f1_scores)),
            'avg_matter_weighted_jaccard': float(np.mean(matter_weighted_jaccard_scores)),
            'avg_matter_weighted_accuracy': float(np.mean(matter_weighted_accuracy_scores)),
            'avg_matter_weighted_recall_fixed': float(np.mean(matter_weighted_recall_fixed_scores)),
            'avg_matter_weighted_precision_fixed': float(np.mean(matter_weighted_precision_fixed_scores)),
            'avg_matter_weighted_f1_fixed': float(np.mean(matter_weighted_f1_fixed_scores)),
            'avg_matter_weighted_jaccard_fixed': float(np.mean(matter_weighted_jaccard_fixed_scores)),
            'avg_matter_weighted_accuracy_fixed': float(np.mean(matter_weighted_accuracy_fixed_scores)),
            'auc_roc': float(auc_roc)
        }
        visualization_data_entry = {
            'per_trial_list': per_trial_list,
            'reference_particles': reference_particles,
            'total_reference_particles': total_reference_particles,
            'all_reference_particles_before_threshold': all_reference_particles_before_threshold,
            'n_all_reference_particles_before_threshold': n_all_reference_particles_before_threshold,
            'n_total_particles_in_scene': n_total_particles_in_scene,
            'num_true_unique_particles_first_frame': num_true_unique_particles,
            'first_frame_n_blobs': first_frame_n_blobs,
            'counting_threshold': counting_threshold,
            'recall_scores': recall_scores,
            'precision_scores': precision_scores,
            'fpr_scores': fpr_scores,
            'jaccard_scores': jaccard_scores,
            'accuracy_scores': accuracy_scores,
            'matter_weighted_recall_scores': matter_weighted_recall_scores,
            'matter_weighted_precision_scores': matter_weighted_precision_scores,
            'matter_weighted_f1_scores': matter_weighted_f1_scores,
            'matter_weighted_jaccard_scores': matter_weighted_jaccard_scores,
            'matter_weighted_accuracy_scores': matter_weighted_accuracy_scores,
            'matter_weighted_recall_fixed_scores': matter_weighted_recall_fixed_scores,
            'matter_weighted_precision_fixed_scores': matter_weighted_precision_fixed_scores,
            'matter_weighted_f1_fixed_scores': matter_weighted_f1_fixed_scores,
            'matter_weighted_jaccard_fixed_scores': matter_weighted_jaccard_fixed_scores,
            'matter_weighted_accuracy_fixed_scores': matter_weighted_accuracy_fixed_scores,
            'tpr_values': tpr_values,
            'fpr_values': fpr_values,
            'num_frames': num_frames,
            'n_outlier_particles': n_outlier_particles,
            'actual_outlier_pixels': actual_outlier_pixels,
            'avg_recall': avg_recall
        }
        all_visualization_trial_data.append(visualization_data_entry)
        all_trials_data.append(trial_data)

        per_frame_recall_all_trials.append(recall_scores)
        per_frame_precision_all_trials.append(precision_scores)
        per_frame_fpr_all_trials.append(fpr_scores)
        per_frame_jaccard_all_trials.append(jaccard_scores)
        per_frame_accuracy_all_trials.append(accuracy_scores)
        per_frame_matter_weighted_f1_all_trials.append(matter_weighted_f1_scores)
        per_frame_matter_weighted_f1_fixed_all_trials.append(matter_weighted_f1_fixed_scores)
        per_frame_matter_weighted_accuracy_all_trials.append(matter_weighted_accuracy_scores)
        per_frame_matter_weighted_accuracy_fixed_all_trials.append(matter_weighted_accuracy_fixed_scores)
    
    print(f"\n📈 Trial Results for {davis_name}:")
    print(f"{'Trial':<8} {'Recall (%)':<12} {'Precision':<10} {'FPR':<8} {'Jaccard':<8} {'Accuracy':<10} {'MW-F1-A':<10} {'MW-F1-F':<10} {'MW-J-A':<8} {'MW-J-F':<8} {'MW-Acc-A':<10} {'MW-Acc-F':<10} {'AUC':<8}")
    print(f"{'-'*134}")
    for trial_idx, trial_data in enumerate(all_trials_data):
        recall_pct = trial_data['avg_recall'] * 100
        print(f"{trial_idx+1:<8} {recall_pct:<12.2f} "
              f"{trial_data['avg_precision']:<10.3f} {trial_data['avg_fpr']:<8.3f} "
              f"{trial_data['avg_jaccard']:<8.3f} {trial_data['avg_accuracy']:<10.3f} "
              f"{trial_data['avg_matter_weighted_f1']:<10.3f} "
              f"{trial_data['avg_matter_weighted_f1_fixed']:<10.3f} {trial_data['avg_matter_weighted_jaccard']:<8.3f} "
              f"{trial_data['avg_matter_weighted_jaccard_fixed']:<8.3f} {trial_data['avg_matter_weighted_accuracy']:<10.3f} "
              f"{trial_data['avg_matter_weighted_accuracy_fixed']:<10.3f} {trial_data['auc_roc']:<8.3f}")
    print(f"{'-'*134}")
    print(f"Note: MW-F1-A = Matter-Weighted F1 (Adaptive weights), MW-F1-F = Matter-Weighted F1 (Fixed frame-0 weights)")
    print(f"      MW-J-A = Matter-Weighted Jaccard (Adaptive weights), MW-J-F = Matter-Weighted Jaccard (Fixed frame-0 weights)")
    print(f"      MW-Acc-A = Matter-Weighted Accuracy (Adaptive weights), MW-Acc-F = Matter-Weighted Accuracy (Fixed frame-0 weights)")
    print(f"      Use MW-F1-F, MW-J-F, and MW-Acc-F for fair comparison with point trackers (CoTracker, TAP-Net, TAPIR, etc.)")
    print(f"      Use MW-F1-A, MW-J-A, and MW-Acc-A to show benefit of adaptive probabilistic matter representation")
    print(f"      Both metrics weight by (pixel_count × blob_weight) to emphasize spatially important particles")

    avg_recall_values = [data['avg_recall'] for data in all_trials_data]
    std_recall_values = [data['std_recall'] for data in all_trials_data]
    overall_mean = float(np.mean(avg_recall_values))
    overall_median = float(np.median(avg_recall_values))
    overall_std = float(np.std(avg_recall_values))
    mean_std = float(np.mean(std_recall_values))
    avg_precision_values = [data['avg_precision'] for data in all_trials_data]
    avg_fpr_values = [data['avg_fpr'] for data in all_trials_data]
    avg_jaccard_values = [data['avg_jaccard'] for data in all_trials_data]
    avg_accuracy_values = [data['avg_accuracy'] for data in all_trials_data]
    avg_matter_weighted_f1_values = [data['avg_matter_weighted_f1'] for data in all_trials_data]
    avg_matter_weighted_f1_fixed_values = [data['avg_matter_weighted_f1_fixed'] for data in all_trials_data]
    avg_matter_weighted_jaccard_values = [data['avg_matter_weighted_jaccard'] for data in all_trials_data]
    avg_matter_weighted_jaccard_fixed_values = [data['avg_matter_weighted_jaccard_fixed'] for data in all_trials_data]
    avg_matter_weighted_accuracy_values = [data['avg_matter_weighted_accuracy'] for data in all_trials_data]
    avg_matter_weighted_accuracy_fixed_values = [data['avg_matter_weighted_accuracy_fixed'] for data in all_trials_data]
    auc_roc_values = [data['auc_roc'] for data in all_trials_data]

    recall_mean = float(np.mean(avg_recall_values))
    precision_mean = float(np.mean(avg_precision_values))
    fpr_mean = float(np.mean(avg_fpr_values))
    jaccard_mean = float(np.mean(avg_jaccard_values))
    accuracy_mean = float(np.mean(avg_accuracy_values))
    matter_weighted_f1_mean = float(np.mean(avg_matter_weighted_f1_values))
    matter_weighted_f1_fixed_mean = float(np.mean(avg_matter_weighted_f1_fixed_values))
    matter_weighted_jaccard_mean = float(np.mean(avg_matter_weighted_jaccard_values))
    matter_weighted_jaccard_fixed_mean = float(np.mean(avg_matter_weighted_jaccard_fixed_values))
    matter_weighted_accuracy_mean = float(np.mean(avg_matter_weighted_accuracy_values))
    matter_weighted_accuracy_fixed_mean = float(np.mean(avg_matter_weighted_accuracy_fixed_values))
    auc_roc_mean = float(np.mean(auc_roc_values))
    recall_pct = recall_mean * 100
    print(f"{'Mean':<8} {recall_pct:<12.2f} "
          f"{precision_mean:<10.3f} {fpr_mean:<8.3f} "
          f"{jaccard_mean:<8.3f} {accuracy_mean:<10.3f} "
          f"{matter_weighted_f1_mean:<10.3f} "
          f"{matter_weighted_f1_fixed_mean:<10.3f} {matter_weighted_jaccard_mean:<8.3f} "
          f"{matter_weighted_jaccard_fixed_mean:<8.3f} {matter_weighted_accuracy_mean:<10.3f} "
          f"{matter_weighted_accuracy_fixed_mean:<10.3f} {auc_roc_mean:<8.3f}")

    fps_mean = None
    fps_std = None
    if fps_list is not None and len(fps_list) > 0:
        fps_mean = float(np.mean(fps_list))
        fps_std = float(np.std(fps_list))
        print(f"{'FPS Mean':<8} {fps_mean:<18.2f} {fps_std:<12.2f}")

    # Compute per-frame mean metrics across all runs for visualizing as framewise average lines (not flat lines)
    per_frame_recall_all_trials = np.stack(per_frame_recall_all_trials, axis=0) if len(per_frame_recall_all_trials) > 0 else None
    per_frame_precision_all_trials = np.stack(per_frame_precision_all_trials, axis=0) if len(per_frame_precision_all_trials) > 0 else None
    per_frame_fpr_all_trials = np.stack(per_frame_fpr_all_trials, axis=0) if len(per_frame_fpr_all_trials) > 0 else None
    per_frame_jaccard_all_trials = np.stack(per_frame_jaccard_all_trials, axis=0) if len(per_frame_jaccard_all_trials) > 0 else None
    per_frame_accuracy_all_trials = np.stack(per_frame_accuracy_all_trials, axis=0) if len(per_frame_accuracy_all_trials) > 0 else None
    per_frame_matter_weighted_f1_all_trials = np.stack(per_frame_matter_weighted_f1_all_trials, axis=0) if len(per_frame_matter_weighted_f1_all_trials) > 0 else None
    per_frame_matter_weighted_f1_fixed_all_trials = np.stack(per_frame_matter_weighted_f1_fixed_all_trials, axis=0) if len(per_frame_matter_weighted_f1_fixed_all_trials) > 0 else None
    per_frame_matter_weighted_accuracy_all_trials = np.stack(per_frame_matter_weighted_accuracy_all_trials, axis=0) if len(per_frame_matter_weighted_accuracy_all_trials) > 0 else None
    per_frame_matter_weighted_accuracy_fixed_all_trials = np.stack(per_frame_matter_weighted_accuracy_fixed_all_trials, axis=0) if len(per_frame_matter_weighted_accuracy_fixed_all_trials) > 0 else None

    # Enhanced visualization: Use the trial with best average recall (not index 0)
    best_visualization_data = None
    if render_results_video:
        print(f"\n🎬 Creating results video (with particle overlay)...")
        try:
            # Determine the best trial by max avg_recall
            best_trial_idx = int(np.argmax([v['avg_recall'] for v in all_visualization_trial_data]))
            best_visualization_data = all_visualization_trial_data[best_trial_idx].copy()
            # Add means and trial info for display in plots, including framewise average metrics
            best_visualization_data['per_frame_recall_mean']    = np.mean(per_frame_recall_all_trials, axis=0) if per_frame_recall_all_trials is not None else None
            best_visualization_data['per_frame_precision_mean'] = np.mean(per_frame_precision_all_trials, axis=0) if per_frame_precision_all_trials is not None else None
            best_visualization_data['per_frame_fpr_mean']       = np.mean(per_frame_fpr_all_trials, axis=0) if per_frame_fpr_all_trials is not None else None
            best_visualization_data['per_frame_jaccard_mean']   = np.mean(per_frame_jaccard_all_trials, axis=0) if per_frame_jaccard_all_trials is not None else None
            best_visualization_data['per_frame_accuracy_mean']  = np.mean(per_frame_accuracy_all_trials, axis=0) if per_frame_accuracy_all_trials is not None else None
            best_visualization_data['per_frame_matter_weighted_f1_mean'] = np.mean(per_frame_matter_weighted_f1_all_trials, axis=0) if per_frame_matter_weighted_f1_all_trials is not None else None
            best_visualization_data['per_frame_matter_weighted_f1_fixed_mean'] = np.mean(per_frame_matter_weighted_f1_fixed_all_trials, axis=0) if per_frame_matter_weighted_f1_fixed_all_trials is not None else None
            best_visualization_data['per_frame_matter_weighted_accuracy_mean'] = np.mean(per_frame_matter_weighted_accuracy_all_trials, axis=0) if per_frame_matter_weighted_accuracy_all_trials is not None else None
            best_visualization_data['per_frame_matter_weighted_accuracy_fixed_mean'] = np.mean(per_frame_matter_weighted_accuracy_fixed_all_trials, axis=0) if per_frame_matter_weighted_accuracy_fixed_all_trials is not None else None
            best_visualization_data['runs_n_trials']            = len(all_trials_data)
            best_visualization_data['selected_trial_idx']       = best_trial_idx
            best_visualization_data['davis_name']               = davis_name
            # create_genmatter_results_video(
            #     best_visualization_data,
            #     annotations_path,
            #     img_dims,
            #     experiment_save_dir
            # )
        except Exception as e:
            print(f"⚠️  Could not plot results video due to error: {str(e)}")

    # Compute means for matter-weighted recall and precision separately
    avg_matter_weighted_recall_values = [data['avg_matter_weighted_recall'] for data in all_trials_data]
    avg_matter_weighted_precision_values = [data['avg_matter_weighted_precision'] for data in all_trials_data]
    avg_matter_weighted_recall_fixed_values = [data['avg_matter_weighted_recall_fixed'] for data in all_trials_data]
    avg_matter_weighted_precision_fixed_values = [data['avg_matter_weighted_precision_fixed'] for data in all_trials_data]

    matter_weighted_recall_mean = float(np.mean(avg_matter_weighted_recall_values))
    matter_weighted_precision_mean = float(np.mean(avg_matter_weighted_precision_values))
    matter_weighted_recall_fixed_mean = float(np.mean(avg_matter_weighted_recall_fixed_values))
    matter_weighted_precision_fixed_mean = float(np.mean(avg_matter_weighted_precision_fixed_values))

    results = {
        'davis_name': davis_name,
        'all_trials_data': all_trials_data,
        'avg_recall': overall_mean,
        'std_recall': overall_std,
        'median_recall': overall_median,
        'avg_precision': precision_mean,
        'avg_fpr': fpr_mean,
        'avg_jaccard': jaccard_mean,
        'avg_accuracy': accuracy_mean,
        'avg_matter_weighted_recall': matter_weighted_recall_mean,
        'avg_matter_weighted_precision': matter_weighted_precision_mean,
        'avg_matter_weighted_f1': matter_weighted_f1_mean,
        'avg_matter_weighted_jaccard': matter_weighted_jaccard_mean,
        'avg_matter_weighted_accuracy': matter_weighted_accuracy_mean,
        'avg_matter_weighted_recall_fixed': matter_weighted_recall_fixed_mean,
        'avg_matter_weighted_precision_fixed': matter_weighted_precision_fixed_mean,
        'avg_matter_weighted_f1_fixed': matter_weighted_f1_fixed_mean,
        'avg_matter_weighted_jaccard_fixed': matter_weighted_jaccard_fixed_mean,
        'avg_matter_weighted_accuracy_fixed': matter_weighted_accuracy_fixed_mean,
        'avg_auc_roc': auc_roc_mean,
        'fps_mean': fps_mean,
        'fps_std': fps_std
    }
    return results, best_visualization_data


def create_genmatter_results_video(viz_data, annotations_path, img_dims, experiment_save_dir = None):
    """
    Create an animated visualization where all mask/assignment/particle/GT plots
    show the best single run (highest mean recall), but the line plot subplot
    overlays the framewise AVERAGE over all runs for each metric as dynamic 
    colored lines (not static horizontal lines). Particles replace "blobs"
    nomenclature throughout. Legends and supertitle are explicit.
    Improved legend placement, layout, and clearer text info for particle
    counts before/after Gibbs, etc. Main title is "GenMatter Tracking Results -- {trial name}".
    """
    import os
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.animation import FuncAnimation
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap
    import matplotlib as mpl
    import numpy as np

    per_trial_list = viz_data['per_trial_list']
    reference_particles = viz_data['reference_particles']
    num_frames = viz_data['num_frames']
    n_total_particles_in_scene = viz_data.get('n_total_particles_in_scene', None)
    n_all_reference_particles_before_threshold = viz_data.get('n_all_reference_particles_before_threshold', None)
    counting_threshold = viz_data.get('counting_threshold', None)
    total_reference_particles = viz_data.get('total_reference_particles', None)
    n_outlier_particles = viz_data.get('n_outlier_particles', 0)
    actual_outlier_pixels = viz_data.get('actual_outlier_pixels', 0)
    first_frame_n_blobs = viz_data.get('first_frame_n_blobs', None)
    num_true_unique_particles_first_frame = viz_data.get('num_true_unique_particles_first_frame', None)
    davis_name = viz_data.get('davis_name', 'Unknown-Stimulus')

    # Per-trial best run (for lines)
    recall_scores      = viz_data['recall_scores']
    precision_scores   = viz_data['precision_scores']
    fpr_scores         = viz_data['fpr_scores']
    iou_scores         = viz_data['iou_scores']
    # Framewise mean across all runs
    per_frame_recall_mean    = viz_data.get('per_frame_recall_mean', None)
    per_frame_precision_mean = viz_data.get('per_frame_precision_mean', None)
    per_frame_fpr_mean       = viz_data.get('per_frame_fpr_mean', None)
    per_frame_iou_mean       = viz_data.get('per_frame_iou_mean', None)
    runs_n_trials            = viz_data.get('runs_n_trials', 1)
    selected_trial_idx       = viz_data.get('selected_trial_idx', None)  # zero-based index

    # --- Color schemes ---
    gt_cmap = LinearSegmentedColormap.from_list('gt', ['#f0f0f0', '#3498db'])
    pred_cmap = LinearSegmentedColormap.from_list('pred', ['#f0f0f0', '#e74c3c'])
    tp_color = [46 / 255, 204 / 255, 113 / 255]  # Green
    fp_color = [231 / 255, 76 / 255, 60 / 255]   # Red
    fn_color = [52 / 255, 152 / 255, 219 / 255]  # Blue
    tn_color = [149 / 255, 165 / 255, 166 / 255] # Gray

    def make_particle_cmap(n):
        hsvs = np.linspace(0, 1, n, endpoint=False)
        cm = mpl.colormaps.get_cmap("hsv")
        colors = [cm(h) for h in hsvs]
        np.random.shuffle(colors)
        return ListedColormap(colors)
    particle_cmap = make_particle_cmap(max(len(reference_particles), 2))

    fig = plt.figure(figsize=(22, 12))
    trial_str = f"(best run, trial {1 + selected_trial_idx if selected_trial_idx is not None else 1})"

    fig.suptitle(f"GenMatter Tracking Results -- {davis_name}", fontsize=22, fontweight='bold')

    plt.subplots_adjust(top=0.94)  # Give room for suptitle and subtitle

    gs = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.25)
    ax_gt = fig.add_subplot(gs[0, 0])
    ax_pred = fig.add_subplot(gs[0, 1])
    ax_particles = fig.add_subplot(gs[0, 2])
    ax_classification = fig.add_subplot(gs[1, 0])
    ax_metrics = fig.add_subplot(gs[1, 1])
    ax_info = fig.add_subplot(gs[1, 2])

    # --- Set up axes ---
    gt_im = ax_gt.imshow(np.zeros(img_dims), cmap=gt_cmap, vmin=0, vmax=1)
    ax_gt.set_title(f"Ground Truth", fontsize=13)
    ax_gt.set_xticks([]); ax_gt.set_yticks([])

    pred_im = ax_pred.imshow(np.zeros(img_dims), cmap=pred_cmap, vmin=0, vmax=1)
    ax_pred.set_title(f"Model Prediction (best run)", fontsize=13)
    ax_pred.set_xticks([]); ax_pred.set_yticks([])

    particle_im = ax_particles.imshow(np.zeros(img_dims), cmap=particle_cmap, interpolation='nearest', vmin=0, vmax=max(len(reference_particles)-1, 1))
    ax_particles.set_title(f"Reference Particle Assignment\n(best run)", fontsize=13)
    ax_particles.set_xticks([]); ax_particles.set_yticks([])

    classification_im = ax_classification.imshow(np.zeros((*img_dims, 3)))
    ax_classification.set_title('Classification Analysis (best run)', fontsize=13)
    ax_classification.set_xticks([]); ax_classification.set_yticks([])

    legend_elements = [
        patches.Patch(color=tp_color, label='True Positive'),
        patches.Patch(color=fp_color, label='False Positive'),
        patches.Patch(color=fn_color, label='False Negative'),
        patches.Patch(color=tn_color, label='True Negative')
    ]
    # Move classification legend to the very top of the plot, above the title
    legend = ax_classification.legend(
        handles=legend_elements,
        loc='upper center',
        bbox_to_anchor=(0.35, 1.35),
        ncol=4,
        fontsize=12,
        frameon=True,
        title="Classification"
    )
    fig.add_artist(legend)  # Ensure it stays above the suptitle

    # METRICS plot: overlay per-frame mean lines (dashed), and best run (solid)
    metrics_title = f"Tracking Metrics per Frame"
    ax_metrics.set_title(metrics_title, fontsize=15)
    ax_metrics.set_xlabel('Frame', fontsize=12)
    ax_metrics.set_ylabel('Score', fontsize=12)
    ax_metrics.set_xlim(0, num_frames-1)
    ax_metrics.set_ylim(0, 1)
    ax_metrics.grid(True, alpha=0.3)
    x_frames = np.arange(num_frames)

    # Per-frame mean as dashed lines (across runs, if multiple)
    lines_mean = {}
    if per_frame_recall_mean is not None:
        lines_mean['recall'], = ax_metrics.plot(x_frames, per_frame_recall_mean, '--', color='#2ecc71', alpha=0.38, lw=2, label='Recall (mean)')
    if per_frame_precision_mean is not None:
        lines_mean['precision'], = ax_metrics.plot(x_frames, per_frame_precision_mean, '--', color='#3498db', alpha=0.38, lw=2, label='Precision (mean)')
    if per_frame_fpr_mean is not None:
        lines_mean['fpr'], = ax_metrics.plot(x_frames, per_frame_fpr_mean, '--', color='#e74c3c', alpha=0.38, lw=2, label='FPR (mean)')
    if per_frame_iou_mean is not None:
        lines_mean['iou'], = ax_metrics.plot(x_frames, per_frame_iou_mean, '--', color='#9b59b6', alpha=0.38, lw=2, label='IoU (mean)')

    # Best run (this trial) as colored solid lines
    recall_line, = ax_metrics.plot([], [], '-', color='#2ecc71', linewidth=2, label='Recall (best run)')
    precision_line, = ax_metrics.plot([], [], '-', color='#3498db', linewidth=2, label='Precision (best run)')
    fpr_line, = ax_metrics.plot([], [], '-', color='#e74c3c', linewidth=2, label='FPR (best run)')
    iou_line, = ax_metrics.plot([], [], '-', color='#9b59b6', linewidth=2, label='IoU (best run)')

    # Move the legend to the right side of the plot
    ax_metrics.legend(
        loc='center left',
        bbox_to_anchor=(1.02, 0.35),
        ncol=1,
        fontsize=11,
        frameon=True,
        title="Metric"
    )

    # ax_info (textbox panel) put text heavily right
    ax_info.axis("off")
    info_text_obj = ax_info.text(
        0.11, 0.99, "", va="top", ha="left", fontsize=14, fontweight="medium", family='monospace', color="#34495e",
        transform=ax_info.transAxes, linespacing=1.32
    )

    def info_lines_fmt(frame_idx):
        lines = []
        total_particles = None
        if n_total_particles_in_scene is not None and first_frame_n_blobs is not None and num_true_unique_particles_first_frame is not None:
            total_particles = int(n_total_particles_in_scene + n_outlier_particles)
            lines.append(
                f"Frame: {frame_idx+1} / {num_frames}"
            )
            lines.append(
                f"First frame: n_particles (initialization): {first_frame_n_blobs}"
            )
            lines.append(
                f"Particles in scene (post- first frame Gibbs): {n_total_particles_in_scene}"
            )
        if n_all_reference_particles_before_threshold is not None:
            lines.append(
                f"Reference particles pre-threshold: {n_all_reference_particles_before_threshold}"
            )
        if total_reference_particles is not None:
            lines.append(
                f"Reference particles post-threshold: {total_reference_particles}"
            )
        if counting_threshold is not None:
            lines.append(
                f"Counting threshold: {counting_threshold}"
            )
        return "\n".join(lines)

    def update_frame(frame_idx):
        gt_mask = get_segmentation_mask(davis_name, frame_idx, annotations_path, img_dims, flatten=False)
        frame_assignments = np.array(per_trial_list[frame_idx]['blob_assignments']).reshape(img_dims)
        frame_n_blobs = per_trial_list[frame_idx]['n_blobs']

        pred_mask = np.isin(frame_assignments, reference_particles) & (frame_assignments != frame_n_blobs)
        gt_im.set_data(gt_mask.astype(float))
        pred_im.set_data(pred_mask.astype(float))

        particle_display = np.full(img_dims, fill_value=np.nan)
        if len(reference_particles) > 0:
            for idx, pid in enumerate(reference_particles):
                particle_display[frame_assignments == pid] = idx
        particle_im.set_data(particle_display)

        # 4-way classification
        classification_viz = np.zeros((*img_dims, 3))
        is_inside_gt = gt_mask.astype(bool)
        is_reference_particle = np.isin(frame_assignments, reference_particles) & (frame_assignments != frame_n_blobs)
        tp_mask = is_inside_gt & is_reference_particle
        fp_mask = is_inside_gt & ~is_reference_particle
        fn_mask = ~is_inside_gt & is_reference_particle
        tn_mask = ~is_inside_gt & ~is_reference_particle
        classification_viz[tp_mask] = tp_color
        classification_viz[fp_mask] = fp_color
        classification_viz[fn_mask] = fn_color
        classification_viz[tn_mask] = tn_color
        classification_im.set_data(classification_viz)

        frames_range = np.arange(frame_idx + 1)
        recall_line.set_data(frames_range, recall_scores[:frame_idx+1])
        precision_line.set_data(frames_range, precision_scores[:frame_idx+1])
        fpr_line.set_data(frames_range, fpr_scores[:frame_idx+1])
        iou_line.set_data(frames_range, iou_scores[:frame_idx+1])

        # Subplot titles
        ax_gt.set_title(f'Ground Truth (best run, Frame {frame_idx+1})', fontsize=13)
        ax_pred.set_title(f'Model Prediction (best run, Frame {frame_idx+1})', fontsize=13)
        ax_classification.set_title(f'Classification Analysis (best run, Frame {frame_idx+1})', fontsize=13)
        ax_particles.set_title(f"Reference Particle Assignment\n(best run, Frame {frame_idx+1})", fontsize=13)

        info_text_obj.set_text(info_lines_fmt(frame_idx))
        info_text_obj.set_color('#34495e')
        info_text_obj.set_fontsize(14)
        return [
            gt_im, pred_im, particle_im, classification_im, recall_line, precision_line, fpr_line, iou_line, info_text_obj
        ]

    from IPython.display import HTML, display

    anim = FuncAnimation(fig, update_frame, frames=num_frames, interval=200, blit=True, repeat=True)

    if experiment_save_dir is None:
        # In notebook/in-memory mode: display as HTML animation and return it
        plt.close(fig)
        print(f"💡 experiment_save_dir is None, returning inline HTML animation (not saving to disk)")
        return HTML(anim.to_html5_video())
    else:
        output_path = os.path.join(experiment_save_dir, f"{davis_name}_results_video.mp4")
        print(f"💾 Saving visualization to: {output_path}")
        video_saved = False

        try:
            anim.save(output_path, writer='ffmpeg', fps=5)
            video_saved = True
        except Exception as e1:
            print(f"⚠️  ffmpeg writer failed: {e1}")
            try:
                anim.save(output_path, writer='pillow', fps=5)
                video_saved = True
            except Exception as e2:
                print(f"⚠️  pillow (as mp4) writer failed: {e2}")
                try:
                    print(f"⚠️  Could not save video. Saving as GIF instead...")
                    gif_path = os.path.join(experiment_save_dir, f"{davis_name}_results_video.gif")
                    anim.save(gif_path, writer='pillow', fps=2)
                    print(f"💾 GIF saved to: {gif_path}")
                except Exception as e3:
                    print(f"❌ Failed to save GIF as well: {e3}")

        plt.close(fig)
        print(f"✅ Visualization complete!")
    
