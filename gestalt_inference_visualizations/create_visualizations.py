"""
Create gestalt inference visualizations for all scenes and textures.

This script processes:
- Scenes: 0-19 (scene_00000 to scene_00019)
- Textures: 00, 07, 13, 16, 21, 22, 25
- Creates visualization videos showing stimulus, object posterior, background posterior, and outlier posterior
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import cv2
import os
from pathlib import Path

# Configuration
BASE_DIR = Path("/home/esli/GenMatter")
VIDEO_GESTALT_MASKS_DIR = BASE_DIR / "video_gestalt_masks"
ASSETS_DIR = BASE_DIR / "assets" / "from_thomas"
OUTPUT_DIR = BASE_DIR / "gestalt_inference_visualizations"
VIDEO_OUTPUT_DIR = OUTPUT_DIR / "videos"  # Subdirectory for all MP4 files

# Scenes: 0-19
SCENES = [f"scene_{i:05d}" for i in range(20)]

# Textures: 00, 07, 13, 16, 21, 22, 25
TEXTURES = ["texture_00", "texture_07", "texture_13", "texture_16", "texture_21", "texture_22", "texture_25"]

# Run ID (using run_0)
RUN_ID = "run_0"

# Create output directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_assignments(scene_id, texture_id, run_id):
    """Load assignments.npz file for a given scene/texture/run."""
    assignments_path = VIDEO_GESTALT_MASKS_DIR / scene_id / texture_id / run_id / "assignments.npz"
    
    if not assignments_path.exists():
        print(f"  Warning: Assignments file not found: {assignments_path}")
        return None
    
    data = np.load(assignments_path)
    assignments_data = {key: data[key] for key in data.files}
    data.close()
    return assignments_data


def load_ground_truth_video(scene_id, texture_id, max_frames=5):
    """Load ground truth video frames."""
    video_path = ASSETS_DIR / scene_id / texture_id / "output_six_frame.mp4"
    
    if not video_path.exists():
        print(f"  Warning: Video file not found: {video_path}")
        return None
    
    cap = cv2.VideoCapture(str(video_path))
    gt_frames = []
    frame_count = 0
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gt_frames.append(frame_rgb)
        frame_count += 1
    
    cap.release()
    
    if len(gt_frames) == 0:
        return None
    
    return np.array(gt_frames)


def compute_posteriors(assignments_data):
    """Compute object, background, and outlier posterior probabilities."""
    pixel_hyperblob_assignments = assignments_data['pixel_hyperblob_assignments']
    hyperblob_sample_counts = assignments_data['hyperblob_sample_counts']
    
    # Convert sample counts to posterior probabilities
    posterior_all = hyperblob_sample_counts.astype(float)
    posterior_sum = posterior_all.sum(axis=-1, keepdims=True)
    posterior_sum[posterior_sum == 0] = 1  # Avoid division by zero
    posterior_all = posterior_all / posterior_sum
    
    # Channel mapping: [0, 1, 2, 3, 4, 5] -> [object, bg1, bg2, bg3, bg4, outliers]
    object_posterior = posterior_all[:, :, :, 0]  # Hyperblob 0 (object)
    background_posterior = posterior_all[:, :, :, 1:5].sum(axis=-1)  # Hyperblobs 1-4 (background)
    outlier_posterior = posterior_all[:, :, :, 5]  # Outliers
    
    return object_posterior, background_posterior, outlier_posterior


def create_visualization_video(scene_id, texture_id, gt_frames, object_posterior, 
                                background_posterior, outlier_posterior, output_path):
    """Create and save visualization video."""
    num_frames = object_posterior.shape[0]
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Initialize the plots
    def init():
        # First: Ground truth video frame
        if len(gt_frames) > 0:
            axes[0].imshow(gt_frames[0], interpolation='nearest')
        axes[0].set_title('Stimulus')
        axes[0].axis('off')
        
        # Second: Object posterior with colorbar (Blue)
        obj_prob = object_posterior[0]
        im1 = axes[1].imshow(obj_prob, cmap='Blues', vmin=0, vmax=1, interpolation='nearest')
        axes[1].set_title('Object Posterior\np(object|stimulus)')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Third: Background posterior with colorbar (Green)
        bg_prob = background_posterior[0]
        im2 = axes[2].imshow(bg_prob, cmap='Greens', vmin=0, vmax=1, interpolation='nearest')
        axes[2].set_title('Background Posterior\np(background|stimulus)')
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        
        # Fourth: Outlier posterior with colorbar (Red)
        out_prob = outlier_posterior[0]
        im3 = axes[3].imshow(out_prob, cmap='Reds', vmin=0, vmax=1, interpolation='nearest')
        axes[3].set_title('Outlier Posterior\np(outlier|stimulus)')
        axes[3].axis('off')
        plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        return axes
    
    def update(frame_idx):
        # Clear only the images, not the colorbars
        for ax in axes:
            for im in ax.images:
                im.remove()
        
        # First: Ground truth video frame
        if frame_idx < len(gt_frames):
            axes[0].imshow(gt_frames[frame_idx], interpolation='nearest')
        
        # Second: Object posterior (Blue)
        obj_prob = object_posterior[frame_idx]
        axes[1].imshow(obj_prob, cmap='Blues', vmin=0, vmax=1, interpolation='nearest')
        
        # Third: Background posterior (Green)
        bg_prob = background_posterior[frame_idx]
        axes[2].imshow(bg_prob, cmap='Greens', vmin=0, vmax=1, interpolation='nearest')
        
        # Fourth: Outlier posterior (Red)
        out_prob = outlier_posterior[frame_idx]
        axes[3].imshow(out_prob, cmap='Reds', vmin=0, vmax=1, interpolation='nearest')
        
        return axes
    
    # Initialize with colorbars
    init()
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=num_frames, interval=500, repeat=True)
    
    # Save as video
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=2, bitrate=1800)
    anim.save(output_path, writer=writer)
    
    plt.close(fig)
    
    return output_path


def process_scene_texture(scene_id, texture_id):
    """Process a single scene/texture combination."""
    print(f"\nProcessing {scene_id} - {texture_id}")
    
    # Load assignments
    assignments_data = load_assignments(scene_id, texture_id, RUN_ID)
    if assignments_data is None:
        print(f"  Skipping {scene_id}/{texture_id}: No assignments file")
        return False
    
    # Load ground truth video
    gt_frames = load_ground_truth_video(scene_id, texture_id)
    if gt_frames is None:
        print(f"  Skipping {scene_id}/{texture_id}: No video file")
        return False
    
    # Compute posteriors
    try:
        object_posterior, background_posterior, outlier_posterior = compute_posteriors(assignments_data)
    except Exception as e:
        print(f"  Error computing posteriors: {e}")
        return False
    
    # Create output filename
    output_filename = f"segmentation_posteriors_{scene_id}_{texture_id}.mp4"
    output_path = VIDEO_OUTPUT_DIR / output_filename
    
    # Create visualization video
    try:
        create_visualization_video(
            scene_id, texture_id, gt_frames, 
            object_posterior, background_posterior, outlier_posterior,
            str(output_path)
        )
        print(f"  ✓ Saved: {output_path}")
        return True
    except Exception as e:
        print(f"  Error creating video: {e}")
        return False


def main():
    """Main function to process all scenes and textures."""
    print("=" * 70)
    print("Gestalt Inference Visualizations")
    print("=" * 70)
    print(f"Output directory: {VIDEO_OUTPUT_DIR}")
    print(f"Scenes: {len(SCENES)} (0-19)")
    print(f"Textures: {len(TEXTURES)} ({', '.join([t.replace('texture_', '') for t in TEXTURES])})")
    print(f"Total combinations: {len(SCENES) * len(TEXTURES)}")
    print("=" * 70)
    
    successful = 0
    failed = 0
    
    for scene_id in SCENES:
        for texture_id in TEXTURES:
            if process_scene_texture(scene_id, texture_id):
                successful += 1
            else:
                failed += 1
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total: {successful + failed}")
    print(f"Output directory: {VIDEO_OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()

