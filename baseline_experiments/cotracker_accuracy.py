import torch
import os
import glob
from PIL import Image
import numpy as np
import imageio.v3 as iio

# List of davis videos that has single deformable object
new_vids = ['blackswan', 'boat', 'breakdance', 'breakdance-flare', 'bus', 'car-roundabout',
             'car-shadow', 'car-turn', 'dance-jump', 'dance-twirl', 'drift-chicane', 'drift-straight', 
             'drift-turn', 'elephant', 'flamingo', 'goat', 'hike', 'libby', 'lucia', 'mallard-water', 'parkour',
             'rhino', 'rollerblade']


for animal in new_vids:
    print(f"\n\n===== Processing {animal} =====")
    
    # Path to the video
    video_path = 'DAVIS_VIDEO_DIRECTORY/' + animal + '.mp4'
    
    # Load video frames
    frames = iio.imread(video_path, plugin="FFMPEG")  # plugin="pyav"
    
    device = 'cuda'
    grid_size = 25
    video = torch.tensor(frames).permute(0, 3, 1, 2)[None].float().to(device)  # B T C H W
    
    # Run Offline CoTracker:
    cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)
    pred_tracks, pred_visibility = cotracker(video, grid_size=grid_size) # B T N 2,  B T N 1
    
    # Path to the segmentation masks
    segmentation_path = 'DAVIS_ANNOTATION_DIRECTORY/'+ animal
    
    # Get all PNG files in the directory
    mask_files = sorted(glob.glob(os.path.join(segmentation_path, '*.png')))
    
    # Read all masks
    segmentation_masks = []
    for mask_file in mask_files:
        mask = np.array(Image.open(mask_file))
        if len(mask.shape) == 3:
            segmentation_masks.append(mask.any(axis=2))
        else:
            segmentation_masks.append(mask)

    print(f"Loaded {len(segmentation_masks)} segmentation masks from {segmentation_path}")
    if segmentation_masks:
        print(f"Mask shape: {segmentation_masks[0].shape}")
    
    # Get the points from the first frame
    points = pred_tracks[0][0].cpu().numpy()  # Shape: [N, 2]
    
    # Check if these points are within the segmentation mask
    if segmentation_masks:
        # Get the first segmentation mask
        mask = segmentation_masks[0]
        
        # Convert points to integer coordinates for indexing
        points_int = points.astype(int)
        
        # Create boolean masks for points within image boundaries
        within_y = (points_int[:, 1] >= 0) & (points_int[:, 1] < mask.shape[0])
        within_x = (points_int[:, 0] >= 0) & (points_int[:, 0] < mask.shape[1])
        valid_points = within_y & within_x
        
        # Get mask values at valid point locations (vectorized)
        mask_values = np.zeros(len(points), dtype=bool)
        if np.any(valid_points):
            y_coords = points_int[valid_points, 1]
            x_coords = points_int[valid_points, 0]
            mask_values[valid_points] = mask[y_coords, x_coords] > 0
        
        # Print summary
        print(f"Total points: {len(points)}")
        print(f"Points in mask: {np.sum(mask_values)}")
        print(f"Percentage in mask: {np.mean(mask_values) * 100:.2f}%")
    else:
        print("No segmentation masks available to check points against")
        continue
    
    # Track points on mask across all frames
    if segmentation_masks and len(segmentation_masks) > 1:
        # Get the first segmentation mask
        first_mask = segmentation_masks[0]
        
        # Get the points from the first frame that were on the mask
        first_frame_points_on_mask = points[mask_values]
        total_mask_points = len(first_frame_points_on_mask)
        
        # Initialize array to store percentage of points on mask for each frame
        percent_on_mask = np.zeros(len(segmentation_masks))
        percent_on_mask[0] = 100.0  # First frame is 100% by definition
        
        # Iterate through all frames after the first
        for frame_idx in range(1, len(segmentation_masks)):
            current_mask = segmentation_masks[frame_idx]
            
            # Get the corresponding points in the current frame
            current_frame_points = pred_tracks[0][frame_idx].cpu().numpy()  # Shape: [N, 2]
            current_frame_mask_points = current_frame_points[mask_values]
            
            # Convert current frame points to integer coordinates for indexing
            current_points_int = current_frame_mask_points.astype(int)
            
            # Create boolean masks for points within image boundaries
            within_y = (current_points_int[:, 1] >= 0) & (current_points_int[:, 1] < current_mask.shape[0])
            within_x = (current_points_int[:, 0] >= 0) & (current_points_int[:, 0] < current_mask.shape[1])
            valid_points = within_y & within_x
            
            # Get mask values at valid point locations in the current frame
            current_mask_values = np.zeros(len(current_points_int), dtype=bool)
            if np.any(valid_points):
                y_coords = current_points_int[valid_points, 1]
                x_coords = current_points_int[valid_points, 0]
                current_mask_values[valid_points] = current_mask[y_coords, x_coords] > 0
            
            # Calculate percentage of points still on mask
            points_on_mask = np.sum(current_mask_values)
            percent_on_mask[frame_idx] = (points_on_mask / total_mask_points) * 100
        
        # Print summary for the last frame
        print(f"Points on mask in first frame: {total_mask_points}")
        print(f"Points still on mask in last frame: {np.sum(current_mask_values)}")
        print(f"Percentage still on mask in last frame: {percent_on_mask[-1]:.2f}%")
        
        # Save the tracked points, predicted visibilities, and percent on mask data to an npz file
        save_dir = f'DAVIS_BENCHMARK_RESULTS/{animal}'
        os.makedirs(save_dir, exist_ok=True)
        np.savez(
            f'{save_dir}/cotracker_results.npz',
            pred_tracks=pred_tracks.cpu().numpy(),
            pred_visibility=pred_visibility.cpu().numpy(),
            percent_on_mask=percent_on_mask
        )
        print(f"Saved tracking results and percent on mask data to {save_dir}/cotracker_results.npz")
        
    else:
        print("Need at least two segmentation masks to track points across frames")

print("\n===== Processing complete for all animals =====")

