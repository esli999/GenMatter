import os
import numpy as np
import glob
from PIL import Image
import cv2
import pickle
import matplotlib.pyplot as plt


SCALE_FACTOR = 4

# Define the list of stimuli
DAVIS_STIMULI = [
    "breakdance",
    "koala", "mallard-fly", "rallye", "soccerball", "varanus-cage",
    "bear", "dog-agility", "dance-twirl", "drift-straight",
    "parkour", "breakdance-flare", "lucia", "drift-chicane", "car-roundabout",
    "blackswan", "boat", "dog", "elephant", "goat",
    "cows", "libby", "bus", "car-shadow", "flamingo",
    "camel", "hike", "car-turn", "mallard-water", "dance-jump",
    "rhino", "rollerblade", "drift-turn"
]

# Directory containing the results
results_dir = "SPATRACKER_RESULTS_DIRECTORY"

for stimulus in DAVIS_STIMULI:
    print(f"\n\n===== Processing {stimulus} =====")
    
    # Path to the result file
    result_path = os.path.join(results_dir, f"{stimulus}_2d.npz")
    
    # Check if result file exists
    if not os.path.exists(result_path):
        print(f"Result file for {stimulus} not found. Skipping.")
        continue
    
    # Load the tracking results
    try:
        result_data = np.load(result_path)
        pred_tracks = result_data['pred_tracks']  # Should be in shape [T, N, 2]
        # Reshape to match CoTracker format [B, T, N, 2]
        pred_tracks = pred_tracks[None, :, :, :]  # Add batch dimension
    except Exception as e:
        print(f"Error loading result file for {stimulus}: {e}")
        continue
    
    # Path to the segmentation masks
    segmentation_path = f'DAVIS_ANNOTATION_DIRECTORY/{stimulus}'
    
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
    
    if stimulus == 'rs_dog':
        # need to reshape to 720x1280
        for i in range(len(segmentation_masks)):
            # Convert boolean mask to uint8 before resizing
            mask_uint8 = segmentation_masks[i].astype(np.uint8) * 255
            # Use nearest neighbor interpolation
            resized_mask = cv2.resize(mask_uint8, (1280, 720), interpolation=cv2.INTER_NEAREST)
            # Convert back to boolean
            segmentation_masks[i] = resized_mask > 0

    print(f"Loaded {len(segmentation_masks)} segmentation masks from {segmentation_path}")
    if segmentation_masks:
        print(f"Mask shape: {segmentation_masks[0].shape}")
    
    pred_tracks = pred_tracks * SCALE_FACTOR

    # Get the points from the first frame
    points = pred_tracks[0, 0]  # Shape: [N, 2]
    print(points)
    
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
            current_frame_points = pred_tracks[0, frame_idx]  # Shape: [N, 2]
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
        
        # Print only the requested analytics
        print(f"Points on mask in first frame: {total_mask_points}")
        print(f"Points still on mask in last frame: {np.sum(current_mask_values)}")
        print(f"Percentage still on mask in last frame: {percent_on_mask[-1]:.2f}%")
        print(f"Average percentage on mask across all frames: {np.mean(percent_on_mask):.2f}%")
        print(f"Minimum percentage on mask: {np.min(percent_on_mask):.2f}% at frame {np.argmin(percent_on_mask)}")
    else:
        print("Need at least two segmentation masks to track points across frames")

print("\n===== Processing complete for all stimuli =====")




# Directory containing the results
results_dir = "SPATRACKER_RESULTS_DIRECTORY"

performance_dict = {}

for stimulus in DAVIS_STIMULI:
    # print(f"stimulus: {stimulus}")
    
    # Path to the result file
    result_path = os.path.join(results_dir, f"{stimulus}_2d.npz")
    
    # Check if result file exists
    if not os.path.exists(result_path):
        print(f"Result file for {stimulus} not found. Skipping.")
        continue
    
    # Load the tracking results
    try:
        result_data = np.load(result_path)
        pred_tracks = result_data['pred_tracks']  # Should be in shape [T, N, 2]
        # Reshape to match CoTracker format [B, T, N, 2]
        pred_tracks = pred_tracks[None, :, :, :]  # Add batch dimension
    except Exception as e:
        print(f"Error loading result file for {stimulus}: {e}")
        continue
    
    # Path to the segmentation masks
    segmentation_path = f'DAVIS_ANNOTATION_DIRECTORY/{stimulus}'
    
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
    
    pred_tracks = pred_tracks * SCALE_FACTOR

    # Get the points from the first frame
    points = pred_tracks[0, 0]  # Shape: [N, 2]
    
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
            current_frame_points = pred_tracks[0, frame_idx]  # Shape: [N, 2]
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
        
        # Print only the requested analytics
        print(stimulus, np.mean(percent_on_mask))
        performance_dict[stimulus] = np.mean(percent_on_mask)
    else:
        print("Need at least two segmentation masks to track points across frames")

print("\n===== Processing complete for all stimuli =====")


out_file = "performance_results.pkl"
with open(out_file, 'wb') as f:
    pickle.dump(performance_dict, f)

print(f"Performance results saved to {out_file}")

