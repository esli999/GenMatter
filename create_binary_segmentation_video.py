"""
Create binary segmentation video from DAVIS annotations.
All background pixels are black (0), all foreground pixels are white (255).
Uses H.264 codec for VSCode compatibility.
"""
import os
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

# Configuration
VIDEO_NAME = "goat"
DAVIS_SEGMASKS_PATH = "/home/esli/BADJA/DAVIS/Annotations/Full-Resolution"
OUTPUT_PATH = f"{VIDEO_NAME}_binary_segmentation.mp4"

# Find all segmentation mask files
segmask_dir = os.path.join(DAVIS_SEGMASKS_PATH, VIDEO_NAME)
mask_files = sorted([f for f in os.listdir(segmask_dir) if f.endswith('.png')])

print(f"Found {len(mask_files)} frames for {VIDEO_NAME}")

# Read first frame to get dimensions
first_mask = Image.open(os.path.join(segmask_dir, mask_files[0]))
first_mask_np = np.array(first_mask)
height, width = first_mask_np.shape[:2]

print(f"Frame dimensions: {height}x{width}")

# Setup video writer with H.264 codec (VSCode compatible)
fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
fps = 30.0
# For grayscale, we need to write as BGR by converting to 3-channel
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height), isColor=True)

# Process each frame
for mask_file in tqdm(mask_files, desc="Creating binary segmentation video"):
    # Load mask
    mask_path = os.path.join(segmask_dir, mask_file)
    mask = Image.open(mask_path)
    mask_np = np.array(mask)

    # Convert to binary: background (0) -> 0, any foreground (>0) -> 255
    binary_mask = np.where(mask_np > 0, 255, 0).astype(np.uint8)

    # Convert grayscale to BGR for video writer
    binary_mask_bgr = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)

    # Write frame
    out.write(binary_mask_bgr)

# Release video writer
out.release()

print(f"\nBinary segmentation video saved to: {OUTPUT_PATH}")
print(f"Total frames: {len(mask_files)}")
print(f"Resolution: {width}x{height}")
print(f"FPS: {fps}")
print(f"Codec: H.264 (avc1)")
