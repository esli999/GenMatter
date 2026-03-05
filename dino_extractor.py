# %%
import torch
import cv2
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
import os
from PIL import Image
import traceback
import gc

# %%
# Configuration
DAVIS_STIMULI = [
    "judo", "bike-packing", "blackswan", "bmx-trees", "breakdance", "camel", "car-roundabout",
    "car-shadow", "cows", "dance-twirl", "dog", "dogs-jump", "drift-chicane", "drift-straight",
    "goat", "gold-fish", "horsejump-high", "india", "kite-surf", "lab-coat",
    "libby", "loading", "mbike-trick", "motocross-jump", "paragliding-launch", "parkour",
    "pigs", "scooter-black", "shooting", "soapbox"
]

BASE_VIDEO_PATH = "/home/esli/GenMatter/assets/tapvid_davis_30_videos_processed/tapvid_davis_rgb_frames"
OUTPUT_DIR = "/home/esli/GenMatter/assets/tapvid_davis_30_videos_processed/tapvid_davis_dino"
TARGET_H, TARGET_W = 520, 960
PATCH_SIZE = 14
N_COMPONENTS = 10

# %%
# Load DINO model once
print("Loading DINO model...")
dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
dinov2_vits14.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dinov2_vits14 = dinov2_vits14.to(device)
print(f"Model loaded on {device}")

# %%
def load_and_preprocess_frames(video_path):
    """Load frames from directory and resize to be divisible by patch size."""
    frame_files = sorted([f for f in os.listdir(video_path) if f.endswith(('.jpg', '.png'))])
    
    if len(frame_files) == 0:
        return None, None, None
    
    frames = []
    for frame_file in tqdm(frame_files, desc="Loading frames"):
        frame_path = os.path.join(video_path, frame_file)
        frame = cv2.imread(frame_path)
        if frame is None:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    
    if len(frames) == 0:
        return None, None, None
    
    # Resize frames to be divisible by patch size
    h_orig, w_orig = frames[0].shape[:2]
    h = (h_orig // PATCH_SIZE) * PATCH_SIZE
    w = (w_orig // PATCH_SIZE) * PATCH_SIZE
    
    frames_resized = []
    for frame in tqdm(frames, desc="Resizing frames"):
        frame_resized = cv2.resize(frame, (w, h))
        frames_resized.append(frame_resized)
    
    return frames_resized, h, w

# %%
def extract_dino_features(frames, model, device):
    """Extract DINO features for all frames."""
    all_patch_features = []
    
    # ImageNet normalization stats
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    
    for frame in tqdm(frames, desc="Extracting DINO features"):
        # Preprocess frame
        img_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)
        img_tensor = (img_tensor - mean) / std
        
        # Extract patch features
        with torch.no_grad():
            features = model.forward_features(img_tensor)
            patch_features = features['x_norm_patchtokens']
        
        all_patch_features.append(patch_features.cpu())
    
    # Concatenate all patch features
    all_patch_features = torch.cat(all_patch_features, dim=1)
    return all_patch_features

# %%
def apply_pca(features_np, n_components=10):
    """Apply PCA to features."""
    pca = PCA(n_components=n_components)
    pca_features = pca.fit_transform(features_np)
    
    # Work with views/slices to avoid unnecessary copies
    pca_unnormalized = pca_features  # Use original, no copy
    
    # Create normalized version for RGB mapping (first 3 components only)
    pca_rgb = pca_features[:, :3]  # Use slice, no copy
    # Normalize in-place to save memory
    for i in range(3):
        col_min = pca_rgb[:, i].min()
        col_max = pca_rgb[:, i].max()
        pca_rgb[:, i] = (pca_rgb[:, i] - col_min) / (col_max - col_min)
    
    return pca, pca_unnormalized, pca_rgb

# %%
def save_npz_memory_efficient(video_name, pca_unnormalized, pca_rgb, num_frames, h, w, pca, output_dir):
    """Save PCA features to npz file with memory-efficient processing."""
    patches_h = h // PATCH_SIZE
    patches_w = w // PATCH_SIZE
    num_patches_per_frame = patches_h * patches_w
    
    print(f"Processing {num_frames} frames with {num_patches_per_frame} patches each")
    
    # Pre-allocate the final array
    final_shape = (num_frames, TARGET_H, TARGET_W, pca_unnormalized.shape[-1])
    print(f"Pre-allocating array with shape: {final_shape}")
    print(f"Estimated memory usage: {np.prod(final_shape) * 4 / (1024**3):.2f} GB")
    
    try:
        pca_frames_resized = np.empty(final_shape, dtype=np.float32)
        
        # Process frames one by one to avoid storing all in memory
        for i in tqdm(range(num_frames), desc="Processing frames"):
            start_idx = i * num_patches_per_frame
            end_idx = start_idx + num_patches_per_frame
            
            # Process unnormalized version (all components for Gaussian)
            frame_pca_unnorm = pca_unnormalized[start_idx:end_idx].reshape(patches_h, patches_w, -1)
            frame_pca_unnorm_upsampled = cv2.resize(frame_pca_unnorm, (w, h), interpolation=cv2.INTER_LINEAR)
            
            # Resize directly to target size and store in pre-allocated array
            frame_resized = cv2.resize(frame_pca_unnorm_upsampled, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)
            pca_frames_resized[i] = frame_resized
            
            # Clean up intermediate variables
            del frame_pca_unnorm, frame_pca_unnorm_upsampled, frame_resized
            
        print(f"Successfully processed all frames! Final shape: {pca_frames_resized.shape}")
        
    except Exception as e:
        print(f"Error during frame processing: {type(e).__name__}: {str(e)}")
        print("Full traceback:")
        traceback.print_exc()
        return None

    print(f"shape of pca_frames_resized: {pca_frames_resized.shape}")
    
    # Compute Gaussian parameters
    pca_flat = pca_frames_resized.reshape(-1, pca_frames_resized.shape[-1])
    print(f"shape of pca_flat: {pca_flat.shape}")
    gaussian_means = np.mean(pca_flat, axis=0)
    gaussian_stds = np.std(pca_flat, axis=0)
    print(f"shape of gaussian_means: {gaussian_means.shape}")
    print(f"shape of gaussian_stds: {gaussian_stds.shape}")

    print(f"  ✓ {video_name}: Computed Gaussian parameters")
    
    # Clean up intermediate variables
    del pca_flat
    
    # Save to npz
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{video_name}_dino_pca_per_pixel.npz')
    
    print(f"  ✓ {video_name}: Saving PCA features to {output_path}")
    np.savez(output_path,
             pca_features_unnormalized=pca_frames_resized,
             gaussian_means=gaussian_means,
             gaussian_stds=gaussian_stds,
             components=pca.components_,
             mean=pca.mean_,
             explained_variance_ratio=pca.explained_variance_ratio_)
    
    return output_path

# %%
def process_video(video_name, base_path, model, device, output_dir):
    """Process a single video: extract features, apply PCA, and save."""
    video_path = os.path.join(base_path, video_name)
    
    # Check if output already exists
    output_path = os.path.join(output_dir, f'{video_name}_dino_pca_per_pixel.npz')
    if os.path.exists(output_path):
        print(f"  ✓ {video_name}: Already processed, skipping")
        return True
    
    # Check if video directory exists
    if not os.path.exists(video_path):
        print(f"  ✗ {video_name}: Directory not found")
        return False
    
    try:
        # Load frames
        frames, h, w = load_and_preprocess_frames(video_path)
        print(f"  ✓ {video_name}: Loaded {len(frames)} frames")
        if frames is None:
            print(f"  ✗ {video_name}: No frames found")
            return False
        
        # Extract DINO features
        all_patch_features = extract_dino_features(frames, model, device)
        print(f"  ✓ {video_name}: Extracted {all_patch_features.shape[1]} patch features")
        
        # # Clean up frames to free memory
        # del frames
        # gc.collect()
        
        # Apply PCA
        features_np = all_patch_features.squeeze(0).numpy()
        pca, pca_unnormalized, pca_rgb = apply_pca(features_np, N_COMPONENTS)
        print(f"  ✓ {video_name}: Applied PCA to {features_np.shape[1]} features")
        
        # Clean up intermediate variables
        del all_patch_features, features_np
        gc.collect()
        
        # Process and save directly without storing all frames in memory
        output_path = save_npz_memory_efficient(video_name, pca_unnormalized, pca_rgb, len(frames), h, w, pca, output_dir)
        print(f"  ✓ {video_name}: Saved PCA features to {output_path}")
        
        # Clean up remaining variables
        del pca_unnormalized, pca_rgb, pca
        gc.collect()
        
        print(f"  ✓ {video_name}: Processing complete → {output_path}")
        return True
        
    except Exception as e:
        print(f"  ✗ {video_name}: Error - {str(e)}")
        traceback.print_exc()
        return False

# %%
# Process all DAVIS stimuli videos
print(f"Processing {len(DAVIS_STIMULI)} videos...")
print(f"Output directory: {OUTPUT_DIR}\n")

successful = 0
failed = 0
skipped = 0

for i, video_name in enumerate(DAVIS_STIMULI, 1):
    print(f"[{i}/{len(DAVIS_STIMULI)}] Processing {video_name}...")
    
    result = process_video(video_name, BASE_VIDEO_PATH, dinov2_vits14, device, OUTPUT_DIR)
    
    if result:
        # Check if it was skipped or newly processed
        output_path = os.path.join(OUTPUT_DIR, f'{video_name}_dino_pca_per_pixel.npz')
        if "skipping" in str(result):
            skipped += 1
        else:
            successful += 1
    else:
        failed += 1
    
    # Force garbage collection between videos
    gc.collect()
    print()

print("=" * 60)
print(f"Summary:")
print(f"  Successfully processed: {successful}")
print(f"  Already existed: {skipped}")
print(f"  Failed: {failed}")
print(f"  Total: {len(DAVIS_STIMULI)}")
print("=" * 60)

# %%
# Optional: View results for a specific video
# Uncomment and run to visualize a specific video's results

# video_to_visualize = "bear"  # Change this
# data = np.load(f'{OUTPUT_DIR}/{video_to_visualize}_dino_pca_per_pixel.npz')
# print(f"Loaded data for {video_to_visualize}:")
# print(f"  Shape: {data['pca_features_unnormalized'].shape}")
# print(f"  Gaussian means: {data['gaussian_means']}")
# print(f"  Gaussian stds: {data['gaussian_stds']}")
# print(f"  Explained variance: {data['explained_variance_ratio']}")

# %%
# To run your script in a specific conda environment (here "rebuttal") from a Jupyter notebook using zsh,
# you should use the following. This ensures conda is loaded and activated correctly for zsh:

# !zsh -i -c "conda activate rebuttal && python /home/esli/GenParticles_NeurIPS/dino_tracking.py"

# If you encounter an error about 'conda: command not found', you may need to run:
# !zsh -i -c 'source ~/.zshrc && conda activate rebuttal && python /home/esli/GenParticles_NeurIPS/dino_tracking.py'
