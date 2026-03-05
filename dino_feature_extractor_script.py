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

# %%
# Configuration
# DAVIS_STIMULI = [
#     "varanus-cage", "dog-agility", "mallard-fly", "rallye", "soccerball",
#     "breakdance", "bear", "koala", "dance-twirl", "drift-straight",
#     "parkour", "breakdance-flare", "lucia", "drift-chicane", "car-roundabout",
#     "blackswan", "boat", "dog", "elephant", "goat",
#     "cows", "libby", "bus", "car-shadow", "flamingo",
#     "camel", "hike", "car-turn", "mallard-water", "dance-jump",
#     "rhino", "rollerblade", "drift-turn"
# ]

# BASE_VIDEO_PATH = "/home/esli/BADJA/DAVIS/JPEGImages/Full-Resolution"

DAVIS_STIMULI = [
    "belt",
    "cloth_bag",
    "eagle_trim",
    "gray_jacket",
    "jello_trim",
    "manta_ray",
    "new_eagle_trim",
    "ostrich_trim",
    "purple_jacket",
    "snake_trim",
    "whiskey_swirl_1",
    "whiskey_swirl_2",
    "wine_swirl"
]

BASE_VIDEO_PATH = "/home/esli/GenMatter/assets/new_demo_vids"

OUTPUT_DIR = "dino_features_pca_l"
TARGET_H, TARGET_W = 520, 960 # These numbers are set by RAFT
PATCH_SIZE = 14
N_COMPONENTS = 10

# %%
# Load DINO model once
print("Loading DINO model...")
dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
dinov2_vits14.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dinov2_vits14 = dinov2_vits14.to(device)
print(f"Model loaded on {device}")

# %%
def load_and_preprocess_frames(video_path, max_dimension=1920):
    """Load frames from directory and resize to be divisible by patch size.

    Args:
        video_path: Path to directory containing video frames
        max_dimension: Maximum width or height (default 1920 for Full HD)
    """
    frame_files = sorted([f for f in os.listdir(video_path) if f.endswith(('.jpg', '.png'))])

    if len(frame_files) == 0:
        return None, None, None

    frames = []
    for frame_file in frame_files:
        frame_path = os.path.join(video_path, frame_file)
        frame = cv2.imread(frame_path)
        if frame is None:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    if len(frames) == 0:
        return None, None, None

    # Get original dimensions
    h_orig, w_orig = frames[0].shape[:2]

    # Downsample if too large (to save GPU memory)
    if max(h_orig, w_orig) > max_dimension:
        scale = max_dimension / max(h_orig, w_orig)
        h_scaled = int(h_orig * scale)
        w_scaled = int(w_orig * scale)
        print(f"    Downsampling from {w_orig}x{h_orig} to {w_scaled}x{h_scaled} to save memory")
    else:
        h_scaled = h_orig
        w_scaled = w_orig

    # Make divisible by patch size
    h = (h_scaled // PATCH_SIZE) * PATCH_SIZE
    w = (w_scaled // PATCH_SIZE) * PATCH_SIZE

    frames_resized = []
    for frame in frames:
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

    for frame in frames:
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

    # Keep unnormalized for Gaussian fitting
    pca_unnormalized = pca_features.copy()

    # Create normalized version for RGB mapping (first 3 components)
    pca_rgb = pca_features[:, :3].copy()
    for i in range(3):
        pca_rgb[:, i] = (pca_rgb[:, i] - pca_rgb[:, i].min()) / (pca_rgb[:, i].max() - pca_rgb[:, i].min())

    return pca, pca_unnormalized, pca_rgb

# %%
def reshape_to_frames(pca_unnormalized, pca_rgb, num_frames, h, w):
    """Reshape PCA features back to frame format."""
    patches_h = h // PATCH_SIZE
    patches_w = w // PATCH_SIZE
    num_patches_per_frame = patches_h * patches_w
    
    pca_frames = []
    pca_frames_unnormalized = []
    
    for i in range(num_frames):
        start_idx = i * num_patches_per_frame
        end_idx = start_idx + num_patches_per_frame
        
        # Normalized version (3 components for RGB viz)
        frame_pca = pca_rgb[start_idx:end_idx].reshape(patches_h, patches_w, 3)
        frame_pca_upsampled = cv2.resize(frame_pca, (w, h), interpolation=cv2.INTER_LINEAR)
        pca_frames.append(frame_pca_upsampled)
        
        # Unnormalized version (all components for Gaussian)
        frame_pca_unnorm = pca_unnormalized[start_idx:end_idx].reshape(patches_h, patches_w, -1)
        frame_pca_unnorm_upsampled = cv2.resize(frame_pca_unnorm, (w, h), interpolation=cv2.INTER_LINEAR)
        pca_frames_unnormalized.append(frame_pca_unnorm_upsampled)
    
    return pca_frames, pca_frames_unnormalized

# %%
def save_npz(video_name, pca_frames_unnormalized, pca, output_dir):
    """Save PCA features to npz file."""
    # Stack and resize
    pca_frames_array = np.stack(pca_frames_unnormalized, axis=0)
    
    pca_frames_resized = []
    for i in range(len(pca_frames_array)):
        frame_resized = cv2.resize(pca_frames_array[i], (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)
        pca_frames_resized.append(frame_resized)
    
    pca_frames_resized = np.stack(pca_frames_resized, axis=0)
    
    # Compute Gaussian parameters
    pca_flat = pca_frames_resized.reshape(-1, pca_frames_resized.shape[-1])
    gaussian_means = np.mean(pca_flat, axis=0)
    gaussian_stds = np.std(pca_flat, axis=0)
    
    # Save to npz
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{video_name}_dino_pca_per_pixel.npz')
    
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
        if frames is None:
            print(f"  ✗ {video_name}: No frames found")
            return False
        
        # Extract DINO features
        all_patch_features = extract_dino_features(frames, model, device)

        # Apply PCA
        features_np = all_patch_features.squeeze(0).numpy()
        pca, pca_unnormalized, pca_rgb = apply_pca(features_np, N_COMPONENTS)
        
        # Reshape to frames
        pca_frames, pca_frames_unnormalized = reshape_to_frames(
            pca_unnormalized, pca_rgb, len(frames), h, w
        )
        
        # Save to npz
        output_path = save_npz(video_name, pca_frames_unnormalized, pca, output_dir)
        
        print(f"  ✓ {video_name}: Processed {len(frames)} frames → {output_path}")
        return True
        
    except Exception as e:
        print(f"  ✗ {video_name}: Error - {str(e)}")
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


