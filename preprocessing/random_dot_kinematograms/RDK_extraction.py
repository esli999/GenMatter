import torch
import torchvision.transforms as T
from torchvision.models.optical_flow import raft_large
import numpy as np
import os
import imageio.v2 as imageio
import random

VIDEO_FOLDER_PATH = "" # SET THIS TO THE PATH OF THE GROUND TRUTH PHYSICS SIMULATION VIDEO FOLDER WITH ALL THE MP4s
OUTPUT_FOLDER_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'assets/RDK')

def preprocess(frames):
    transforms = T.Compose([
        T.ToTensor(),
        T.ConvertImageDtype(torch.float32),
        T.Normalize(mean=0.5, std=0.5),
    ])
    processed_frames = []
    for frame in frames:
        if frame.shape[2] == 4:
            frame = frame[:, :, :3]
        processed_frames.append(transforms(frame))
    return torch.stack(processed_frames)

def compute_optical_flow(frames):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = raft_large(pretrained=True, progress=False).to(device).eval()
    processed_frames = preprocess(frames)
    flow_predictions = []
    for i in range(len(processed_frames) - 1):
        img1 = processed_frames[i:i+1].to(device)
        img2 = processed_frames[i+1:i+2].to(device)
        with torch.no_grad():
            predictions = model(img1, img2)
        predicted_flow = predictions[-1]
        flow_predictions.append(predicted_flow.cpu())
    return torch.cat(flow_predictions, dim=0)

def create_2d_kinematogram(flow_array, frames, n_points=300, point_size=5, resample_ratio=0.1, h=480, w=640):
    flow_array = flow_array.permute(0, 2, 3, 1).numpy()
    n_frames = flow_array.shape[0] + 1
    points = np.random.randint(0, [w, h], size=(n_points, 2)).astype(np.float32)
    points_data = np.zeros((n_frames, n_points, 2), dtype=np.float32)
    points_data[0] = points
    for i in range(n_frames):
        current_points = points_data[i]
        current_points_int = np.round(current_points).astype(int)
        valid_mask = (current_points_int[:, 0] >= 0) & (current_points_int[:, 0] < w) & \
                     (current_points_int[:, 1] >= 0) & (current_points_int[:, 1] < h)
        if i < n_frames - 1:
            points_data[i+1] = points_data[i].copy()
            resample_indices = np.random.choice(n_points, int(resample_ratio * n_points), replace=False)
            non_resample_mask = np.ones(n_points, dtype=bool)
            non_resample_mask[resample_indices] = False
            current_points = points_data[i][non_resample_mask]
            current_points_int = np.round(current_points).astype(int)
            valid_mask = (current_points_int[:, 0] >= 0) & (current_points_int[:, 0] < w) & \
                         (current_points_int[:, 1] >= 0) & (current_points_int[:, 1] < h)
            valid_indices = np.where(valid_mask)[0]
            for idx in valid_indices:
                j = np.where(non_resample_mask)[0][idx]
                x_int, y_int = current_points_int[idx]
                if y_int < frames[i].shape[0] and x_int < frames[i].shape[1]:
                    pixel = frames[i][y_int, x_int]
                    if np.all(pixel > 240):
                        continue
                if y_int < flow_array.shape[1] and x_int < flow_array.shape[2]:
                    flow_vector = flow_array[i, y_int, x_int]
                    new_x = current_points[idx, 0] + flow_vector[0]
                    new_y = current_points[idx, 1] + flow_vector[1]
                    new_x = new_x % w
                    new_y = new_y % h
                    points_data[i+1, j] = [new_x, new_y]
            for j in resample_indices:
                new_x = random.randint(0, w-1)
                new_y = random.randint(0, h-1)
                points_data[i+1, j] = [new_x, new_y]
    return points_data

for config_n in range(10):
    if config_n == 2:
        continue # skip this config as it is not in this dataset
    CONFIG_NAME = f'config_{config_n+1}'
    video_path = os.path.join(VIDEO_FOLDER_PATH, f'{CONFIG_NAME}.mp4')
    frames = [frame for frame in imageio.get_reader(video_path)]
    flow_data = compute_optical_flow(frames)
    points_data = create_2d_kinematogram(
        flow_data, frames, n_points=5000, point_size=3, resample_ratio=0.3,
        h=frames[0].shape[0], w=frames[0].shape[1]
    )
    config_output_path = os.path.join(OUTPUT_FOLDER_PATH, CONFIG_NAME)
    os.makedirs(config_output_path, exist_ok=True)
    np.savez_compressed(os.path.join(config_output_path, 'data.npz'), 
                        points_data=points_data)