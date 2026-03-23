"""RAFT optical flow → 2D random-dot kinematogram ``points_data`` (``data.npz``)."""

from __future__ import annotations

import random
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
from tqdm import tqdm


def preprocess(frames: list) -> torch.Tensor:
    transforms = T.Compose(
        [
            T.ToTensor(),
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=0.5, std=0.5),
        ]
    )
    processed_frames = []
    for frame in frames:
        if frame.shape[2] == 4:
            frame = frame[:, :, :3]
        processed_frames.append(transforms(frame))
    return torch.stack(processed_frames)


def compute_optical_flow(frames: list, device: torch.device) -> torch.Tensor:
    # C_T_SKHT_V2 matches deprecated ``pretrained=True`` (torchvision 0.13+ API).
    model = raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2, progress=False).to(device).eval()
    processed_frames = preprocess(frames)
    flow_predictions = []
    n_pairs = len(processed_frames) - 1
    for i in range(n_pairs):
        img1 = processed_frames[i : i + 1].to(device)
        img2 = processed_frames[i + 1 : i + 2].to(device)
        with torch.no_grad():
            predictions = model(img1, img2)
        predicted_flow = predictions[-1]
        flow_predictions.append(predicted_flow.cpu())
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return torch.cat(flow_predictions, dim=0)


def create_2d_kinematogram(
    flow_array: torch.Tensor,
    frames: list,
    n_points: int = 5000,
    point_size: int = 3,
    resample_ratio: float = 0.3,
    h: int = 480,
    w: int = 640,
) -> np.ndarray:
    flow_array = flow_array.permute(0, 2, 3, 1).numpy()
    n_frames = flow_array.shape[0] + 1
    points = np.random.randint(0, [w, h], size=(n_points, 2)).astype(np.float32)
    points_data = np.zeros((n_frames, n_points, 2), dtype=np.float32)
    points_data[0] = points
    for i in range(n_frames):
        current_points = points_data[i]
        current_points_int = np.round(current_points).astype(int)
        valid_mask = (
            (current_points_int[:, 0] >= 0)
            & (current_points_int[:, 0] < w)
            & (current_points_int[:, 1] >= 0)
            & (current_points_int[:, 1] < h)
        )
        if i < n_frames - 1:
            points_data[i + 1] = points_data[i].copy()
            resample_indices = np.random.choice(n_points, int(resample_ratio * n_points), replace=False)
            non_resample_mask = np.ones(n_points, dtype=bool)
            non_resample_mask[resample_indices] = False
            current_points = points_data[i][non_resample_mask]
            current_points_int = np.round(current_points).astype(int)
            valid_mask = (
                (current_points_int[:, 0] >= 0)
                & (current_points_int[:, 0] < w)
                & (current_points_int[:, 1] >= 0)
                & (current_points_int[:, 1] < h)
            )
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
                    points_data[i + 1, j] = [new_x, new_y]
            for j in resample_indices:
                new_x = random.randint(0, w - 1)
                new_y = random.randint(0, h - 1)
                points_data[i + 1, j] = [new_x, new_y]
    return points_data


def _set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_config_npz(
    video_path: Path,
    out_npz: Path,
    *,
    seed: int,
    device: torch.device,
    n_points: int = 5000,
    point_size: int = 3,
    resample_ratio: float = 0.3,
) -> None:
    _set_global_seeds(seed)
    frames = [frame for frame in imageio.get_reader(str(video_path))]
    if not frames:
        raise RuntimeError(f"No frames in {video_path}")
    h, w = frames[0].shape[0], frames[0].shape[1]
    flow_data = compute_optical_flow(frames, device)
    points_data = create_2d_kinematogram(
        flow_data,
        frames,
        n_points=n_points,
        point_size=point_size,
        resample_ratio=resample_ratio,
        h=h,
        w=w,
    )
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out_npz), points_data=points_data)


def run_npz_extraction(
    out_root: Path,
    console,
    master_seed: int = 0,
    config_nums: list[int] | None = None,
) -> None:
    """Read ``out_root/physics_rgb/config_<n>.mp4``, write ``out_root/config_<n>/data.npz``."""
    physics_dir = out_root / "physics_rgb"
    if not physics_dir.is_dir():
        raise FileNotFoundError(f"Missing physics videos: {physics_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print("[bold cyan]Stage 2/4[/bold cyan] RAFT → kinematogram → data.npz")
    console.print(f"Device: [yellow]{device}[/yellow]  master_seed: [yellow]{master_seed}[/yellow]")

    if config_nums is None:
        config_nums = list(range(1, 11))

    for k in tqdm(config_nums, desc="NPZ configs", unit="cfg"):
        vid = physics_dir / f"config_{k}.mp4"
        if not vid.is_file():
            tqdm.write(f"Skip missing video: {vid}")
            continue
        out_npz = out_root / f"config_{k}" / "data.npz"
        seed_k = master_seed + k * 100_003
        extract_config_npz(vid, out_npz, seed=seed_k, device=device)


if __name__ == "__main__":
    raise SystemExit(
        "Use: python preprocessing/random_dot_kinematograms/run_rdk_preprocess.py\n"
        "Or:  uv run python run_experiments.py rdk-preprocess"
    )
