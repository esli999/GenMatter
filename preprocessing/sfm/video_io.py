"""Read-only decoding of the SFM stimulus videos (mpeg4 mp4s, 24 frames, 1024x1024)."""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.sfm import sfm_config as cfg


def decode_video(stim_id: int) -> np.ndarray:
    """All frames as (T, H, W, 3) uint8 BGR. Opens the tpo file strictly read-only."""
    path = cfg.video_path(stim_id)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open {path}")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    arr = np.stack(frames)
    if arr.shape[0] != cfg.N_FRAMES or arr.shape[1] != cfg.VIDEO_RES:
        raise ValueError(f"{path}: unexpected shape {arr.shape}")
    return arr


def decode_moving_gray(stim_id: int) -> np.ndarray:
    """The 12 moving frames as (12, H, W) uint8 grayscale."""
    frames = decode_video(stim_id)[list(cfg.MOVING_FRAMES)]
    return np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames])
