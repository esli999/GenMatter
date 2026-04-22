#!/usr/bin/env python3
"""SAM2 frame-0 pseudo-color PNGs for TAP-Vid DAVIS → ``tapvid_davis_SAM_frame0/{video}_SAM_frame0.png``."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import config  # noqa: E402


def _resolve_model_weights(model: str, weights_dir: Path) -> str:
    """Bare filenames use ``assets/deeplearning_weights/<name>``; download if missing."""
    from ultralytics.utils.downloads import attempt_download_asset

    weights_dir = Path(weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    p = Path(model).expanduser()
    # Path with a directory component: require that file to exist (no auto-download here).
    if p.parent != Path("."):
        if not p.is_file():
            raise FileNotFoundError(f"Model weights not found: {p}")
        return str(p.resolve())
    dest = weights_dir / p.name
    if not dest.is_file():
        print(f"Downloading SAM weights → {dest}", flush=True)
        attempt_download_asset(dest, progress=True)
    return str(dest.resolve())


def _combined_mask(masks_tensor, min_threshold: float, seed: int) -> np.ndarray:
    masks = masks_tensor
    n, h, w = int(masks.shape[0]), int(masks.shape[1]), int(masks.shape[2])
    out = np.full((h, w, 3), 255, dtype=np.uint8)
    rng = np.random.default_rng(seed)
    colors = []
    for _ in range(n):
        c = rng.integers(0, 255, size=3, dtype=np.int64)
        while np.all(c > 200):
            c = rng.integers(0, 255, size=3, dtype=np.int64)
        colors.append(c.astype(np.uint8))
    prio = np.zeros((h, w), dtype=np.float32)
    for i in range(n):
        m = masks[i].detach().cpu().numpy()
        upd = (m > min_threshold) & (m > prio)
        out[upd] = colors[i]
        prio[upd] = m[upd]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="SAM2 frame-0 export (Ultralytics).")
    p.add_argument("--rgb-dir", type=Path, default=config.DAVIS_RGB_PATH)
    p.add_argument("--output-dir", type=Path, default=config.DAVIS_SAM_FRAME0_PATH)
    p.add_argument(
        "--model",
        type=str,
        default="sam2.1_l.pt",
        help="Checkpoint filename or path (default: sam2.1_l.pt under assets/deeplearning_weights/)",
    )
    p.add_argument(
        "--weights-dir",
        type=Path,
        default=config.DEEPLEARNING_WEIGHTS_DIR,
        help="Where to store/find downloaded .pt files (default: config.DEEPLEARNING_WEIGHTS_DIR)",
    )
    p.add_argument("--min-threshold", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--frame-name", type=str, default="00000.jpg")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--videos", nargs="*", default=None)
    args = p.parse_args()

    try:
        from ultralytics import SAM
    except ImportError as e:
        print("Install deps: `uv sync` (needs ultralytics).", file=sys.stderr)
        raise SystemExit(1) from e

    logging.getLogger("ultralytics").setLevel(logging.WARNING)

    rgb_base = args.rgb_dir.resolve()
    out_base = args.output_dir.resolve()
    if not rgb_base.is_dir():
        sys.exit(f"RGB directory missing: {rgb_base}\n")

    names = list(args.videos) if args.videos else [
        v for v in config.TAPVID_DAVIS_VIDEO_NAMES if (rgb_base / v).is_dir()
    ]
    if not names:
        sys.exit(f"No video folders under {rgb_base}\n")

    weights_path = _resolve_model_weights(args.model, args.weights_dir.resolve())
    model = SAM(weights_path)
    failed: list[str] = []

    for name in tqdm(names, desc="sam-frame0", mininterval=0.5):
        outp = out_base / f"{name}_SAM_frame0.png"
        if args.skip_existing and outp.is_file():
            continue
        frame = rgb_base / name / args.frame_name
        if not frame.is_file():
            failed.append(f"{name} (no {args.frame_name})")
            continue
        r = model(str(frame), verbose=False)[0]
        if r.masks is None or len(r.masks.data) == 0:
            failed.append(f"{name} (no masks)")
            continue
        out_base.mkdir(parents=True, exist_ok=True)
        rgb = _combined_mask(r.masks.data, args.min_threshold, args.seed)
        cv2.imwrite(str(outp), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    if failed:
        tqdm.write("Skipped / failed:\n  " + "\n  ".join(failed), file=sys.stderr)


if __name__ == "__main__":
    main()
