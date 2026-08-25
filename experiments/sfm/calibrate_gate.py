"""
Calibrate EVIDENCE_GRAD_THRESH for the v2 evidence gate (luminance OR dilated
Sobel magnitude).

The threshold must sit ABOVE the mp4-compression gradient noise on the shaded
condition's true-black far-background (else the gate stops rescuing shaded) and
BELOW the gradient energy inside dark textures (else texture_21 stays broken).
Prints the distributions on both sides of that margin for exemplar stimuli.

CPU-only; run on mit_quicktest.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from experiments.sfm import sfm_config as cfg                        # noqa: E402
from preprocessing.sfm.video_io import decode_video                  # noqa: E402
from preprocessing.sfm.derive_masks import load_mask_grid            # noqa: E402
from preprocessing.sfm.preprocess_videos import (                    # noqa: E402
    grad_energy, EVIDENCE_LUM_THRESH, EVIDENCE_GRAD_THRESH)


def pct(x, ps=(50, 90, 99, 99.9)):
    return " ".join(f"p{p:g}={np.percentile(x, p):7.2f}" for p in ps)


def region_masks(sid, frame):
    """(object, near-bg ring <=20px, far-bg) boolean masks at 1024^2."""
    m12 = load_mask_grid(sid)
    from experiments.sfm import windows as W
    m = m12[W.frame_to_mask_index(frame)].astype(np.uint8)
    m = cv2.resize(m, (1024, 1024), interpolation=cv2.INTER_NEAREST) > 0
    dil = cv2.dilate(m.astype(np.uint8), np.ones((41, 41), np.uint8)) > 0
    return m, dil & ~m, ~dil


def main():
    # same object/viewpoint across textures: shaded, mid texture, the dark texture
    exemplars = [(1680, "shaded"), (1694, "texture_07"), (1708, "texture_21"),
                 (2100, "shaded"), (2128, "texture_21")]
    for sid, label in exemplars:
        frames = decode_video(sid)
        for frame in (2, 12):
            gray = cv2.cvtColor(frames[frame], cv2.COLOR_BGR2GRAY)
            g = grad_energy(gray)
            obj, near, far = region_masks(sid, frame)
            dark_obj = obj & (gray < EVIDENCE_LUM_THRESH)
            print(f"[{sid} {label} f{frame}] "
                  f"obj grad: {pct(g[obj])} | dark-obj({dark_obj.sum()}px): "
                  f"{pct(g[dark_obj]) if dark_obj.any() else 'none'}")
            print(f"    near-bg: {pct(g[near])} | far-bg: {pct(g[far])} "
                  f"| far-bg lum p99.9={np.percentile(gray[far], 99.9):.1f}")
            for tau in (6, 9, 12, 18, 25):
                lum = gray > EVIDENCE_LUM_THRESH
                ev = lum | (g > tau)
                print(f"    tau={tau:>4}: evidence covers obj "
                      f"{ev[obj].mean():5.1%}  near-bg {ev[near].mean():5.1%}  "
                      f"far-bg {ev[far].mean():6.2%}  (lum alone: obj "
                      f"{lum[obj].mean():5.1%})")
    print(f"current EVIDENCE_GRAD_THRESH={EVIDENCE_GRAD_THRESH}")


if __name__ == "__main__":
    main()
