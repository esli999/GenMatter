"""
Whole-video segment plumbing: seg0..seg5 definitions, mask-index clamping, gating
defaults, and generic-T bundle validation. Pure CPU, synthetic arrays — safe anywhere.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.sfm import sfm_config as cfg
from experiments.sfm import windows as W
from experiments.sfm import bundles


def test_segment_layout():
    frames = [f for i in range(6) for f in W.window_frames(f"seg{i}")]
    assert frames == list(range(24)), "segments must tile all 24 frames in order"
    for i in range(6):
        v = f"seg{i}"
        assert W.window_frames(v) == tuple(range(4 * i, 4 * i + 4))
        assert len(W.flow_pairs(v)) == 3
        assert all(b == a + 1 for a, b in W.flow_pairs(v)), "consecutive pairs"
    print("segment layout: PASS")


def test_mask_index_clamp():
    # statics hold render frame init (mask 0) / init+11 (mask 11); moving = f-6
    want = [0] * 6 + list(range(12)) + [11] * 6
    got = [W.frame_to_mask_index(f) for f in range(24)]
    assert got == want, f"mask clamp mismatch: {got}"
    for bad in (-1, 24):
        try:
            W.frame_to_mask_index(bad)
            raise AssertionError(f"frame {bad} accepted")
        except ValueError:
            pass
    # old moving-window variants keep their phase-1 mask indices
    assert W.mask_indices("tiledA") == (0, 1, 2, 3, 4, 5)
    assert W.mask_indices("tiledB_g") == (6, 7, 8, 9, 10, 11)
    assert W.mask_indices("seg0") == (0, 0, 0, 0)
    assert W.mask_indices("seg1") == (0, 0, 0, 1)
    assert W.mask_indices("seg4") == (10, 11, 11, 11)
    assert W.mask_indices("seg5") == (11, 11, 11, 11)
    print("mask-index clamping: PASS")


def test_gating_defaults():
    assert all(W.default_gated(f"seg{i}") for i in range(6))
    assert W.default_gated("tiledA_g") and not W.default_gated("tiledA")
    assert W.base_variant("seg2") == "seg2" and not W.is_gated("seg2")
    print("gating defaults: PASS")


def _synth(T):
    G, N = cfg.GRID_HW, cfg.N_DATAPOINTS
    return dict(points_3d=np.zeros((T, N, 3), np.float32),
                motion_3d=np.zeros((T, N, 3), np.float32),
                motion_valid=np.ones((T, N), bool),
                depth_sq=np.zeros((T + 1, G, G), np.float16),
                flow_sq=np.zeros((T, G, G, 2), np.float16),
                gt_masks=np.zeros((T + 1, G, G), bool))


def test_generic_t_validation(tmp_root):
    for T in (3, 5):
        p = tmp_root / f"t{T}" / "0000.npz"
        bundles.save_bundle(p, **_synth(T), meta={"stim_id": 0, "variant": f"t{T}"})
        arrays, meta = bundles.load_bundle(p)
        assert arrays["points_3d"].shape[0] == T
    bad = _synth(3)
    bad["depth_sq"] = bad["depth_sq"][:3]          # T instead of T+1
    try:
        bundles.save_bundle(tmp_root / "bad" / "0000.npz", **bad,
                            meta={"stim_id": 0, "variant": "bad"})
        raise AssertionError("inconsistent-T bundle accepted")
    except ValueError:
        pass
    print("generic-T bundle validation: PASS")


if __name__ == "__main__":
    import tempfile
    test_segment_layout()
    test_mask_index_clamp()
    test_gating_defaults()
    with tempfile.TemporaryDirectory() as d:
        test_generic_t_validation(Path(d))
    print("test_segments: PASS")
