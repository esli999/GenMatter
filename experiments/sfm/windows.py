"""
Single source of truth for temporal windows over the SFM videos.

Video layout (24 frames @60fps): frames 0-5 static (render frame init), frames 6-17
moving (render frames init..init+11; frame 6 duplicates the static hold), frames 18-23
static (render frame init+11). Only moving frames are consumed.

A window is 6 video frames -> 6 depth frames + 5 flow pairs -> 5 model timesteps.
"""

WINDOW_VARIANTS = {
    "tiledA":   (6, 7, 8, 9, 10, 11),
    "tiledB":   (12, 13, 14, 15, 16, 17),
    "centered": (9, 10, 11, 12, 13, 14),
    "stride2":  (6, 8, 10, 12, 14, 16),
}

PILOT_VARIANTS = ("tiledA", "tiledB", "centered", "stride2")


def base_variant(variant: str) -> str:
    """A '_g' suffix marks evidence-gated preprocessing of the same frame window."""
    return variant.removesuffix("_g")


def is_gated(variant: str) -> bool:
    return variant.endswith("_g")


def window_frames(variant: str):
    return WINDOW_VARIANTS[base_variant(variant)]


def flow_pairs(variant: str):
    """Consecutive frame pairs within the window (5 pairs)."""
    f = WINDOW_VARIANTS[base_variant(variant)]
    return tuple(zip(f[:-1], f[1:]))


def moving_index(video_frame: int) -> int:
    """Video frame index (6..17) -> index into the 12-frame moving arrays (0..11)."""
    if not (6 <= video_frame <= 17):
        raise ValueError(f"frame {video_frame} is not in the moving segment 6..17")
    return video_frame - 6


def mask_indices(variant: str):
    """Indices into the per-(size,object,viewpoint) 12-frame mask stack for this window."""
    return tuple(moving_index(f) for f in WINDOW_VARIANTS[base_variant(variant)])
