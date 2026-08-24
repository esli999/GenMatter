"""
Single source of truth for temporal windows over the SFM videos.

Video layout (24 frames @60fps): frames 0-5 static (render frame init), frames 6-17
moving (render frames init..init+11; frame 6 duplicates the static hold), frames 18-23
static (render frame init+11).

A window of F video frames -> F depth frames + F-1 flow pairs -> F-1 model timesteps.
Phase-1 variants are 6-frame excerpts of the moving segment; the seg0..seg5 variants
tile the WHOLE 24-frame video as six consecutive 4-frame segments (seg0/seg5 are the
static holds — reported separately, kept for ephys-aligned model output).
"""

SEGMENT_VARIANTS = {f"seg{i}": tuple(range(4 * i, 4 * i + 4)) for i in range(6)}

WINDOW_VARIANTS = {
    "tiledA":   (6, 7, 8, 9, 10, 11),
    "tiledB":   (12, 13, 14, 15, 16, 17),
    "centered": (9, 10, 11, 12, 13, 14),
    "stride2":  (6, 8, 10, 12, 14, 16),
    **SEGMENT_VARIANTS,
}

PILOT_VARIANTS = ("tiledA", "tiledB", "centered", "stride2")


def base_variant(variant: str) -> str:
    """A '_g' suffix marks evidence-gated preprocessing of the same frame window."""
    return variant.removesuffix("_g")


def is_gated(variant: str) -> bool:
    return variant.endswith("_g")


def default_gated(variant: str) -> bool:
    """Segments are always evidence-gated (a no-op on textured stimuli); phase-1
    variants opt in via the '_g' suffix so gated/ungated A-B results stay distinct."""
    return is_gated(variant) or base_variant(variant) in SEGMENT_VARIANTS


def window_frames(variant: str):
    return WINDOW_VARIANTS[base_variant(variant)]


def flow_pairs(variant: str):
    """Consecutive frame pairs within the window (len(frames)-1 pairs)."""
    f = WINDOW_VARIANTS[base_variant(variant)]
    return tuple(zip(f[:-1], f[1:]))


def moving_index(video_frame: int) -> int:
    """Video frame index (6..17) -> index into the 12-frame moving arrays (0..11)."""
    if not (6 <= video_frame <= 17):
        raise ValueError(f"frame {video_frame} is not in the moving segment 6..17")
    return video_frame - 6


def frame_to_mask_index(video_frame: int) -> int:
    """Any video frame (0..23) -> index into the 12-frame GT mask stack: the static
    holds show render frame init (mask 0) / init+11 (mask 11)."""
    if not (0 <= video_frame <= 23):
        raise ValueError(f"frame {video_frame} is not a video frame 0..23")
    return min(max(video_frame - 6, 0), 11)


def mask_indices(variant: str):
    """Indices into the per-(size,object,viewpoint) 12-frame mask stack for this window."""
    return tuple(frame_to_mask_index(f) for f in WINDOW_VARIANTS[base_variant(variant)])
