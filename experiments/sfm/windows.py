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

# LEGACY blind 4-frame tiles (first phase-2 pass): cuts at 4/8/12/16/20 straddle
# motion onset (6) and offset (18) — kept only to read existing results. New work
# should use EXP_SEGMENTS below.
SEGMENT_VARIANTS = {f"seg{i}": tuple(range(4 * i, 4 * i + 4)) for i in range(6)}

# Metadata-aligned segmentation: the MWorks protocol presents each clip once for
# 400 ms; the generation notebook assembles 100 ms static -> 200 ms motion ->
# 100 ms static at 60 fps, so the EXPERIMENT's divisions are motion onset (frame 6)
# and offset (frame 18). These segments never straddle either event: motion windows
# are pure motion, and onset/offset are between-segment transitions.
EXP_SEGMENTS = {
    "hold1": (0, 1, 2, 3, 4, 5),
    "m0":    (6, 7, 8, 9),
    "m1":    (10, 11, 12, 13),
    "m2":    (14, 15, 16, 17),
    "hold2": (18, 19, 20, 21, 22, 23),
}

# EXP2: same cuts, one-frame forward overlap so every frame gets a model output.
# A window of F frames evaluates frames[:-1] (each needs its outgoing flow pair),
# so EXP_SEGMENTS leaves the five segment-final frames (5, 9, 13, 17, 23)
# unevaluated. Extending each window one frame into the next segment closes the
# gap without moving any init frame off its epoch start and without evaluating
# any frame twice: the boundary pairs are (5,6) and (17,18), both zero-motion by
# construction (frame 6 re-shows the hold; frame 18 holds init+11). Frame 23 has
# no successor, so hold2x repeats it — a self-pair with identically zero flow
# (the preprocessor short-circuits i==j pairs), which is exact for a static hold.
EXP2_SEGMENTS = {
    "hold1x": (0, 1, 2, 3, 4, 5, 6),
    "m0x":    (6, 7, 8, 9, 10),
    "m1x":    (10, 11, 12, 13, 14),
    "m2x":    (14, 15, 16, 17, 18),
    "hold2x": (18, 19, 20, 21, 22, 23, 23),
}

WINDOW_VARIANTS = {
    "tiledA":   (6, 7, 8, 9, 10, 11),
    "tiledB":   (12, 13, 14, 15, 16, 17),
    "centered": (9, 10, 11, 12, 13, 14),
    "stride2":  (6, 8, 10, 12, 14, 16),
    **SEGMENT_VARIANTS,
    **EXP_SEGMENTS,
    **EXP2_SEGMENTS,
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
    return (is_gated(variant) or base_variant(variant) in SEGMENT_VARIANTS
            or base_variant(variant) in EXP_SEGMENTS
            or base_variant(variant) in EXP2_SEGMENTS)


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
