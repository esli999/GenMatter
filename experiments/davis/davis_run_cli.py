"""Shared CLI flags for DAVIS GenMatter experiment scripts (e.g. ``--use-sam``)."""

from __future__ import annotations

import argparse
from types import ModuleType

import config

_KIND_TO_PAIR = {
    "tracking": (
        config.DAVIS_TRACKING_OUTPUT_DIR_SAM,
        config.DAVIS_TRACKING_OUTPUT_DIR_NO_SAM,
    ),
    "subsampling": (
        config.DAVIS_SUBSAMPLING_OUTPUT_DIR_SAM,
        config.DAVIS_SUBSAMPLING_OUTPUT_DIR_NO_SAM,
    ),
    "ablation": (
        config.DAVIS_ABLATION_OUTPUT_DIR_SAM,
        config.DAVIS_ABLATION_OUTPUT_DIR_NO_SAM,
    ),
    "ablation_2": (
        config.DAVIS_ABLATION_2_OUTPUT_DIR_SAM,
        config.DAVIS_ABLATION_2_OUTPUT_DIR_NO_SAM,
    ),
    "ablation_3": (
        config.DAVIS_ABLATION_3_OUTPUT_DIR_SAM,
        config.DAVIS_ABLATION_3_OUTPUT_DIR_NO_SAM,
    ),
    "ablation_4": (
        config.DAVIS_ABLATION_4_OUTPUT_DIR_SAM,
        config.DAVIS_ABLATION_4_OUTPUT_DIR_NO_SAM,
    ),
}


def add_use_sam_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--use-sam",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use SAM frame-0 PNGs for initialization (default). "
            "Pass --no-use-sam to use the TAP-Vid frame-0 mask and ratio-based K-means init."
        ),
    )


def add_save_3wide_video_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--save-3wide-video",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Write GenMatter 3-wide visualization MP4s under 3wide_videos/ (default: off). "
            "Pass --save-3wide-video to enable encoding and disk I/O."
        ),
    )


def configure_experiment_module(
    module: ModuleType, args: argparse.Namespace, kind: str
) -> None:
    if kind not in _KIND_TO_PAIR:
        raise ValueError(f"unknown DAVIS experiment kind: {kind!r}")
    sam_dir, no_sam_dir = _KIND_TO_PAIR[kind]
    use_sam = bool(args.use_sam)
    module.USE_SAM_FRAME0 = use_sam
    module.EXPERIMENT_SAVE_DIR = str(sam_dir if use_sam else no_sam_dir)
    module.SAVE_3WIDE_VIDEO = bool(getattr(args, "save_3wide_video", False))
