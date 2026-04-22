#!/usr/bin/env python3
"""
Copy Gestalt inputs into ``<GENMATTER_DATA_DIR>/assets/gestalt_stimuli/`` (default: ``assets/gestalt_stimuli/``) from a source tree that has
``gestalt_scenes/`` plus full-length RAFT ``*.npz`` (defaults: real ``<repo>/assets``
and ``<repo>/raft_flows``, else ``GENMATTER_LEGACY_DATA_ROOT`` when set; see ``config.py``).

RAFT ``.npz`` files are written trimmed to ``GESTALT_RAFT_NUM_FLOW_FRAMES`` and
compressed (see ``config.py``).

  python scripts/populate_genmatter_data.py
  python scripts/populate_genmatter_data.py --raft-only
  python scripts/populate_genmatter_data.py --dry-run
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config  # noqa: E402

GESTALT_MASK_FRAMES = range(1, 7)
FLOWSAM_FRAME_INDICES = range(5)


def _remove_path(p: Path) -> None:
    if not p.exists() and not p.is_symlink():
        return
    if p.is_symlink() or p.is_file():
        p.unlink()
    else:
        shutil.rmtree(p)


def _copy_file(src: Path, dst: Path, *, dry_run: bool) -> None:
    if not src.is_file():
        return
    if dry_run:
        print(f"  [dry-run] copy {src} -> {dst}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    shutil.copy2(src, dst)


def _write_trimmed_raft_npz(
    src: Path,
    dst: Path,
    *,
    num_frames: int,
    dry_run: bool,
) -> None:
    if not src.is_file():
        return
    if dry_run:
        print(f"  [dry-run] trim RAFT ({num_frames} fr) {src} -> {dst}")
        return
    with np.load(src, allow_pickle=False) as data:
        if "flow" in data:
            flow = np.asarray(data["flow"])
        else:
            keys = list(data.files)
            if not keys:
                return
            flow = np.asarray(data[keys[0]])
    n = min(int(num_frames), flow.shape[0])
    flow_trim = np.ascontiguousarray(flow[:n])
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    np.savez_compressed(dst, flow=flow_trim)


def populate_gestalt_gestalt_scenes(
    source_gestalt_scenes_parent: Path, dry_run: bool
) -> None:
    """Copy minimal files into ``config.GESTALT_BASE_PATH``."""
    src_gestalt_scenes = source_gestalt_scenes_parent / "gestalt_scenes"
    dst_gestalt_scenes = config.GESTALT_BASE_PATH
    print(f"Gestalt → {dst_gestalt_scenes}")
    for scene in config.GESTALT_SCENES:
        mask_dir = src_gestalt_scenes / scene / "render_passes" / "masks"
        for fi in GESTALT_MASK_FRAMES:
            name = f"Image{fi:04d}.png"
            _copy_file(
                mask_dir / name,
                dst_gestalt_scenes / scene / "render_passes" / "masks" / name,
                dry_run=dry_run,
            )
        for tex in config.GESTALT_TEXTURES:
            depth = src_gestalt_scenes / scene / tex / "output_six_frame_depths.npz"
            _copy_file(
                depth,
                dst_gestalt_scenes / scene / tex / "output_six_frame_depths.npz",
                dry_run=dry_run,
            )
            flowsam_dir = src_gestalt_scenes / scene / tex / config.FLOWSAM_MASKS_SUBPATH
            if flowsam_dir.is_dir():
                for idx in FLOWSAM_FRAME_INDICES:
                    fname = f"frame_{idx:05d}_matched.png"
                    _copy_file(
                        flowsam_dir / fname,
                        dst_gestalt_scenes
                        / scene
                        / tex
                        / config.FLOWSAM_MASKS_SUBPATH
                        / fname,
                        dry_run=dry_run,
                    )


def populate_minimal_raft(source_raft: Path, dry_run: bool) -> None:
    dst_raft = config.RAFT_FLOWS_PATH
    n = config.GESTALT_RAFT_NUM_FLOW_FRAMES
    print(f"RAFT → {dst_raft} (first {n} frames, compressed)")
    for scene in config.GESTALT_SCENES:
        for tex in config.GESTALT_TEXTURES:
            name = f"raft_flows_{scene}_{tex}.npz"
            _write_trimmed_raft_npz(
                source_raft / name,
                dst_raft / name,
                num_frames=n,
                dry_run=dry_run,
            )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source",
        type=Path,
        default=config.POPULATE_GESTALT_SOURCE_DIR,
        help=(
            "Directory that contains ``gestalt_scenes/`` "
            "(default: real <repo>/assets, else GENMATTER_LEGACY_DATA_ROOT/assets when set; "
            "see config.py)"
        ),
    )
    p.add_argument(
        "--raft-source",
        type=Path,
        default=config.POPULATE_FULL_RAFT_SOURCE_DIR,
        help=(
            "Directory of full-length RAFT ``raft_flows_*.npz`` "
            "(default: real <repo>/raft_flows, else GENMATTER_LEGACY_DATA_ROOT/raft_flows when set)"
        ),
    )
    p.add_argument(
        "--raft-only",
        action="store_true",
        help="Only rewrite trimmed RAFT npz under RAFT_FLOWS_PATH",
    )
    p.add_argument("--dry-run", action="store_true", help="Print planned actions only")
    args = p.parse_args()

    src = args.source.resolve()
    raft_src = args.raft_source.resolve()

    print(f"Gestalt dest:  {config.GESTALT_BASE_PATH}")
    print(f"RAFT dest:     {config.RAFT_FLOWS_PATH}")
    print(f"gestalt_scenes ← {src / 'gestalt_scenes'}")
    print(f"RAFT src:      {raft_src}")
    print()

    if not args.dry_run:
        config.GESTALT_STIMULI_DIR.mkdir(parents=True, exist_ok=True)

    if args.raft_only:
        populate_minimal_raft(raft_src, args.dry_run)
    else:
        populate_gestalt_gestalt_scenes(src, args.dry_run)
        populate_minimal_raft(raft_src, args.dry_run)

    print("\nDone.")


if __name__ == "__main__":
    main()
