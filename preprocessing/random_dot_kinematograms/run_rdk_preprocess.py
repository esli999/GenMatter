#!/usr/bin/env python3
"""Orchestrate RDK preprocessing: physics MP4 → RAFT NPZ → canonical JSON → GIFs."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from preprocessing.random_dot_kinematograms.RDK_extraction import run_npz_extraction


def emit_canonical_json(out_dir: Path, console: Console) -> None:
    canon_dir = Path(__file__).resolve().parent / "canonical"
    for name in ("RDK_configs.json", "reproducibility_keys.json"):
        src = canon_dir / name
        if not src.is_file():
            raise FileNotFoundError(f"Missing canonical file: {src}")
        dst = out_dir / name
        shutil.copy2(src, dst)
        console.print(f"  Copied [green]{name}[/green] → {dst}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build GenMatter RDK assets under assets/RDK (or --out).")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output root (default: <repo>/assets/RDK)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Master RNG seed for NPZ extraction.")
    parser.add_argument("--skip-physics", action="store_true", help="Skip pymunk MP4 generation.")
    parser.add_argument("--skip-npz", action="store_true", help="Skip RAFT / data.npz.")
    parser.add_argument("--skip-json", action="store_true", help="Skip copying canonical JSON.")
    parser.add_argument("--skip-gifs", action="store_true", help="Skip point-light GIF export.")
    parser.add_argument("--frame-pbar", action="store_true", help="Per-frame tqdm during physics simulation.")
    args = parser.parse_args()

    repo = _REPO
    out_root = (args.out or (repo / "assets" / "RDK")).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    console = Console()
    console.print(
        Panel.fit(
            "[bold]GenMatter RDK preprocessing[/bold]\n"
            f"Output: [cyan]{out_root}[/cyan]\n"
            f"Seed: [yellow]{args.seed}[/yellow]",
            title="RDK pipeline",
        )
    )

    if not args.skip_physics:
        from preprocessing.random_dot_kinematograms.physics_mp4 import run_physics_mp4s

        run_physics_mp4s(out_root, console, frame_pbar=args.frame_pbar)
    else:
        console.print("[dim]Skipping Stage 1 (physics MP4s)[/dim]")

    if not args.skip_npz:
        run_npz_extraction(out_root, console, master_seed=args.seed)
    else:
        console.print("[dim]Skipping Stage 2 (RAFT NPZ)[/dim]")

    if not args.skip_json:
        console.print("[bold cyan]Stage 3/4[/bold cyan] Canonical JSON (RDK_configs + reproducibility_keys)")
        emit_canonical_json(out_root, console)
    else:
        console.print("[dim]Skipping Stage 3 (JSON copy)[/dim]")

    if not args.skip_gifs:
        from preprocessing.random_dot_kinematograms.point_light_gif import run_all_gifs

        cfg_path = out_root / "RDK_configs.json"
        if not cfg_path.is_file():
            console.print("[red]RDK_configs.json missing; run without --skip-json first.[/red]")
            return 1
        run_all_gifs(out_root, cfg_path, console=console)
    else:
        console.print("[dim]Skipping Stage 4 (GIFs)[/dim]")

    table = Table(title="Done")
    table.add_column("Artifact", style="cyan")
    table.add_column("Path")
    table.add_row("physics_rgb/", str(out_root / "physics_rgb"))
    table.add_row("config_*/data.npz", str(out_root / "config_*"))
    table.add_row("RDK JSON", str(out_root / "RDK_configs.json"))
    table.add_row("stimuli_gifs/", str(out_root / "stimuli_gifs"))
    console.print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
