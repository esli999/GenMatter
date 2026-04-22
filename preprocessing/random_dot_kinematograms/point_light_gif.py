"""Point-light GIF export (matplotlib), ported from a reference notebook."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
from rich.console import Console


def _video_frame_hw(out_root: Path, config_num: int, cache: dict[int, tuple[int, int]]) -> tuple[int, int]:
    """Height × width of first decoded frame (matches RAFT / ``points_data`` coordinates)."""
    if config_num in cache:
        return cache[config_num]
    path = out_root / "physics_rgb" / f"config_{config_num}.mp4"
    if not path.is_file():
        cache[config_num] = (608, 800)
        return cache[config_num]
    r = imageio.get_reader(path)
    try:
        fr = r.get_data(0)
    finally:
        r.close()
    h, w = int(fr.shape[0]), int(fr.shape[1])
    cache[config_num] = (h, w)
    return cache[config_num]


def create_point_light_gif(
    points_data: np.ndarray,
    start_frame: int,
    end_frame: int,
    point_size: int = 5,
    fps: int = 30,
    dpi: int = 100,
    red_point: tuple | list | None = None,
    green_point: tuple | list | None = None,
    probe_timestep: int | None = None,
    highlight_time_length: int | None = None,
    gif_path: str | Path | None = None,
    show_probe_point: bool = True,
    *,
    canvas_height: int | None = None,
    canvas_width: int | None = None,
) -> Path | Any:
    """
    Point-light display: forward then backward. Saves GIF via Pillow writer when ``gif_path`` is set.

    ``canvas_height`` / ``canvas_width`` should match the decoded physics MP4 (H, W), same as
    ``points_data`` coordinates. If omitted, defaults to 608×800 (legacy); wrong aspect breaks
    perceived motion timing relative to the probe.
    """
    hl = int(highlight_time_length or 0)

    frames_to_use = points_data[start_frame : end_frame + 1]
    if canvas_height is not None and canvas_width is not None:
        height, width = canvas_height, canvas_width
    else:
        height, width = 608, 800

    num_frames = len(frames_to_use)
    forward_indices = list(range(num_frames))
    backward_indices = list(range(num_frames - 1, -1, -1))

    frame_sequence: list[tuple[int, bool]] = []
    for i in forward_indices:
        frame_sequence.append((i, False))
        if probe_timestep is not None and i == (probe_timestep - start_frame):
            for _ in range(hl):
                frame_sequence.append((i, True))

    for i in backward_indices:
        frame_sequence.append((i, False))
        if probe_timestep is not None and i == (probe_timestep - start_frame):
            for _ in range(hl):
                frame_sequence.append((i, True))

    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi, facecolor="black")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal")
    ax.axis("off")

    scatter = ax.scatter([], [], s=point_size**2, color="white", zorder=1)

    red_scatter = None
    green_scatter = None
    if show_probe_point:
        if red_point is not None:
            red_scatter = ax.scatter(
                [],
                [],
                s=(3 * 3) ** 2,
                color="#FF0000",
                edgecolors="white",
                linewidth=0.5,
                alpha=1.0,
                zorder=3,
            )
        if green_point is not None:
            green_scatter = ax.scatter(
                [],
                [],
                s=(3 * 3) ** 2,
                color="#00FF00",
                edgecolors="white",
                linewidth=0.5,
                alpha=1.0,
                zorder=3,
            )

    def init():
        scatter.set_offsets(np.empty((0, 2)))
        if red_scatter:
            red_scatter.set_offsets(np.empty((0, 2)))
        if green_scatter:
            green_scatter.set_offsets(np.empty((0, 2)))
        out = [scatter]
        if red_scatter:
            out.append(red_scatter)
        if green_scatter:
            out.append(green_scatter)
        return out

    def update(frame_data):
        frame_idx, highlight = frame_data
        scatter.set_offsets(frames_to_use[frame_idx])
        show_highlights = highlight or (
            probe_timestep is not None and frame_idx == (probe_timestep - start_frame)
        )
        if red_scatter:
            red_scatter.set_offsets(
                [red_point] if show_highlights and red_point is not None else np.empty((0, 2))
            )
        if green_scatter:
            green_scatter.set_offsets(
                [green_point] if show_highlights and green_point is not None else np.empty((0, 2))
            )
        out = [scatter]
        if red_scatter:
            out.append(red_scatter)
        if green_scatter:
            out.append(green_scatter)
        return out

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=frame_sequence,
        init_func=init,
        blit=False,
        interval=1000 / fps,
        repeat=True,
    )

    if gif_path is not None:
        gif_path = Path(gif_path)
        gif_path.parent.mkdir(parents=True, exist_ok=True)
        anim.save(str(gif_path), writer="pillow", fps=fps)
        plt.close(fig)
        return gif_path

    plt.close(fig)
    return anim


def run_all_gifs(
    out_root: Path,
    rdk_configs_path: Path,
    *,
    console: Console | None = None,
    fps: int = 8,
    point_size: int = 2,
) -> None:
    """Create one GIF per stimulus: probe visible + freeze frames at probe."""
    import json

    from tqdm import tqdm

    con = console or Console()

    with open(rdk_configs_path) as f:
        configs = json.load(f)

    highlight_time_length = max(1, fps // 2)
    out_dir = out_root / "stimuli_gifs"
    out_dir.mkdir(parents=True, exist_ok=True)

    con.print("[bold cyan]Stage 4/4[/bold cyan] Point-light GIFs → `stimuli_gifs/`")

    hw_cache: dict[int, tuple[int, int]] = {}
    for config_key, config_data in tqdm(configs.items(), desc="RDK stimuli", unit="stim"):
        config_num = config_data["config_num"]
        start_frame = config_data["start_frame"]
        end_frame = config_data["end_frame"]
        probe_timestep = config_data["probe_timestep"]
        red_point = config_data["point_1"]
        green_point = config_data["point_2"]

        npz_path = out_root / f"config_{config_num}" / "data.npz"
        if not npz_path.is_file():
            con.print(f"[yellow]Skip[/yellow] {config_key}: missing {npz_path}")
            continue

        data = np.load(npz_path)
        points_data = data["points_data"]
        gif_name = f"{config_key}.gif"

        ch, cw = _video_frame_hw(out_root, config_num, hw_cache)
        create_point_light_gif(
            points_data,
            start_frame,
            end_frame,
            point_size,
            red_point=red_point,
            green_point=green_point,
            probe_timestep=probe_timestep,
            highlight_time_length=highlight_time_length,
            fps=fps,
            show_probe_point=True,
            gif_path=out_dir / gif_name,
            canvas_height=ch,
            canvas_width=cw,
        )
