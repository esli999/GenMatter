"""Physics-based RDK stimulus videos (pymunk + pygame).

The physics export intentionally uses only ``impulse_magnitude`` from the per-config dict.
If it is not present, a default value of ``200000`` is used.
"""

from __future__ import annotations

import logging
import math
import os
from contextlib import contextmanager
from math import pi
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pymunk
import pymunk.pygame_util
import pygame
from PIL import Image
from rich.console import Console
from tqdm import tqdm


def _configure_sdl_for_rdk_physics() -> None:
    """Video-only export: avoid ALSA errors when no sound card. Optional dummy display for headless."""
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    if os.environ.get("GENMATTER_RDK_HEADLESS", "").lower() in ("1", "true", "yes"):
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


@contextmanager
def _suppress_imageio_ffmpeg_macroblock_warnings():
    """imageio_ffmpeg logs a resize warning when H/W are not multiples of 16."""
    log = logging.getLogger("imageio_ffmpeg")
    prev = log.level
    log.setLevel(logging.ERROR)
    try:
        yield
    finally:
        log.setLevel(prev)


class PhysicsSimulation:
    """2D rigid-body simulation rendered to RGB frames and exported as MP4."""

    def __init__(self, config: dict):
        self.config = config
        self.bottom_shape_type = config.get("bottom_shape_type", "cube")
        self.top_shape_type = config.get("top_shape_type", "cube")
        self.top_rotation = config.get("top_rotation", 0)
        self.force_target = config.get("force_target", "bottom")
        self.top_offset_x = config.get("top_offset_x", 0)
        self.top_offset_y = config.get("top_offset_y", 0)

        self.friction = config.get("friction", 1.0)
        self.elasticity = config.get("elasticity", 0.2)
        self.density = config.get("density", 1.0)
        self.cube_size = config.get("cube_size", 75)
        self.rod_width = config.get("rod_width", 75)
        self.rod_height = config.get("rod_height", 150)

        self.dt = config.get("dt", 1 / 60.0)
        self.num_frames = config.get("num_frames", 120)
        self.settle_frames = config.get("settle_frames", 30)
        self.force_duration = config.get("force_duration", 30)
        # Legacy behavior: per-scenario ``impulse`` is ignored unless ``impulse_magnitude`` is provided.
        self.impulse_magnitude = float(config.get("impulse_magnitude", 200000))

        self.file_name = config.get("file_name", None)

        self.width = config.get("width", 800)
        self.height = config.get("height", 600)

        _configure_sdl_for_rdk_physics()
        pygame.init()
        self.load_textures()
        self.setup_space()
        self.setup_floor()
        self.create_shapes()
        self.setup_rendering()
        self.frames: list[np.ndarray] = []

    def load_textures(self) -> None:
        self.textures = {}

        bottom_texture = pygame.Surface((self.cube_size * 2, self.cube_size * 2))
        checker_size = self.cube_size // 4
        for y in range(0, self.cube_size * 2, checker_size):
            for x in range(0, self.cube_size * 2, checker_size):
                color = (0, 0, 255) if (x // checker_size + y // checker_size) % 2 == 0 else (100, 100, 255)
                pygame.draw.rect(bottom_texture, color, (x, y, checker_size, checker_size))
        self.textures["bottom"] = bottom_texture

        top_texture = pygame.Surface((self.cube_size * 2, self.cube_size * 2))
        for y in range(0, self.cube_size * 2, checker_size):
            for x in range(0, self.cube_size * 2, checker_size):
                color = (255, 0, 0) if (x // checker_size + y // checker_size) % 2 == 0 else (255, 100, 100)
                pygame.draw.rect(top_texture, color, (x, y, checker_size, checker_size))
        self.textures["top"] = top_texture

        bg_texture = pygame.Surface((self.width, self.height))
        bg_checker_size = 50
        for y in range(0, self.height, bg_checker_size):
            for x in range(0, self.width, bg_checker_size):
                color = (240, 240, 240) if (x // bg_checker_size + y // bg_checker_size) % 2 == 0 else (200, 200, 200)
                pygame.draw.rect(bg_texture, color, (x, y, bg_checker_size, bg_checker_size))
        self.textures["background"] = bg_texture

        floor_texture = pygame.Surface((800, 20))
        floor_checker_size = 10
        for y in range(0, 20, floor_checker_size):
            for x in range(0, 800, floor_checker_size):
                color = (139, 69, 19) if (x // floor_checker_size + y // floor_checker_size) % 2 == 0 else (160, 82, 45)
                pygame.draw.rect(floor_texture, color, (x, y, floor_checker_size, floor_checker_size))
        self.textures["floor"] = floor_texture

    def setup_space(self) -> None:
        self.space = pymunk.Space()
        self.space.gravity = (0, -900)

    def setup_floor(self) -> None:
        self.floor_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.floor_body.position = (400, 100)
        floor_shape = pymunk.Poly.create_box(self.floor_body, (800, 20))
        floor_shape.friction = self.friction
        floor_shape.elasticity = self.elasticity
        self.space.add(self.floor_body, floor_shape)
        self.floor_top_y = self.floor_body.position.y + 10

    def get_shape_dimensions(self, shape_type: str) -> tuple[float, float]:
        if shape_type == "L":
            return self.cube_size * 2, self.cube_size * 2
        if shape_type == "T":
            return self.cube_size * 3, self.cube_size * 2
        if shape_type == "cube":
            return self.cube_size * 2, self.cube_size * 2
        if shape_type == "rod":
            return self.rod_width, self.rod_height
        raise ValueError(shape_type)

    def create_shape_by_type(self, shape_type: str, position, color, angle: float = 0):
        if shape_type == "L":
            return self.create_L_shape(position, color, angle)
        if shape_type == "T":
            return self.create_T_shape(position, color, angle)
        if shape_type == "cube":
            return self.create_cube(position, color, angle)
        if shape_type == "rod":
            return self.create_rod(position, color, angle)
        raise ValueError(shape_type)

    def create_shapes(self) -> None:
        bottom_width, bottom_height = self.get_shape_dimensions(self.bottom_shape_type)
        bottom_y = self.floor_top_y + bottom_height / 2
        self.bottom_shape = self.create_shape_by_type(
            self.bottom_shape_type,
            (400, bottom_y),
            (0, 0, 255, 255),
            angle=0,
        )

        top_width, top_height = self.get_shape_dimensions(self.top_shape_type)
        top_y = bottom_y + (bottom_height + top_height) / 2
        self.top_shape = self.create_shape_by_type(
            self.top_shape_type,
            (400 + self.top_offset_x, top_y + self.top_offset_y),
            (255, 0, 0, 255),
            angle=self.top_rotation,
        )

    def create_L_shape(self, position, color, angle: float = 0):
        body = pymunk.Body(body_type=pymunk.Body.DYNAMIC)
        body.position = position
        body.angle = angle

        h_positions = [(-self.cube_size / 2, self.cube_size / 2), (self.cube_size / 2, self.cube_size / 2)]
        v_positions = [(-self.cube_size / 2, -self.cube_size / 2)]
        shapes = []

        for pos in h_positions:
            shape = pymunk.Poly.create_box(body, (self.cube_size, self.cube_size))
            vertices = [pymunk.Vec2d(v.x + pos[0], v.y + pos[1]) for v in shape.get_vertices()]
            shape = pymunk.Poly(body, vertices)
            shape.density = self.density
            shape.friction = self.friction
            shape.elasticity = self.elasticity
            shape.color = color
            shapes.append(shape)

        for pos in v_positions:
            shape = pymunk.Poly.create_box(body, (self.cube_size, self.cube_size))
            vertices = [pymunk.Vec2d(v.x + pos[0], v.y + pos[1]) for v in shape.get_vertices()]
            shape = pymunk.Poly(body, vertices)
            shape.density = self.density
            shape.friction = self.friction
            shape.elasticity = self.elasticity
            shape.color = color
            shapes.append(shape)

        self.space.add(body, *shapes)
        return body

    def create_T_shape(self, position, color, angle: float = 0):
        body = pymunk.Body(body_type=pymunk.Body.DYNAMIC)
        body.position = position
        body.angle = angle

        h_positions = [(-self.cube_size, self.cube_size / 2), (0, self.cube_size / 2), (self.cube_size, self.cube_size / 2)]
        v_positions = [(0, -self.cube_size / 2)]
        shapes = []

        for pos in h_positions:
            shape = pymunk.Poly.create_box(body, (self.cube_size, self.cube_size))
            vertices = [pymunk.Vec2d(v.x + pos[0], v.y + pos[1]) for v in shape.get_vertices()]
            shape = pymunk.Poly(body, vertices)
            shape.density = self.density
            shape.friction = self.friction
            shape.elasticity = self.elasticity
            shape.color = color
            shapes.append(shape)

        for pos in v_positions:
            shape = pymunk.Poly.create_box(body, (self.cube_size, self.cube_size))
            vertices = [pymunk.Vec2d(v.x + pos[0], v.y + pos[1]) for v in shape.get_vertices()]
            shape = pymunk.Poly(body, vertices)
            shape.density = self.density
            shape.friction = self.friction
            shape.elasticity = self.elasticity
            shape.color = color
            shapes.append(shape)

        self.space.add(body, *shapes)
        return body

    def create_cube(self, position, color, angle: float = 0):
        body = pymunk.Body(body_type=pymunk.Body.DYNAMIC)
        body.position = position
        body.angle = angle

        shape = pymunk.Poly.create_box(body, (self.cube_size * 2, self.cube_size * 2))
        shape.density = self.density
        shape.friction = self.friction
        shape.elasticity = self.elasticity
        shape.color = color

        self.space.add(body, shape)
        return body

    def create_rod(self, position, color, angle: float = 0):
        body = pymunk.Body(body_type=pymunk.Body.DYNAMIC)

        if abs(angle - (math.pi / 2)) < 0.01 or abs(angle + (math.pi / 2)) < 0.01:
            offset = (self.rod_height - self.rod_width) / 2
            adjusted_position = (position[0], position[1] - offset)
            body.position = adjusted_position
        else:
            body.position = position

        body.angle = angle

        shape = pymunk.Poly.create_box(body, (self.rod_width, self.rod_height))
        shape.density = self.density
        shape.friction = self.friction
        shape.elasticity = self.elasticity
        shape.color = color

        self.space.add(body, shape)
        return body

    def setup_rendering(self) -> None:
        self.screen = pygame.Surface((self.width, self.height))
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)

    def get_force_point(self, shape, shape_type: str):
        if shape_type == "L":
            return (shape.position.x - self.cube_size, shape.position.y + self.cube_size)
        if shape_type == "T":
            bb_width = self.cube_size * 3
            bb_height = self.cube_size * 2
            return (shape.position.x + (bb_width * 0.5), shape.position.y + (bb_height * 0.5))
        if shape_type == "cube":
            return (shape.position.x, shape.position.y)
        if shape_type == "rod":
            return (shape.position.x, shape.position.y)
        raise ValueError(shape_type)

    def apply_force(self) -> None:
        impulse = (self.impulse_magnitude, 0)
        if self.force_target == "top":
            target_shape = self.top_shape
            target_shape_type = self.top_shape_type
        else:
            target_shape = self.bottom_shape
            target_shape_type = self.bottom_shape_type
        force_point = self.get_force_point(target_shape, target_shape_type)
        target_shape.apply_impulse_at_world_point(impulse, force_point)

    def run_simulation(self, frame_pbar: bool = False) -> None:
        iterator = range(self.num_frames)
        if frame_pbar:
            iterator = tqdm(iterator, desc="Sim frames", leave=False, unit="fr")
        for i in iterator:
            self.screen.blit(self.textures["background"], (0, 0))
            if self.settle_frames < i <= self.settle_frames + self.force_duration:
                self.apply_force()
            self.space.step(self.dt)

            floor_pos = self.floor_body.position
            floor_rect = pygame.Rect(floor_pos.x - 400, self.height - floor_pos.y - 10, 800, 20)
            self.screen.blit(self.textures["floor"], floor_rect)
            self.space.debug_draw(self.draw_options)

            pygame_surface_data = pygame.image.tostring(self.screen, "RGB")
            img = Image.frombytes("RGB", (self.width, self.height), pygame_surface_data)
            self.frames.append(np.array(img))

    def create_animation(self, physics_dir: Path) -> Path:
        if not self.file_name:
            config_name = f"{self.bottom_shape_type}_{self.top_shape_type}"
            rel = f"{config_name}_animation.mp4"
        else:
            rel = f"{self.file_name}.mp4"
        physics_dir.mkdir(parents=True, exist_ok=True)
        mp4_path = physics_dir / rel
        # Flip vertical, then libx264 via imageio (may resize
        # e.g. 800×600 → 800×608 for macroblock compatibility—matches original pipeline).
        # Duration: len(frames) / fps == num_frames * dt (e.g. 120 frames @ 60 fps -> 2 s).
        corrected_frames = [np.flipud(frame) for frame in self.frames]
        with _suppress_imageio_ffmpeg_macroblock_warnings():
            imageio.mimsave(
                str(mp4_path),
                corrected_frames,
                fps=1 / self.dt,
                codec="libx264",
                quality=8,
            )
        return mp4_path

    def cleanup(self) -> None:
        pygame.quit()


# Canonical 10-scenario schedule.
# Note: dynamics depend on ``impulse_magnitude``; the per-config dicts in this file use ``impulse``.
PHYSICS_CONFIGS: list[dict] = [
    {
        "bottom_shape_type": "cube",
        "top_shape_type": "L",
        "top_rotation": 0,
        "cube_size": 75,
        "impulse": 50000,
        "force_target": "top",
        "file_name": "config_1",
    },
    {
        "bottom_shape_type": "rod",
        "top_shape_type": "T",
        "top_rotation": pi,
        "cube_size": 75,
        "rod_width": 75,
        "rod_height": 150,
        "impulse": 100000,
        "force_target": "top",
        "file_name": "config_2",
    },
    {
        "bottom_shape_type": "T",
        "top_shape_type": "cube",
        "top_rotation": 0,
        "cube_size": 75,
        "impulse": 200000,
        "force_target": "top",
        "file_name": "config_3",
    },
    {
        "bottom_shape_type": "L",
        "top_shape_type": "rod",
        "top_rotation": pi / 2,
        "rod_width": 75,
        "rod_height": 150,
        "cube_size": 75,
        "impulse": 75000,
        "force_target": "top",
        "file_name": "config_4",
    },
    {
        "bottom_shape_type": "rod",
        "top_shape_type": "rod",
        "top_rotation": pi / 2,
        "rod_width": 75,
        "rod_height": 150,
        "impulse": 100000,
        "force_target": "top",
        "file_name": "config_5",
    },
    {
        "bottom_shape_type": "cube",
        "top_shape_type": "rod",
        "top_rotation": 0,
        "rod_width": 75,
        "rod_height": 150,
        "cube_size": 75,
        "impulse": 150000,
        "force_target": "top",
        "file_name": "config_6",
    },
    {
        "bottom_shape_type": "rod",
        "top_shape_type": "L",
        "top_rotation": 0,
        "rod_width": 75,
        "rod_height": 150,
        "cube_size": 75,
        "impulse": 75000,
        "force_target": "top",
        "top_offset_x": 75 / 2,
        "top_offset_y": 0,
        "file_name": "config_7",
    },
    {
        "bottom_shape_type": "L",
        "top_shape_type": "rod",
        "top_rotation": 0,
        "rod_width": 75,
        "rod_height": 150,
        "cube_size": 75,
        "impulse": 75000,
        "force_target": "top",
        "top_offset_x": 75 / 2,
        "top_offset_y": 0,
        "file_name": "config_8",
    },
    {
        "bottom_shape_type": "T",
        "top_shape_type": "rod",
        "top_rotation": 0,
        "cube_size": 75,
        "rod_width": 75,
        "rod_height": 150,
        "impulse": 150000,
        "force_target": "top",
        "file_name": "config_9",
    },
    {
        "bottom_shape_type": "rod",
        "top_shape_type": "rod",
        "top_rotation": 0,
        "rod_width": 75,
        "rod_height": 150,
        "impulse": 200000,
        "force_target": "top",
        "file_name": "config_10",
    },
]


def run_physics_mp4s(out_root: Path, console: Console, frame_pbar: bool = False) -> Path:
    """Render all ``PHYSICS_CONFIGS`` MP4s under ``out_root/physics_rgb``."""
    _configure_sdl_for_rdk_physics()
    physics_dir = out_root / "physics_rgb"
    physics_dir.mkdir(parents=True, exist_ok=True)
    console.print("[bold cyan]Stage 1/4[/bold cyan] Physics simulation → MP4")
    console.print(f"Output directory: [green]{physics_dir}[/green]")

    for cfg in tqdm(PHYSICS_CONFIGS, desc="Physics configs", unit="cfg"):
        sim = PhysicsSimulation(cfg)
        try:
            sim.run_simulation(frame_pbar=frame_pbar)
            sim.create_animation(physics_dir)
        finally:
            sim.cleanup()

    return physics_dir
