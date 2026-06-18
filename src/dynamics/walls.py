from typing import Optional, Tuple

import numpy as np
from src.geometry.channel import RectangularChannel


def _apply_x_boundary(position: np.ndarray, velocity: np.ndarray, radius: float,
                      channel: RectangularChannel, x_boundary_type: str) -> Tuple[np.ndarray, np.ndarray]:
    if x_boundary_type == "periodic":
        position = position.copy()
        position[0] = position[0] % channel.width
        return position, velocity

    # Reflective x walls behave as specular bounces
    if position[0] < radius:
        position = position.copy()
        position[0] = radius
        velocity = velocity.copy()
        velocity[0] = -velocity[0]
    elif position[0] > channel.width - radius:
        position = position.copy()
        position[0] = channel.width - radius
        velocity = velocity.copy()
        velocity[0] = -velocity[0]

    return position, velocity


def _apply_y_reflective(position: np.ndarray, velocity: np.ndarray, radius: float,
                        channel: RectangularChannel) -> Tuple[np.ndarray, np.ndarray]:
    position = position.copy()
    velocity = velocity.copy()
    if position[1] < radius:
        position[1] = radius
        velocity[1] = -velocity[1]
    elif position[1] > channel.height - radius:
        position[1] = channel.height - radius
        velocity[1] = -velocity[1]
    return position, velocity


def apply_specular_wall(position: np.ndarray, velocity: np.ndarray, radius: float,
                        channel: RectangularChannel, x_boundary_type: str) -> Tuple[np.ndarray, np.ndarray]:
    new_position, new_velocity = _apply_x_boundary(position, velocity, radius, channel, x_boundary_type)
    new_position, new_velocity = _apply_y_reflective(new_position, new_velocity, radius, channel)
    if x_boundary_type == "periodic":
        new_position, new_velocity = _apply_x_boundary(new_position, new_velocity, radius, channel, x_boundary_type)
    return new_position, new_velocity


def apply_diffuse_same_speed_wall(position: np.ndarray, velocity: np.ndarray, radius: float,
                                  channel: RectangularChannel, x_boundary_type: str,
                                  rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, np.ndarray]:
    if rng is None:
        rng = np.random.default_rng()
    new_position, new_velocity = _apply_x_boundary(position, velocity, radius, channel, x_boundary_type)
    collide_bottom = new_position[1] < radius
    collide_top = new_position[1] > channel.height - radius

    if collide_bottom or collide_top:
        speed = np.linalg.norm(new_velocity)
        if speed == 0.0:
            speed = 1e-6

        if collide_bottom:
            angle = rng.uniform(0.0, np.pi)
            new_position[1] = radius
        else:
            angle = rng.uniform(-np.pi, 0.0)
            new_position[1] = channel.height - radius

        new_velocity = np.array([
            speed * np.cos(angle),
            speed * np.sin(angle)
        ])
    return new_position, new_velocity


def apply_diffuse_thermal_wall(position: np.ndarray, velocity: np.ndarray, radius: float,
                               channel: RectangularChannel, x_boundary_type: str,
                               wall_temperature: float,
                               rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, np.ndarray]:
    if rng is None:
        rng = np.random.default_rng()
    new_position, new_velocity = _apply_x_boundary(position, velocity, radius, channel, x_boundary_type)
    collide_bottom = new_position[1] < radius
    collide_top = new_position[1] > channel.height - radius

    if collide_bottom or collide_top:
        sigma = np.sqrt(wall_temperature)
        if sigma == 0.0:
            sigma = 1e-6

        new_position = new_position.copy()
        if collide_bottom:
            new_position[1] = radius
        else:
            new_position[1] = channel.height - radius

        # Reject samples that are directed into the wall
        while True:
            sampled = rng.normal(loc=0.0, scale=sigma, size=2)
            if collide_bottom and sampled[1] > 0:
                new_velocity = sampled
                break
            if collide_top and sampled[1] < 0:
                new_velocity = sampled
                break

    return new_position, new_velocity


def handle_wall_collision(position: np.ndarray, velocity: np.ndarray, radius: float,
                          channel: RectangularChannel, wall_model: str,
                          x_boundary_type: str, wall_temperature: float,
                          rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, np.ndarray]:
    if wall_model == "specular":
        return apply_specular_wall(position, velocity, radius, channel, x_boundary_type)
    if wall_model == "diffuse_same_speed":
        return apply_diffuse_same_speed_wall(position, velocity, radius, channel, x_boundary_type, rng=rng)
    if wall_model == "diffuse_thermal":
        return apply_diffuse_thermal_wall(position, velocity, radius, channel, x_boundary_type, wall_temperature, rng=rng)
    raise ValueError(f"Unknown wall model: {wall_model}")
