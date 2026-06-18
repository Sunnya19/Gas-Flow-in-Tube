from typing import Optional, Tuple
import numpy as np

from src.dynamics.walls import handle_wall_collision
from src.geometry.channel import RectangularChannel


def create_wall_model(model_type: str, channel: RectangularChannel,
                      x_boundary_type: str = "reflective") -> str:
    if model_type in ("specular", "diffuse_same_speed", "diffuse_thermal"):
        return model_type
    raise ValueError(f"Unknown wall model type: {model_type}")


def process_wall_collisions(positions: np.ndarray, velocities: np.ndarray,
                            wall_model: str, radius: float,
                            dt: float, channel: RectangularChannel,
                            x_boundary_type: str,
                            wall_temperature: float,
                            rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, np.ndarray, int, int, int, int]:
    N = positions.shape[0]
    new_positions = positions.copy()
    new_velocities = velocities.copy()
    total_wall_collisions = 0
    specular_collisions = 0
    diffuse_collisions = 0
    thermal_collisions = 0

    for i in range(N):
        pos = positions[i]
        vel = velocities[i]
        new_pos, new_vel = handle_wall_collision(
            pos, vel, radius, channel, wall_model,
            x_boundary_type, wall_temperature, rng=rng,
        )
        new_positions[i] = new_pos
        new_velocities[i] = new_vel

        if not np.array_equal(new_pos, pos) or not np.array_equal(new_vel, vel):
            total_wall_collisions += 1
            if wall_model == "specular":
                specular_collisions += 1
            elif wall_model == "diffuse_same_speed":
                diffuse_collisions += 1
            elif wall_model == "diffuse_thermal":
                thermal_collisions += 1

    return (
        new_positions,
        new_velocities,
        total_wall_collisions,
        specular_collisions,
        diffuse_collisions,
        thermal_collisions,
    )


def check_wall_collisions_simple(positions: np.ndarray, radius: float,
                                 channel: RectangularChannel) -> np.ndarray:
    N = positions.shape[0]
    colliding = np.zeros(N, dtype=bool)

    for i in range(N):
        x, y = positions[i]
        if (x < radius or x > channel.width - radius or
                y < radius or y > channel.height - radius):
            colliding[i] = True
    return colliding
