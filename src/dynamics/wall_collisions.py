from typing import Tuple
import numpy as np

from src.wall_models.base import WallModel
from src.wall_models.specular import SpecularWall
from src.geometry.channel import RectangularChannel


def create_wall_model(model_type: str, channel: RectangularChannel) -> WallModel:
    if model_type == "specular":
        return SpecularWall(channel)
    else:
        raise ValueError(f"Unknown wall model type: {model_type}")


def process_wall_collisions(positions: np.ndarray, velocities: np.ndarray,
                            wall_model: WallModel, radius: float,
                            dt: float) -> Tuple[np.ndarray, np.ndarray]:
    N = positions.shape[0]
    new_positions = positions.copy()
    new_velocities = velocities.copy()

    for i in range(N):
        pos = positions[i]
        vel = velocities[i]
        new_pos, new_vel = wall_model.handle_collision(pos, vel, radius, dt)
        new_positions[i] = new_pos
        new_velocities[i] = new_vel

    return new_positions, new_velocities


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
