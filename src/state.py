from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class SystemState:

    positions: np.ndarray  # shape (N, 2)
    velocities: np.ndarray  # shape (N, 2)
    time: float = 0.0

    def __post_init__(self):
        if self.positions.shape != self.velocities.shape:
            raise ValueError(
                f"Positions shape {self.positions.shape} does not match "
                f"velocities shape {self.velocities.shape}"
            )
        if len(self.positions.shape) != 2 or self.positions.shape[1] != 2:
            raise ValueError(
                f"Positions must have shape (N, 2), got {self.positions.shape}"
            )

    @property
    def num_particles(self) -> int:
        return self.positions.shape[0]

    def copy(self) -> 'SystemState':
        return SystemState(
            positions=self.positions.copy(),
            velocities=self.velocities.copy(),
            time=self.time
        )

    def get_particle_state(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.positions[idx], self.velocities[idx]

    def set_particle_state(self, idx: int, position: np.ndarray, velocity: np.ndarray):
        self.positions[idx] = position
        self.velocities[idx] = velocity
