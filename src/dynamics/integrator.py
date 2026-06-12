import numpy as np
from ..state import SystemState


class EulerIntegrator:
    def __init__(self, dt: float):
        self.dt = dt

    def step(self, state: SystemState) -> SystemState:
        new_positions = state.positions + state.velocities * self.dt
        new_time = state.time + self.dt

        return SystemState(
            positions=new_positions,
            velocities=state.velocities.copy(),
            time=new_time
        )

    def step_positions_only(self, positions: np.ndarray, velocities: np.ndarray) -> np.ndarray:
        return positions + velocities * self.dt
