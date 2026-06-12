from typing import Tuple
import numpy as np

from src.state import SystemState
from src.geometry.channel import RectangularChannel


def generate_random_positions(
        num_particles: int,
        radius: float,
        channel: RectangularChannel,
        max_attempts: int = 10000,
) -> np.ndarray:

    positions = np.zeros((num_particles, 2))
    diameter = 2 * radius
    diameter_sq = diameter ** 2

    for i in range(num_particles):
        placed = False
        attempts = 0

        while not placed and attempts < max_attempts:
            # Generate random position within channel boundaries
            x = np.random.uniform(radius, channel.width - radius)
            y = np.random.uniform(radius, channel.height - radius)
            candidate = np.array([x, y])

            # Check overlap with already placed particles
            overlap = False
            for j in range(i):
                dx = candidate[0] - positions[j, 0]
                dy = candidate[1] - positions[j, 1]
                dist_sq = dx * dx + dy * dy

                if dist_sq < diameter_sq:
                    overlap = True
                    break

            if not overlap:
                positions[i] = candidate
                placed = True

            attempts += 1

        if not placed:
            raise RuntimeError(
                f"Failed to place particle {i+1} after {max_attempts} attempts. "
                f"Try increasing channel size or decreasing particle radius."
            )

    return positions


def generate_maxwell_velocities(num_particles: int, temperature: float) -> np.ndarray:
    # Generate velocities from normal distribution
    # For 2D, variance is sqrt(temperature) with m = 1, k_B = 1
    std = np.sqrt(temperature)
    velocities = np.random.normal(0, std, size=(num_particles, 2))

    # Subtract center-of-mass velocity
    com_velocity = np.mean(velocities, axis=0)
    velocities -= com_velocity

    return velocities


def scale_velocities_to_temperature(velocities: np.ndarray, target_temperature: float) -> np.ndarray:
    # Current kinetic temperature (for 2D)
    # T_kin = (1/N) * sum(0.5 * m * v^2) / (k_B * 1) with m=1, k_B=1
    # 2D: <K> = N * k_B * T, so T = (1/(N*k_B)) * sum(0.5*m*v^2)
    # With m=1, k_B=1: T = (1/N) * sum(0.5*v^2)
    current_kinetic_energy = 0.5 * np.sum(velocities ** 2)
    current_temperature = current_kinetic_energy / len(velocities)

    if current_temperature > 0:
        scale_factor = np.sqrt(target_temperature / current_temperature)
        return velocities * scale_factor
    else:
        return generate_maxwell_velocities(len(velocities), target_temperature)


def initialize_system(config) -> Tuple[SystemState, RectangularChannel]:
    # channel geometry
    channel = RectangularChannel(width=config.width, height=config.height)

    positions = generate_random_positions(
        num_particles=config.num_particles,
        radius=config.particle_radius,
        channel=channel
    )

    # M-B distr
    velocities = generate_maxwell_velocities(
        num_particles=config.num_particles,
        temperature=config.initial_temperature
    )

    # Scale to exact temperature
    velocities = scale_velocities_to_temperature(velocities, config.initial_temperature)

    # Create state
    state = SystemState(positions=positions, velocities=velocities, time=0.0)

    return state, channel
