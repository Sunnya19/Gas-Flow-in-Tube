from typing import Tuple
import numpy as np


def calculate_kinetic_energy(velocities: np.ndarray, mass: float = 1.0) -> float:
    # v^2 for each particle
    speed_sq = np.sum(velocities ** 2, axis=1)
    # Sum over particles: 0.5 * m * v^2
    return 0.5 * mass * np.sum(speed_sq)


def calculate_temperature(velocities: np.ndarray, mass: float = 1.0) -> float:
    N = velocities.shape[0]
    if N == 0:
        return 0.0

    kinetic_energy = calculate_kinetic_energy(velocities, mass)
    # For 2D: <K> = N * k_B * T, with k_B = 1
    return kinetic_energy / N


def calculate_total_energy(velocities: np.ndarray, mass: float = 1.0) -> float:
    return calculate_kinetic_energy(velocities, mass)


def calculate_energy_conservation(energy_history: np.ndarray) -> Tuple[float, float]:
    if len(energy_history) == 0:
        return 0.0, 0.0

    initial_energy = energy_history[0]
    if initial_energy == 0:
        return 0.0, 0.0

    # Relative error: (E(t) - E(0)) / E(0)
    relative_errors = (energy_history - initial_energy) / initial_energy

    max_error = np.max(np.abs(relative_errors))
    std_error = np.std(relative_errors)

    return max_error, std_error


def calculate_energy_drift(energy_history: np.ndarray) -> float:
    if len(energy_history) < 2:
        return 0.0

    # Simple linear regression
    times = np.arange(len(energy_history))
    slope = np.polyfit(times, energy_history, 1)[0]

    return slope
