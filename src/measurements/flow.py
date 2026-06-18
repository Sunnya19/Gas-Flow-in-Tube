import numpy as np


def compute_mean_flow_velocity(velocities: np.ndarray) -> np.ndarray:
    """
    Compute the mean flow velocity vector for the particle ensemble.

    Returns:
        A numpy array [mean_vx, mean_vy].
    """
    return np.mean(velocities, axis=0)


def compute_flow_temperature(velocities: np.ndarray, mass: float) -> float:
    """
    Compute the temperature of the moving gas relative to its mean flow.

    The temperature is defined from the thermal velocity fluctuations about the
    mean flow velocity.

    Args:
        velocities: Particle velocities shape (N, 2).
        mass: Particle mass.

    Returns:
        Scalar temperature from the thermal velocity variance.
    """
    mean_velocity = compute_mean_flow_velocity(velocities)
    thermal_velocities = velocities - mean_velocity
    cx = thermal_velocities[:, 0]
    cy = thermal_velocities[:, 1]
    return 0.5 * mass * np.mean(cx**2 + cy**2)
