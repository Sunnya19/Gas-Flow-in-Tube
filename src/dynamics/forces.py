import numpy as np


def apply_external_force(
    velocities: np.ndarray,
    force_x: float,
    mass: float,
    dt: float,
) -> np.ndarray:
    """
    Apply a constant body force along the x direction.

    The force is applied to every particle velocity using Euler integration.

    Args:
        velocities: Particle velocities with shape (N, 2).
        force_x: External force value applied in the x direction.
        mass: Particle mass.
        dt: Time step.

    Returns:
        Updated velocity array with the same shape as the input.
    """
    updated_velocities = velocities.copy()
    updated_velocities[:, 0] += force_x / mass * dt
    return updated_velocities
