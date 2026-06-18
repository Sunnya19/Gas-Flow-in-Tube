"""
Tests for external force application.

Verifies that:
- vx increases by force_x / mass * dt.
- vy is not modified.
"""
import numpy as np

from src.dynamics.forces import apply_external_force


def test_force_increases_vx():
    """vx should increase by force_x / mass * dt."""
    velocities = np.zeros((10, 2), dtype=float)
    force_x = 0.1
    mass = 1.0
    dt = 0.01

    updated = apply_external_force(velocities, force_x, mass, dt)

    expected_dv = force_x / mass * dt
    assert np.allclose(updated[:, 0], expected_dv), \
        f"Expected vx={expected_dv}, got {updated[:, 0]}"
    assert np.allclose(updated[:, 1], 0.0), "vy should remain zero"


def test_force_does_not_change_vy():
    """External force along x must not affect vy."""
    rng = np.random.default_rng(42)
    velocities = rng.normal(0, 1.0, size=(10, 2))
    vy_before = velocities[:, 1].copy()

    updated = apply_external_force(velocities, 0.05, 1.0, 0.01)

    assert np.allclose(updated[:, 1], vy_before), \
        "vy should be unchanged by x-directed force"


def test_zero_force_no_change():
    """Zero external force should leave velocities unchanged."""
    rng = np.random.default_rng(42)
    velocities = rng.normal(0, 1.0, size=(10, 2))
    v_before = velocities.copy()

    updated = apply_external_force(velocities, 0.0, 1.0, 0.01)

    assert np.allclose(updated, v_before), \
        "Velocities should be unchanged with zero force"


def test_force_with_nonzero_initial_vx():
    """Force should add to existing vx correctly."""
    velocities = np.ones((5, 2)) * 2.0  # vx=2, vy=2
    force_x = 0.5
    mass = 2.0
    dt = 0.02

    updated = apply_external_force(velocities, force_x, mass, dt)

    expected_vx = 2.0 + force_x / mass * dt  # 2.0 + 0.5/2.0*0.02 = 2.005
    assert np.allclose(updated[:, 0], expected_vx), \
        f"Expected vx={expected_vx}, got {updated[:, 0]}"
    assert np.allclose(updated[:, 1], 2.0), "vy should remain 2.0"


if __name__ == "__main__":
    test_force_increases_vx()
    test_force_does_not_change_vy()
    test_zero_force_no_change()
    test_force_with_nonzero_initial_vx()
    print("All force tests passed!")