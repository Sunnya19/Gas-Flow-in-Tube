"""
Tests for flow-related functions: external force, periodic boundaries,
mean flow velocity, and velocity profile.
"""
import numpy as np

from src.dynamics.forces import apply_external_force
from src.measurements.flow import compute_mean_flow_velocity, compute_flow_temperature
from src.measurements.velocity_profile import compute_velocity_profile


def test_apply_external_force_increases_vx():
    """External force should increase vx by force_x / mass * dt."""
    velocities = np.zeros((10, 2), dtype=float)
    force_x = 0.1
    mass = 1.0
    dt = 0.01

    updated = apply_external_force(velocities, force_x, mass, dt)

    expected_dv = force_x / mass * dt
    assert np.allclose(updated[:, 0], expected_dv), \
        f"Expected vx={expected_dv}, got {updated[:, 0]}"
    assert np.allclose(updated[:, 1], 0.0), "vy should remain unchanged"


def test_apply_external_force_preserves_vy():
    """External force along x should not affect vy."""
    velocities = np.random.randn(10, 2)
    vy_before = velocities[:, 1].copy()

    updated = apply_external_force(velocities, 0.05, 1.0, 0.01)

    assert np.allclose(updated[:, 1], vy_before), \
        "vy should be unchanged by x-directed force"


def test_apply_external_force_zero_force():
    """Zero external force should leave velocities unchanged."""
    velocities = np.random.randn(10, 2)
    v_before = velocities.copy()

    updated = apply_external_force(velocities, 0.0, 1.0, 0.01)

    assert np.allclose(updated, v_before), \
        "Velocities should be unchanged with zero force"


def test_compute_mean_flow_velocity():
    """Mean flow velocity should be the average of all particle velocities."""
    velocities = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
    ])
    mean_v = compute_mean_flow_velocity(velocities)
    expected = np.array([3.0, 4.0])
    assert np.allclose(mean_v, expected), \
        f"Expected {expected}, got {mean_v}"


def test_compute_mean_flow_velocity_single_particle():
    """With one particle, mean velocity equals its velocity."""
    velocities = np.array([[2.5, -1.5]])
    mean_v = compute_mean_flow_velocity(velocities)
    assert np.allclose(mean_v, [2.5, -1.5]), \
        f"Expected [2.5, -1.5], got {mean_v}"


def test_compute_flow_temperature():
    """Temperature should be computed from thermal velocity fluctuations."""
    # All particles moving with same velocity -> zero temperature
    velocities = np.ones((10, 2)) * 5.0
    temp = compute_flow_temperature(velocities, mass=1.0)
    assert np.isclose(temp, 0.0), \
        f"Expected temperature 0.0 for uniform flow, got {temp}"

    # Maxwell-like distribution should give positive temperature
    rng = np.random.default_rng(42)
    velocities = rng.normal(0, 1.0, size=(1000, 2))
    temp = compute_flow_temperature(velocities, mass=1.0)
    assert temp > 0.0, f"Expected positive temperature, got {temp}"


def test_compute_velocity_profile_shape():
    """Velocity profile should return arrays of length n_bins."""
    positions = np.random.rand(100, 2) * 10.0
    velocities = np.random.randn(100, 2)
    height = 10.0
    n_bins = 20

    y_centers, ux_profile = compute_velocity_profile(
        positions, velocities, height, n_bins
    )

    assert len(y_centers) == n_bins, \
        f"Expected {n_bins} y_centers, got {len(y_centers)}"
    assert len(ux_profile) == n_bins, \
        f"Expected {n_bins} ux_profile, got {len(ux_profile)}"


def test_compute_velocity_profile_uniform_flow():
    """Uniform flow should give constant ux across all bins."""
    positions = np.random.rand(100, 2) * 10.0
    velocities = np.ones((100, 2)) * 3.0  # all moving at vx=3
    height = 10.0
    n_bins = 10

    y_centers, ux_profile = compute_velocity_profile(
        positions, velocities, height, n_bins
    )

    # All non-NaN bins should have ux ≈ 3.0
    valid = ~np.isnan(ux_profile)
    if np.any(valid):
        assert np.allclose(ux_profile[valid], 3.0), \
            f"Expected ux=3.0 in all bins, got {ux_profile}"


def test_periodic_boundary_in_simulation():
    """Test that periodic x boundary works in a short simulation."""
    from src.config import SimulationConfig
    from src.simulation import Simulation

    config = SimulationConfig(
        width=20.0,
        height=15.0,
        num_particles=20,
        particle_radius=0.2,
        particle_mass=1.0,
        initial_temperature=1.0,
        time_step=0.01,
        total_time=0.5,
        save_interval=5,
        num_trajectory_particles=3,
        wall_model_type="specular",
        save_vtk=False,
        x_boundary_type="periodic",
        external_force_x=0.0,
    )

    sim = Simulation(config)
    history = sim.run()

    # Check that positions are within [0, width] for x
    final_positions = sim.state.positions
    assert np.all(final_positions[:, 0] >= 0.0), "x positions below 0"
    assert np.all(final_positions[:, 0] <= config.width), \
        f"x positions above width {config.width}"

    # Check that history has expected keys
    assert "mean_vx" in history
    assert "mean_vy" in history
    assert "temperature" in history


# ---------------------------------------------------------------------------
# Velocity profile averaging tests
# ---------------------------------------------------------------------------


def test_velocity_profile_averaging_simple():
    """Test that averaging multiple identical profiles gives the same profile."""
    n_bins = 5
    height = 10.0

    # Create a simple profile: uniform vx=2.0 across all bins
    positions = np.random.rand(100, 2) * height
    velocities = np.ones((100, 2)) * 2.0

    y_centers, ux_profile = compute_velocity_profile(
        positions, velocities, height, n_bins
    )

    # Simulate time-averaging: accumulate the same profile 10 times
    profile_sum = np.zeros(n_bins)
    profile_count = np.zeros(n_bins, dtype=int)

    for _ in range(10):
        _, ux = compute_velocity_profile(positions, velocities, height, n_bins)
        for i in range(n_bins):
            if np.isfinite(ux[i]):
                profile_sum[i] += ux[i]
                profile_count[i] += 1

    average_ux = np.full(n_bins, np.nan)
    nonzero = profile_count > 0
    average_ux[nonzero] = profile_sum[nonzero] / profile_count[nonzero]

    # All non-NaN bins should have ux ≈ 2.0
    valid = ~np.isnan(average_ux)
    if np.any(valid):
        assert np.allclose(average_ux[valid], 2.0), \
            f"Expected averaged ux=2.0, got {average_ux}"


def test_velocity_profile_averaging_nan_bins():
    """Test that NaN bins in individual profiles do not corrupt the average."""
    n_bins = 5
    height = 10.0

    # Profile 1: all bins have data
    pos1 = np.random.rand(100, 2) * height
    vel1 = np.ones((100, 2)) * 2.0
    _, ux1 = compute_velocity_profile(pos1, vel1, height, n_bins)

    # Profile 2: all bins have data
    pos2 = np.random.rand(100, 2) * height
    vel2 = np.ones((100, 2)) * 4.0
    _, ux2 = compute_velocity_profile(pos2, vel2, height, n_bins)

    profile_sum = np.zeros(n_bins)
    profile_count = np.zeros(n_bins, dtype=int)

    for ux in [ux1, ux2]:
        for i in range(n_bins):
            if np.isfinite(ux[i]):
                profile_sum[i] += ux[i]
                profile_count[i] += 1

    average_ux = np.full(n_bins, np.nan)
    nonzero = profile_count > 0
    average_ux[nonzero] = profile_sum[nonzero] / profile_count[nonzero]

    # All bins should have finite values (both profiles had data in all bins)
    assert np.all(np.isfinite(average_ux)), \
        f"Expected all finite, got {average_ux}"

    # Each bin should have count=2
    assert np.all(profile_count == 2), \
        f"Expected all counts=2, got {profile_count}"


if __name__ == "__main__":
    test_apply_external_force_increases_vx()
    test_apply_external_force_preserves_vy()
    test_apply_external_force_zero_force()
    test_compute_mean_flow_velocity()
    test_compute_mean_flow_velocity_single_particle()
    test_compute_flow_temperature()
    test_compute_velocity_profile_shape()
    test_compute_velocity_profile_uniform_flow()
    test_periodic_boundary_in_simulation()
    test_velocity_profile_averaging_simple()
    test_velocity_profile_averaging_nan_bins()
    print("All flow tests passed!")