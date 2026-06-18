"""
Tests for reproducibility with random_seed.
"""
import numpy as np

from src.config import SimulationConfig
from src.initialization import initialize_system


def test_random_seed_reproducible_positions():
    """Two initializations with the same seed should give identical positions."""
    config1 = SimulationConfig(
        width=20.0,
        height=15.0,
        num_particles=50,
        particle_radius=0.1,
        particle_mass=1.0,
        initial_temperature=1.0,
        random_seed=42,
    )
    config2 = SimulationConfig(
        width=20.0,
        height=15.0,
        num_particles=50,
        particle_radius=0.1,
        particle_mass=1.0,
        initial_temperature=1.0,
        random_seed=42,
    )

    state1, _ = initialize_system(config1)
    state2, _ = initialize_system(config2)

    assert np.allclose(state1.positions, state2.positions), \
        "Positions differ with same random_seed"
    assert np.allclose(state1.velocities, state2.velocities), \
        "Velocities differ with same random_seed"


def test_random_seed_different_seeds_different_positions():
    """Different seeds should (almost certainly) give different positions."""
    config1 = SimulationConfig(
        width=20.0,
        height=15.0,
        num_particles=50,
        particle_radius=0.1,
        particle_mass=1.0,
        initial_temperature=1.0,
        random_seed=42,
    )
    config2 = SimulationConfig(
        width=20.0,
        height=15.0,
        num_particles=50,
        particle_radius=0.1,
        particle_mass=1.0,
        initial_temperature=1.0,
        random_seed=99,
    )

    state1, _ = initialize_system(config1)
    state2, _ = initialize_system(config2)

    # Extremely unlikely to be equal with different seeds
    assert not np.allclose(state1.positions, state2.positions), \
        "Positions should differ with different random_seed"


def test_random_seed_none_is_not_reproducible():
    """When random_seed is None, two runs should (almost certainly) differ."""
    config1 = SimulationConfig(
        width=20.0,
        height=15.0,
        num_particles=50,
        particle_radius=0.1,
        particle_mass=1.0,
        initial_temperature=1.0,
        random_seed=None,
    )
    config2 = SimulationConfig(
        width=20.0,
        height=15.0,
        num_particles=50,
        particle_radius=0.1,
        particle_mass=1.0,
        initial_temperature=1.0,
        random_seed=None,
    )

    state1, _ = initialize_system(config1)
    state2, _ = initialize_system(config2)

    # With no seed, they should differ (extremely unlikely to match)
    assert not np.allclose(state1.positions, state2.positions), \
        "Positions should differ when random_seed is None"


def test_random_seed_short_simulation_reproducible():
    """A short simulation with the same seed should give reproducible results."""
    config1 = SimulationConfig(
        width=20.0,
        height=15.0,
        num_particles=20,
        particle_radius=0.2,
        particle_mass=1.0,
        initial_temperature=1.0,
        time_step=0.01,
        total_time=0.2,
        save_interval=5,
        num_trajectory_particles=3,
        wall_model_type="specular",
        save_vtk=False,
        random_seed=42,
    )
    config2 = SimulationConfig(
        width=20.0,
        height=15.0,
        num_particles=20,
        particle_radius=0.2,
        particle_mass=1.0,
        initial_temperature=1.0,
        time_step=0.01,
        total_time=0.2,
        save_interval=5,
        num_trajectory_particles=3,
        wall_model_type="specular",
        save_vtk=False,
        random_seed=42,
    )

    from src.simulation import Simulation
    sim1 = Simulation(config1)
    history1 = sim1.run()

    sim2 = Simulation(config2)
    history2 = sim2.run()

    # Energy history should be identical
    assert np.allclose(history1["total_energy"], history2["total_energy"]), \
        "Energy history differs with same random_seed"

    # Mean free path should be identical
    mfp1 = history1["mean_free_path"]
    mfp2 = history2["mean_free_path"]
    if np.isfinite(mfp1) and np.isfinite(mfp2):
        assert np.isclose(mfp1, mfp2), \
            f"Mean free path differs: {mfp1} vs {mfp2}"


if __name__ == "__main__":
    test_random_seed_reproducible_positions()
    test_random_seed_different_seeds_different_positions()
    test_random_seed_none_is_not_reproducible()
    test_random_seed_short_simulation_reproducible()
    print("All reproducibility tests passed!")