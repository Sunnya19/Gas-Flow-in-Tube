"""
Tests for reproducible initialization with random_seed.

Verifies that:
- Same seed -> identical positions and velocities.
- Different seeds -> positions or velocities differ.
"""
import numpy as np

from src.config import SimulationConfig
from src.initialization import initialize_system


def test_same_seed_identical():
    """Two initializations with the same seed must give identical state."""
    config1 = SimulationConfig(
        width=20.0, height=15.0, num_particles=50,
        particle_radius=0.1, particle_mass=1.0,
        initial_temperature=1.0, random_seed=42,
    )
    config2 = SimulationConfig(
        width=20.0, height=15.0, num_particles=50,
        particle_radius=0.1, particle_mass=1.0,
        initial_temperature=1.0, random_seed=42,
    )

    state1, _ = initialize_system(config1)
    state2, _ = initialize_system(config2)

    assert np.allclose(state1.positions, state2.positions), \
        "Positions differ with same random_seed"
    assert np.allclose(state1.velocities, state2.velocities), \
        "Velocities differ with same random_seed"


def test_different_seeds_differ():
    """Different seeds should (almost certainly) give different positions."""
    config1 = SimulationConfig(
        width=20.0, height=15.0, num_particles=50,
        particle_radius=0.1, particle_mass=1.0,
        initial_temperature=1.0, random_seed=42,
    )
    config2 = SimulationConfig(
        width=20.0, height=15.0, num_particles=50,
        particle_radius=0.1, particle_mass=1.0,
        initial_temperature=1.0, random_seed=99,
    )

    state1, _ = initialize_system(config1)
    state2, _ = initialize_system(config2)

    # Extremely unlikely to be equal with different seeds
    assert not np.allclose(state1.positions, state2.positions), \
        "Positions should differ with different random_seed"


if __name__ == "__main__":
    test_same_seed_identical()
    test_different_seeds_differ()
    print("All random seed tests passed!")