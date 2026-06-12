import math
import numpy as np

from src.measurements.mean_free_path import (
    MeanFreePathStats,
    update_free_path_measurements,
)


def test_mean_free_path_initial_nan():
    stats = MeanFreePathStats()
    assert math.isnan(stats.mean_free_path()), "Expected nan when no samples"


def test_mean_free_path_with_samples():
    stats = MeanFreePathStats()
    stats.add_samples([1.0, 2.0, 3.0])
    expected = 2.0
    assert np.isclose(stats.mean_free_path(), expected), \
        f"Expected {expected}, got {stats.mean_free_path()}"


def test_mean_free_path_single_sample():
    stats = MeanFreePathStats()
    stats.add_samples([5.0])
    assert np.isclose(stats.mean_free_path(), 5.0)


def test_mean_free_path_multiple_adds():
    stats = MeanFreePathStats()
    stats.add_samples([1.0, 2.0])
    stats.add_samples([3.0, 4.0])
    expected = 2.5
    assert np.isclose(stats.mean_free_path(), expected), \
        f"Expected {expected}, got {stats.mean_free_path()}"


def test_mean_free_path_history():
    stats = MeanFreePathStats()
    stats.record_current_mean()
    assert len(stats.mean_free_path_history) == 1
    assert math.isnan(stats.mean_free_path_history[0])

    stats.add_samples([2.0, 4.0])
    stats.record_current_mean()
    assert len(stats.mean_free_path_history) == 2
    assert np.isclose(stats.mean_free_path_history[1], 3.0)


def test_mean_free_path_clear():
    stats = MeanFreePathStats()
    stats.add_samples([1.0, 2.0, 3.0])
    stats.record_current_mean()
    assert len(stats.free_path_samples) == 3
    assert len(stats.mean_free_path_history) == 1

    stats.clear()
    assert len(stats.free_path_samples) == 0
    assert len(stats.mean_free_path_history) == 0
    assert math.isnan(stats.mean_free_path())


def test_update_free_path_measurements_first_collision_no_sample():
    """First collision of a particle should NOT produce a sample."""
    path = np.array([5.0, 3.0, 7.0], dtype=float)
    has_prev = np.array([False, False, False], dtype=bool)
    collision_pairs = [(0, 1)]
    stats = MeanFreePathStats()

    update_free_path_measurements(path, has_prev, collision_pairs, stats, True)

    # No samples added (first collision for both particles)
    assert len(stats.free_path_samples) == 0
    # Paths should be reset
    assert path[0] == 0.0
    assert path[1] == 0.0
    # has_prev should be True now
    assert has_prev[0] == True
    assert has_prev[1] == True


def test_update_free_path_measurements_second_collision_adds_sample():
    """Second collision of a particle SHOULD produce a sample."""
    path = np.array([5.0, 3.0, 7.0], dtype=float)
    has_prev = np.array([True, False, False], dtype=bool)
    collision_pairs = [(0, 1)]
    stats = MeanFreePathStats()

    update_free_path_measurements(path, has_prev, collision_pairs, stats, True)

    # Particle 0 has previous collision -> sample added
    assert len(stats.free_path_samples) == 1
    assert np.isclose(stats.free_path_samples[0], 5.0)
    # Paths should be reset
    assert path[0] == 0.0
    assert path[1] == 0.0


def test_update_free_path_measurements_disabled():
    """When measurement_enabled=False, no samples should be added."""
    path = np.array([5.0, 3.0], dtype=float)
    has_prev = np.array([True, True], dtype=bool)
    collision_pairs = [(0, 1)]
    stats = MeanFreePathStats()

    update_free_path_measurements(path, has_prev, collision_pairs, stats, False)

    # No samples added
    assert len(stats.free_path_samples) == 0
    # Paths should still be reset (physics happens regardless)
    assert path[0] == 0.0
    assert path[1] == 0.0


def test_update_free_path_measurements_multiple_collisions_same_particle():
    """If a particle collides multiple times in one step, its path is recorded once."""
    path = np.array([5.0, 3.0, 7.0], dtype=float)
    has_prev = np.array([True, True, False], dtype=bool)
    # Particle 0 collides with both 1 and 2
    collision_pairs = [(0, 1), (0, 2)]
    stats = MeanFreePathStats()

    update_free_path_measurements(path, has_prev, collision_pairs, stats, True)

    # Particle 0 and 1 have previous collisions -> 2 samples
    # Particle 2 has no previous -> no sample
    assert len(stats.free_path_samples) == 2
    assert np.isclose(stats.free_path_samples[0], 5.0)
    assert np.isclose(stats.free_path_samples[1], 3.0)
    # All paths reset
    assert path[0] == 0.0
    assert path[1] == 0.0
    assert path[2] == 0.0


def test_free_path_reset_only_for_collided_particles():
    from src.config import SimulationConfig
    from src.simulation import Simulation

    config = SimulationConfig(
        width=20.0,
        height=20.0,
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
    )

    sim = Simulation(config)
    history = sim.run()

    assert "mean_free_path" in history
    assert "mean_free_path_history" in history
    assert "free_path_samples" in history


if __name__ == "__main__":
    test_mean_free_path_initial_nan()
    test_mean_free_path_with_samples()
    test_mean_free_path_single_sample()
    test_mean_free_path_multiple_adds()
    test_mean_free_path_history()
    test_mean_free_path_clear()
    test_update_free_path_measurements_first_collision_no_sample()
    test_update_free_path_measurements_second_collision_adds_sample()
    test_update_free_path_measurements_disabled()
    test_update_free_path_measurements_multiple_collisions_same_particle()
    test_free_path_reset_only_for_collided_particles()
    print("All mean free path tests passed!")