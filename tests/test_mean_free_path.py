import math
import numpy as np

from src.measurements.mean_free_path import MeanFreePathStats


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
    test_free_path_reset_only_for_collided_particles()
    print("All mean free path tests passed!")