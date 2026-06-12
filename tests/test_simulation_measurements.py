import numpy as np

from src.config import SimulationConfig
from src.simulation import Simulation


def test_simulation_measurements_run_without_errors():
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

    required_keys = [
        "particle_collisions",
        "wall_collisions",
        "mean_free_path",
        "mean_free_path_history",
        "free_path_samples",
    ]
    for key in required_keys:
        assert key in history, f"History missing required key: {key}"


def test_simulation_measurements_positive_mfp_if_collisions():
    config = SimulationConfig(
        width=10.0,
        height=10.0,
        num_particles=50,
        particle_radius=0.15,
        particle_mass=1.0,
        initial_temperature=2.0,
        time_step=0.005,
        total_time=1.0,
        save_interval=10,
        num_trajectory_particles=3,
        wall_model_type="specular",
        save_vtk=False,
    )

    sim = Simulation(config)
    history = sim.run()

    total_particle = history["total_particle_collisions"]
    mfp = history["mean_free_path"]

    if total_particle > 0:
        assert mfp > 0, f"Expected positive mean_free_path, got {mfp}"
    else:
        import math
        assert math.isnan(mfp) or mfp > 0, \
            f"Expected nan or positive mean_free_path, got {mfp}"


def test_simulation_measurements_history_lengths():
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

    n_time = len(history["time"])
    assert len(history["particle_collisions"]) == n_time
    assert len(history["wall_collisions"]) == n_time
    assert len(history["mean_free_path_history"]) == n_time


if __name__ == "__main__":
    test_simulation_measurements_run_without_errors()
    test_simulation_measurements_positive_mfp_if_collisions()
    test_simulation_measurements_history_lengths()
    print("All simulation measurements tests passed!")