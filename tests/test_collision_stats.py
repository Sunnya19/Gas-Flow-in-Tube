import numpy as np

from src.measurements.collision_stats import CollisionStats


def test_collision_stats_initial_state():
    stats = CollisionStats()
    assert stats.particle_collision_count == 0
    assert stats.wall_collision_count == 0
    assert stats.particle_collisions_history == []
    assert stats.wall_collisions_history == []


def test_collision_stats_record_step():
    stats = CollisionStats()
    stats.record_step(particle_collisions=5, wall_collisions=3)
    assert stats.particle_collision_count == 5
    assert stats.wall_collision_count == 3
    assert stats.particle_collisions_history == [5]
    assert stats.wall_collisions_history == [3]


def test_collision_stats_multiple_steps():
    stats = CollisionStats()
    stats.record_step(particle_collisions=2, wall_collisions=1)
    stats.record_step(particle_collisions=3, wall_collisions=4)
    stats.record_step(particle_collisions=0, wall_collisions=0)
    assert stats.particle_collision_count == 5
    assert stats.wall_collision_count == 5
    assert stats.particle_collisions_history == [2, 3, 0]
    assert stats.wall_collisions_history == [1, 4, 0]


def test_collision_stats_history_in_run():
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

    assert "particle_collisions" in history, "History missing particle_collisions"
    assert "wall_collisions" in history, "History missing wall_collisions"
    assert "total_particle_collisions" in history, "History missing total_particle_collisions"
    assert "total_wall_collisions" in history, "History missing total_wall_collisions"

    assert len(history["particle_collisions"]) > 0
    assert len(history["wall_collisions"]) > 0


def test_collision_counts_per_interval():
    """
    Verify that collision history stores accumulated counts per save interval,
    not per-step values. The sum of all history entries should equal the total.
    """
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

    # Sum of per-interval values should equal the total
    total_from_history = sum(history["particle_collisions"])
    assert total_from_history == history["total_particle_collisions"], (
        f"Sum of per-interval collisions ({total_from_history}) "
        f"does not match total ({history['total_particle_collisions']})"
    )

    total_wall_from_history = sum(history["wall_collisions"])
    assert total_wall_from_history == history["total_wall_collisions"], (
        f"Sum of per-interval wall collisions ({total_wall_from_history}) "
        f"does not match total ({history['total_wall_collisions']})"
    )


if __name__ == "__main__":
    test_collision_stats_initial_state()
    test_collision_stats_record_step()
    test_collision_stats_multiple_steps()
    test_collision_stats_history_in_run()
    test_collision_counts_per_interval()
    print("All collision stats tests passed!")