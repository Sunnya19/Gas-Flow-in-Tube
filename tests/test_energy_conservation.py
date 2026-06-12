"""
Tests for energy conservation in molecular dynamics.
"""
import numpy as np

from src.config import SimulationConfig
from src.simulation import Simulation
from src.measurements.energy import (
    calculate_total_energy,
    calculate_energy_conservation
)


def test_energy_calculation():
    velocities = np.array([
        [1.0, 0.0],
        [0.0, 2.0],
        [-1.0, 1.0]
    ])
    
    # Mass = 1.0
    energy = calculate_total_energy(velocities, mass=1.0)
    expected_energy = 3.5
    
    assert np.isclose(energy, expected_energy), \
        f"Expected energy {expected_energy}, got {energy}"


def test_energy_conservation_metrics():
    # Perfectly conserved energy
    energy_history1 = np.array([100.0, 100.0, 100.0, 100.0])
    max_error1, std_error1 = calculate_energy_conservation(energy_history1)
    
    assert np.isclose(max_error1, 0.0), f"Expected max error 0.0, got {max_error1}"
    assert np.isclose(std_error1, 0.0), f"Expected std error 0.0, got {std_error1}"
    
    # Energy drifting upward
    energy_history2 = np.array([100.0, 101.0, 102.0, 103.0])
    max_error2, std_error2 = calculate_energy_conservation(energy_history2)
    
    # Relative errors: [0.0, 0.01, 0.02, 0.03]
    expected_max_error = 0.03  # 3%
    expected_std_error = np.std([0.0, 0.01, 0.02, 0.03])
    
    assert np.isclose(max_error2, expected_max_error, rtol=1e-10), \
        f"Expected max error {expected_max_error}, got {max_error2}"
    assert np.isclose(std_error2, expected_std_error, rtol=1e-10), \
        f"Expected std error {expected_std_error}, got {std_error2}"


def test_short_simulation_energy_conservation():
    # fast testing
    config = SimulationConfig(
        width=20.0,
        height=20.0,
        num_particles=20,
        particle_radius=0.2,
        particle_mass=1.0,
        initial_temperature=1.0,
        time_step=0.01,
        total_time=1.0,
        save_interval=5,
        num_trajectory_particles=3,
        wall_model_type="specular"
    )
    
    sim = Simulation(config)
    history = sim.run()
    
    energy_history = np.array(history['total_energy'])
    
    # Check that we have any data
    assert len(energy_history) > 1, "Energy history should have more than one point"
    
    # Calculate energy conservation
    max_error, std_error = calculate_energy_conservation(energy_history)
    
    max_allowed_error = 0.05
    
    print(f"Energy conservation test:")
    print(f"  Initial energy: {energy_history[0]:.6f}")
    print(f"  Final energy: {energy_history[-1]:.6f}")
    print(f"  Max relative error: {max_error:.6f} ({max_error*100:.2f}%)")
    print(f"  Std relative error: {std_error:.6f} ({std_error*100:.2f}%)")
    
    assert max_error < max_allowed_error, \
        f"Energy conservation violated: max error {max_error*100:.2f}% > {max_allowed_error*100}%"
    
    energy_change = abs(energy_history[-1] - energy_history[0]) / energy_history[0]
    assert energy_change < max_allowed_error, \
        f"Energy change too large: {energy_change*100:.2f}% > {max_allowed_error*100}%"


def test_energy_after_wall_collision():
    from src.geometry.channel import RectangularChannel
    from src.wall_models.specular import SpecularWall
    
    channel = RectangularChannel(width=10.0, height=10.0)
    wall_model = SpecularWall(channel)
    
    # Particle hitting left wall
    position = np.array([0.5, 5.0])
    velocity = np.array([-2.0, 1.0])
    radius = 1.0
    
    # Energy before collision
    energy_before = 0.5 * np.sum(velocity ** 2)  # m=1.0
    
    # Apply collision
    new_pos, new_vel = wall_model.handle_collision(position, velocity, radius, dt=0.01)
    
    # Energy after collision
    energy_after = 0.5 * np.sum(new_vel ** 2)
    
    # Energy should be exactly conserved for specular reflection
    assert np.isclose(energy_before, energy_after), \
        f"Energy not conserved in wall collision: {energy_before} != {energy_after}"
    
    # Speed should be the same (elastic collision)
    speed_before = np.linalg.norm(velocity)
    speed_after = np.linalg.norm(new_vel)
    assert np.isclose(speed_before, speed_after), \
        f"Speed changed in elastic collision: {speed_before} != {speed_after}"


def test_energy_after_particle_collision():
    from src.dynamics.particle_collisions import handle_elastic_collision
    
    # Two particles colliding
    positions = np.array([
        [0.0, 0.0],
        [1.0, 0.0]
    ])
    velocities = np.array([
        [2.0, 0.0],
        [-1.0, 0.0]
    ])
    
    # Energy before collision
    energy_before = 0.5 * np.sum(velocities ** 2)
    
    # Apply collision
    new_positions, new_velocities = handle_elastic_collision(
        positions, velocities, 0, 1, mass=1.0
    )
    
    # Energy after collision
    energy_after = 0.5 * np.sum(new_velocities ** 2)
    
    # Energy should be exactly conserved for elastic collision
    assert np.isclose(energy_before, energy_after), \
        f"Energy not conserved in particle collision: {energy_before} != {energy_after}"
    
    # Total momentum should be conserved
    momentum_before = np.sum(velocities, axis=0)
    momentum_after = np.sum(new_velocities, axis=0)
    assert np.allclose(momentum_before, momentum_after), \
        f"Momentum not conserved: {momentum_before} != {momentum_after}"


if __name__ == "__main__":
    # Run tests
    test_energy_calculation()
    test_energy_conservation_metrics()
    test_energy_after_wall_collision()
    test_energy_after_particle_collision()
    
    # Run the integration test
    print("Running short simulation energy conservation test...")
    test_short_simulation_energy_conservation()
    
    print("All energy conservation tests passed!")