"""
Tests for particle-particle collision handling.
"""
import numpy as np

from src.dynamics.particle_collisions import (
    find_colliding_pairs,
    check_approaching,
    handle_elastic_collision,
    separate_overlapping_particles,
    process_all_collisions
)


def test_find_colliding_pairs():
    positions = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [3.0, 0.0],  # Not overlapping with anyone
        [0.0, 1.0]   # Overlapping with particle 0
    ])
    radius = 0.6  # Diameter = 1.2
    
    pairs = find_colliding_pairs(positions, radius)
    
    # Expected pairs: (0,1) distance=1.0 < 1.2, (0,3) distance=1.0 < 1.2
    # Note: (1,3) distance=sqrt2 ~= 1.414 > 1.2
    expected_pairs = [(0, 1), (0, 3)]
    
    assert len(pairs) == len(expected_pairs), \
        f"Expected {len(expected_pairs)} pairs, got {len(pairs)}"
    
    for pair in expected_pairs:
        assert pair in pairs, f"Expected pair {pair} not found in {pairs}"


def test_check_approaching():
    positions = np.array([
        [0.0, 0.0],
        [2.0, 0.0]
    ])
    
    # 1: Particles moving toward each other
    velocities1 = np.array([
        [1.0, 0.0],  # Moving right
        [-1.0, 0.0]  # Moving left
    ])
    assert check_approaching(positions, velocities1, 0, 1) == True
    
    # 2: Particles moving away from each other
    velocities2 = np.array([
        [-1.0, 0.0],  # Moving left
        [1.0, 0.0]    # Moving right
    ])
    assert check_approaching(positions, velocities2, 0, 1) == False
    
    # 3: Particles moving perpendicular (not approaching along line)
    velocities3 = np.array([
        [0.0, 1.0],  # Moving up
        [0.0, -1.0]  # Moving down
    ])
    assert check_approaching(positions, velocities3, 0, 1) == False


def test_handle_elastic_collision_head_on():
    """Test head-on elastic collision of identical particles."""
    positions = np.array([
        [0.0, 0.0],
        [2.0, 0.0]
    ])
    velocities = np.array([
        [3.0, 0.0],  # Moving right
        [-1.0, 0.0]  # Moving left
    ])
    
    new_positions, new_velocities = handle_elastic_collision(
        positions, velocities, 0, 1, mass=1.0
    )
    
    # For identical masses in 1d head-on collision, velocities swap
    # Particle 0: was 3.0, should become -1.0
    # Particle 1: was -1.0, should become 3.0
    expected_velocities = np.array([
        [-1.0, 0.0],
        [3.0, 0.0]
    ])
    
    assert np.allclose(new_velocities, expected_velocities), \
        f"Expected {expected_velocities}, got {new_velocities}"
    
    assert np.allclose(new_positions, positions)


def test_handle_elastic_collision_glancing():
    """Test glancing elastic collision."""
    positions = np.array([
        [0.0, 0.0],
        [1.0, 1.0]
    ])
    velocities = np.array([
        [1.0, 0.0],  # Moving right
        [0.0, -1.0]  # Moving down
    ])
    
    new_positions, new_velocities = handle_elastic_collision(
        positions, velocities, 0, 1, mass=1.0
    )
    
    # Check energy conservation
    kinetic_before = 0.5 * np.sum(velocities ** 2)
    kinetic_after = 0.5 * np.sum(new_velocities ** 2)
    
    assert np.isclose(kinetic_before, kinetic_after), \
        f"Kinetic energy not conserved: {kinetic_before} != {kinetic_after}"
    
    # Check momentum conservation
    momentum_before = np.sum(velocities, axis=0)
    momentum_after = np.sum(new_velocities, axis=0)
    
    assert np.allclose(momentum_before, momentum_after), \
        f"Momentum not conserved: {momentum_before} != {momentum_after}"


def test_separate_overlapping_particles():
    positions = np.array([
        [0.0, 0.0],
        [0.5, 0.0]  # Overlapping with particle 0 (distance=0.5)
    ])
    radius = 0.5  # Diameter=1.0, so particles should be at least 1.0 apart
    
    new_positions = separate_overlapping_particles(positions, 0, 1, radius)
    
    # Distance after separation should be >= 2*radius
    dist = np.linalg.norm(new_positions[0] - new_positions[1])
    assert dist >= 2 * radius - 1e-10, \
        f"Particles still overlapping: distance={dist}, expected >= {2*radius}"
    
    # Center of mass should be preserved (for equal masses)
    com_before = np.mean(positions, axis=0)
    com_after = np.mean(new_positions, axis=0)
    assert np.allclose(com_before, com_after), \
        f"Center of mass changed: {com_before} != {com_after}"


def test_process_all_collisions():
    """Test processing of multiple collisions."""
    # Create 3 particles in a line
    positions = np.array([
        [0.0, 0.0],
        [1.5, 0.0],  # Overlapping with both neighbors
        [3.0, 0.0]
    ])
    velocities = np.array([
        [1.0, 0.0],   # Moving right
        [0.0, 0.0],   # Stationary
        [-1.0, 0.0]   # Moving left
    ])
    radius = 1.0  # All particles overlapping
    
    new_positions, new_velocities = process_all_collisions(
        positions, velocities, radius, mass=1.0
    )
    
    # Check that no particles are overlapping after separation
    for i in range(3):
        for j in range(i + 1, 3):
            dist = np.linalg.norm(new_positions[i] - new_positions[j])
            assert dist >= 2 * radius - 1e-10, \
                f"Particles {i} and {j} still overlapping: distance={dist}"
    
    # Check energy conservation
    kinetic_before = 0.5 * np.sum(velocities ** 2)
    kinetic_after = 0.5 * np.sum(new_velocities ** 2)
    
    # Energy should be conserved (elastic collisions)
    assert np.isclose(kinetic_before, kinetic_after, rtol=1e-10), \
        f"Kinetic energy not conserved: {kinetic_before} != {kinetic_after}"
    
    # Check momentum conservation
    momentum_before = np.sum(velocities, axis=0)
    momentum_after = np.sum(new_velocities, axis=0)
    
    assert np.allclose(momentum_before, momentum_after), \
        f"Momentum not conserved: {momentum_before} != {momentum_after}"


def test_collision_with_zero_distance():
    """Test collision handling when particles are exactly on top of each other."""
    positions = np.array([
        [0.0, 0.0],
        [0.0, 0.0]  # Exactly same position
    ])
    velocities = np.array([
        [1.0, 0.0],
        [-1.0, 0.0]
    ])
    radius = 0.5
    
    # This should not crash
    new_positions, new_velocities = process_all_collisions(
        positions, velocities, radius, mass=1.0
    )
    
    # Particles should be separated
    dist = np.linalg.norm(new_positions[0] - new_positions[1])
    assert dist > 0, "Particles should be separated"


if __name__ == "__main__":
    # Run tests
    test_find_colliding_pairs()
    test_check_approaching()
    test_handle_elastic_collision_head_on()
    test_handle_elastic_collision_glancing()
    test_separate_overlapping_particles()
    test_process_all_collisions()
    test_collision_with_zero_distance()
    print("All particle collision tests passed!")