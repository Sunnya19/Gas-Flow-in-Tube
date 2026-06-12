"""
Tests for wall collision handling.
"""
import numpy as np

from src.geometry.channel import RectangularChannel
from src.wall_models.specular import SpecularWall


def test_specular_wall_left():
    channel = RectangularChannel(width=10.0, height=10.0)
    wall_model = SpecularWall(channel)
    
    # Particle moving left into left wall
    position = np.array([0.5, 5.0])  # x=0.5, radius=1.0 means collision
    velocity = np.array([-2.0, 1.0])  # Moving left
    radius = 1.0
    
    new_pos, new_vel = wall_model.handle_collision(position, velocity, radius, dt=0.01)
    
    # Velocity should be reflected in x-direction
    assert np.allclose(new_vel, [2.0, 1.0]), f"Expected [2.0, 1.0], got {new_vel}"
    # Position should be corrected to be at radius
    assert np.isclose(new_pos[0], radius), f"Expected x={radius}, got {new_pos[0]}"
    assert np.isclose(new_pos[1], 5.0), f"Expected y=5.0, got {new_pos[1]}"


def test_specular_wall_right():
    channel = RectangularChannel(width=10.0, height=10.0)
    wall_model = SpecularWall(channel)
    
    # Particle moving right into right wall
    position = np.array([9.5, 5.0])  # x=9.5, radius=1.0 means collision
    velocity = np.array([2.0, 1.0])  # Moving right
    radius = 1.0
    
    new_pos, new_vel = wall_model.handle_collision(position, velocity, radius, dt=0.01)
    
    # Velocity should be reflected in x-direction
    assert np.allclose(new_vel, [-2.0, 1.0]), f"Expected [-2.0, 1.0], got {new_vel}"
    # Position should be corrected to be at width - radius
    assert np.isclose(new_pos[0], channel.width - radius), \
        f"Expected x={channel.width - radius}, got {new_pos[0]}"


def test_specular_wall_bottom():
    channel = RectangularChannel(width=10.0, height=10.0)
    wall_model = SpecularWall(channel)
    
    # Particle moving down into bottom wall
    position = np.array([5.0, 0.5])  # y=0.5, radius=1.0 means collision
    velocity = np.array([1.0, -2.0])  # Moving down
    radius = 1.0
    
    new_pos, new_vel = wall_model.handle_collision(position, velocity, radius, dt=0.01)
    
    # Velocity should be reflected in y-direction
    assert np.allclose(new_vel, [1.0, 2.0]), f"Expected [1.0, 2.0], got {new_vel}"
    # Position should be corrected to be at radius
    assert np.isclose(new_pos[1], radius), f"Expected y={radius}, got {new_pos[1]}"


def test_specular_wall_top():
    channel = RectangularChannel(width=10.0, height=10.0)
    wall_model = SpecularWall(channel)
    
    # Particle moving up into top wall
    position = np.array([5.0, 9.5])  # y=9.5, radius=1.0 means collision
    velocity = np.array([1.0, 2.0])  # Moving up
    radius = 1.0
    
    new_pos, new_vel = wall_model.handle_collision(position, velocity, radius, dt=0.01)
    
    # Velocity should be reflected in y-direction
    assert np.allclose(new_vel, [1.0, -2.0]), f"Expected [1.0, -2.0], got {new_vel}"
    # Position should be corrected to be at height - radius
    assert np.isclose(new_pos[1], channel.height - radius), \
        f"Expected y={channel.height - radius}, got {new_pos[1]}"


def test_specular_wall_corner():
    channel = RectangularChannel(width=10.0, height=10.0)
    wall_model = SpecularWall(channel)
    
    # Particle in top-right corner moving up and right
    position = np.array([9.5, 9.5])  # Colliding with both right and top walls
    velocity = np.array([2.0, 2.0])  # Moving up and right
    radius = 1.0
    
    new_pos, new_vel = wall_model.handle_collision(position, velocity, radius, dt=0.01)
    
    # Both velocity components should be reflected
    assert np.allclose(new_vel, [-2.0, -2.0]), f"Expected [-2.0, -2.0], got {new_vel}"
    # Position should be corrected
    assert np.isclose(new_pos[0], channel.width - radius), \
        f"Expected x={channel.width - radius}, got {new_pos[0]}"
    assert np.isclose(new_pos[1], channel.height - radius), \
        f"Expected y={channel.height - radius}, got {new_pos[1]}"


def test_no_collision():
    channel = RectangularChannel(width=10.0, height=10.0)
    wall_model = SpecularWall(channel)
    
    # Particle well inside the channel
    position = np.array([5.0, 5.0])
    velocity = np.array([1.0, 2.0])
    radius = 0.5
    
    new_pos, new_vel = wall_model.handle_collision(position, velocity, radius, dt=0.01)
    
    # Position and velocity should be unchanged
    assert np.allclose(new_pos, position), f"Expected {position}, got {new_pos}"
    assert np.allclose(new_vel, velocity), f"Expected {velocity}, got {new_vel}"


def test_simple_reflection():
    channel = RectangularChannel(width=10.0, height=10.0)
    wall_model = SpecularWall(channel)
    
    # Test left wall collision
    position = np.array([0.5, 5.0])
    velocity = np.array([-2.0, 1.0])
    radius = 1.0
    
    new_pos, new_vel = wall_model.handle_collision_simple(position, velocity, radius)
    
    # Position should be unchanged
    assert np.allclose(new_pos, position), f"Expected {position}, got {new_pos}"
    # Velocity x-component should be flipped
    assert np.isclose(new_vel[0], 2.0), f"Expected vx=2.0, got {new_vel[0]}"
    assert np.isclose(new_vel[1], 1.0), f"Expected vy=1.0, got {new_vel[1]}"


if __name__ == "__main__":
    # Run tests
    test_specular_wall_left()
    test_specular_wall_right()
    test_specular_wall_bottom()
    test_specular_wall_top()
    test_specular_wall_corner()
    test_no_collision()
    test_simple_reflection()
    print("All wall collision tests passed!")