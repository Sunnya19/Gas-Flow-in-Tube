"""
Tests for periodic x-boundary condition.

Verifies that:
- Particle exiting right reappears on the left.
- Particle exiting left reappears on the right.
- vx preserves sign and magnitude.
"""
import numpy as np

from src.geometry.channel import RectangularChannel
from src.dynamics.walls import _apply_x_boundary


def _make_particle(x, y, vx, vy):
    return np.array([x, y]), np.array([vx, vy])


def test_periodic_right_to_left():
    """Particle beyond right wall wraps to left side."""
    channel = RectangularChannel(width=10.0, height=10.0)
    radius = 0.1

    pos, vel = _make_particle(10.5, 5.0, 2.0, 1.0)
    new_pos, new_vel = _apply_x_boundary(pos, vel, radius, channel, "periodic")

    # x should wrap: 10.5 % 10.0 = 0.5
    assert np.isclose(new_pos[0], 0.5), \
        f"Expected x=0.5, got {new_pos[0]}"
    # y unchanged
    assert np.isclose(new_pos[1], 5.0), \
        f"Expected y=5.0, got {new_pos[1]}"
    # vx unchanged (no reflection)
    assert np.isclose(new_vel[0], 2.0), \
        f"Expected vx=2.0, got {new_vel[0]}"
    # vy unchanged
    assert np.isclose(new_vel[1], 1.0), \
        f"Expected vy=1.0, got {new_vel[1]}"


def test_periodic_left_to_right():
    """Particle beyond left wall wraps to right side."""
    channel = RectangularChannel(width=10.0, height=10.0)
    radius = 0.1

    pos, vel = _make_particle(-0.5, 5.0, -2.0, 1.0)
    new_pos, new_vel = _apply_x_boundary(pos, vel, radius, channel, "periodic")

    # x should wrap: -0.5 % 10.0 = 9.5
    assert np.isclose(new_pos[0], 9.5), \
        f"Expected x=9.5, got {new_pos[0]}"
    # y unchanged
    assert np.isclose(new_pos[1], 5.0), \
        f"Expected y=5.0, got {new_pos[1]}"
    # vx unchanged (no reflection)
    assert np.isclose(new_vel[0], -2.0), \
        f"Expected vx=-2.0, got {new_vel[0]}"
    # vy unchanged
    assert np.isclose(new_vel[1], 1.0), \
        f"Expected vy=1.0, got {new_vel[1]}"


def test_periodic_exactly_on_boundary():
    """Particle exactly at boundary should not be moved."""
    channel = RectangularChannel(width=10.0, height=10.0)
    radius = 0.1

    pos, vel = _make_particle(0.0, 5.0, 1.0, 0.0)
    new_pos, new_vel = _apply_x_boundary(pos, vel, radius, channel, "periodic")

    # x=0.0 % 10.0 = 0.0
    assert np.isclose(new_pos[0], 0.0), \
        f"Expected x=0.0, got {new_pos[0]}"
    assert np.isclose(new_vel[0], 1.0), \
        "vx should be unchanged"


def test_periodic_vx_sign_preserved():
    """vx sign must be preserved under periodic wrapping."""
    channel = RectangularChannel(width=10.0, height=10.0)
    radius = 0.1

    # Particle moving left beyond left wall
    pos, vel = _make_particle(-1.0, 5.0, -3.0, 0.5)
    new_pos, new_vel = _apply_x_boundary(pos, vel, radius, channel, "periodic")

    assert new_vel[0] < 0, "vx should remain negative"
    assert np.isclose(new_vel[0], -3.0), "vx magnitude should be preserved"

    # Particle moving right beyond right wall
    pos, vel = _make_particle(11.0, 5.0, 3.0, 0.5)
    new_pos, new_vel = _apply_x_boundary(pos, vel, radius, channel, "periodic")

    assert new_vel[0] > 0, "vx should remain positive"
    assert np.isclose(new_vel[0], 3.0), "vx magnitude should be preserved"


if __name__ == "__main__":
    test_periodic_right_to_left()
    test_periodic_left_to_right()
    test_periodic_exactly_on_boundary()
    test_periodic_vx_sign_preserved()
    print("All periodic boundary tests passed!")