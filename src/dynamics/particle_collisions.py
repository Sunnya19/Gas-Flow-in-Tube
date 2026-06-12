from typing import Tuple, List
import numpy as np


def find_colliding_pairs(positions: np.ndarray, radius: float) -> List[Tuple[int, int]]:
    N = positions.shape[0]
    pairs = []
    diameter_sq = (2 * radius) ** 2

    for i in range(N):
        pos_i = positions[i]
        for j in range(i + 1, N):
            pos_j = positions[j]
            dx = pos_i[0] - pos_j[0]
            dy = pos_i[1] - pos_j[1]
            dist_sq = dx * dx + dy * dy

            if dist_sq < diameter_sq:
                pairs.append((i, j))

    return pairs


def check_approaching(positions: np.ndarray, velocities: np.ndarray,
                      i: int, j: int) -> bool:
    pos_i = positions[i]
    pos_j = positions[j]
    vel_i = velocities[i]
    vel_j = velocities[j]

    r_ij = pos_i - pos_j

    v_ij = vel_i - vel_j

    return np.dot(r_ij, v_ij) < 0


def handle_elastic_collision(positions: np.ndarray, velocities: np.ndarray,
                             i: int, j: int, mass: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    pos_i = positions[i]
    pos_j = positions[j]
    vel_i = velocities[i]
    vel_j = velocities[j]
    r_ij = pos_i - pos_j
    dist = np.linalg.norm(r_ij)
    if dist == 0:
        return positions.copy(), velocities.copy()

    n = r_ij / dist
    v_rel = vel_i - vel_j
    v_rel_n = np.dot(v_rel, n)
    delta_v = v_rel_n * n
    new_vel_i = vel_i - delta_v
    new_vel_j = vel_j + delta_v

    new_positions = positions.copy()
    new_velocities = velocities.copy()

    new_velocities[i] = new_vel_i
    new_velocities[j] = new_vel_j

    return new_positions, new_velocities


def separate_overlapping_particles(positions: np.ndarray, i: int, j: int,
                                   radius: float, epsilon: float = 1e-6) -> np.ndarray:
    pos_i = positions[i]
    pos_j = positions[j]
    r_ij = pos_i - pos_j
    dist = np.linalg.norm(r_ij)

    if dist == 0:
        r_ij = np.array([1.0, 0.0])
        dist = 1.0

    desired_dist = 2 * radius + epsilon

    overlap = desired_dist - dist
    if overlap <= 0:
        return positions.copy()

    move_vector = (overlap / 2) * (r_ij / dist)

    new_positions = positions.copy()
    new_positions[i] = pos_i + move_vector
    new_positions[j] = pos_j - move_vector

    return new_positions


def process_all_collisions(positions: np.ndarray, velocities: np.ndarray,
                           radius: float, mass: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    new_positions = positions.copy()
    new_velocities = velocities.copy()

    pairs = find_colliding_pairs(new_positions, radius)

    for i, j in pairs:
        if check_approaching(new_positions, new_velocities, i, j):
            new_positions, new_velocities = handle_elastic_collision(
                new_positions, new_velocities, i, j, mass
            )

        new_positions = separate_overlapping_particles(
            new_positions, i, j, radius)

    return new_positions, new_velocities
