from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np


@dataclass
class MeanFreePathStats:
    free_path_samples: list[float] = field(default_factory=list)
    mean_free_path_history: list[float] = field(default_factory=list)

    def add_samples(self, samples: list[float]) -> None:
        self.free_path_samples.extend(samples)

    def mean_free_path(self) -> float:
        if len(self.free_path_samples) == 0:
            return float("nan")
        return float(np.mean(self.free_path_samples))

    def record_current_mean(self) -> None:
        self.mean_free_path_history.append(self.mean_free_path())

    def clear(self) -> None:
        self.free_path_samples.clear()
        self.mean_free_path_history.clear()


def update_free_path_measurements(
    path_since_last_collision: np.ndarray,
    has_previous_particle_collision: np.ndarray,
    collision_pairs: List[Tuple[int, int]],
    mfp_stats: MeanFreePathStats,
    measurement_enabled: bool,
) -> None:
    if not collision_pairs:
        return

    collided_particles = set()
    for i, j in collision_pairs:
        collided_particles.add(i)
        collided_particles.add(j)

    samples = []
    for p in collided_particles:
        if measurement_enabled and has_previous_particle_collision[p]:
            samples.append(path_since_last_collision[p])
        path_since_last_collision[p] = 0.0
        has_previous_particle_collision[p] = True

    if samples:
        mfp_stats.add_samples(samples)