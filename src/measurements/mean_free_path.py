from dataclasses import dataclass, field
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