
from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class RectangularChannel:
    
    width: float
    height: float
    
    def __post_init__(self):
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Channel dimensions must be positive")
    
    def contains(self, position: np.ndarray, radius: float = 0.0) -> bool:
        
        x, y = position
        return (radius <= x <= self.width - radius and 
                radius <= y <= self.height - radius)
    
    def clip_to_boundary(self, position: np.ndarray, radius: float) -> np.ndarray:
        
        x, y = position
        x_clipped = np.clip(x, radius, self.width - radius)
        y_clipped = np.clip(y, radius, self.height - radius)
        return np.array([x_clipped, y_clipped])
    
    def distance_to_walls(self, position: np.ndarray) -> Tuple[float, float, float, float]:
        
        x, y = position
        return (
            x,  # distance to left wall (x=0)
            self.width - x,  # distance to right wall (x=width)
            y,  # distance to bottom wall (y=0)
            self.height - y  # distance to top wall (y=height)
        )
    
    def check_wall_collision(self, position: np.ndarray, radius: float) -> Tuple[bool, bool, bool, bool]:
        
        dist_left, dist_right, dist_bottom, dist_top = self.distance_to_walls(position)
        return (
            dist_left < radius,
            dist_right < radius,
            dist_bottom < radius,
            dist_top < radius
        )
    
    @property
    def area(self) -> float:
        return self.width * self.height
    
    @property
    def center(self) -> np.ndarray:
        return np.array([self.width / 2, self.height / 2])