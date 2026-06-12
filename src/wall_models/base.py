from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
from ..geometry.channel import RectangularChannel


class WallModel(ABC):
    def __init__(self, channel: RectangularChannel):
        self.channel = channel
    
    @abstractmethod
    def handle_collision(self, position: np.ndarray, velocity: np.ndarray, 
                        radius: float, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        pass
    
    def check_and_handle(self, position: np.ndarray, velocity: np.ndarray,
                        radius: float, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        if self.channel.contains(position, radius):
            return position.copy(), velocity.copy()
        return self.handle_collision(position, velocity, radius, dt)