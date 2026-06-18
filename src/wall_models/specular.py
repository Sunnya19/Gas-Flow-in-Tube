from typing import Tuple
import numpy as np

from src.wall_models.base import WallModel
from src.geometry.channel import RectangularChannel


class SpecularWall(WallModel):
    def __init__(self, channel: RectangularChannel, x_boundary_type: str = "reflective"):
        super().__init__(channel)
        self.x_boundary_type = x_boundary_type

    def _apply_x_boundary(self, position: np.ndarray) -> np.ndarray:
        if self.x_boundary_type == "periodic":
            width = self.channel.width
            new_x = position[0] % width
            return np.array([new_x, position[1]])
        return position

    def handle_collision(self, position: np.ndarray, velocity: np.ndarray,
                        radius: float, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        new_position = position.copy()
        new_velocity = velocity.copy()
        
        x, y = position
        vx, vy = velocity
        
        collide_left, collide_right, collide_bottom, collide_top = \
            self.channel.check_wall_collision(position, radius)
        
        if collide_left or collide_right:
            if self.x_boundary_type == "periodic":
                new_position[0] = position[0] % self.channel.width
            else:
                if collide_left:
                    new_position[0] = radius
                else:
                    new_position[0] = self.channel.width - radius
                new_velocity[0] = -vx
        
        if collide_bottom:
            new_position[1] = radius  
            new_velocity[1] = -vy  
        elif collide_top:
            new_position[1] = self.channel.height - radius  
            new_velocity[1] = -vy  
        
        if self.x_boundary_type == "periodic":
            new_position[0] = new_position[0] % self.channel.width
            if new_position[1] < radius or new_position[1] > self.channel.height - radius:
                new_position = self.channel.clip_to_boundary(new_position, radius)
        elif not self.channel.contains(new_position, radius):
            new_position = self.channel.clip_to_boundary(new_position, radius)
        return new_position, new_velocity
    
    def handle_collision_simple(self, position: np.ndarray, velocity: np.ndarray,
                               radius: float) -> Tuple[np.ndarray, np.ndarray]:
        new_velocity = velocity.copy()
        x, y = position

        if x < radius:  
            new_velocity[0] = -velocity[0]
        elif x > self.channel.width - radius:  
            new_velocity[0] = -velocity[0]
        
        if y < radius:  
            new_velocity[1] = -velocity[1]
        elif y > self.channel.height - radius:  
            new_velocity[1] = -velocity[1]
        
        return position.copy(), new_velocity