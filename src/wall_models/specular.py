from typing import Tuple
import numpy as np

from src.wall_models.base import WallModel
from src.geometry.channel import RectangularChannel


class SpecularWall(WallModel):
    def handle_collision(self, position: np.ndarray, velocity: np.ndarray,
                        radius: float, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        new_position = position.copy()
        new_velocity = velocity.copy()
        
        x, y = position
        vx, vy = velocity
        
        collide_left, collide_right, collide_bottom, collide_top = \
            self.channel.check_wall_collision(position, radius)
        
        if collide_left:
            new_position[0] = radius  
            new_velocity[0] = -vx  
        elif collide_right:
            new_position[0] = self.channel.width - radius  
            new_velocity[0] = -vx  
        
        if collide_bottom:
            new_position[1] = radius  
            new_velocity[1] = -vy  
        elif collide_top:
            new_position[1] = self.channel.height - radius  
            new_velocity[1] = -vy  
        
        if not self.channel.contains(new_position, radius):
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