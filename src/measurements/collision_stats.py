from dataclasses import dataclass, field


@dataclass
class CollisionStats:
    particle_collision_count: int = 0
    wall_collision_count: int = 0
    particle_collisions_history: list[int] = field(default_factory=list)
    wall_collisions_history: list[int] = field(default_factory=list)

    def record_step(
        self,
        particle_collisions: int,
        wall_collisions: int,
    ) -> None:
        self.particle_collision_count += particle_collisions
        self.wall_collision_count += wall_collisions
        self.particle_collisions_history.append(particle_collisions)
        self.wall_collisions_history.append(wall_collisions)