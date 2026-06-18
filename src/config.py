from dataclasses import dataclass


@dataclass
class SimulationConfig:
    # Domain dimensions
    width: float = 10.0  # L
    height: float = 10.0  # H

    # Particle properties
    num_particles: int = 100  # N
    particle_radius: float = 0.1  # r
    particle_mass: float = 1.0  # m

    # Initial conditions
    initial_temperature: float = 1.0  # T (k_B = 1)

    # Simulation parameters
    time_step: float = 0.01  # dt
    total_time: float = 10.0  # total simulation time
    save_interval: int = 10  # save history every N steps

    # Equilibration / burn-in
    equilibration_steps: int = 0  # steps to run before starting measurements

    # Sampling
    num_trajectory_particles: int = 5  # number of particles to track for trajectories

    # Wall model type
    wall_model_type: str = "specular"  # options: "specular", "diffuse", etc.

    # VTK export settings
    save_vtk: bool = False  # whether to save VTK files for ParaView
    save_vtk_every: int = 20  # save VTK frame every N steps
    vtk_output_dir: str = "outputs/vtk"  # directory for VTK files
    # Driven flow settings
    external_force_x: float = 0.0  # constant force applied along x
    x_boundary_type: str = "reflective"  # options: "reflective", "periodic"

    def __post_init__(self):
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Domain dimensions must be positive")
        if self.num_particles <= 0:
            raise ValueError("Number of particles must be positive")
        if self.particle_radius <= 0:
            raise ValueError("Particle radius must be positive")
        if self.particle_mass <= 0:
            raise ValueError("Particle mass must be positive")
        if self.initial_temperature < 0:
            raise ValueError("Temperature cannot be negative")
        if self.time_step <= 0:
            raise ValueError("Time step must be positive")
        if self.total_time <= 0:
            raise ValueError("Total time must be positive")
        if self.save_interval <= 0:
            raise ValueError("Save interval must be positive")
        if self.equilibration_steps < 0:
            raise ValueError("equilibration_steps must be non-negative")
        if self.num_trajectory_particles <= 0:
            raise ValueError("Number of trajectory particles must be positive")
        if self.num_trajectory_particles > self.num_particles:
            raise ValueError("Cannot track more particles than exist in simulation")

        # Validate VTK export settings
        if self.save_vtk_every <= 0:
            raise ValueError("save_vtk_every must be positive")
        if not self.vtk_output_dir:
            raise ValueError("vtk_output_dir cannot be empty")

        if self.x_boundary_type not in ("reflective", "periodic"):
            raise ValueError(
                f"x_boundary_type must be 'reflective' or 'periodic', got {self.x_boundary_type}"
            )

        # Check that particles can fit in the domain
        min_dim = min(self.width, self.height)
        if 2 * self.particle_radius >= min_dim:
            raise ValueError(f"Particle diameter {2*self.particle_radius} is too large for domain size {min_dim}")

    @property
    def num_steps(self) -> int:
        return int(self.total_time / self.time_step)

    @property
    def save_every(self) -> int:
        return self.save_interval
