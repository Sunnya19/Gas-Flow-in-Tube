
from typing import Dict, List, Any
import numpy as np

from .config import SimulationConfig
from .state import SystemState
from .geometry.channel import RectangularChannel
from .initialization import initialize_system
from .dynamics.integrator import EulerIntegrator
from .dynamics.particle_collisions import process_all_collisions
from .dynamics.wall_collisions import create_wall_model, process_wall_collisions
from .measurements.energy import calculate_total_energy
from .io.vtk_writer import save_particles_vtp, save_pvd_file


class Simulation:
    def __init__(self, config: SimulationConfig):

        self.config = config

        # Initialize system
        self.state, self.channel = initialize_system(config)

        # Create integrator
        self.integrator = EulerIntegrator(dt=config.time_step)

        # Create wall model
        self.wall_model = create_wall_model(config.wall_model_type, self.channel)

        # History storage
        self.history: Dict[str, List[Any]] = {
            'time': [],
            'total_energy': [],
            'positions_sample': []  # Store positions of tracked particles
        }

        # Select particles to track for trajectories
        self.tracked_particles = np.random.choice(
            config.num_particles,
            size=min(config.num_trajectory_particles, config.num_particles),
            replace=False
        )

        # Save initial state
        self._save_history()

    def _save_history(self):
        """Save current state to history.Perform one simulation step.
        Run the complete simulation.

        Returns:
            Dictionary with simulation history

        Run simulation with progress reporting.

        Args:
            progress_callback: Optional callback function that receives
                (current_step, total_steps, current_time)

        Returns:
            Dictionary with simulation history
        Get a copy of the current system state.
        Convert history lists to numpy arrays.

        Returns:
            Dictionary with numpy arrays instead of lists
        """
        return {
            'time': np.array(self.history['time']),
            'total_energy': np.array(self.history['total_energy']),
            'positions_sample': np.array(self.history['positions_sample'])
        }
