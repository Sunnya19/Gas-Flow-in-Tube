"""
Main simulation class for molecular dynamics.
"""
from typing import Dict, List, Any
import numpy as np

from src.config import SimulationConfig
from src.state import SystemState
from src.geometry.channel import RectangularChannel
from src.initialization import initialize_system
from src.dynamics.integrator import EulerIntegrator
from src.dynamics.particle_collisions import process_all_collisions
from src.dynamics.wall_collisions import create_wall_model, process_wall_collisions
from src.measurements.energy import calculate_total_energy
from src.io.vtk_writer import save_particles_vtp, save_pvd_file


class Simulation:
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        
        # init sys
        self.state, self.channel = initialize_system(config)
        self.integrator = EulerIntegrator(dt=config.time_step)
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
        """Save current state to history."""
        self.history['time'].append(self.state.time)
        
        # Calculate and save total energy
        energy = calculate_total_energy(
            self.state.velocities,
            self.config.particle_mass
        )
        self.history['total_energy'].append(energy)
        
        # Save positions of tracked particles
        tracked_positions = self.state.positions[self.tracked_particles].copy()
        self.history['positions_sample'].append(tracked_positions)
    
    def step(self):
        """Perform one simulation step."""
        # 1. Integrate positions
        self.state = self.integrator.step(self.state)
        
        # 2. Handle particle-particle collisions
        new_positions, new_velocities = process_all_collisions(
            self.state.positions,
            self.state.velocities,
            self.config.particle_radius,
            self.config.particle_mass
        )
        self.state.positions = new_positions
        self.state.velocities = new_velocities
        
        # 3. Handle wall collisions
        new_positions, new_velocities = process_wall_collisions(
            self.state.positions,
            self.state.velocities,
            self.wall_model,
            self.config.particle_radius,
            self.config.time_step
        )
        self.state.positions = new_positions
        self.state.velocities = new_velocities
    
    def run(self) -> Dict[str, List[Any]]:
        """
        Run the complete simulation.
        
        Returns:
            Dictionary with simulation history
        """
        total_steps = self.config.num_steps
        
        # VTK export collections
        vtk_frame_paths = []
        vtk_frame_times = []
        
        for step in range(total_steps):
            self.step()
            
            # Save to history at specified intervals
            if step % self.config.save_interval == 0:
                self._save_history()
            
            # Save VTK frame if enabled
            if self.config.save_vtk and step % self.config.save_vtk_every == 0:
                from pathlib import Path
                frame_path = Path(self.config.vtk_output_dir) / f"particles_{step:06d}.vtp"
                
                save_particles_vtp(
                    positions=self.state.positions,
                    velocities=self.state.velocities,
                    radius=self.config.particle_radius,
                    output_path=frame_path,
                )
                vtk_frame_paths.append(frame_path)
                vtk_frame_times.append(self.state.time)
        
        # Save final state to history
        self._save_history()
        
        # Save final VTK frame if enabled and not already saved
        if self.config.save_vtk and (total_steps - 1) % self.config.save_vtk_every != 0:
            from pathlib import Path
            final_frame_path = Path(self.config.vtk_output_dir) / f"particles_{total_steps:06d}.vtp"
            save_particles_vtp(
                positions=self.state.positions,
                velocities=self.state.velocities,
                radius=self.config.particle_radius,
                output_path=final_frame_path,
            )
            vtk_frame_paths.append(final_frame_path)
            vtk_frame_times.append(self.state.time)
        
        # Create PVD collection file if any frames were saved
        if self.config.save_vtk and vtk_frame_paths:
            from pathlib import Path
            pvd_path = Path(self.config.vtk_output_dir) / "particles.pvd"
            save_pvd_file(
                frame_paths=vtk_frame_paths,
                times=vtk_frame_times,
                output_path=pvd_path,
            )
        
        return self.history
    
    def run_with_progress(self, progress_callback=None) -> Dict[str, List[Any]]:
        """
        Run simulation with progress reporting.
        
        Args:
            progress_callback: Optional callback function that receives
                (current_step, total_steps, current_time)
                
        Returns:
            Dictionary with simulation history
        """
        total_steps = self.config.num_steps
        
        # VTK export collections
        vtk_frame_paths = []
        vtk_frame_times = []
        
        for step in range(total_steps):
            self.step()
            
            # Save to history at specified intervals
            if step % self.config.save_interval == 0:
                self._save_history()
            
            # Save VTK frame if enabled
            if self.config.save_vtk and step % self.config.save_vtk_every == 0:
                from pathlib import Path
                frame_path = Path(self.config.vtk_output_dir) / f"particles_{step:06d}.vtp"
                
                save_particles_vtp(
                    positions=self.state.positions,
                    velocities=self.state.velocities,
                    radius=self.config.particle_radius,
                    output_path=frame_path,
                )
                vtk_frame_paths.append(frame_path)
                vtk_frame_times.append(self.state.time)
            
            # Report progress
            if progress_callback:
                progress_callback(step, total_steps, self.state.time)
        
        # Save final state to history
        self._save_history()
        
        # Save final VTK frame if enabled and not already saved
        if self.config.save_vtk and (total_steps - 1) % self.config.save_vtk_every != 0:
            from pathlib import Path
            final_frame_path = Path(self.config.vtk_output_dir) / f"particles_{total_steps:06d}.vtp"
            save_particles_vtp(
                positions=self.state.positions,
                velocities=self.state.velocities,
                radius=self.config.particle_radius,
                output_path=final_frame_path,
            )
            vtk_frame_paths.append(final_frame_path)
            vtk_frame_times.append(self.state.time)
        
        # Create PVD collection file if any frames were saved
        if self.config.save_vtk and vtk_frame_paths:
            from pathlib import Path
            pvd_path = Path(self.config.vtk_output_dir) / "particles.pvd"
            save_pvd_file(
                frame_paths=vtk_frame_paths,
                times=vtk_frame_times,
                output_path=pvd_path,
            )
        
        return self.history
    
    def get_current_state(self) -> SystemState:
        return self.state.copy()
    
    def get_history_array(self) -> Dict[str, np.ndarray]:
        return {
            'time': np.array(self.history['time']),
            'total_energy': np.array(self.history['total_energy']),
            'positions_sample': np.array(self.history['positions_sample'])
        }