from typing import Dict, List, Any
import numpy as np

from src.config import SimulationConfig
from src.state import SystemState
from src.geometry.channel import RectangularChannel
from src.initialization import initialize_system
from src.dynamics.integrator import EulerIntegrator
from src.dynamics.forces import apply_external_force
from src.dynamics.particle_collisions import process_all_collisions
from src.dynamics.wall_collisions import (
    create_wall_model,
    process_wall_collisions,
)
from src.measurements.energy import calculate_total_energy
from src.measurements.collision_stats import CollisionStats
from src.measurements.mean_free_path import MeanFreePathStats, update_free_path_measurements
from src.measurements.flow import compute_flow_temperature, compute_mean_flow_velocity
from src.measurements.velocity_profile import compute_velocity_profile
from src.io.vtk_writer import save_particles_vtp, save_pvd_file


class Simulation:

    def __init__(self, config: SimulationConfig):
        self.config = config

        # init sys
        self.state, self.channel = initialize_system(config)
        self.integrator = EulerIntegrator(dt=config.time_step)
        self.wall_model = create_wall_model(
            config.wall_model,
            self.channel,
            x_boundary_type=config.x_boundary_type,
        )

        # History storage
        self.history: Dict[str, List[Any]] = {
            'time': [],
            'total_energy': [],
            'positions_sample': [],
            'particle_collisions': [],
            'wall_collisions': [],
            'total_particle_collisions': 0,
            'total_wall_collisions': 0,
            'specular_wall_collisions': 0,
            'diffuse_wall_collisions': 0,
            'thermal_wall_collisions': 0,
            'mean_free_path': float("nan"),
            'mean_free_path_history': [],
            'free_path_samples': [],
            'mean_vx': [],
            'mean_vy': [],
            'temperature': [],
        }

        # Collision stats
        self.collision_stats = CollisionStats()

        # Mean free path tracking
        self.mfp_stats = MeanFreePathStats()
        self.path_since_last_collision = np.zeros(config.num_particles)
        self.has_previous_particle_collision = np.zeros(config.num_particles, dtype=bool)

        # Accumulators for collisions per save interval
        self.particle_collisions_since_save = 0
        self.wall_collisions_since_save = 0

        # Equilibration tracking
        self._measurement_enabled = (config.equilibration_steps == 0)

        # Select particles to track for trajectories
        rng = np.random.default_rng(config.random_seed) if config.random_seed is not None else np.random.default_rng()
        self.tracked_particles = rng.choice(
            config.num_particles,
            size=min(config.num_trajectory_particles, config.num_particles),
            replace=False
        )

        # Save initial state
        self._save_history()

    def _save_history(self):
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

        # Save collision stats — accumulated values since last save
        self.history['particle_collisions'].append(self.particle_collisions_since_save)
        self.history['wall_collisions'].append(self.wall_collisions_since_save)

        # Reset accumulators
        self.particle_collisions_since_save = 0
        self.wall_collisions_since_save = 0
        self.specular_wall_collisions = 0
        self.diffuse_wall_collisions = 0
        self.thermal_wall_collisions = 0

        # Save mean free path history
        self.mfp_stats.record_current_mean()
        self.history['mean_free_path_history'].append(self.mfp_stats.mean_free_path())
        mean_flow = compute_mean_flow_velocity(self.state.velocities)
        self.history['mean_vx'].append(float(mean_flow[0]))
        self.history['mean_vy'].append(float(mean_flow[1]))
        self.history['temperature'].append(
            compute_flow_temperature(self.state.velocities, self.config.particle_mass)
        )

    def step(self):
        dt = self.config.time_step

        # 1. Apply external force to particle velocities
        self.state.velocities = apply_external_force(
            self.state.velocities,
            force_x=self.config.external_force_x,
            mass=self.config.particle_mass,
            dt=dt,
        )

        # 2. Integrate positions
        self.state = self.integrator.step(self.state)

        # 3. Apply periodic x boundaries before particle collisions.
        if self.config.x_boundary_type == "periodic":
            self.state.positions[:, 0] = self.state.positions[:, 0] % self.channel.width

        # Accumulate path length for mean free path
        speeds = np.linalg.norm(self.state.velocities, axis=1)
        self.path_since_last_collision += speeds * dt

        # 2. Handle particle-particle collisions
        new_positions, new_velocities, collision_pairs = process_all_collisions(
            self.state.positions,
            self.state.velocities,
            self.config.particle_radius,
            self.config.particle_mass
        )
        self.state.positions = new_positions
        self.state.velocities = new_velocities

        # Record particle collisions for this step
        particle_collisions_this_step = len(collision_pairs)
        self.particle_collisions_since_save += particle_collisions_this_step

        # Update free path measurements using the new logic
        update_free_path_measurements(
            self.path_since_last_collision,
            self.has_previous_particle_collision,
            collision_pairs,
            self.mfp_stats,
            self._measurement_enabled,
        )

        if self.config.x_boundary_type == "periodic":
            self.state.positions[:, 0] = self.state.positions[:, 0] % self.channel.width

        # 3. Handle wall collisions
        new_positions, new_velocities, wall_collision_count, specular_count, diffuse_count, thermal_count = process_wall_collisions(
            self.state.positions,
            self.state.velocities,
            self.wall_model,
            self.config.particle_radius,
            dt,
            self.channel,
            self.config.x_boundary_type,
            self.config.wall_temperature,
        )
        self.state.positions = new_positions
        self.state.velocities = new_velocities

        self.wall_collisions_since_save += wall_collision_count
        self.specular_wall_collisions += specular_count
        self.diffuse_wall_collisions += diffuse_count
        self.thermal_wall_collisions += thermal_count

        # Record collision stats
        self.collision_stats.record_step(
            particle_collisions=particle_collisions_this_step,
            wall_collisions=wall_collision_count,
        )

    def _handle_equilibration(self, step: int):
        """Handle equilibration step logic."""
        eq_steps = self.config.equilibration_steps
        if eq_steps == 0:
            return
        if step == eq_steps - 1:
            # Last equilibration step: reset all measurement state
            self.path_since_last_collision[:] = 0.0
            self.has_previous_particle_collision[:] = False
            self.mfp_stats.clear()
            self._measurement_enabled = True

    def run(self) -> Dict[str, List[Any]]:
        total_steps = self.config.num_steps

        # VTK export collections
        vtk_frame_paths = []
        vtk_frame_times = []

        for step in range(total_steps):
            self.step()

            # Handle equilibration boundary
            self._handle_equilibration(step)

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

        # Populate final totals in history
        self.history['total_particle_collisions'] = self.collision_stats.particle_collision_count
        self.history['total_wall_collisions'] = self.collision_stats.wall_collision_count
        self.history['specular_wall_collisions'] = self.specular_wall_collisions
        self.history['diffuse_wall_collisions'] = self.diffuse_wall_collisions
        self.history['thermal_wall_collisions'] = self.thermal_wall_collisions
        self.history['mean_free_path'] = self.mfp_stats.mean_free_path()
        self.history['free_path_samples'] = self.mfp_stats.free_path_samples

        # Compute velocity profile from final state
        try:
            y_centers, ux_profile = compute_velocity_profile(
                self.state.positions,
                self.state.velocities,
                height=self.config.height,
                n_bins=self.config.velocity_profile_bins,
            )
            # store as lists for JSON/npz friendliness
            self.history['velocity_profile_y'] = list(map(float, y_centers.tolist()))
            self.history['velocity_profile_ux'] = list(map(float, ux_profile.tolist()))
        except Exception:
            # if something goes wrong, leave profile absent
            self.history['velocity_profile_y'] = []
            self.history['velocity_profile_ux'] = []

        return self.history

    def run_with_progress(self, progress_callback=None) -> Dict[str, List[Any]]:
        total_steps = self.config.num_steps

        # VTK export collections
        vtk_frame_paths = []
        vtk_frame_times = []

        for step in range(total_steps):
            self.step()

            # Handle equilibration boundary
            self._handle_equilibration(step)

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

        # Populate final totals in history
        self.history['total_particle_collisions'] = self.collision_stats.particle_collision_count
        self.history['total_wall_collisions'] = self.collision_stats.wall_collision_count
        self.history['mean_free_path'] = self.mfp_stats.mean_free_path()
        self.history['free_path_samples'] = self.mfp_stats.free_path_samples

        return self.history

    def get_current_state(self) -> SystemState:
        return self.state.copy()

    def get_history_array(self) -> Dict[str, np.ndarray]:
        return {
            'time': np.array(self.history['time']),
            'total_energy': np.array(self.history['total_energy']),
            'positions_sample': np.array(self.history['positions_sample'])
        }