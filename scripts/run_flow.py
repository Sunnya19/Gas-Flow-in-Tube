"""
Forced flow simulation in a 2D channel.

Periodic boundary along x, specular walls along y, constant external force
along x to drive the flow.  Saves diagnostic plots and prints summary.

Usage:
    python scripts/run_flow.py
    python scripts/run_flow.py --force 0.005 --particles 200 --time 20.0
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.config import SimulationConfig
from src.measurements.energy import calculate_total_energy
from src.simulation import Simulation
from src.visualization.plots import (
    plot_flow_energy,
    plot_mean_velocity,
    plot_temperature_vs_time,
    plot_velocity_profile,
)


def build_config(
    num_particles: int,
    total_time: float,
    external_force_x: float,
    initial_temperature: float,
    save_vtk: bool,
) -> SimulationConfig:
    """Build a SimulationConfig for the flow simulation."""
    return SimulationConfig(
        width=20.0,
        height=15.0,
        num_particles=num_particles,
        particle_radius=0.1,
        particle_mass=1.0,
        initial_temperature=initial_temperature,
        time_step=0.005,
        total_time=total_time,
        save_interval=20,
        equilibration_steps=500,
        num_trajectory_particles=5,
        wall_model_type="specular",
        wall_temperature=1.0,
        save_vtk=save_vtk,
        save_vtk_every=50,
        vtk_output_dir="outputs/vtk_flow",
        external_force_x=external_force_x,
        x_boundary_type="periodic",
        velocity_profile_bins=20,
    )


def run_flow_simulation(args: argparse.Namespace) -> None:
    """Run the forced flow simulation and produce outputs."""
    config = build_config(
        num_particles=args.particles,
        total_time=args.time,
        external_force_x=args.force,
        initial_temperature=args.temperature,
        save_vtk=args.save_vtk,
    )

    print("=" * 60)
    print("Forced flow simulation")
    print("=" * 60)
    print(f"  Domain: {config.width} x {config.height}")
    print(f"  Particles: {config.num_particles} (radius={config.particle_radius})")
    print(f"  Initial temperature: {config.initial_temperature}")
    print(f"  Time step: {config.time_step}, Total time: {config.total_time}")
    print(f"  Steps: {config.num_steps}")
    print(f"  Equilibration steps: {config.equilibration_steps}")
    print(f"  External force x: {config.external_force_x}")
    print(f"  X boundary: {config.x_boundary_type}")
    print(f"  Wall model: {config.wall_model}")
    print(f"  VTK export: {'enabled' if config.save_vtk else 'disabled'}")
    print()

    # Run simulation
    print("Initializing simulation...")
    sim = Simulation(config)

    print("Running simulation...")
    history = sim.run()
    print("Simulation completed!\n")

    # Extract results
    energy_history = history["total_energy"]
    initial_energy = float(energy_history[0])
    final_energy = float(energy_history[-1])
    relative_energy_change = (
        (final_energy - initial_energy) / abs(initial_energy)
        if initial_energy != 0.0
        else float("nan")
    )

    mean_vx_history = np.array(history.get("mean_vx", []), dtype=float)
    mean_vy_history = np.array(history.get("mean_vy", []), dtype=float)
    temperature_history = np.array(history.get("temperature", []), dtype=float)

    # Filter out NaN for final values
    valid_vx = mean_vx_history[np.isfinite(mean_vx_history)]
    valid_vy = mean_vy_history[np.isfinite(mean_vy_history)]
    valid_temp = temperature_history[np.isfinite(temperature_history)]

    final_mean_vx = float(valid_vx[-1]) if len(valid_vx) > 0 else float("nan")
    final_mean_vy = float(valid_vy[-1]) if len(valid_vy) > 0 else float("nan")
    initial_temp = float(valid_temp[0]) if len(valid_temp) > 0 else float("nan")
    final_temp = float(valid_temp[-1]) if len(valid_temp) > 0 else float("nan")

    # Print summary
    print("-" * 60)
    print("Flow simulation completed.")
    print(f"  external_force_x: {config.external_force_x}")
    print(f"  x_boundary_type: {config.x_boundary_type}")
    print(f"  wall_model_type: {config.wall_model}")
    print(f"  mean_vx final: {final_mean_vx:.6f}")
    print(f"  mean_vy final: {final_mean_vy:.6f}")
    print(f"  temperature initial: {initial_temp:.6f}")
    print(f"  temperature final: {final_temp:.6f}")
    print(f"  energy initial: {initial_energy:.6f}")
    print(f"  energy final: {final_energy:.6f}")
    print(f"  relative energy change: {relative_energy_change:.6f} ({relative_energy_change*100:.4f}%)")
    print("-" * 60)

    # Create output directories
    output_dir = ROOT_DIR / "outputs"
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Save plots
    print("\nGenerating plots...")

    plot_mean_velocity(history, plots_dir / "flow_mean_velocity.png")
    print(f"  Saved: {plots_dir / 'flow_mean_velocity.png'}")

    plot_temperature_vs_time(history, plots_dir / "flow_temperature.png")
    print(f"  Saved: {plots_dir / 'flow_temperature.png'}")

    plot_flow_energy(history, plots_dir / "flow_energy.png")
    print(f"  Saved: {plots_dir / 'flow_energy.png'}")

    # Velocity profile
    y_profile = history.get("velocity_profile_y", [])
    ux_profile = history.get("velocity_profile_ux", [])
    if y_profile and ux_profile:
        plot_velocity_profile(
            np.array(y_profile),
            np.array(ux_profile),
            plots_dir / "velocity_profile.png",
        )
        print(f"  Saved: {plots_dir / 'velocity_profile.png'}")
    else:
        print("  (velocity profile not available)")

    print("\nFlow simulation completed successfully!")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run forced flow simulation in a 2D channel."
    )
    parser.add_argument(
        "--force", type=float, default=0.005,
        help="External force along x (default: 0.005)",
    )
    parser.add_argument(
        "--particles", type=int, default=200,
        help="Number of particles (default: 200)",
    )
    parser.add_argument(
        "--time", type=float, default=20.0,
        help="Total simulation time (default: 20.0)",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="Initial temperature (default: 1.0)",
    )
    parser.add_argument(
        "--save-vtk", action="store_true",
        help="Enable VTK export (default: disabled)",
    )
    parser.add_argument(
        "--no-vtk", action="store_false", dest="save_vtk",
        help="Disable VTK export",
    )
    parser.set_defaults(save_vtk=False)

    args = parser.parse_args()
    run_flow_simulation(args)


if __name__ == "__main__":
    main()