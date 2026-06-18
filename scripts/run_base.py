import sys
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path so imports work when running as python scripts/run_base.py
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import SimulationConfig
from src.measurements.knudsen import (
    classify_knudsen_number,
    compute_knudsen_number,
)
from src.simulation import Simulation
from src.visualization.plots import (
    plot_energy,
    plot_trajectories,
    plot_collision_counts,
    plot_mean_free_path,
    plot_free_path_histogram,
    plot_mean_vx_vs_time,
    plot_mean_vy_vs_time,
    plot_temperature_vs_time,
    plot_velocity_profile,
)


def load_config_from_yaml(yaml_path):

    with open(yaml_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    # use defaults
    config = SimulationConfig(
        width=config_dict.get('width', 20.0),
        height=config_dict.get('height', 15.0),
        num_particles=config_dict.get('num_particles', 100),
        particle_radius=config_dict.get('particle_radius', 0.1),
        particle_mass=config_dict.get('particle_mass', 1.0),
        initial_temperature=config_dict.get('initial_temperature', 1.0),
        time_step=config_dict.get('time_step', 0.005),
        total_time=config_dict.get('total_time', 10.0),
        save_interval=config_dict.get('save_interval', 20),
        num_trajectory_particles=config_dict.get('num_trajectory_particles', 5),
        wall_model_type=config_dict.get('wall_model_type', 'specular'),
        save_vtk=True,
        save_vtk_every=20,
        vtk_output_dir="outputs/vtk",
        external_force_x=config_dict.get('external_force_x', 0.0),
        x_boundary_type=config_dict.get('x_boundary_type', 'reflective'),
    )

    return config


def run_base_simulation():
    print("Starting basic molecular dynamics simulation...")

    # Create configuration with default hardcoded values
    config = SimulationConfig(
        width=20.0,
        height=15.0,
        num_particles=100,
        particle_radius=0.1,
        particle_mass=1.0,
        initial_temperature=1.0,
        time_step=0.005,
        total_time=10.0,
        save_interval=20,
        equilibration_steps=500,
        num_trajectory_particles=5,
        wall_model_type="specular",
        save_vtk=True,
        save_vtk_every=20,
        vtk_output_dir="outputs/vtk",
        external_force_x=0.0,
        x_boundary_type="reflective",
    )

    print(f"Configuration:")
    print(f"  Domain: {config.width} x {config.height}")
    print(f"  Particles: {config.num_particles} (radius={config.particle_radius})")
    print(f"  Temperature: {config.initial_temperature}")
    print(f"  Time step: {config.time_step}, Total time: {config.total_time}")
    print(f"  Steps: {config.num_steps}, Save every: {config.save_interval} steps")
    print(f"  Equilibration steps: {config.equilibration_steps}")
    print(f"  External force x: {config.external_force_x}")
    print(f"  X boundary: {config.x_boundary_type}")
    print(f"  Wall model: {config.wall_model_type}")
    if config.save_vtk:
        print(f"  VTK export: enabled (every {config.save_vtk_every} steps)")
        print(f"  VTK output directory: {config.vtk_output_dir}")
    else:
        print(f"  VTK export: disabled")

    # Create and run simulation
    print("\nInitializing simulation...")
    sim = Simulation(config)

    print("Running simulation...")
    history = sim.run()

    print(f"Simulation completed!")
    print(f"  Final time: {sim.state.time:.3f}")
    print(f"  History points: {len(history['time'])}")

    # Calculate energy conservation
    from src.measurements.energy import calculate_energy_conservation
    energy_history = history['total_energy']
    max_error, std_error = calculate_energy_conservation(energy_history)
    print(f"  Energy conservation:")
    print(f"    Initial energy: {energy_history[0]:.6f}")
    print(f"    Final energy: {energy_history[-1]:.6f}")
    print(f"    Max relative error: {max_error:.6f} ({max_error*100:.2f}%)")
    print(f"    Std relative error: {std_error:.6f} ({std_error*100:.2f}%)")

    # Print measurement statistics
    print(f"\n  Collision statistics:")
    print(f"    Total particle collisions: {history['total_particle_collisions']}")
    print(f"    Total wall collisions: {history['total_wall_collisions']}")

    print(f"\n  Mean free path diagnostics:")
    mfp = history['mean_free_path']
    samples = history['free_path_samples']
    n_samples = len(samples)
    characteristic_length = config.height / 2.0
    kn = compute_knudsen_number(
        mean_free_path=mfp,
        characteristic_length=characteristic_length,
    )
    regime = classify_knudsen_number(kn) if np.isfinite(kn) else "unknown"

    print(f"    lambda_MD: {mfp:.6f}")
    print(f"    Number of free path samples: {n_samples}")
    print(f"    Knudsen number: {kn:.6f}")
    print(f"    Regime: {regime}")

    if n_samples > 0:
        import math
        samples_arr = np.array(samples)
        print(f"    Min free path: {np.min(samples_arr):.6f}")
        print(f"    Max free path: {np.max(samples_arr):.6f}")
        radius = config.particle_radius
        frac_lt_r = np.sum(samples_arr < radius) / n_samples
        frac_lt_2r = np.sum(samples_arr < 2 * radius) / n_samples
        print(f"    Fraction of free paths < particle_radius: {frac_lt_r:.6f} ({frac_lt_r*100:.2f}%)")
        print(f"    Fraction of free paths < 2*particle_radius: {frac_lt_2r:.6f} ({frac_lt_2r*100:.2f}%)")
    else:
        print(f"    (not enough samples)")

    # Relative energy drift
    initial_energy = energy_history[0]
    final_energy = energy_history[-1]
    if initial_energy != 0:
        rel_drift = abs(final_energy - initial_energy) / abs(initial_energy)
        print(f"\n  Relative energy drift: {rel_drift:.6f} ({rel_drift*100:.4f}%)")

    # Create output directories
    output_dir = Path(__file__).parent.parent / 'outputs'
    plots_dir = output_dir / 'plots'
    data_dir = output_dir / 'data'

    plots_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Save plots
    print("\nGenerating plots...")

    energy_plot_path = plots_dir / 'energy.png'
    plot_energy(history, energy_plot_path)
    print(f"  Energy plot saved to: {energy_plot_path}")

    trajectories_plot_path = plots_dir / 'trajectories.png'
    plot_trajectories(history, trajectories_plot_path)
    print(f"  Trajectories plot saved to: {trajectories_plot_path}")

    collisions_plot_path = plots_dir / 'collisions.png'
    plot_collision_counts(history, collisions_plot_path)
    print(f"  Collisions plot saved to: {collisions_plot_path}")

    mean_free_path_plot_path = plots_dir / 'mean_free_path.png'
    plot_mean_free_path(history, mean_free_path_plot_path)
    print(f"  Mean free path plot saved to: {mean_free_path_plot_path}")

    free_path_histogram_path = plots_dir / 'free_path_histogram.png'
    plot_free_path_histogram(history, free_path_histogram_path)
    print(f"  Free path histogram saved to: {free_path_histogram_path}")

    flow_history_path = data_dir / 'flow_history.csv'
    flow_df = pd.DataFrame({
        'time': history['time'],
        'mean_vx': history.get('mean_vx', []),
        'mean_vy': history.get('mean_vy', []),
        'temperature': history.get('temperature', []),
    })
    flow_df.to_csv(flow_history_path, index=False)
    print(f"  Flow history saved to: {flow_history_path}")

    mean_vx_plot_path = plots_dir / 'mean_vx_vs_time.png'
    plot_mean_vx_vs_time(history, mean_vx_plot_path)
    print(f"  Mean v_x plot saved to: {mean_vx_plot_path}")

    mean_vy_plot_path = plots_dir / 'mean_vy_vs_time.png'
    plot_mean_vy_vs_time(history, mean_vy_plot_path)
    print(f"  Mean v_y plot saved to: {mean_vy_plot_path}")

    temperature_plot_path = plots_dir / 'temperature_vs_time.png'
    plot_temperature_vs_time(history, temperature_plot_path)
    print(f"  Temperature plot saved to: {temperature_plot_path}")

    # Save velocity profile if available
    y_profile = history.get('velocity_profile_y', [])
    ux_profile = history.get('velocity_profile_ux', [])
    if y_profile and ux_profile:
        vp_csv = data_dir / 'velocity_profile.csv'
        vp_df = pd.DataFrame({'y': y_profile, 'ux': ux_profile})
        vp_df.to_csv(vp_csv, index=False)
        print(f"  Velocity profile saved to: {vp_csv}")

        vp_plot = plots_dir / 'velocity_profile.png'
        # convert lists to numpy arrays for plotting
        import numpy as _np
        plot_velocity_profile(_np.array(y_profile), _np.array(ux_profile), vp_plot)
        print(f"  Velocity profile plot saved to: {vp_plot}")

    flow_history_path = data_dir / 'flow_history.csv'
    flow_df = pd.DataFrame({
        'time': history['time'],
        'mean_vx': history.get('mean_vx', []),
        'mean_vy': history.get('mean_vy', []),
        'temperature': history.get('temperature', []),
    })
    flow_df.to_csv(flow_history_path, index=False)
    print(f"  Flow history saved to: {flow_history_path}")

    # Save history data
    history_data = {
        'time': np.array(history['time']),
        'total_energy': np.array(history['total_energy']),
        'positions_sample': np.array(history['positions_sample']),
        'particle_collisions': np.array(history['particle_collisions']),
        'wall_collisions': np.array(history['wall_collisions']),
        'mean_free_path_history': np.array(history['mean_free_path_history']),
    }

    data_path = data_dir / 'simulation_history.npz'
    np.savez(data_path, **history_data)
    print(f"  History data saved to: {data_path}")

    # Save final state
    final_state = sim.get_current_state()
    final_state_data = {
        'positions': final_state.positions,
        'velocities': final_state.velocities,
        'time': final_state.time
    }

    final_state_path = data_dir / 'final_state.npz'
    np.savez(final_state_path, **final_state_data)
    print(f"  Final state saved to: {final_state_path}")

    print("\nSimulation completed successfully!")
    return history


def run_custom_simulation(config_file=None):

    if config_file:
        print(f"Loading configuration from: {config_file}")
        try:
            config = load_config_from_yaml(config_file)
            print(f"Configuration loaded successfully from YAML.")
        except Exception as e:
            print(f"Error loading YAML config: {e}")
            print(f"Falling back to default configuration.")
            config = None
    else:
        config = None

    if config is None:
        # Use default configuration
        return run_base_simulation()
    else:
        # Run simulation with custom configuration
        print("Starting molecular dynamics simulation with custom configuration...")

        print(f"Configuration:")
        print(f"  Domain: {config.width} x {config.height}")
        print(f"  Particles: {config.num_particles} (radius={config.particle_radius})")
        print(f"  Temperature: {config.initial_temperature}")
        print(f"  Time step: {config.time_step}, Total time: {config.total_time}")
        print(f"  Steps: {config.num_steps}, Save every: {config.save_interval} steps")
        print(f"  Wall model: {config.wall_model_type}")
        print(f"  External force x: {config.external_force_x}")
        print(f"  X boundary: {config.x_boundary_type}")
        if config.save_vtk:
            print(f"  VTK export: enabled (every {config.save_vtk_every} steps)")
            print(f"  VTK output directory: {config.vtk_output_dir}")
        else:
            print(f"  VTK export: disabled")

        # Create and run simulation
        print("\nInitializing simulation...")
        sim = Simulation(config)

        print("Running simulation...")
        history = sim.run()

        print(f"Simulation completed!")
        print(f"  Final time: {sim.state.time:.3f}")
        print(f"  History points: {len(history['time'])}")

        # Calculate energy conservation
        from src.measurements.energy import calculate_energy_conservation
        energy_history = history['total_energy']
        max_error, std_error = calculate_energy_conservation(energy_history)
        print(f"  Energy conservation:")
        print(f"    Initial energy: {energy_history[0]:.6f}")
        print(f"    Final energy: {energy_history[-1]:.6f}")
        print(f"    Max relative error: {max_error:.6f} ({max_error*100:.2f}%)")
        print(f"    Std relative error: {std_error:.6f} ({std_error*100:.2f}%)")

        # Print measurement statistics
        print(f"\n  Collision statistics:")
        print(f"    Total particle collisions: {history['total_particle_collisions']}")
        print(f"    Total wall collisions: {history['total_wall_collisions']}")

        print(f"\n  Mean free path:")
        mfp = history['mean_free_path']
        n_samples = len(history['free_path_samples'])
        characteristic_length = config.height / 2.0
        kn = compute_knudsen_number(
            mean_free_path=mfp,
            characteristic_length=characteristic_length,
        )
        regime = classify_knudsen_number(kn) if np.isfinite(kn) else "unknown"

        print(f"    lambda_MD: {mfp:.6f}")
        print(f"    Free path samples: {n_samples}")
        print(f"    Knudsen number: {kn:.6f}")
        print(f"    Regime: {regime}")

        # Relative energy drift
        initial_energy = energy_history[0]
        final_energy = energy_history[-1]
        if initial_energy != 0:
            rel_drift = abs(final_energy - initial_energy) / abs(initial_energy)
            print(f"\n  Relative energy drift: {rel_drift:.6f} ({rel_drift*100:.4f}%)")

        # Create output directories
        output_dir = Path(__file__).parent.parent / 'outputs'
        plots_dir = output_dir / 'plots'
        data_dir = output_dir / 'data'

        plots_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)

        # Save plots
        print("\nGenerating plots...")

        energy_plot_path = plots_dir / 'energy.png'
        plot_energy(history, energy_plot_path)
        print(f"  Energy plot saved to: {energy_plot_path}")

        trajectories_plot_path = plots_dir / 'trajectories.png'
        plot_trajectories(history, trajectories_plot_path)
        print(f"  Trajectories plot saved to: {trajectories_plot_path}")

        collisions_plot_path = plots_dir / 'collisions.png'
        plot_collision_counts(history, collisions_plot_path)
        print(f"  Collisions plot saved to: {collisions_plot_path}")

        mean_free_path_plot_path = plots_dir / 'mean_free_path.png'
        plot_mean_free_path(history, mean_free_path_plot_path)
        print(f"  Mean free path plot saved to: {mean_free_path_plot_path}")

        free_path_histogram_path = plots_dir / 'free_path_histogram.png'
        plot_free_path_histogram(history, free_path_histogram_path)
        print(f"  Free path histogram saved to: {free_path_histogram_path}")

        flow_history_path = data_dir / 'flow_history.csv'
        flow_df = pd.DataFrame({
            'time': history['time'],
            'mean_vx': history.get('mean_vx', []),
            'mean_vy': history.get('mean_vy', []),
            'temperature': history.get('temperature', []),
        })
        flow_df.to_csv(flow_history_path, index=False)
        print(f"  Flow history saved to: {flow_history_path}")

        mean_vx_plot_path = plots_dir / 'mean_vx_vs_time.png'
        plot_mean_vx_vs_time(history, mean_vx_plot_path)
        print(f"  Mean v_x plot saved to: {mean_vx_plot_path}")

        mean_vy_plot_path = plots_dir / 'mean_vy_vs_time.png'
        plot_mean_vy_vs_time(history, mean_vy_plot_path)
        print(f"  Mean v_y plot saved to: {mean_vy_plot_path}")

        temperature_plot_path = plots_dir / 'temperature_vs_time.png'
        plot_temperature_vs_time(history, temperature_plot_path)
        print(f"  Temperature plot saved to: {temperature_plot_path}")

        # Save velocity profile if available
        y_profile = history.get('velocity_profile_y', [])
        ux_profile = history.get('velocity_profile_ux', [])
        if y_profile and ux_profile:
            vp_csv = data_dir / 'velocity_profile.csv'
            vp_df = pd.DataFrame({'y': y_profile, 'ux': ux_profile})
            vp_df.to_csv(vp_csv, index=False)
            print(f"  Velocity profile saved to: {vp_csv}")

            vp_plot = plots_dir / 'velocity_profile.png'
            import numpy as _np
            plot_velocity_profile(_np.array(y_profile), _np.array(ux_profile), vp_plot)
            print(f"  Velocity profile plot saved to: {vp_plot}")

        # Save history data
        history_data = {
            'time': np.array(history['time']),
            'total_energy': np.array(history['total_energy']),
            'positions_sample': np.array(history['positions_sample']),
            'particle_collisions': np.array(history['particle_collisions']),
            'wall_collisions': np.array(history['wall_collisions']),
            'mean_free_path_history': np.array(history['mean_free_path_history']),
        }

        data_path = data_dir / 'simulation_history.npz'
        np.savez(data_path, **history_data)
        print(f"  History data saved to: {data_path}")

        # Save final state
        final_state = sim.get_current_state()
        final_state_data = {
            'positions': final_state.positions,
            'velocities': final_state.velocities,
            'time': final_state.time
        }

        final_state_path = data_dir / 'final_state.npz'
        np.savez(final_state_path, **final_state_data)
        print(f"  Final state saved to: {final_state_path}")

        print("\nSimulation completed successfully!")
        return history


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run molecular dynamics simulation of hard disks."
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to configuration YAML file (e.g., configs/base_specular.yaml)"
    )

    args = parser.parse_args()

    try:
        run_custom_simulation(args.config)
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during simulation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)