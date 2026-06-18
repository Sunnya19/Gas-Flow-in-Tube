"""
Visualization functions for molecular dynamics results.
"""
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import matplotlib.pyplot as plt


def plot_energy(history: Dict[str, List[Any]], output_path: Optional[Path] = None):
    """
    Plot total energy over time.

    Args:
        history: Simulation history dictionary
        output_path: Path to save the plot (if None, display interactively)
    """
    time = np.array(history['time'])
    energy = np.array(history['total_energy'])

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(time, energy, 'b-', linewidth=2, label='Total Energy')
    ax.axhline(y=energy[0], color='r', linestyle='--', alpha=0.7,
               label=f'Initial: {energy[0]:.4f}')

    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Total Energy', fontsize=12)
    ax.set_title('Energy Conservation in Molecular Dynamics', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    # Add text box with energy conservation metrics
    if len(energy) > 1:
        initial_energy = energy[0]
        final_energy = energy[-1]
        relative_error = abs(final_energy - initial_energy) / initial_energy

        text_str = (
            f'Initial: {initial_energy:.4f}\n'
            f'Final: {final_energy:.4f}\n'
            f'Relative error: {relative_error:.2%}'
        )
        ax.text(0.02, 0.98, text_str, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_trajectories(history: Dict[str, List[Any]], output_path: Optional[Path] = None):
    """
    Plot trajectories of tracked particles.

    Args:
        history: Simulation history dictionary
        output_path: Path to save the plot (if None, display interactively)
    """
    positions_sample = np.array(history['positions_sample'])  # shape: (time, tracked, 2)
    time = np.array(history['time'])

    if positions_sample.size == 0:
        print("No trajectory data to plot")
        return

    num_tracked = positions_sample.shape[1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: All trajectories in 2D space
    ax1 = axes[0]
    for i in range(num_tracked):
        x = positions_sample[:, i, 0]
        y = positions_sample[:, i, 1]
        ax1.plot(x, y, '-', alpha=0.7, linewidth=1.5, label=f'Particle {i}')

    # Mark start and end points
    for i in range(num_tracked):
        start_x = positions_sample[0, i, 0]
        start_y = positions_sample[0, i, 1]
        end_x = positions_sample[-1, i, 0]
        end_y = positions_sample[-1, i, 1]

        ax1.plot(start_x, start_y, 'go', markersize=8, label='Start' if i == 0 else "")
        ax1.plot(end_x, end_y, 'ro', markersize=8, label='End' if i == 0 else "")

    ax1.set_xlabel('X position', fontsize=12)
    ax1.set_ylabel('Y position', fontsize=12)
    ax1.set_title(f'Trajectories of {num_tracked} Tracked Particles', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10, loc='upper right')
    ax1.set_aspect('equal', adjustable='box')

    # Plot 2: X and Y positions vs time for first few particles
    ax2 = axes[1]
    num_to_plot = min(3, num_tracked)

    for i in range(num_to_plot):
        x = positions_sample[:, i, 0]
        y = positions_sample[:, i, 1]

        ax2.plot(time, x, '-', linewidth=2, alpha=0.8, label=f'Particle {i} X')
        ax2.plot(time, y, '--', linewidth=2, alpha=0.8, label=f'Particle {i} Y')

    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylabel('Position', fontsize=12)
    ax2.set_title('Position vs Time for Selected Particles', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_snapshot(positions: np.ndarray, velocities: np.ndarray,
                  channel_width: float, channel_height: float,
                  particle_radius: float, output_path: Optional[Path] = None):
    """
    Plot a snapshot of the system at a given time.

    Args:
        positions: Particle positions (N, 2)
        velocities: Particle velocities (N, 2)
        channel_width: Width of the channel
        channel_height: Height of the channel
        particle_radius: Particle radius
        output_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw channel boundaries
    ax.plot([0, channel_width, channel_width, 0, 0],
            [0, 0, channel_height, channel_height, 0],
            'k-', linewidth=2)

    # Draw particles as circles
    for i, (x, y) in enumerate(positions):
        circle = plt.Circle((x, y), particle_radius,
                            facecolor='blue', edgecolor='black',
                            alpha=0.7, linewidth=1)
        ax.add_patch(circle)

        # Draw velocity vector
        vx, vy = velocities[i]
        speed = np.sqrt(vx**2 + vy**2)
        if speed > 0:
            scale = 0.5  # Scale factor for visibility
            ax.arrow(x, y, vx*scale, vy*scale,
                     head_width=particle_radius*0.5, head_length=particle_radius*0.7,
                     fc='red', ec='red', alpha=0.8)

    ax.set_xlim(-particle_radius, channel_width + particle_radius)
    ax.set_ylim(-particle_radius, channel_height + particle_radius)
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title(f'System Snapshot (N={len(positions)})', fontsize=14)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)

    # Add statistics
    avg_speed = np.mean(np.sqrt(np.sum(velocities**2, axis=1)))
    text_str = f'Particles: {len(positions)}\nAvg speed: {avg_speed:.3f}'
    ax.text(0.02, 0.98, text_str, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_collision_counts(history: Dict[str, List[Any]], output_path: Optional[Path] = None):
    time = np.array(history['time'])
    particle_collisions = np.array(history['particle_collisions'])
    wall_collisions = np.array(history['wall_collisions'])

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(time, particle_collisions, 'b-', linewidth=1.5, label='Particle-Particle')
    ax.plot(time, wall_collisions, 'r-', linewidth=1.5, label='Particle-Wall')

    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Collisions per save interval', fontsize=12)
    ax.set_title('Collision Counts per Save Interval', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    total_particle = history.get('total_particle_collisions', 0)
    total_wall = history.get('total_wall_collisions', 0)
    text_str = f'Total particle collisions: {total_particle}\nTotal wall collisions: {total_wall}'
    ax.text(0.02, 0.98, text_str, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_mean_free_path(history: Dict[str, List[Any]], output_path: Optional[Path] = None):
    time = np.array(history['time'])
    mfp_history = np.array(history['mean_free_path_history'], dtype=float)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Mask NaN values (pre-measurement points)
    mask = ~np.isnan(mfp_history)
    if np.any(mask):
        ax.plot(time[mask], mfp_history[mask], 'g-', linewidth=2,
                label=r'Cumulative $\lambda_{MD}(t)$')
    else:
        ax.text(0.5, 0.5, 'No measurement data', transform=ax.transAxes,
                ha='center', va='center', fontsize=14)

    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel(r'Cumulative mean free path $\lambda_{MD}$', fontsize=12)
    ax.set_title('Mean Free Path vs Time', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    final_mfp = history.get('mean_free_path', float("nan"))
    n_samples = len(history.get('free_path_samples', []))
    text_str = f'Final λ_MD: {final_mfp:.4f}\nSamples: {n_samples}'
    ax.text(0.02, 0.98, text_str, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_velocity_profile(y_centers: np.ndarray, ux_profile: np.ndarray, output_path: Optional[Path] = None):
    """
    Plot u_x(y) profile: x-axis u_x, y-axis y
    """
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.plot(ux_profile, y_centers, '-o')
    ax.set_xlabel(r'$u_x$')
    ax.set_ylabel('y')
    ax.set_title('Velocity profile $u_x(y)$')
    ax.grid(True)
    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_mean_vx_vs_time(history: Dict[str, List[Any]], output_path: Optional[Path] = None):
    time = np.array(history['time'])
    mean_vx = np.array(history.get('mean_vx', []), dtype=float)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time, mean_vx, 'b-o', linewidth=2, markersize=4)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Mean $v_x$', fontsize=12)
    ax.set_title('Mean Flow Velocity $v_x$ vs Time', fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_mean_vy_vs_time(history: Dict[str, List[Any]], output_path: Optional[Path] = None):
    time = np.array(history['time'])
    mean_vy = np.array(history.get('mean_vy', []), dtype=float)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time, mean_vy, 'g-o', linewidth=2, markersize=4)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Mean $v_y$', fontsize=12)
    ax.set_title('Mean Flow Velocity $v_y$ vs Time', fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_temperature_vs_time(history: Dict[str, List[Any]], output_path: Optional[Path] = None):
    time = np.array(history['time'])
    temperature = np.array(history.get('temperature', []), dtype=float)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time, temperature, 'r-o', linewidth=2, markersize=4)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Temperature', fontsize=12)
    ax.set_title('Flow Temperature vs Time', fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_free_path_histogram(history: Dict[str, List[Any]], output_path: Optional[Path] = None):
    samples = history.get('free_path_samples', [])

    if not samples:
        print("No free path samples to plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(samples, bins=50, density=True, alpha=0.7, color='green', edgecolor='black')

    ax.set_xlabel('Free path length', fontsize=12)
    ax.set_ylabel('Probability density', fontsize=12)
    ax.set_title('Distribution of Free Paths Between Particle Collisions', fontsize=14)
    ax.grid(True, alpha=0.3)

    mean_val = np.mean(samples)
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
               label=f'Mean: {mean_val:.4f}')
    ax.legend(fontsize=11)

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close()
    else:
        plt.show()
