"""
Wall Model Comparison Script
=============================

Compares forced-flow gas dynamics in a 2D channel under three different
wall boundary conditions:

    - specular
    - diffuse_same_speed
    - diffuse_thermal

Physical context:
-----------------
* specular wall:
    Does not change the tangential (streamwise) velocity upon collision with
    a horizontal wall.  The flow can therefore remain nearly plug-like with
    minimal wall drag.

* diffuse_same_speed wall:
    Randomises the direction of the reflected velocity while preserving its
    magnitude.  This introduces stronger wall drag and modifies the velocity
    profile compared to the specular case.

* diffuse_thermal wall:
    Re-thermalises the reflected particle to a Maxwellian at the prescribed
    wall temperature.  This can stabilise the gas temperature better than
    the specular wall and provides the strongest coupling between the gas
    and the wall.

Goal:
-----
Show that, under the same external_force_x, different wall boundary
conditions produce measurably different flow fields, velocity profiles,
and energy/temperature histories.

Usage:
------
    python scripts/run_wall_model_comparison.py
    python scripts/run_wall_model_comparison.py --force 0.01 --particles 150 --time 10.0
    python scripts/run_wall_model_comparison.py --particles 50 --time 0.5 --seed 1  # quick test

For more stable physical profiles use:
    python scripts/run_wall_model_comparison.py --particles 150 --time 10.0 --force 0.005 --seed 1
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.config import SimulationConfig
from src.measurements.energy import calculate_total_energy
from src.measurements.flow import compute_flow_temperature, compute_mean_flow_velocity
from src.measurements.velocity_profile import compute_velocity_profile
from src.simulation import Simulation
from src.visualization.plots import (
    plot_time_series_comparison,
    plot_velocity_profile_comparison,
)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def extract_flow_summary(history: Dict[str, Any], wall_model: str) -> Dict[str, Any]:
    """
    Extract a summary dictionary from a simulation history.

    Parameters
    ----------
    history : dict
        The history returned by ``Simulation.run()``.
    wall_model : str
        One of "specular", "diffuse_same_speed", "diffuse_thermal".

    Returns
    -------
    summary : dict
        Keys matching the CSV columns required by the comparison script.
    """
    energy_history = history["total_energy"]
    initial_energy = float(energy_history[0]) if energy_history else float("nan")
    final_energy = float(energy_history[-1]) if energy_history else float("nan")

    relative_energy_change = (
        (final_energy - initial_energy) / abs(initial_energy)
        if initial_energy != 0.0
        else float("nan")
    )

    mean_vx_history = np.array(history.get("mean_vx", []), dtype=float)
    mean_vy_history = np.array(history.get("mean_vy", []), dtype=float)
    temperature_history = np.array(history.get("temperature", []), dtype=float)

    valid_vx = mean_vx_history[np.isfinite(mean_vx_history)]
    valid_vy = mean_vy_history[np.isfinite(mean_vy_history)]
    valid_temp = temperature_history[np.isfinite(temperature_history)]

    # Late-time statistics: use last 50% of valid data
    late_fraction = 0.5
    if len(valid_vx) > 0:
        late_vx = valid_vx[int(len(valid_vx) * (1 - late_fraction)):]
        mean_vx_late_mean = float(np.mean(late_vx))
        mean_vx_late_std = float(np.std(late_vx, ddof=1)) if len(late_vx) > 1 else float("nan")
    else:
        mean_vx_late_mean = float("nan")
        mean_vx_late_std = float("nan")

    return {
        "wall_model": wall_model,
        "mean_vx_final": float(valid_vx[-1]) if len(valid_vx) > 0 else float("nan"),
        "mean_vx_late_mean": mean_vx_late_mean,
        "mean_vx_late_std": mean_vx_late_std,
        "mean_vy_final": float(valid_vy[-1]) if len(valid_vy) > 0 else float("nan"),
        "temperature_initial": float(valid_temp[0]) if len(valid_temp) > 0 else float("nan"),
        "temperature_final": float(valid_temp[-1]) if len(valid_temp) > 0 else float("nan"),
        "energy_initial": initial_energy,
        "energy_final": final_energy,
        "relative_energy_change": relative_energy_change,
        "total_particle_collisions": history.get("total_particle_collisions", 0),
        "total_wall_collisions": history.get("total_wall_collisions", 0),
        "specular_wall_collisions": history.get("specular_wall_collisions", 0),
        "diffuse_wall_collisions": history.get("diffuse_wall_collisions", 0),
        "thermal_wall_collisions": history.get("thermal_wall_collisions", 0),
        "mean_free_path": history.get("mean_free_path", float("nan")),
        "free_path_samples": len(history.get("free_path_samples", [])),
    }


def save_velocity_profile_csv(
    y_centers: np.ndarray,
    ux_profile: np.ndarray,
    output_path: Path,
) -> None:
    """
    Save a velocity profile to a CSV file with columns ``y`` and ``ux``.

    Parameters
    ----------
    y_centers : np.ndarray
        Bin centre y-coordinates.
    ux_profile : np.ndarray
        Mean streamwise velocity in each bin.
    output_path : Path
        Destination CSV path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Build CSV manually to avoid pandas dependency in this helper
    lines = ["y,ux"]
    for y, ux in zip(y_centers, ux_profile):
        ux_str = "nan" if np.isnan(ux) else f"{ux:.10e}"
        lines.append(f"{y:.10e},{ux_str}")
    output_path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------


def build_config(
    wall_model: str,
    num_particles: int,
    total_time: float,
    external_force_x: float,
    random_seed: int,
    save_vtk: bool,
    height: float = 15.0,
    bins: int = 15,
) -> SimulationConfig:
    """
    Build a SimulationConfig for the wall model comparison.

    Parameters
    ----------
    wall_model : str
        One of "specular", "diffuse_same_speed", "diffuse_thermal".
    num_particles : int
    total_time : float
    external_force_x : float
    random_seed : int
    save_vtk : bool
    height : float
        Channel height.
    bins : int
        Number of velocity profile bins.

    Returns
    -------
    SimulationConfig
    """
    time_step = 0.005
    save_interval = 20
    total_steps = int(total_time / time_step)

    # Adaptive equilibration: don't waste steps on short runs
    if total_steps < 4 * save_interval:
        equilibration_steps = 0
    else:
        equilibration_steps = min(500, total_steps // 5)

    measurement_time_start = equilibration_steps * time_step

    print(f"  total_steps           = {total_steps}")
    print(f"  equilibration_steps   = {equilibration_steps}")
    print(f"  measurement_time_start = {measurement_time_start:.3f}")
    print()

    return SimulationConfig(
        width=20.0,
        height=height,
        num_particles=num_particles,
        particle_radius=0.1,
        particle_mass=1.0,
        initial_temperature=1.0,
        time_step=time_step,
        total_time=total_time,
        save_interval=save_interval,
        equilibration_steps=equilibration_steps,
        num_trajectory_particles=5,
        wall_model=wall_model,
        wall_model_type=wall_model,
        wall_temperature=1.0,
        save_vtk=save_vtk,
        save_vtk_every=50,
        vtk_output_dir="outputs/vtk_wall_compare",
        external_force_x=external_force_x,
        x_boundary_type="periodic",
        velocity_profile_bins=bins,
        random_seed=random_seed,
    )


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _plot_bar_chart(
    labels: List[str],
    values: List[float],
    ylabel: str,
    title: str,
    output_path: Path,
    errors: Optional[List[float]] = None,
) -> None:
    """Simple bar chart using matplotlib (no seaborn), with optional error bars."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos = np.arange(len(labels))
    colours = ["steelblue", "darkorange", "seagreen"]
    bars = ax.bar(x_pos, values, color=colours[: len(labels)], width=0.5, edgecolor="black",
                  yerr=[errors[i] if errors and np.isfinite(errors[i]) else 0.0 for i in range(len(labels))] if errors else None,
                  capsize=5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=15, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(True, alpha=0.3, axis="y")

    # Annotate bars with values
    for bar, val in zip(bars, values):
        if np.isfinite(val):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main comparison runner
# ---------------------------------------------------------------------------


def run_wall_model_comparison(args: argparse.Namespace) -> None:
    """
    Run forced-flow simulations for all three wall models and produce
    comparison outputs (CSV data + plots).
    """
    wall_models = ["specular", "diffuse_same_speed", "diffuse_thermal"]

    print("=" * 70)
    print("Wall Model Comparison — Forced Flow in a 2D Channel")
    print("=" * 70)
    print()
    print("Physical context:")
    print("  specular          — tangential velocity preserved at wall")
    print("  diffuse_same_speed — direction randomised, speed preserved")
    print("  diffuse_thermal   — re-thermalised to wall temperature")
    print()
    print(f"  external_force_x = {args.force}")
    print(f"  num_particles    = {args.particles}")
    print(f"  total_time       = {args.time}")
    print(f"  random_seed      = {args.seed}")
    print(f"  save_vtk         = {args.save_vtk}")
    print()

    # Storage for per-model results
    histories: Dict[str, Dict[str, Any]] = {}
    summaries: List[Dict[str, Any]] = []
    velocity_profiles: Dict[str, tuple] = {}

    for wm in wall_models:
        print(f"--- Running wall model: {wm} ---")
        config = build_config(
            wall_model=wm,
            num_particles=args.particles,
            total_time=args.time,
            external_force_x=args.force,
            random_seed=args.seed,
            save_vtk=args.save_vtk,
            height=args.height,
            bins=args.bins,
        )
        sim = Simulation(config)
        history = sim.run()
        histories[wm] = history

        # Extract summary
        summary = extract_flow_summary(history, wm)
        summaries.append(summary)

        # Print per-model summary
        print(f"  mean_vx late-time mean: {summary['mean_vx_late_mean']:.6f}")
        print(f"  mean_vx late-time std:  {summary['mean_vx_late_std']:.6f}")
        print(f"  mean_vx final:          {summary['mean_vx_final']:.6f}")
        print(f"  mean_vy final:          {summary['mean_vy_final']:.6f}")
        print(f"  temperature initial:    {summary['temperature_initial']:.6f}")
        print(f"  temperature final:      {summary['temperature_final']:.6f}")
        print(f"  energy initial:         {summary['energy_initial']:.6f}")
        print(f"  energy final:           {summary['energy_final']:.6f}")
        print(f"  relative energy change: {summary['relative_energy_change']:.6f}")
        print(f"  total particle collisions: {summary['total_particle_collisions']}")
        print(f"  total wall collisions:     {summary['total_wall_collisions']}")
        print(f"  specular wall collisions:  {summary['specular_wall_collisions']}")
        print(f"  diffuse wall collisions:   {summary['diffuse_wall_collisions']}")
        print(f"  thermal wall collisions:   {summary['thermal_wall_collisions']}")
        print(f"  mean free path:         {summary['mean_free_path']:.6f}")
        print()

        # --- Profile diagnostics (Task 4) ---
        profile_counts = history.get("velocity_profile_counts", [])
        ux_profile = history.get("velocity_profile_ux", [])
        finite_bins = sum(1 for v in ux_profile if np.isfinite(v))
        total_profile_samples = int(sum(profile_counts))
        print(f"  profile finite bins: {finite_bins} / {len(ux_profile) if hasattr(ux_profile, '__len__') else 0}")
        print(f"  profile samples:     {total_profile_samples}")
        if finite_bins < 3 or total_profile_samples == 0:
            print("  WARNING: velocity profile has too little data. Increase --time or reduce equilibration.")
        total_wc = summary["total_wall_collisions"]
        if total_wc < 20:
            print(f"  WARNING: very few wall collisions ({total_wc}); wall model comparison may be noisy.")
        print()

        # Extract velocity profile from history
        y_profile = history.get("velocity_profile_y", [])
        ux_profile = history.get("velocity_profile_ux", [])
        if y_profile and ux_profile:
            velocity_profiles[wm] = (np.array(y_profile), np.array(ux_profile))
        else:
            velocity_profiles[wm] = (np.array([]), np.array([]))

    # ------------------------------------------------------------------
    # Save comparison CSV
    # ------------------------------------------------------------------
    data_dir = ROOT_DIR / "outputs" / "data"
    plots_dir = ROOT_DIR / "outputs" / "plots"
    data_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / "wall_model_comparison.csv"
    field_names = [
        "wall_model",
        "mean_vx_final",
        "mean_vx_late_mean",
        "mean_vx_late_std",
        "mean_vy_final",
        "temperature_initial",
        "temperature_final",
        "energy_initial",
        "energy_final",
        "relative_energy_change",
        "total_particle_collisions",
        "total_wall_collisions",
        "specular_wall_collisions",
        "diffuse_wall_collisions",
        "thermal_wall_collisions",
        "mean_free_path",
        "free_path_samples",
    ]
    lines = [",".join(field_names)]
    for s in summaries:
        row = ",".join(str(s.get(f, "")) for f in field_names)
        lines.append(row)
    csv_path.write_text("\n".join(lines) + "\n")
    print(f"Saved comparison CSV: {csv_path}")

    # ------------------------------------------------------------------
    # Save per-model velocity profile CSVs
    # ------------------------------------------------------------------
    for wm in wall_models:
        yp, uxp = velocity_profiles[wm]
        if len(yp) > 0:
            vp_path = data_dir / f"velocity_profile_{wm}.csv"
            save_velocity_profile_csv(yp, uxp, vp_path)
            print(f"Saved velocity profile: {vp_path}")

    # ------------------------------------------------------------------
    # Generate comparison plots
    # ------------------------------------------------------------------
    print("\nGenerating comparison plots...")

    # 1. mean_vx(t) for all three models
    time_arrays = []
    vx_series = []
    for wm in wall_models:
        t = np.array(histories[wm]["time"])
        vx = np.array(histories[wm].get("mean_vx", []), dtype=float)
        time_arrays.append(t)
        vx_series.append(vx)
    # Use the shortest time array for plotting
    min_len = min(len(t) for t in time_arrays)
    common_time = time_arrays[0][:min_len]
    vx_common = [vx[:min_len] for vx in vx_series]

    plot_time_series_comparison(
        common_time,
        np.array(vx_common),
        labels=wall_models,
        xlabel="Time",
        ylabel=r"$\langle v_x \rangle$",
        title="Mean Streamwise Velocity vs Time",
        output_path=plots_dir / "wall_models_mean_vx.png",
    )
    print(f"  Saved: {plots_dir / 'wall_models_mean_vx.png'}")

    # 2. Temperature(t) for all three models with T_wall reference line
    temp_series = []
    for wm in wall_models:
        temp = np.array(histories[wm].get("temperature", []), dtype=float)
        temp_series.append(temp[:min_len])

    # Use plot_time_series_comparison but add T_wall line via a custom plot
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    for line, label in zip(temp_series, wall_models):
        ax.plot(common_time, line, linewidth=2, label=label)
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label=r"$T_{\mathrm{wall}}$")
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Flow temperature", fontsize=12)
    ax.set_title("Flow Temperature vs Time", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    temp_path = plots_dir / "wall_models_temperature.png"
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(temp_path, dpi=150)
    plt.close()
    print(f"  Saved: {temp_path}")

    # 3. Relative energy change plot: 100 * (E(t) - E(0)) / E(0)
    fig, ax = plt.subplots(figsize=(10, 6))
    for wm in wall_models:
        e = np.array(histories[wm]["total_energy"][:min_len])
        e0 = e[0] if len(e) > 0 and e[0] != 0.0 else 1.0
        relative_change = 100.0 * (e - e0) / abs(e0)
        ax.plot(common_time, relative_change, linewidth=2, label=wm)
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Relative energy change, %", fontsize=12)
    ax.set_title("Relative total energy change vs time", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    energy_path = plots_dir / "wall_models_energy.png"
    energy_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(energy_path, dpi=150)
    plt.close()
    print(f"  Saved: {energy_path}")

    # 4. Velocity profile ux(y) for all three models
    vp_labels = []
    vp_profiles = []
    for wm in wall_models:
        yp, uxp = velocity_profiles[wm]
        if len(yp) > 0:
            vp_labels.append(wm)
            vp_profiles.append(uxp)

    if vp_profiles:
        # Use the y-centers from the first available profile
        ref_y = velocity_profiles[wall_models[0]][0]
        plot_velocity_profile_comparison(
            ref_y,
            np.array(vp_profiles),
            labels=vp_labels,
            output_path=plots_dir / "wall_models_velocity_profile.png",
        )
        print(f"  Saved: {plots_dir / 'wall_models_velocity_profile.png'}")

    # 5. Bar chart: late-time mean_vx for each model with error bars
    late_mean_values = [s["mean_vx_late_mean"] for s in summaries]
    late_std_values = [s["mean_vx_late_std"] for s in summaries]
    _plot_bar_chart(
        labels=wall_models,
        values=late_mean_values,
        ylabel=r"Late-time $\langle v_x \rangle$",
        title="Late-time Mean Streamwise Velocity by Wall Model",
        output_path=plots_dir / "wall_models_final_mean_vx_bar.png",
        errors=late_std_values,
    )
    # Also save under the more descriptive name
    _plot_bar_chart(
        labels=wall_models,
        values=late_mean_values,
        ylabel=r"Late-time $\langle v_x \rangle$",
        title="Late-time Mean Streamwise Velocity by Wall Model",
        output_path=plots_dir / "wall_models_late_mean_vx_bar.png",
        errors=late_std_values,
    )

    # ------------------------------------------------------------------
    # Final summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Comparison Summary")
    print("=" * 70)
    print(f"{'Wall Model':<22} {'Late <v_x>':<14} {'Final T':<14} {'Energy change':<14}")
    print("-" * 70)
    for s in summaries:
        ec = s["relative_energy_change"]
        ec_str = f"{ec:.4f}" if np.isfinite(ec) else "nan"
        lv = s["mean_vx_late_mean"]
        lv_str = f"{lv:.6f}" if np.isfinite(lv) else "nan"
        print(
            f"{s['wall_model']:<22} {lv_str:<14} "
            f"{s['temperature_final']:<14.6f} {ec_str:<14}"
        )
    print("-" * 70)
    print()
    print("Interpretation:")
    print("  - Specular walls preserve tangential velocity; this often leads to")
    print("    weaker wall drag, but finite-time noisy runs should be interpreted")
    print("    using late-time averages.")
    print("  - Diffuse_same_speed walls randomise outgoing direction while preserving")
    print("    speed, changing the drift and velocity profile.")
    print("  - Diffuse_thermal walls also exchange energy with a wall-temperature")
    print("    reservoir, so temperature and total energy can change.")
    print()
    print("Wall model comparison completed successfully!")
    print()
    print("Recommended informative run:")
    print("  python scripts/run_wall_model_comparison.py --particles 80 --time 3.0 --height 6.0 --bins 15 --force 0.005 --seed 1 --no-vtk")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare wall boundary conditions in forced 2D channel flow."
    )
    parser.add_argument(
        "--force", type=float, default=0.005,
        help="External force along x (default: 0.005)",
    )
    parser.add_argument(
        "--particles", type=int, default=100,
        help="Number of particles (default: 100)",
    )
    parser.add_argument(
        "--time", type=float, default=3.0,
        help="Total simulation time (default: 3.0)",
    )
    parser.add_argument(
        "--seed", type=int, default=1,
        help="Random seed for reproducibility (default: 1)",
    )
    parser.add_argument(
        "--height", type=float, default=15.0,
        help="Channel height (default: 15.0). For clearer wall-model differences, try --height 6.0.",
    )
    parser.add_argument(
        "--bins", type=int, default=15,
        help="Number of velocity profile bins (default: 15).",
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
    run_wall_model_comparison(args)


if __name__ == "__main__":
    main()