"""
Wall Model Seed Sweep — multi-seed comparison with viscosity and Reynolds estimates.

For each wall model (specular, diffuse_same_speed, diffuse_thermal) and each
random seed, a forced-flow simulation is run.  The script computes:

    - 2D mass density
    - mean free path (lambda_MD)
    - Knudsen number (Kn)
    - kinetic viscosity estimate (eta_kin)
    - Reynolds number estimate (Re)

Results are saved to per-run and summary CSVs, and several diagnostic plots
are generated.

Usage:
    python scripts/run_wall_model_seed_sweep.py
    python scripts/run_wall_model_seed_sweep.py --particles 80 --time 3.0 --height 6.0 --bins 15 --force 0.005 --seeds 1,2,3,4,5
"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.config import SimulationConfig
from src.measurements.energy import calculate_total_energy
from src.measurements.flow import compute_flow_temperature, compute_mean_flow_velocity
from src.measurements.knudsen import classify_knudsen_number, compute_knudsen_number
from src.measurements.viscosity import (
    classify_reynolds_number,
    compute_2d_mass_density,
    compute_mean_thermal_speed_2d,
    compute_reynolds_number,
    estimate_kinetic_viscosity_2d,
)
from src.simulation import Simulation

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OUTPUT_ROOT = ROOT_DIR / "outputs"
DATA_DIR = OUTPUT_ROOT / "data"
PLOTS_DIR = OUTPUT_ROOT / "plots"

RESULTS_CSV = DATA_DIR / "wall_model_seed_sweep_results.csv"
SUMMARY_CSV = DATA_DIR / "wall_model_seed_sweep_summary.csv"

# ---------------------------------------------------------------------------
# Default sweep parameters
# ---------------------------------------------------------------------------
WALL_MODELS: List[str] = ["specular", "diffuse_same_speed", "diffuse_thermal"]
DEFAULT_PARTICLE_COUNTS: List[int] = [80]
DEFAULT_SEEDS: List[int] = [1, 2, 3, 4, 5, 6, 7, 8]


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------


def build_config(
    wall_model: str,
    num_particles: int,
    total_time: float,
    external_force_x: float,
    random_seed: int,
    height: float = 6.0,
    bins: int = 15,
) -> SimulationConfig:
    """
    Build a SimulationConfig for the wall model seed sweep.
    """
    time_step = 0.005
    save_interval = 20
    total_steps = int(total_time / time_step)

    if total_steps < 4 * save_interval:
        equilibration_steps = 0
    else:
        equilibration_steps = min(500, total_steps // 5)

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
        save_vtk=False,
        external_force_x=external_force_x,
        x_boundary_type="periodic",
        velocity_profile_bins=bins,
        random_seed=random_seed,
    )


# ---------------------------------------------------------------------------
# Per-run computation
# ---------------------------------------------------------------------------


def compute_viscosity_reynolds(
    history: Dict[str, Any],
    config: SimulationConfig,
) -> Dict[str, Any]:
    """
    Compute viscosity and Reynolds number estimates from a simulation history.

    Parameters
    ----------
    history : dict
        The history returned by ``Simulation.run()``.
    config : SimulationConfig
        The configuration used for the simulation.

    Returns
    -------
    extra : dict
        Keys: density_2d, temperature_late_mean, mean_thermal_speed,
        kinetic_viscosity, reynolds_number, reynolds_regime, knudsen_number.
    """
    # 2D mass density
    density_2d = compute_2d_mass_density(
        num_particles=config.num_particles,
        mass=config.particle_mass,
        width=config.width,
        height=config.height,
    )

    # Late-time mean temperature
    temperature_history = np.array(history.get("temperature", []), dtype=float)
    valid_temp = temperature_history[np.isfinite(temperature_history)]
    if len(valid_temp) > 0:
        late_fraction = 0.5
        late_temp = valid_temp[int(len(valid_temp) * (1 - late_fraction)):]
        temperature_late_mean = float(np.mean(late_temp)) if len(late_temp) > 0 else float("nan")
    else:
        temperature_late_mean = float("nan")

    # Fallback to temperature_final if no late-time data
    if not np.isfinite(temperature_late_mean):
        temperature_late_mean = history.get("temperature", [float("nan")])[-1]

    # Mean thermal speed
    mean_thermal_speed = compute_mean_thermal_speed_2d(
        temperature=temperature_late_mean,
        mass=config.particle_mass,
    )

    # Mean free path from history
    mean_free_path = history.get("mean_free_path", float("nan"))

    # Kinetic viscosity estimate
    eta_kinetic = estimate_kinetic_viscosity_2d(
        density=density_2d,
        mean_thermal_speed=mean_thermal_speed,
        mean_free_path=mean_free_path,
        coefficient=0.5,
    )

    # Late-time mean flow speed
    mean_vx_history = np.array(history.get("mean_vx", []), dtype=float)
    valid_vx = mean_vx_history[np.isfinite(mean_vx_history)]
    if len(valid_vx) > 0:
        late_vx = valid_vx[int(len(valid_vx) * (1 - 0.5)):]
        mean_vx_late_mean = float(np.mean(late_vx)) if len(late_vx) > 0 else float("nan")
    else:
        mean_vx_late_mean = float("nan")

    flow_speed = abs(mean_vx_late_mean)

    # Characteristic length: channel half-height
    characteristic_length = config.height / 2.0

    # Reynolds number
    reynolds_number = compute_reynolds_number(
        density=density_2d,
        flow_speed=flow_speed,
        characteristic_length=characteristic_length,
        dynamic_viscosity=eta_kinetic,
    )

    reynolds_regime = classify_reynolds_number(reynolds_number)

    # Knudsen number
    knudsen_number = compute_knudsen_number(
        mean_free_path=mean_free_path,
        characteristic_length=characteristic_length,
    )

    return {
        "density_2d": density_2d,
        "temperature_late_mean": temperature_late_mean,
        "mean_thermal_speed": mean_thermal_speed,
        "kinetic_viscosity": eta_kinetic,
        "reynolds_number": reynolds_number,
        "reynolds_regime": reynolds_regime,
        "knudsen_number": knudsen_number,
    }


# ---------------------------------------------------------------------------
# Summary extraction
# ---------------------------------------------------------------------------


def extract_flow_summary(history: Dict[str, Any], wall_model: str) -> Dict[str, Any]:
    """
    Extract a summary dictionary from a simulation history.
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
    hlines: Optional[List[float]] = None,
    hline_labels: Optional[List[str]] = None,
) -> None:
    """Bar chart with optional error bars and horizontal reference lines."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos = np.arange(len(labels))
    colours = ["steelblue", "darkorange", "seagreen"]
    bars = ax.bar(
        x_pos, values,
        color=colours[: len(labels)],
        width=0.5,
        edgecolor="black",
        yerr=[errors[i] if errors and np.isfinite(errors[i]) else 0.0 for i in range(len(labels))] if errors else None,
        capsize=5,
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=15, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(True, alpha=0.3, axis="y")

    # Annotate bars
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

    # Horizontal reference lines
    if hlines:
        for hl, hl_label in zip(hlines, hline_labels or []):
            ax.axhline(y=hl, color="red", linestyle="--", linewidth=1.0, alpha=0.6)
            ax.text(
                x_pos[-1] + 0.5, hl, hl_label,
                va="bottom", ha="left", fontsize=8, color="red", alpha=0.7,
            )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def _plot_scatter(
    x_values: List[float],
    y_values: List[float],
    labels: List[str],
    xlabel: str,
    ylabel: str,
    title: str,
    output_path: Path,
    x_errors: Optional[List[float]] = None,
    y_errors: Optional[List[float]] = None,
    log_x: bool = False,
    log_y: bool = False,
) -> None:
    """Scatter plot with optional error bars and point labels."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    colours = ["steelblue", "darkorange", "seagreen"]

    for i in range(len(x_values)):
        ax.errorbar(
            x_values[i], y_values[i],
            xerr=x_errors[i] if x_errors and np.isfinite(x_errors[i]) else None,
            yerr=y_errors[i] if y_errors and np.isfinite(y_errors[i]) else None,
            fmt="o", color=colours[i % len(colours)], markersize=8, capsize=4,
        )
        ax.annotate(
            labels[i],
            (x_values[i], y_values[i]),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=9,
        )

    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------


def save_results_csv(results: List[Dict[str, Any]], output_path: Path) -> None:
    """Save detailed per-run results to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "wall_model",
        "seed",
        "particle_count",
        "density_2d",
        "temperature_late_mean",
        "mean_thermal_speed",
        "mean_free_path",
        "kinetic_viscosity",
        "reynolds_number",
        "reynolds_regime",
        "knudsen_number",
        "mean_vx_late_mean",
        "mean_vx_late_std",
        "total_particle_collisions",
        "total_wall_collisions",
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved detailed results to {output_path}")


def compute_summary(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compute summary statistics grouped by wall_model."""
    by_wm: Dict[str, List[Dict[str, Any]]] = {}
    for row in results:
        by_wm.setdefault(row["wall_model"], []).append(row)

    summaries: List[Dict[str, Any]] = []
    for wm in sorted(by_wm.keys()):
        rows = by_wm[wm]

        def _mean_std(key: str):
            vals = np.array([r[key] for r in rows if np.isfinite(r.get(key, float("nan")))])
            if len(vals) == 0:
                return float("nan"), float("nan")
            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            return mean, std

        density_mean, _ = _mean_std("density_2d")
        eta_mean, eta_std = _mean_std("kinetic_viscosity")
        re_mean, re_std = _mean_std("reynolds_number")
        kn_mean, kn_std = _mean_std("knudsen_number")
        mfp_mean, mfp_std = _mean_std("mean_free_path")
        vx_mean, vx_std = _mean_std("mean_vx_late_mean")

        # Regime modes
        re_regimes = [r.get("reynolds_regime", "undefined") for r in rows]
        re_regime_mode = Counter(re_regimes).most_common(1)[0][0] if re_regimes else "undefined"

        kn_regimes = []
        for r in rows:
            kn = r.get("knudsen_number", float("nan"))
            kn_regimes.append(classify_knudsen_number(kn) if np.isfinite(kn) else "unknown")
        kn_regime_mode = Counter(kn_regimes).most_common(1)[0][0] if kn_regimes else "unknown"

        summaries.append({
            "wall_model": wm,
            "runs": len(rows),
            "density_2d_mean": density_mean,
            "kinetic_viscosity_mean": eta_mean,
            "kinetic_viscosity_std": eta_std,
            "reynolds_number_mean": re_mean,
            "reynolds_number_std": re_std,
            "knudsen_number_mean": kn_mean,
            "knudsen_number_std": kn_std,
            "mean_free_path_mean": mfp_mean,
            "mean_free_path_std": mfp_std,
            "mean_vx_late_mean_over_seeds": vx_mean,
            "mean_vx_late_std_over_seeds": vx_std,
            "reynolds_regime_mode": re_regime_mode,
            "knudsen_regime_mode": kn_regime_mode,
        })
    return summaries


def save_summary_csv(summaries: List[Dict[str, Any]], output_path: Path) -> None:
    """Save summary statistics to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "wall_model",
        "runs",
        "density_2d_mean",
        "kinetic_viscosity_mean",
        "kinetic_viscosity_std",
        "reynolds_number_mean",
        "reynolds_number_std",
        "knudsen_number_mean",
        "knudsen_number_std",
        "mean_free_path_mean",
        "mean_free_path_std",
        "mean_vx_late_mean_over_seeds",
        "mean_vx_late_std_over_seeds",
        "reynolds_regime_mode",
        "knudsen_regime_mode",
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)
    print(f"Saved summary to {output_path}")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_wall_model_seed_sweep(args: argparse.Namespace) -> None:
    """Run the wall model seed sweep with viscosity and Reynolds estimates."""
    wall_models = WALL_MODELS
    particle_counts = args.particles
    seeds = args.seeds
    total_time = args.time
    external_force_x = args.force
    height = args.height
    bins = args.bins

    print("=" * 70)
    print("Wall Model Seed Sweep — Viscosity and Reynolds Estimates")
    print("=" * 70)
    print()
    print(f"  wall_models       = {wall_models}")
    print(f"  particle_counts   = {particle_counts}")
    print(f"  seeds             = {seeds}")
    print(f"  total_time        = {total_time}")
    print(f"  external_force_x  = {external_force_x}")
    print(f"  height            = {height}")
    print(f"  bins              = {bins}")
    print()

    results: List[Dict[str, Any]] = []
    total_runs = len(wall_models) * len(particle_counts) * len(seeds)
    run_idx = 0

    for wm in wall_models:
        for num_particles in particle_counts:
            for seed in seeds:
                run_idx += 1
                print(f"[{run_idx}/{total_runs}] wall={wm}, N={num_particles}, seed={seed} ...", end=" ")
                sys.stdout.flush()

                config = build_config(
                    wall_model=wm,
                    num_particles=num_particles,
                    total_time=total_time,
                    external_force_x=external_force_x,
                    random_seed=seed,
                    height=height,
                    bins=bins,
                )
                sim = Simulation(config)
                history = sim.run()

                # Extract flow summary
                summary = extract_flow_summary(history, wm)

                # Compute viscosity and Reynolds
                vr = compute_viscosity_reynolds(history, config)

                row: Dict[str, Any] = {
                    "wall_model": wm,
                    "seed": seed,
                    "particle_count": num_particles,
                    "density_2d": vr["density_2d"],
                    "temperature_late_mean": vr["temperature_late_mean"],
                    "mean_thermal_speed": vr["mean_thermal_speed"],
                    "mean_free_path": summary["mean_free_path"],
                    "kinetic_viscosity": vr["kinetic_viscosity"],
                    "reynolds_number": vr["reynolds_number"],
                    "reynolds_regime": vr["reynolds_regime"],
                    "knudsen_number": vr["knudsen_number"],
                    "mean_vx_late_mean": summary["mean_vx_late_mean"],
                    "mean_vx_late_std": summary["mean_vx_late_std"],
                    "total_particle_collisions": summary["total_particle_collisions"],
                    "total_wall_collisions": summary["total_wall_collisions"],
                }
                results.append(row)

                mfp = summary["mean_free_path"]
                eta = vr["kinetic_viscosity"]
                re = vr["reynolds_number"]
                print(f"lambda={mfp:.4f}, eta={eta:.6f}, Re={re:.4f}")

    # ------------------------------------------------------------------
    # Save CSVs
    # ------------------------------------------------------------------
    save_results_csv(results, RESULTS_CSV)
    summaries = compute_summary(results)
    save_summary_csv(summaries, SUMMARY_CSV)

    # ------------------------------------------------------------------
    # Generate plots
    # ------------------------------------------------------------------
    print("\nGenerating plots...")

    # 1. Viscosity bar chart
    wm_labels = [s["wall_model"] for s in summaries]
    eta_means = [s["kinetic_viscosity_mean"] for s in summaries]
    eta_stds = [s["kinetic_viscosity_std"] for s in summaries]
    _plot_bar_chart(
        labels=wm_labels,
        values=eta_means,
        ylabel="Estimated dynamic viscosity eta_kin",
        title="Kinetic viscosity estimate by wall model",
        output_path=PLOTS_DIR / "wall_seed_viscosity_bar.png",
        errors=eta_stds,
    )

    # 2. Reynolds bar chart with reference line at Re = 1
    re_means = [s["reynolds_number_mean"] for s in summaries]
    re_stds = [s["reynolds_number_std"] for s in summaries]
    _plot_bar_chart(
        labels=wm_labels,
        values=re_means,
        ylabel="Estimated Re",
        title="Estimated Reynolds number by wall model",
        output_path=PLOTS_DIR / "wall_seed_reynolds_bar.png",
        errors=re_stds,
        hlines=[1.0],
        hline_labels=["Re = 1"],
    )

    # 3. Knudsen-Reynolds scatter
    kn_means = [s["knudsen_number_mean"] for s in summaries]
    kn_stds = [s["knudsen_number_std"] for s in summaries]
    _plot_scatter(
        x_values=kn_means,
        y_values=re_means,
        labels=wm_labels,
        xlabel="Knudsen number Kn",
        ylabel="Estimated Reynolds number Re",
        title="Knudsen-Reynolds map by wall model",
        output_path=PLOTS_DIR / "wall_seed_kn_re_scatter.png",
        x_errors=kn_stds,
        y_errors=re_stds,
    )

    # 4. Viscosity vs mean free path
    mfp_means = [s["mean_free_path_mean"] for s in summaries]
    mfp_stds = [s["mean_free_path_std"] for s in summaries]
    _plot_scatter(
        x_values=mfp_means,
        y_values=eta_means,
        labels=wm_labels,
        xlabel="Mean free path lambda_MD",
        ylabel="Estimated viscosity eta_kin",
        title="Viscosity estimate vs mean free path",
        output_path=PLOTS_DIR / "wall_seed_viscosity_vs_mean_free_path.png",
        x_errors=mfp_stds,
        y_errors=eta_stds,
    )

    # ------------------------------------------------------------------
    # Console output
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("Viscosity and Reynolds Summary")
    print("=" * 70)
    print()
    print("  eta formula used:")
    print("    eta_kin = C * rho_2D * v_rms * lambda_MD, C = 0.5")
    print()
    print("  Re formula:")
    print("    Re = rho_2D * U * R / eta_kin, R = H / 2")
    print()
    print("  These are qualitative estimates for a 2D hard-disk gas")
    print("  and should not be compared directly with real 3D gas viscosity.")
    print()
    print("  Physical interpretation:")
    print("    - lambda_MD decreases with density.")
    print("    - Kn decreases with density.")
    print("    - eta_kin is an order-of-magnitude transport estimate.")
    print("    - Re depends both on viscosity and on the measured drift speed U.")
    print()

    # Per-model summary table
    print(f"{'Wall Model':<22} {'eta_kin':<14} {'Re':<14} {'Kn':<14} {'Regime':<20}")
    print("-" * 84)
    for s in summaries:
        eta_str = f"{s['kinetic_viscosity_mean']:.6f}" if np.isfinite(s['kinetic_viscosity_mean']) else "nan"
        re_str = f"{s['reynolds_number_mean']:.4f}" if np.isfinite(s['reynolds_number_mean']) else "nan"
        kn_str = f"{s['knudsen_number_mean']:.4f}" if np.isfinite(s['knudsen_number_mean']) else "nan"
        print(
            f"{s['wall_model']:<22} {eta_str:<14} {re_str:<14} {kn_str:<14} {s['reynolds_regime_mode']:<20}"
        )
    print("-" * 84)
    print()

    # Check if all Re values are below 1 (creeping flow regime)
    all_re_below_1 = all(
        np.isfinite(s["reynolds_number_mean"]) and s["reynolds_number_mean"] < 1.0
        for s in summaries
    )
    if all_re_below_1:
        print("  All values in this run are below Re = 1, corresponding to")
        print("  a creeping-flow-like qualitative regime.")
        print()

    print("Wall model seed sweep completed successfully!")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_comma_ints(value: str) -> List[int]:
    """Parse a comma-separated list of integers."""
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wall model seed sweep with viscosity and Reynolds estimates."
    )
    parser.add_argument(
        "--particles", type=parse_comma_ints, default="80",
        help="Comma-separated particle counts (default: 80)",
    )
    parser.add_argument(
        "--seeds", type=parse_comma_ints, default="1,2,3,4,5,6,7,8",
        help="Comma-separated random seeds (default: 1,2,3,4,5,6,7,8)",
    )
    parser.add_argument(
        "--time", type=float, default=3.0,
        help="Total simulation time (default: 3.0)",
    )
    parser.add_argument(
        "--force", type=float, default=0.005,
        help="External force along x (default: 0.005)",
    )
    parser.add_argument(
        "--height", type=float, default=6.0,
        help="Channel height (default: 6.0)",
    )
    parser.add_argument(
        "--bins", type=int, default=15,
        help="Number of velocity profile bins (default: 15)",
    )
    args = parser.parse_args()
    run_wall_model_seed_sweep(args)


if __name__ == "__main__":
    main()