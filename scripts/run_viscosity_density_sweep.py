"""
Viscosity Density Sweep — how mean free path, Kn, eta_kin, and Re change with N.

For each particle count and each seed, a forced-flow simulation is run with
the ``diffuse_same_speed`` wall model.  Results are saved to per-run and
summary CSVs, and several diagnostic plots are generated.

Usage:
    python scripts/run_viscosity_density_sweep.py
    python scripts/run_viscosity_density_sweep.py --particles 80,120,160,240 --seeds 1,2,3 --time 3.0 --height 6.0 --force 0.005

    For quick smoke tests use:
        python scripts/run_viscosity_density_sweep.py --particles 40,60 --seeds 1,2 --time 0.5

    For report-quality plots use:
        python scripts/run_viscosity_density_sweep.py --particles 80,120,160,240 --seeds 1,2,3 --time 3.0 --height 6.0 --force 0.005
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.config import SimulationConfig
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

RESULTS_CSV = DATA_DIR / "viscosity_density_sweep_results.csv"
SUMMARY_CSV = DATA_DIR / "viscosity_density_sweep_summary.csv"

# ---------------------------------------------------------------------------
# Default sweep parameters
# ---------------------------------------------------------------------------
DEFAULT_WALL_MODEL = "diffuse_same_speed"
DEFAULT_PARTICLE_COUNTS: List[int] = [80, 120, 160, 240]
DEFAULT_SEEDS: List[int] = [1, 2, 3]


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------


def build_config(
    num_particles: int,
    total_time: float,
    external_force_x: float,
    random_seed: int,
    height: float = 6.0,
    bins: int = 15,
    wall_model: str = DEFAULT_WALL_MODEL,
) -> SimulationConfig:
    """
    Build a SimulationConfig for the viscosity density sweep.
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


def run_single_simulation(config: SimulationConfig) -> Dict[str, Any]:
    """
    Run a single simulation and compute all quantities of interest.
    """
    sim = Simulation(config)
    history = sim.run()

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

    if not np.isfinite(temperature_late_mean):
        temperature_late_mean = history.get("temperature", [float("nan")])[-1]

    # Mean thermal speed
    mean_thermal_speed = compute_mean_thermal_speed_2d(
        temperature=temperature_late_mean,
        mass=config.particle_mass,
    )

    # Mean free path
    mean_free_path = history.get("mean_free_path", float("nan"))

    # Characteristic length: channel half-height
    characteristic_length = config.height / 2.0

    # Knudsen number
    knudsen_number = compute_knudsen_number(
        mean_free_path=mean_free_path,
        characteristic_length=characteristic_length,
    )
    knudsen_regime = classify_knudsen_number(knudsen_number) if np.isfinite(knudsen_number) else "unknown"

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

    # Reynolds number
    reynolds_number = compute_reynolds_number(
        density=density_2d,
        flow_speed=flow_speed,
        characteristic_length=characteristic_length,
        dynamic_viscosity=eta_kinetic,
    )
    reynolds_regime = classify_reynolds_number(reynolds_number)

    return {
        "particle_count": config.num_particles,
        "seed": config.random_seed,
        "density_2d": density_2d,
        "temperature_late_mean": temperature_late_mean,
        "mean_thermal_speed": mean_thermal_speed,
        "mean_free_path": mean_free_path,
        "knudsen_number": knudsen_number,
        "knudsen_regime": knudsen_regime,
        "kinetic_viscosity": eta_kinetic,
        "reynolds_number": reynolds_number,
        "reynolds_regime": reynolds_regime,
        "mean_vx_late_mean": mean_vx_late_mean,
        "total_particle_collisions": history.get("total_particle_collisions", 0),
        "total_wall_collisions": history.get("total_wall_collisions", 0),
    }


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------


def save_results_csv(results: List[Dict[str, Any]], output_path: Path) -> None:
    """Save detailed per-run results to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "particle_count",
        "seed",
        "density_2d",
        "temperature_late_mean",
        "mean_thermal_speed",
        "mean_free_path",
        "knudsen_number",
        "knudsen_regime",
        "kinetic_viscosity",
        "reynolds_number",
        "reynolds_regime",
        "mean_vx_late_mean",
        "total_particle_collisions",
        "total_wall_collisions",
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved detailed results to {output_path}")


def _is_valid_row(row: Dict[str, Any]) -> bool:
    """Check if a row has finite values for all required viscosity/Re fields."""
    required_keys = [
        "mean_free_path",
        "kinetic_viscosity",
        "reynolds_number",
        "knudsen_number",
    ]
    for key in required_keys:
        val = row.get(key, float("nan"))
        if not np.isfinite(val):
            return False
    return True


def compute_summary(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compute summary statistics grouped by particle_count.

    For each particle count, the summary includes:
    - total_runs, valid_runs, invalid_runs, valid_fraction
    - means and stds computed over valid rows only.
    """
    by_n: Dict[int, List[Dict[str, Any]]] = {}
    for row in results:
        by_n.setdefault(row["particle_count"], []).append(row)

    summaries: List[Dict[str, Any]] = []
    for n in sorted(by_n.keys()):
        rows = by_n[n]
        total_runs = len(rows)

        # Separate valid and invalid rows
        valid_rows = [r for r in rows if _is_valid_row(r)]
        valid_runs = len(valid_rows)
        invalid_runs = total_runs - valid_runs
        valid_fraction = valid_runs / total_runs if total_runs > 0 else 0.0

        def _mean_std(key: str):
            vals = np.array([r[key] for r in valid_rows if np.isfinite(r.get(key, float("nan")))])
            if len(vals) == 0:
                return float("nan"), float("nan")
            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            return mean, std

        density_mean, _ = _mean_std("density_2d")
        mfp_mean, mfp_std = _mean_std("mean_free_path")
        kn_mean, kn_std = _mean_std("knudsen_number")
        eta_mean, eta_std = _mean_std("kinetic_viscosity")
        re_mean, re_std = _mean_std("reynolds_number")
        vx_mean, vx_std = _mean_std("mean_vx_late_mean")

        summaries.append({
            "particle_count": n,
            "runs": total_runs,
            "valid_runs": valid_runs,
            "invalid_runs": invalid_runs,
            "valid_fraction": valid_fraction,
            "density_2d_mean": density_mean,
            "mean_free_path_mean": mfp_mean,
            "mean_free_path_std": mfp_std,
            "knudsen_number_mean": kn_mean,
            "knudsen_number_std": kn_std,
            "kinetic_viscosity_mean": eta_mean,
            "kinetic_viscosity_std": eta_std,
            "reynolds_number_mean": re_mean,
            "reynolds_number_std": re_std,
            "mean_vx_late_mean_over_seeds": vx_mean,
            "mean_vx_late_std_over_seeds": vx_std,
        })
    return summaries


def save_summary_csv(summaries: List[Dict[str, Any]], output_path: Path) -> None:
    """Save summary statistics to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "particle_count",
        "runs",
        "valid_runs",
        "invalid_runs",
        "valid_fraction",
        "density_2d_mean",
        "mean_free_path_mean",
        "mean_free_path_std",
        "knudsen_number_mean",
        "knudsen_number_std",
        "kinetic_viscosity_mean",
        "kinetic_viscosity_std",
        "reynolds_number_mean",
        "reynolds_number_std",
        "mean_vx_late_mean_over_seeds",
        "mean_vx_late_std_over_seeds",
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)
    print(f"Saved summary to {output_path}")


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _plot_with_errors(
    x_values: List[float],
    y_values: List[float],
    xlabel: str,
    ylabel: str,
    title: str,
    output_path: Path,
    y_errors: Optional[List[float]] = None,
    x_errors: Optional[List[float]] = None,
    log_x: bool = False,
    log_y: bool = False,
    connect_line: bool = True,
) -> None:
    """Line/scatter plot with optional error bars.

    Parameters
    ----------
    connect_line : bool
        If True, points are connected with a line (fmt='-o').
        If False, only markers are shown (fmt='o').
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    fmt = "-o" if connect_line else "o"
    ax.errorbar(
        x_values, y_values,
        xerr=x_errors if x_errors else None,
        yerr=y_errors if y_errors else None,
        fmt=fmt, color="steelblue", markersize=8, capsize=4,
        ecolor="gray", elinewidth=1.5,
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


def _plot_scatter_labeled(
    x_values: List[float],
    y_values: List[float],
    labels: List[str],
    xlabel: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    """Scatter plot with point labels."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x_values, y_values, color="steelblue", s=60, zorder=5)

    for x, y, label in zip(x_values, y_values, labels):
        ax.annotate(
            f"N={label}",
            (x, y),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=9,
        )

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
# Main runner
# ---------------------------------------------------------------------------


def run_viscosity_density_sweep(args: argparse.Namespace) -> None:
    """Run the viscosity density sweep."""
    particle_counts = args.particles
    seeds = args.seeds
    total_time = args.time
    external_force_x = args.force
    height = args.height
    wall_model = args.wall_model

    print("=" * 70)
    print("Viscosity Density Sweep — diffuse_same_speed wall model")
    print("=" * 70)
    print()
    print(f"  wall_model        = {wall_model}")
    print(f"  particle_counts   = {particle_counts}")
    print(f"  seeds             = {seeds}")
    print(f"  total_time        = {total_time}")
    print(f"  external_force_x  = {external_force_x}")
    print(f"  height            = {height}")
    print()

    results: List[Dict[str, Any]] = []
    total_runs = len(particle_counts) * len(seeds)
    run_idx = 0

    for num_particles in particle_counts:
        for seed in seeds:
            run_idx += 1
            print(f"[{run_idx}/{total_runs}] N={num_particles}, seed={seed} ...", end=" ")
            sys.stdout.flush()

            config = build_config(
                num_particles=num_particles,
                total_time=total_time,
                external_force_x=external_force_x,
                random_seed=seed,
                height=height,
                wall_model=wall_model,
            )
            result = run_single_simulation(config)
            results.append(result)

            mfp = result["mean_free_path"]
            eta = result["kinetic_viscosity"]
            re = result["reynolds_number"]
            collisions = result["total_particle_collisions"]

            if not np.isfinite(mfp):
                print(f"lambda=NaN, eta={eta:.6f}, Re={re:.4f}")
                print(
                    f"  WARNING: mean_free_path is NaN for N={num_particles}, seed={seed}.\n"
                    f"  This run had too few molecular collisions for viscosity/Re estimate."
                )
            else:
                print(f"lambda={mfp:.4f}, eta={eta:.6f}, Re={re:.4f}")

            if collisions < 10:
                print(
                    f"  WARNING: total_particle_collisions={collisions} for N={num_particles}, seed={seed}.\n"
                    f"  Very few collisions — results may be unreliable."
                )

    # ------------------------------------------------------------------
    # Save CSVs
    # ------------------------------------------------------------------
    save_results_csv(results, RESULTS_CSV)
    summaries = compute_summary(results)
    save_summary_csv(summaries, SUMMARY_CSV)

    # ------------------------------------------------------------------
    # Console warnings for low valid_fraction
    # ------------------------------------------------------------------
    for s in summaries:
        if s["valid_fraction"] < 0.5:
            print(
                f"  WARNING: N={s['particle_count']} has low valid_fraction={s['valid_fraction']:.3f}.\n"
                f"  Increase total_time, particle_count, or number of seeds."
            )

    # ------------------------------------------------------------------
    # Generate plots
    # ------------------------------------------------------------------
    print("\nGenerating plots...")

    # Filter summary points: only include those with valid_runs >= 2
    valid_summaries = [s for s in summaries if s["valid_runs"] >= 2]
    has_sufficient_points = len(valid_summaries) >= 2

    if len(valid_summaries) == 0:
        print("  No summary points with valid_runs >= 2. Skipping plots.")
        print()
        print("Viscosity density sweep completed successfully!")
        print()
        return

    density_means = [s["density_2d_mean"] for s in valid_summaries]
    mfp_means = [s["mean_free_path_mean"] for s in valid_summaries]
    mfp_stds = [s["mean_free_path_std"] for s in valid_summaries]
    kn_means = [s["knudsen_number_mean"] for s in valid_summaries]
    kn_stds = [s["knudsen_number_std"] for s in valid_summaries]
    eta_means = [s["kinetic_viscosity_mean"] for s in valid_summaries]
    eta_stds = [s["kinetic_viscosity_std"] for s in valid_summaries]
    re_means = [s["reynolds_number_mean"] for s in valid_summaries]
    re_stds = [s["reynolds_number_std"] for s in valid_summaries]
    n_labels = [str(s["particle_count"]) for s in valid_summaries]

    # 1. lambda vs density
    _plot_with_errors(
        x_values=density_means,
        y_values=mfp_means,
        xlabel="2D mass density rho_2D",
        ylabel="Mean free path lambda_MD",
        title="Mean free path vs density",
        output_path=PLOTS_DIR / "viscosity_density_lambda_vs_density.png",
        y_errors=mfp_stds,
        connect_line=has_sufficient_points,
    )

    # 2. eta vs density
    _plot_with_errors(
        x_values=density_means,
        y_values=eta_means,
        xlabel="2D mass density rho_2D",
        ylabel="Estimated viscosity eta_kin",
        title="Viscosity estimate vs density",
        output_path=PLOTS_DIR / "viscosity_density_eta_vs_density.png",
        y_errors=eta_stds,
        connect_line=has_sufficient_points,
    )

    # 3. Kn vs density
    _plot_with_errors(
        x_values=density_means,
        y_values=kn_means,
        xlabel="2D mass density rho_2D",
        ylabel="Knudsen number Kn",
        title="Knudsen number vs density",
        output_path=PLOTS_DIR / "viscosity_density_kn_vs_density.png",
        y_errors=kn_stds,
        connect_line=has_sufficient_points,
    )

    # 4. Re vs density
    _plot_with_errors(
        x_values=density_means,
        y_values=re_means,
        xlabel="2D mass density rho_2D",
        ylabel="Estimated Reynolds number Re",
        title="Reynolds number vs density",
        output_path=PLOTS_DIR / "viscosity_density_re_vs_density.png",
        y_errors=re_stds,
        connect_line=has_sufficient_points,
    )

    # 5. Kn-Re map
    _plot_scatter_labeled(
        x_values=kn_means,
        y_values=re_means,
        labels=n_labels,
        xlabel="Knudsen number Kn",
        ylabel="Estimated Reynolds number Re",
        title="Knudsen-Reynolds map by particle count",
        output_path=PLOTS_DIR / "viscosity_density_kn_re_map.png",
    )

    # ------------------------------------------------------------------
    # Console output
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("Viscosity Density Sweep Summary")
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

    # Per-N summary table
    print(f"{'N':<8} {'valid':<8} {'total':<8} {'vfrac':<8} {'rho_2D':<12} {'eta_kin':<14} {'Re':<14} {'Kn':<14}")
    print("-" * 86)
    for s in summaries:
        rho_str = f"{s['density_2d_mean']:.4f}" if np.isfinite(s['density_2d_mean']) else "nan"
        eta_str = f"{s['kinetic_viscosity_mean']:.6f}" if np.isfinite(s['kinetic_viscosity_mean']) else "nan"
        re_str = f"{s['reynolds_number_mean']:.4f}" if np.isfinite(s['reynolds_number_mean']) else "nan"
        kn_str = f"{s['knudsen_number_mean']:.4f}" if np.isfinite(s['knudsen_number_mean']) else "nan"
        print(
            f"{s['particle_count']:<8} "
            f"{s['valid_runs']:<8} "
            f"{s['runs']:<8} "
            f"{s['valid_fraction']:<8.3f} "
            f"{rho_str:<12} "
            f"{eta_str:<14} "
            f"{re_str:<14} "
            f"{kn_str:<14}"
        )
    print("-" * 86)
    print()

    print("Viscosity density sweep completed successfully!")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_comma_ints(value: str) -> List[int]:
    """Parse a comma-separated list of integers."""
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Viscosity density sweep — how eta_kin and Re change with N.",
        epilog=(
            "Examples:\n"
            "  Quick smoke test:\n"
            "    python scripts/run_viscosity_density_sweep.py --particles 40,60 --seeds 1,2 --time 0.5\n\n"
            "  Report-quality plots:\n"
            "    python scripts/run_viscosity_density_sweep.py --particles 80,120,160,240 --seeds 1,2,3 --time 3.0 --height 6.0 --force 0.005"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--wall-model", type=str, default=DEFAULT_WALL_MODEL,
        help=f"Wall model to use (default: {DEFAULT_WALL_MODEL})",
    )
    parser.add_argument(
        "--particles", type=parse_comma_ints, default="80,120,160,240",
        help="Comma-separated particle counts (default: 80,120,160,240)",
    )
    parser.add_argument(
        "--seeds", type=parse_comma_ints, default="1,2,3",
        help="Comma-separated random seeds (default: 1,2,3)",
    )
    parser.add_argument(
        "--time", type=float, default=3.0,
        help="Total simulation time (default: 3.0)",
    )
    parser.add_argument(
        "--height", type=float, default=6.0,
        help="Channel height (default: 6.0)",
    )
    parser.add_argument(
        "--force", type=float, default=0.005,
        help="External force along x (default: 0.005)",
    )
    args = parser.parse_args()
    run_viscosity_density_sweep(args)


if __name__ == "__main__":
    main()