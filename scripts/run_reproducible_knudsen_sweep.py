"""
Reproducible Knudsen sweep over particle counts with multiple random seeds.

For each (N, seed) pair, a simulation is run and results are saved to CSV.
Summary statistics and plots with error bars are generated.

Usage:
    python scripts/run_reproducible_knudsen_sweep.py
"""

import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.config import SimulationConfig
from src.measurements.energy import calculate_total_energy
from src.measurements.knudsen import classify_knudsen_number, compute_knudsen_number
from src.simulation import Simulation
from src.visualization.plots import plot_sweep_kn_vs_N, plot_sweep_lambda_vs_N

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OUTPUT_ROOT = ROOT_DIR / "outputs"
DATA_DIR = OUTPUT_ROOT / "data"
PLOTS_DIR = OUTPUT_ROOT / "plots"

RESULTS_CSV = DATA_DIR / "sweep_results.csv"
SUMMARY_CSV = DATA_DIR / "sweep_summary.csv"
LAMBDA_PLOT = PLOTS_DIR / "lambda_vs_N.png"
KN_PLOT = PLOTS_DIR / "kn_vs_N.png"

# ---------------------------------------------------------------------------
# Sweep parameters
# ---------------------------------------------------------------------------
PARTICLE_COUNTS: List[int] = [100, 200, 400, 800]
SEEDS: List[int] = [1, 2, 3, 4, 5]


def build_config(num_particles: int, seed: int) -> SimulationConfig:
    """Build a SimulationConfig for the given particle count and seed."""
    return SimulationConfig(
        width=20.0,
        height=15.0,
        num_particles=num_particles,
        particle_radius=0.1,
        particle_mass=1.0,
        initial_temperature=1.0,
        time_step=0.005,
        total_time=10.0,
        save_interval=20,
        equilibration_steps=500,
        num_trajectory_particles=5,
        wall_model_type="specular",
        save_vtk=False,
        random_seed=seed,
    )


def run_single_simulation(config: SimulationConfig) -> Dict[str, Any]:
    """Run a single simulation and extract results."""
    sim = Simulation(config)
    history = sim.run()

    mean_free_path = history["mean_free_path"]
    characteristic_length = config.height / 2.0
    kn = compute_knudsen_number(
        mean_free_path=mean_free_path,
        characteristic_length=characteristic_length,
    )
    regime = classify_knudsen_number(kn) if np.isfinite(kn) else "unknown"

    energy_history = history["total_energy"]
    initial_energy = float(energy_history[0])
    final_energy = float(energy_history[-1])
    relative_energy_drift = (
        abs(final_energy - initial_energy) / abs(initial_energy)
        if initial_energy != 0.0
        else float("nan")
    )

    return {
        "mean_free_path": mean_free_path,
        "knudsen_number": kn,
        "regime": regime,
        "total_particle_collisions": history["total_particle_collisions"],
        "total_wall_collisions": history["total_wall_collisions"],
        "free_path_samples": len(history.get("free_path_samples", [])),
        "initial_energy": initial_energy,
        "final_energy": final_energy,
        "relative_energy_drift": relative_energy_drift,
    }


def run_sweep() -> List[Dict[str, Any]]:
    """Run the full sweep over all (N, seed) combinations."""
    results: List[Dict[str, Any]] = []
    total_runs = len(PARTICLE_COUNTS) * len(SEEDS)
    run_idx = 0

    for num_particles in PARTICLE_COUNTS:
        for seed in SEEDS:
            run_idx += 1
            print(f"[{run_idx}/{total_runs}] N={num_particles}, seed={seed} ...", end=" ")
            sys.stdout.flush()

            config = build_config(num_particles, seed)
            sim_result = run_single_simulation(config)

            row: Dict[str, Any] = {
                "particle_count": num_particles,
                "seed": seed,
                "mean_free_path": sim_result["mean_free_path"],
                "knudsen_number": sim_result["knudsen_number"],
                "regime": sim_result["regime"],
                "total_particle_collisions": sim_result["total_particle_collisions"],
                "total_wall_collisions": sim_result["total_wall_collisions"],
                "free_path_samples": sim_result["free_path_samples"],
                "initial_energy": sim_result["initial_energy"],
                "final_energy": sim_result["final_energy"],
                "relative_energy_drift": sim_result["relative_energy_drift"],
            }
            results.append(row)
            print(f"lambda={sim_result['mean_free_path']:.4f}, Kn={sim_result['knudsen_number']:.4f}")

    return results


def save_results_csv(results: List[Dict[str, Any]], output_path: Path) -> None:
    """Save detailed per-run results to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "particle_count",
        "seed",
        "mean_free_path",
        "knudsen_number",
        "regime",
        "total_particle_collisions",
        "total_wall_collisions",
        "free_path_samples",
        "initial_energy",
        "final_energy",
        "relative_energy_drift",
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved detailed results to {output_path}")


def compute_summary(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compute summary statistics grouped by particle_count."""
    by_n: Dict[int, List[Dict[str, Any]]] = {}
    for row in results:
        by_n.setdefault(row["particle_count"], []).append(row)

    summaries: List[Dict[str, Any]] = []
    for n in sorted(by_n.keys()):
        rows = by_n[n]
        mfp_values = np.array([r["mean_free_path"] for r in rows if np.isfinite(r["mean_free_path"])])
        kn_values = np.array([r["knudsen_number"] for r in rows if np.isfinite(r["knudsen_number"])])
        drift_values = np.array([r["relative_energy_drift"] for r in rows if np.isfinite(r["relative_energy_drift"])])

        # Regime mode (most frequent)
        regimes = [r["regime"] for r in rows]
        regime_mode = Counter(regimes).most_common(1)[0][0] if regimes else "unknown"

        summaries.append({
            "particle_count": n,
            "runs": len(rows),
            "mean_free_path_mean": float(np.mean(mfp_values)) if len(mfp_values) > 0 else float("nan"),
            "mean_free_path_std": float(np.std(mfp_values, ddof=1)) if len(mfp_values) > 1 else 0.0,
            "knudsen_number_mean": float(np.mean(kn_values)) if len(kn_values) > 0 else float("nan"),
            "knudsen_number_std": float(np.std(kn_values, ddof=1)) if len(kn_values) > 1 else 0.0,
            "energy_drift_mean": float(np.mean(drift_values)) if len(drift_values) > 0 else float("nan"),
            "energy_drift_std": float(np.std(drift_values, ddof=1)) if len(drift_values) > 1 else 0.0,
            "regime_mode": regime_mode,
        })
    return summaries


def save_summary_csv(summaries: List[Dict[str, Any]], output_path: Path) -> None:
    """Save summary statistics to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "particle_count",
        "runs",
        "mean_free_path_mean",
        "mean_free_path_std",
        "knudsen_number_mean",
        "knudsen_number_std",
        "energy_drift_mean",
        "energy_drift_std",
        "regime_mode",
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)
    print(f"Saved summary to {output_path}")


def print_summary_table(summaries: List[Dict[str, Any]]) -> None:
    """Print a human-readable summary table to the console."""
    print("\n" + "=" * 80)
    print(f"{'N':>6} | {'lambda_mean ± lambda_std':>28} | {'Kn_mean ± Kn_std':>24} | {'regime':>16}")
    print("-" * 80)
    for s in summaries:
        n = s["particle_count"]
        lm = s["mean_free_path_mean"]
        ls = s["mean_free_path_std"]
        km = s["knudsen_number_mean"]
        ks = s["knudsen_number_std"]
        regime = s["regime_mode"]
        if np.isfinite(lm):
            lambda_str = f"{lm:.4f} ± {ls:.4f}"
        else:
            lambda_str = "nan"
        if np.isfinite(km):
            kn_str = f"{km:.4f} ± {ks:.4f}"
        else:
            kn_str = "nan"
        print(f"{n:>6} | {lambda_str:>28} | {kn_str:>24} | {regime:>16}")
    print("=" * 80 + "\n")


def plot_results(summaries: List[Dict[str, Any]]) -> None:
    """Generate lambda_vs_N and kn_vs_N plots with error bars."""
    n_vals = [s["particle_count"] for s in summaries]
    lambda_means = np.array([s["mean_free_path_mean"] for s in summaries])
    lambda_stds = np.array([s["mean_free_path_std"] for s in summaries])
    kn_means = np.array([s["knudsen_number_mean"] for s in summaries])
    kn_stds = np.array([s["knudsen_number_std"] for s in summaries])

    plot_sweep_lambda_vs_N(n_vals, lambda_means, lambda_stds, LAMBDA_PLOT)
    print(f"Saved lambda plot to {LAMBDA_PLOT}")

    plot_sweep_kn_vs_N(n_vals, kn_means, kn_stds, KN_PLOT)
    print(f"Saved Kn plot to {KN_PLOT}")


def main() -> None:
    print("=" * 60)
    print("Reproducible Knudsen sweep")
    print("=" * 60)
    print(f"Particle counts: {PARTICLE_COUNTS}")
    print(f"Seeds: {SEEDS}")
    print(f"Total runs: {len(PARTICLE_COUNTS) * len(SEEDS)}")
    print()

    results = run_sweep()
    save_results_csv(results, RESULTS_CSV)

    summaries = compute_summary(results)
    save_summary_csv(summaries, SUMMARY_CSV)

    print_summary_table(summaries)
    plot_results(summaries)

    print("Sweep completed successfully!")


if __name__ == "__main__":
    main()