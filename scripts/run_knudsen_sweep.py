import csv
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.config import SimulationConfig
from src.measurements.knudsen import classify_knudsen_number, compute_knudsen_number
from src.simulation import Simulation

OUTPUT_ROOT = Path(__file__).parent.parent / "outputs"
PLOTS_DIR = OUTPUT_ROOT / "plots"
DATA_DIR = OUTPUT_ROOT / "data"
CSV_PATH = DATA_DIR / "sweep_results.csv"
LAMBDA_PLOT_PATH = PLOTS_DIR / "lambda_vs_N.png"
KN_PLOT_PATH = PLOTS_DIR / "kn_vs_N.png"

PARTICLE_COUNTS = [100, 200, 400, 800]


def build_simulation_config(num_particles: int) -> SimulationConfig:
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
    )


def run_sweep() -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for num_particles in PARTICLE_COUNTS:
        print(f"Running sweep simulation for N={num_particles}...")
        config = build_simulation_config(num_particles)
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
        initial_energy = energy_history[0]
        final_energy = energy_history[-1]
        energy_drift = (
            abs(final_energy - initial_energy) / abs(initial_energy)
            if initial_energy != 0.0
            else float("nan")
        )

        results.append({
            "N": num_particles,
            "mean_free_path": mean_free_path,
            "Kn": kn,
            "regime": regime,
            "particle_collisions": history["total_particle_collisions"],
            "wall_collisions": history["total_wall_collisions"],
            "energy_drift": energy_drift,
        })
    return results


def save_sweep_csv(results: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "N",
                "mean_free_path",
                "Kn",
                "regime",
                "particle_collisions",
                "wall_collisions",
                "energy_drift",
            ],
        )
        writer.writeheader()
        writer.writerows(results)


def plot_sweep_metric(
    results: list[dict[str, Any]],
    metric: str,
    output_path: Path,
    y_label: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    N_values = [row["N"] for row in results]
    values = [row[metric] for row in results]

    plt.figure(figsize=(7, 4))
    plt.plot(N_values, values, marker="o")
    plt.xlabel("Number of particles N")
    plt.ylabel(y_label)
    plt.title(f"{y_label} vs Number of particles")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    results = run_sweep()
    save_sweep_csv(results, CSV_PATH)
    print(f"Saved sweep results to {CSV_PATH}")

    plot_sweep_metric(
        results,
        metric="mean_free_path",
        output_path=LAMBDA_PLOT_PATH,
        y_label="Mean free path lambda",
    )
    print(f"Saved mean free path plot to {LAMBDA_PLOT_PATH}")

    plot_sweep_metric(
        results,
        metric="Kn",
        output_path=KN_PLOT_PATH,
        y_label="Knudsen number",
    )
    print(f"Saved Knudsen number plot to {KN_PLOT_PATH}")


if __name__ == "__main__":
    main()
