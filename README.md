# Gas-Flow-in-Tube

Thermodynamics project — molecular dynamics simulation of 2D hard-disk gas.

## Project Structure

```
Gas-Flow-in-Tube/
├── configs/              # YAML configuration files
├── scripts/
│   └── run_base.py       # Main simulation runner
├── src/
│   ├── config.py         # SimulationConfig dataclass
│   ├── simulation.py     # Main Simulation class
│   ├── state.py          # SystemState dataclass
│   ├── initialization.py # Initial position/velocity generation
│   ├── dynamics/
│   │   ├── integrator.py         # Euler time integration
│   │   ├── particle_collisions.py # Particle-particle collisions
│   │   └── wall_collisions.py    # Particle-wall collisions
│   ├── geometry/
│   │   └── channel.py    # RectangularChannel
│   ├── measurements/
│   │   ├── energy.py            # Energy calculation & conservation
│   │   ├── collision_stats.py   # Collision statistics (v0.2)
│   │   └── mean_free_path.py    # Mean free path measurement (v0.3)
│   ├── visualization/
│   │   └── plots.py      # Plotting functions
│   ├── io/
│   │   └── vtk_writer.py # VTK/ParaView export
│   └── wall_models/
│       ├── base.py       # Abstract WallModel
│       └── specular.py   # Specular reflection wall model
├── tests/                # Unit tests
├── outputs/              # Generated plots, data, VTK files
└── requirements.txt
```

## Features

### Physics Model
- 2D hard-disk molecular dynamics
- Elastic collisions between particles (momentum and energy conserving)
- Specular reflection at walls (energy conserving)
- Euler integration of equations of motion
- Maxwell-Boltzmann initial velocity distribution

### Measurements

#### Version 0.2 — Collision Statistics
- **Particle-particle collisions**: each inter-molecular collision is counted and recorded per time step
- **Particle-wall collisions**: each boundary crossing is counted as a separate collision (if a particle crosses a corner, both x and y boundary crossings are counted)
- Collision counts are stored in `CollisionStats` dataclass and exported to history
- The collision history stores **accumulated counts per save interval** (sum of collisions over all steps between saves), not per-step values

#### Version 0.3 — Mean Free Path (λ_MD)
- **λ_MD**: the average distance a particle travels between successive inter-molecular collisions
- Wall collisions do **not** reset the free path — only particle-particle collisions reset it
- If a particle participates in multiple collisions in the same time step, its free path is recorded only once
- The mean free path is computed as the running (cumulative) average over all collected samples

#### Version 0.4 — Corrected Free Path Measurement
- **First-flight exclusion**: the first free path of each particle (from simulation start to its first collision) is **not** recorded. Only paths between two consecutive particle-particle collisions are counted. This prevents artificially small λ_MD values at the beginning of the simulation.
- **`has_previous_particle_collision`**: a per-particle boolean array tracks whether a particle has already experienced at least one inter-molecular collision. Free path samples are only recorded for particles with `has_previous_particle_collision=True`.
- **Equilibration / burn-in**: the `equilibration_steps` config parameter allows running a number of steps before measurements begin. At the end of equilibration, all free path tracking state is reset, ensuring clean measurement data.
- **`update_free_path_measurements()`**: the free path update logic is extracted into a standalone function for testability.

### Output Plots
- `outputs/plots/energy.png` — Energy conservation over time
- `outputs/plots/trajectories.png` — Tracked particle trajectories
- `outputs/plots/collisions.png` — Particle-particle and particle-wall collision counts per save interval
- `outputs/plots/mean_free_path.png` — Cumulative λ_MD(t) evolution (NaN values before measurements begin are masked)
- `outputs/plots/free_path_histogram.png` — Distribution of free path lengths

### VTK Export
- Particle positions and velocities can be exported as `.vtp` files for visualization in ParaView
- A `.pvd` collection file is created for easy loading of time series

## Usage

### Run Simulation
```bash
cd Gas-Flow-in-Tube
python scripts/run_base.py
```

### Run with Custom Config
```bash
python scripts/run_base.py --config configs/base_specular.yaml
```

### Run Tests
```bash
cd Gas-Flow-in-Tube
python -m pytest tests/ -v
```

Or run individual test files:
```bash
python tests/test_collision_stats.py
python tests/test_mean_free_path.py
python tests/test_simulation_measurements.py
python tests/test_energy_conservation.py
python tests/test_particle_collision.py
python tests/test_wall_collision.py
python tests/test_vtk_writer.py
python tests/test_knudsen.py
python tests/test_reproducibility.py
python tests/test_flow.py
```

### Flow Simulation

A forced flow simulation in a 2D channel with periodic x-boundaries and specular walls.

**How to run:**
```bash
python scripts/run_flow.py
```

With custom parameters:
```bash
python scripts/run_flow.py --force 0.005 --particles 200 --time 20.0 --temperature 1.0
```

**What it does:**
- Periodic boundary along x (particles wrap around left/right edges)
- Specular walls along y (elastic reflection at top/bottom)
- Constant external force `external_force_x` along x, mimicking a pressure gradient
- Measures mean flow velocity `⟨v_x⟩(t)`, `⟨v_y⟩(t)`, temperature `T(t)`, total energy `E(t)`
- Computes velocity profile `u_x(y)` at the final state

**Output plots:**
- `outputs/plots/flow_mean_velocity.png` — `⟨v_x⟩(t)` and `⟨v_y⟩(t)` on the same axes
- `outputs/plots/flow_temperature.png` — temperature vs time
- `outputs/plots/flow_energy.png` — total energy vs time
- `outputs/plots/velocity_profile.png` — `u_x(y)` profile

**Physical notes:**
- With `external_force_x > 0`, `⟨v_x⟩` becomes positive (gas flows along x)
- `⟨v_y⟩` remains near zero (no force in y-direction)
- Temperature and energy may increase because there is no thermostat in this version
- On specular walls, the velocity profile `u_x(y)` is nearly flat because specular walls do not impose a no-slip condition
- This is expected physical behaviour for a first-order flow model

**CLI arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--force` | 0.005 | External force along x |
| `--particles` | 200 | Number of particles |
| `--time` | 20.0 | Total simulation time |
| `--temperature` | 1.0 | Initial temperature |
| `--save-vtk` | (off) | Enable VTK export |
| `--no-vtk` | (default) | Disable VTK export |

### Reproducible Knudsen Sweep

A sweep over particle counts with multiple random seeds for reproducible results.

**How to run:**
```bash
python scripts/run_reproducible_knudsen_sweep.py
```

**Output files:**
- `outputs/data/sweep_results.csv` — detailed per-run results (N, seed, lambda, Kn, regime, collisions, energy drift)
- `outputs/data/sweep_summary.csv` — summary statistics per N (mean ± std over seeds)
- `outputs/plots/lambda_vs_N.png` — mean free path vs N with error bars
- `outputs/plots/kn_vs_N.png` — Knudsen number vs N with error bars and Kn=1 line

**Physical meaning:**
- As N increases, density increases, mean free path decreases, and Kn decreases.
- The Kn=1 line on the Kn plot marks the boundary between transition and continuum/slip regimes.

**Reproducibility:**
- Each run uses `random_seed` in `SimulationConfig` for deterministic initialization.
- Two runs with the same seed produce identical initial positions and velocities.

## Requirements
- Python 3.12+
- numpy, matplotlib, pyvista, vtk, pytest, pyyaml
