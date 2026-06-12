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

#### Version 0.3 — Mean Free Path (λ_MD)
- **λ_MD**: the average distance a particle travels between successive inter-molecular collisions
- Wall collisions do **not** reset the free path — only particle-particle collisions reset it
- If a particle participates in multiple collisions in the same time step, its free path is recorded only once
- The mean free path is computed as the running average over all collected samples

### Output Plots
- `outputs/plots/energy.png` — Energy conservation over time
- `outputs/plots/trajectories.png` — Tracked particle trajectories
- `outputs/plots/collisions.png` — Particle-particle and particle-wall collision counts over time
- `outputs/plots/mean_free_path.png` — λ_MD(t) evolution
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
```

## Requirements
- Python 3.12+
- numpy, matplotlib, pyvista, vtk, pytest, pyyaml
