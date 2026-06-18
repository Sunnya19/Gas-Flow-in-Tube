"""
Tests for the viscosity density sweep script.

These tests verify the helper functions (parse_comma_ints, build_config,
compute_summary, NaN handling) without running any full simulation.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.config import SimulationConfig
from scripts.run_viscosity_density_sweep import (
    _is_valid_row,
    build_config,
    compute_summary,
    parse_comma_ints,
    run_single_simulation,
)


# ---------------------------------------------------------------------------
# parse_comma_ints
# ---------------------------------------------------------------------------


class TestParseCommaInts:
    def test_basic(self):
        assert parse_comma_ints("1,2,3") == [1, 2, 3]

    def test_with_spaces(self):
        assert parse_comma_ints("80, 120, 160") == [80, 120, 160]

    def test_single_value(self):
        assert parse_comma_ints("42") == [42]

    def test_empty_after_strip(self):
        assert parse_comma_ints("") == []


# ---------------------------------------------------------------------------
# build_config
# ---------------------------------------------------------------------------


class TestBuildConfig:
    def test_basic_config(self):
        config = build_config(
            num_particles=80,
            total_time=0.1,
            external_force_x=0.005,
            random_seed=1,
        )
        assert isinstance(config, SimulationConfig)
        assert config.num_particles == 80
        assert config.external_force_x == 0.005
        assert config.random_seed == 1
        assert config.wall_model == "diffuse_same_speed"
        assert config.x_boundary_type == "periodic"
        assert config.save_vtk is False

    def test_custom_wall_model(self):
        config = build_config(
            num_particles=80,
            total_time=0.1,
            external_force_x=0.005,
            random_seed=1,
            wall_model="specular",
        )
        assert config.wall_model == "specular"

    def test_custom_height(self):
        config = build_config(
            num_particles=80,
            total_time=0.1,
            external_force_x=0.005,
            random_seed=1,
            height=10.0,
        )
        assert config.height == 10.0


# ---------------------------------------------------------------------------
# _is_valid_row
# ---------------------------------------------------------------------------


class TestIsValidRow:
    def test_valid_row(self):
        row = {
            "mean_free_path": 0.5,
            "kinetic_viscosity": 0.8,
            "reynolds_number": 5.0,
            "knudsen_number": 0.167,
        }
        assert _is_valid_row(row) is True

    def test_nan_mean_free_path(self):
        row = {
            "mean_free_path": float("nan"),
            "kinetic_viscosity": 0.8,
            "reynolds_number": 5.0,
            "knudsen_number": 0.167,
        }
        assert _is_valid_row(row) is False

    def test_nan_kinetic_viscosity(self):
        row = {
            "mean_free_path": 0.5,
            "kinetic_viscosity": float("nan"),
            "reynolds_number": 5.0,
            "knudsen_number": 0.167,
        }
        assert _is_valid_row(row) is False

    def test_nan_reynolds_number(self):
        row = {
            "mean_free_path": 0.5,
            "kinetic_viscosity": 0.8,
            "reynolds_number": float("nan"),
            "knudsen_number": 0.167,
        }
        assert _is_valid_row(row) is False

    def test_nan_knudsen_number(self):
        row = {
            "mean_free_path": 0.5,
            "kinetic_viscosity": 0.8,
            "reynolds_number": 5.0,
            "knudsen_number": float("nan"),
        }
        assert _is_valid_row(row) is False

    def test_missing_key(self):
        row = {
            "mean_free_path": 0.5,
            "kinetic_viscosity": 0.8,
            "reynolds_number": 5.0,
            # knudsen_number missing
        }
        assert _is_valid_row(row) is False


# ---------------------------------------------------------------------------
# compute_summary
# ---------------------------------------------------------------------------


def _make_fake_results() -> List[Dict[str, Any]]:
    """Create fake per-run results for testing compute_summary."""
    return [
        {
            "particle_count": 80,
            "seed": 1,
            "density_2d": 4.0,
            "mean_free_path": 0.5,
            "knudsen_number": 0.167,
            "kinetic_viscosity": 0.8,
            "reynolds_number": 5.0,
            "mean_vx_late_mean": 0.2,
        },
        {
            "particle_count": 80,
            "seed": 2,
            "density_2d": 4.0,
            "mean_free_path": 0.6,
            "knudsen_number": 0.2,
            "kinetic_viscosity": 0.9,
            "reynolds_number": 4.5,
            "mean_vx_late_mean": 0.18,
        },
        {
            "particle_count": 120,
            "seed": 1,
            "density_2d": 6.0,
            "mean_free_path": 0.3,
            "knudsen_number": 0.1,
            "kinetic_viscosity": 1.2,
            "reynolds_number": 3.0,
            "mean_vx_late_mean": 0.15,
        },
    ]


class TestComputeSummary:
    def test_grouped_by_particle_count(self):
        results = _make_fake_results()
        summaries = compute_summary(results)
        assert len(summaries) == 2  # 80 and 120

        # Check N=80 summary
        s80 = [s for s in summaries if s["particle_count"] == 80][0]
        assert s80["runs"] == 2
        assert s80["density_2d_mean"] == pytest.approx(4.0)
        assert s80["mean_free_path_mean"] == pytest.approx(0.55)
        assert s80["kinetic_viscosity_mean"] == pytest.approx(0.85)
        assert s80["reynolds_number_mean"] == pytest.approx(4.75)

        # Check N=120 summary
        s120 = [s for s in summaries if s["particle_count"] == 120][0]
        assert s120["runs"] == 1
        assert s120["density_2d_mean"] == pytest.approx(6.0)

    def test_nan_handling(self):
        results = [
            {
                "particle_count": 80,
                "seed": 1,
                "density_2d": 4.0,
                "mean_free_path": float("nan"),
                "knudsen_number": float("nan"),
                "kinetic_viscosity": float("nan"),
                "reynolds_number": float("nan"),
                "mean_vx_late_mean": float("nan"),
            },
        ]
        summaries = compute_summary(results)
        s = summaries[0]
        assert np.isnan(s["mean_free_path_mean"])
        assert np.isnan(s["kinetic_viscosity_mean"])
        assert np.isnan(s["reynolds_number_mean"])

    def test_empty_results(self):
        summaries = compute_summary([])
        assert summaries == []

    def test_valid_runs_counts_correctly(self):
        """3 rows for N=80: 2 valid, 1 with NaN mean_free_path."""
        results = [
            {
                "particle_count": 80,
                "seed": 1,
                "density_2d": 4.0,
                "mean_free_path": 0.5,
                "knudsen_number": 0.167,
                "kinetic_viscosity": 0.8,
                "reynolds_number": 5.0,
                "mean_vx_late_mean": 0.2,
            },
            {
                "particle_count": 80,
                "seed": 2,
                "density_2d": 4.0,
                "mean_free_path": 0.6,
                "knudsen_number": 0.2,
                "kinetic_viscosity": 0.9,
                "reynolds_number": 4.5,
                "mean_vx_late_mean": 0.18,
            },
            {
                "particle_count": 80,
                "seed": 3,
                "density_2d": 4.0,
                "mean_free_path": float("nan"),
                "knudsen_number": float("nan"),
                "kinetic_viscosity": float("nan"),
                "reynolds_number": float("nan"),
                "mean_vx_late_mean": float("nan"),
            },
        ]
        summaries = compute_summary(results)
        s = summaries[0]
        assert s["runs"] == 3
        assert s["valid_runs"] == 2
        assert s["invalid_runs"] == 1
        assert s["valid_fraction"] == pytest.approx(2.0 / 3.0)

    def test_means_over_valid_rows_only(self):
        """With 2 valid + 1 invalid, means should be over the 2 valid rows."""
        results = [
            {
                "particle_count": 80,
                "seed": 1,
                "density_2d": 4.0,
                "mean_free_path": 0.5,
                "knudsen_number": 0.167,
                "kinetic_viscosity": 0.8,
                "reynolds_number": 5.0,
                "mean_vx_late_mean": 0.2,
            },
            {
                "particle_count": 80,
                "seed": 2,
                "density_2d": 4.0,
                "mean_free_path": 0.6,
                "knudsen_number": 0.2,
                "kinetic_viscosity": 0.9,
                "reynolds_number": 4.5,
                "mean_vx_late_mean": 0.18,
            },
            {
                "particle_count": 80,
                "seed": 3,
                "density_2d": 4.0,
                "mean_free_path": float("nan"),
                "knudsen_number": float("nan"),
                "kinetic_viscosity": float("nan"),
                "reynolds_number": float("nan"),
                "mean_vx_late_mean": float("nan"),
            },
        ]
        summaries = compute_summary(results)
        s = summaries[0]

        # Means over valid rows only (seed 1 and 2)
        assert s["mean_free_path_mean"] == pytest.approx(0.55)  # (0.5 + 0.6) / 2
        assert s["kinetic_viscosity_mean"] == pytest.approx(0.85)  # (0.8 + 0.9) / 2
        assert s["reynolds_number_mean"] == pytest.approx(4.75)  # (5.0 + 4.5) / 2
        assert s["knudsen_number_mean"] == pytest.approx(0.1835, abs=1e-4)  # (0.167 + 0.2) / 2

    def test_all_invalid_rows_returns_nan_means(self):
        """If all rows are invalid, valid_runs == 0 and means are NaN."""
        results = [
            {
                "particle_count": 80,
                "seed": 1,
                "density_2d": 4.0,
                "mean_free_path": float("nan"),
                "knudsen_number": float("nan"),
                "kinetic_viscosity": float("nan"),
                "reynolds_number": float("nan"),
                "mean_vx_late_mean": float("nan"),
            },
            {
                "particle_count": 80,
                "seed": 2,
                "density_2d": 4.0,
                "mean_free_path": float("nan"),
                "knudsen_number": float("nan"),
                "kinetic_viscosity": float("nan"),
                "reynolds_number": float("nan"),
                "mean_vx_late_mean": float("nan"),
            },
        ]
        summaries = compute_summary(results)
        s = summaries[0]
        assert s["valid_runs"] == 0
        assert s["invalid_runs"] == 2
        assert s["valid_fraction"] == 0.0
        assert np.isnan(s["mean_free_path_mean"])
        assert np.isnan(s["mean_free_path_std"])
        assert np.isnan(s["knudsen_number_mean"])
        assert np.isnan(s["knudsen_number_std"])
        assert np.isnan(s["kinetic_viscosity_mean"])
        assert np.isnan(s["kinetic_viscosity_std"])
        assert np.isnan(s["reynolds_number_mean"])
        assert np.isnan(s["reynolds_number_std"])


# ---------------------------------------------------------------------------
# run_single_simulation — not run, just verify it's importable
# ---------------------------------------------------------------------------


class TestRunSingleSimulation:
    def test_function_exists(self):
        """Verify run_single_simulation is importable and callable."""
        assert callable(run_single_simulation)