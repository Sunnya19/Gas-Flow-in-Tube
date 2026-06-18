"""
Tests for the wall model comparison script and its helper functions.

These tests verify:
- ``build_config`` produces a valid ``SimulationConfig`` for each wall model.
- An unsupported wall model raises an error.
- ``extract_flow_summary`` works correctly with a fake history.
- ``save_velocity_profile_csv`` writes the expected CSV format.

Full wall model comparison simulations are NOT run here (too slow).
"""

import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.config import SimulationConfig
from scripts.run_wall_model_comparison import (
    build_config,
    extract_flow_summary,
    save_velocity_profile_csv,
)


# ---------------------------------------------------------------------------
# build_config tests
# ---------------------------------------------------------------------------


class TestBuildConfig:
    """Verify that build_config creates a valid SimulationConfig."""

    @pytest.mark.parametrize(
        "wall_model",
        ["specular", "diffuse_same_speed", "diffuse_thermal"],
    )
    def test_valid_wall_models(self, wall_model: str):
        config = build_config(
            wall_model=wall_model,
            num_particles=50,
            total_time=0.1,
            external_force_x=0.005,
            random_seed=1,
            save_vtk=False,
        )
        assert isinstance(config, SimulationConfig)
        assert config.wall_model == wall_model
        assert config.num_particles == 50
        assert config.external_force_x == 0.005
        assert config.random_seed == 1
        assert config.x_boundary_type == "periodic"
        assert config.save_vtk is False

    def test_unsupported_wall_model_raises(self):
        with pytest.raises(ValueError, match="wall_model must be"):
            build_config(
                wall_model="invalid_model",
                num_particles=50,
                total_time=0.1,
                external_force_x=0.005,
                random_seed=1,
                save_vtk=False,
            )


# ---------------------------------------------------------------------------
# extract_flow_summary tests
# ---------------------------------------------------------------------------


def _make_fake_history() -> Dict[str, Any]:
    """Create a minimal fake history for testing extract_flow_summary."""
    n_points = 10
    return {
        "total_energy": [100.0 + i for i in range(n_points)],
        "mean_vx": [0.01 * i for i in range(n_points)],
        "mean_vy": [0.001 * i for i in range(n_points)],
        "temperature": [1.0 + 0.01 * i for i in range(n_points)],
        "total_particle_collisions": 500,
        "total_wall_collisions": 300,
        "specular_wall_collisions": 100,
        "diffuse_wall_collisions": 100,
        "thermal_wall_collisions": 100,
        "mean_free_path": 2.5,
        "free_path_samples": [1.0, 2.0, 3.0, 4.0],
    }


class TestExtractFlowSummary:
    """Verify extract_flow_summary returns correct fields."""

    def test_basic_extraction(self):
        history = _make_fake_history()
        summary = extract_flow_summary(history, "specular")

        assert summary["wall_model"] == "specular"
        assert summary["mean_vx_final"] == pytest.approx(0.09)
        assert summary["mean_vy_final"] == pytest.approx(0.009)
        assert summary["temperature_initial"] == pytest.approx(1.0)
        assert summary["temperature_final"] == pytest.approx(1.09)
        assert summary["energy_initial"] == pytest.approx(100.0)
        assert summary["energy_final"] == pytest.approx(109.0)
        assert summary["relative_energy_change"] == pytest.approx(0.09)
        assert summary["total_particle_collisions"] == 500
        assert summary["total_wall_collisions"] == 300
        assert summary["specular_wall_collisions"] == 100
        assert summary["diffuse_wall_collisions"] == 100
        assert summary["thermal_wall_collisions"] == 100
        assert summary["mean_free_path"] == pytest.approx(2.5)
        assert summary["free_path_samples"] == 4

    def test_late_time_mean_vx(self):
        """Verify late-time mean_vx and std are computed correctly."""
        history = _make_fake_history()
        summary = extract_flow_summary(history, "specular")

        # mean_vx = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]
        # late_fraction=0.5 -> last 5 values: [0.05, 0.06, 0.07, 0.08, 0.09]
        # mean = 0.07, std(ddof=1) = sqrt(((0.02^2+0.01^2+0+0.01^2+0.02^2)/4)) = sqrt(0.001/4) = sqrt(0.00025) ≈ 0.015811
        assert summary["mean_vx_late_mean"] == pytest.approx(0.07)
        assert summary["mean_vx_late_std"] == pytest.approx(np.std([0.05, 0.06, 0.07, 0.08, 0.09], ddof=1))

    def test_late_time_empty_history(self):
        """Empty mean_vx history should give nan for late-time stats."""
        history: Dict[str, Any] = {
            "total_energy": [],
            "mean_vx": [],
            "mean_vy": [],
            "temperature": [],
        }
        summary = extract_flow_summary(history, "specular")
        assert np.isnan(summary["mean_vx_late_mean"])
        assert np.isnan(summary["mean_vx_late_std"])

    def test_late_time_single_value(self):
        """Single valid mean_vx value: late_mean = that value, late_std = nan."""
        history = _make_fake_history()
        history["mean_vx"] = [0.05]
        summary = extract_flow_summary(history, "specular")
        assert summary["mean_vx_late_mean"] == pytest.approx(0.05)
        assert np.isnan(summary["mean_vx_late_std"])

    def test_missing_fields_default_to_zero_or_nan(self):
        """If history lacks collision subtype fields, should not crash."""
        history = _make_fake_history()
        # Remove optional fields
        del history["specular_wall_collisions"]
        del history["diffuse_wall_collisions"]
        del history["thermal_wall_collisions"]

        summary = extract_flow_summary(history, "diffuse_thermal")
        # These should default to 0 via .get()
        assert summary["specular_wall_collisions"] == 0
        assert summary["diffuse_wall_collisions"] == 0
        assert summary["thermal_wall_collisions"] == 0

    def test_empty_history_does_not_crash(self):
        """Edge case: completely empty history."""
        history: Dict[str, Any] = {
            "total_energy": [],
            "mean_vx": [],
            "mean_vy": [],
            "temperature": [],
        }
        summary = extract_flow_summary(history, "specular")
        assert np.isnan(summary["mean_vx_final"])
        assert np.isnan(summary["mean_vy_final"])
        assert np.isnan(summary["temperature_initial"])
        assert np.isnan(summary["temperature_final"])


# ---------------------------------------------------------------------------
# save_velocity_profile_csv tests
# ---------------------------------------------------------------------------


class TestSaveVelocityProfileCSV:
    """Verify save_velocity_profile_csv writes correct CSV format."""

    def test_basic_csv_content(self, tmp_path: Path):
        y = np.array([0.0, 0.5, 1.0])
        ux = np.array([0.1, 0.2, 0.3])
        out = tmp_path / "profile.csv"

        save_velocity_profile_csv(y, ux, out)

        assert out.exists()
        text = out.read_text().strip()
        lines = text.split("\n")
        assert lines[0] == "y,ux"
        assert len(lines) == 4  # header + 3 data rows

    def test_csv_with_nan(self, tmp_path: Path):
        y = np.array([0.0, 0.5, 1.0])
        ux = np.array([0.1, np.nan, 0.3])
        out = tmp_path / "profile_nan.csv"

        save_velocity_profile_csv(y, ux, out)

        text = out.read_text().strip()
        assert "nan" in text


if __name__ == "__main__":
    pytest.main([__file__])