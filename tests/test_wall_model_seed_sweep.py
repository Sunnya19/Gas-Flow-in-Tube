"""
Tests for the wall model seed sweep script.

These tests verify the helper functions (parse_comma_ints, build_config,
compute_viscosity_reynolds, NaN handling) without running any full simulation.
"""

import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.config import SimulationConfig
from scripts.run_wall_model_seed_sweep import (
    build_config,
    compute_viscosity_reynolds,
    parse_comma_ints,
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
    def test_specular_config(self):
        config = build_config(
            wall_model="specular",
            num_particles=80,
            total_time=0.1,
            external_force_x=0.005,
            random_seed=1,
        )
        assert isinstance(config, SimulationConfig)
        assert config.num_particles == 80
        assert config.external_force_x == 0.005
        assert config.random_seed == 1
        assert config.wall_model == "specular"
        assert config.x_boundary_type == "periodic"
        assert config.save_vtk is False

    def test_diffuse_same_speed_config(self):
        config = build_config(
            wall_model="diffuse_same_speed",
            num_particles=120,
            total_time=0.2,
            external_force_x=0.01,
            random_seed=5,
        )
        assert config.wall_model == "diffuse_same_speed"
        assert config.num_particles == 120

    def test_diffuse_thermal_config(self):
        config = build_config(
            wall_model="diffuse_thermal",
            num_particles=60,
            total_time=0.15,
            external_force_x=0.005,
            random_seed=3,
        )
        assert config.wall_model == "diffuse_thermal"
        assert config.num_particles == 60

    def test_custom_height(self):
        config = build_config(
            wall_model="specular",
            num_particles=80,
            total_time=0.1,
            external_force_x=0.005,
            random_seed=1,
            height=10.0,
        )
        assert config.height == 10.0

    def test_custom_bins(self):
        config = build_config(
            wall_model="specular",
            num_particles=80,
            total_time=0.1,
            external_force_x=0.005,
            random_seed=1,
            bins=20,
        )
        assert config.velocity_profile_bins == 20


# ---------------------------------------------------------------------------
# compute_viscosity_reynolds
# ---------------------------------------------------------------------------


def _make_fake_history(
    mean_vx: float = 0.2,
    temperature: float = 1.0,
    mean_free_path: float = 0.5,
) -> Dict[str, Any]:
    """Create a fake simulation history for testing compute_viscosity_reynolds."""
    return {
        "mean_vx": [0.0, 0.05, 0.1, 0.15, mean_vx],
        "mean_vy": [0.0, 0.0, 0.0, 0.0, 0.0],
        "temperature": [1.0, 1.0, 1.0, 1.0, temperature],
        "mean_free_path": mean_free_path,
        "free_path_samples": [0.1, 0.2, 0.3],
        "total_energy": [100.0, 100.0, 100.0, 100.0, 100.0],
        "total_particle_collisions": 500,
        "total_wall_collisions": 200,
        "specular_wall_collisions": 100,
        "diffuse_wall_collisions": 100,
        "thermal_wall_collisions": 0,
    }


def _make_fake_config(
    num_particles: int = 80,
    height: float = 6.0,
    width: float = 20.0,
    particle_mass: float = 1.0,
) -> SimulationConfig:
    """Create a minimal SimulationConfig for testing."""
    return SimulationConfig(
        width=width,
        height=height,
        num_particles=num_particles,
        particle_radius=0.1,
        particle_mass=particle_mass,
        initial_temperature=1.0,
        time_step=0.005,
        total_time=0.1,
        save_interval=20,
        equilibration_steps=0,
        num_trajectory_particles=5,
        wall_model="specular",
        wall_model_type="specular",
        wall_temperature=1.0,
        save_vtk=False,
        external_force_x=0.005,
        x_boundary_type="periodic",
        velocity_profile_bins=15,
        random_seed=1,
    )


class TestComputeViscosityReynolds:
    def test_returns_finite_values_for_valid_input(self):
        """With valid fake history, all returned values should be finite."""
        history = _make_fake_history(
            mean_vx=0.2,
            temperature=1.0,
            mean_free_path=0.5,
        )
        config = _make_fake_config(num_particles=80, height=6.0)
        result = compute_viscosity_reynolds(history, config)

        # Check all expected keys exist
        expected_keys = [
            "density_2d",
            "temperature_late_mean",
            "mean_thermal_speed",
            "kinetic_viscosity",
            "reynolds_number",
            "reynolds_regime",
            "knudsen_number",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

        # Check finite values
        assert np.isfinite(result["density_2d"])
        assert np.isfinite(result["temperature_late_mean"])
        assert np.isfinite(result["mean_thermal_speed"])
        assert np.isfinite(result["kinetic_viscosity"])
        assert np.isfinite(result["reynolds_number"])
        assert np.isfinite(result["knudsen_number"])

        # Check positive values
        assert result["density_2d"] > 0.0
        assert result["kinetic_viscosity"] > 0.0
        assert result["reynolds_number"] > 0.0
        assert result["knudsen_number"] > 0.0

    def test_known_values(self):
        """Verify against hand-calculated values.

        The late-time mean_vx is computed over the last 50% of valid values.
        With history [0.0, 0.05, 0.1, 0.15, 0.2] (5 elements):
          int(5 * 0.5) = 2, so late_vx = valid_vx[2:] = [0.1, 0.15, 0.2]
          mean = 0.15
        """
        history = _make_fake_history(
            mean_vx=0.2,
            temperature=1.0,
            mean_free_path=0.5,
        )
        config = _make_fake_config(num_particles=80, height=6.0)
        result = compute_viscosity_reynolds(history, config)

        # density_2d = N * m / (W * H) = 80 * 1.0 / (20.0 * 6.0) = 80/120 = 2/3
        assert result["density_2d"] == pytest.approx(80.0 / (20.0 * 6.0))

        # v_rms = sqrt(2 * k_B * T / m) = sqrt(2 * 1 * 1 / 1) = sqrt(2)
        v_rms = np.sqrt(2.0)
        assert result["mean_thermal_speed"] == pytest.approx(v_rms)

        # eta = 0.5 * rho * v_rms * lambda = 0.5 * (2/3) * sqrt(2) * 0.5
        expected_eta = 0.5 * (80.0 / 120.0) * v_rms * 0.5
        assert result["kinetic_viscosity"] == pytest.approx(expected_eta)

        # Late-time mean_vx: valid_vx[2:] = [0.1, 0.15, 0.2], mean = 0.15
        # flow_speed = abs(mean_vx_late_mean) = 0.15
        # Re = rho * U * R / eta, R = H/2 = 3.0
        late_mean_vx = 0.15
        expected_re = (80.0 / 120.0) * late_mean_vx * 3.0 / expected_eta
        assert result["reynolds_number"] == pytest.approx(expected_re)

        # Kn = lambda / R = 0.5 / 3.0
        assert result["knudsen_number"] == pytest.approx(0.5 / 3.0)

    def test_nan_mean_free_path_returns_nan_eta_re(self):
        """If mean_free_path is NaN, eta and Re should be NaN."""
        history = _make_fake_history(
            mean_vx=0.2,
            temperature=1.0,
            mean_free_path=float("nan"),
        )
        config = _make_fake_config(num_particles=80, height=6.0)
        result = compute_viscosity_reynolds(history, config)

        assert np.isnan(result["kinetic_viscosity"])
        assert np.isnan(result["reynolds_number"])
        assert np.isnan(result["knudsen_number"])
        # density_2d should still be finite (it doesn't depend on history)
        assert np.isfinite(result["density_2d"])

    def test_nan_temperature_returns_nan_eta_re(self):
        """If all temperatures are NaN, thermal speed is NaN -> eta and Re should be NaN."""
        history = _make_fake_history(
            mean_vx=0.2,
            temperature=float("nan"),
            mean_free_path=0.5,
        )
        # Override temperature history to be all NaN
        history["temperature"] = [float("nan")] * 5
        config = _make_fake_config(num_particles=80, height=6.0)
        result = compute_viscosity_reynolds(history, config)

        assert np.isnan(result["mean_thermal_speed"])
        assert np.isnan(result["kinetic_viscosity"])
        assert np.isnan(result["reynolds_number"])

    def test_zero_mean_free_path_returns_nan_eta_re(self):
        """If mean_free_path is zero, eta should be NaN (non-positive)."""
        history = _make_fake_history(
            mean_vx=0.2,
            temperature=1.0,
            mean_free_path=0.0,
        )
        config = _make_fake_config(num_particles=80, height=6.0)
        result = compute_viscosity_reynolds(history, config)

        assert np.isnan(result["kinetic_viscosity"])
        assert np.isnan(result["reynolds_number"])

    def test_reynolds_regime_creeping(self):
        """With small flow speed, Re should be < 1 -> creeping."""
        history = _make_fake_history(
            mean_vx=0.001,
            temperature=1.0,
            mean_free_path=0.5,
        )
        config = _make_fake_config(num_particles=80, height=6.0)
        result = compute_viscosity_reynolds(history, config)

        assert result["reynolds_number"] < 1.0
        assert result["reynolds_regime"] == "creeping"