"""
Tests for the viscosity and Reynolds number estimation module.

These tests verify the helper functions in ``src.measurements.viscosity``
without running any full simulation.
"""

import math

import numpy as np
import pytest

from src.measurements.viscosity import (
    classify_reynolds_number,
    compute_2d_mass_density,
    compute_2d_number_density,
    compute_mean_thermal_speed_2d,
    compute_reynolds_number,
    estimate_kinetic_viscosity_2d,
)


# ---------------------------------------------------------------------------
# compute_2d_number_density
# ---------------------------------------------------------------------------


class TestCompute2DNumberDensity:
    def test_basic(self):
        assert compute_2d_number_density(100, 10, 5) == 2.0

    def test_zero_area_returns_nan(self):
        assert math.isnan(compute_2d_number_density(100, 0, 5))
        assert math.isnan(compute_2d_number_density(100, 10, 0))

    def test_negative_area_returns_nan(self):
        assert math.isnan(compute_2d_number_density(100, -10, 5))

    def test_zero_particles(self):
        assert compute_2d_number_density(0, 10, 5) == 0.0


# ---------------------------------------------------------------------------
# compute_2d_mass_density
# ---------------------------------------------------------------------------


class TestCompute2DMassDensity:
    def test_basic(self):
        assert compute_2d_mass_density(100, 1.0, 10, 5) == 2.0

    def test_with_mass(self):
        assert compute_2d_mass_density(100, 2.0, 10, 5) == 4.0

    def test_zero_area_returns_nan(self):
        assert math.isnan(compute_2d_mass_density(100, 1.0, 0, 5))

    def test_zero_particles(self):
        assert compute_2d_mass_density(0, 1.0, 10, 5) == 0.0


# ---------------------------------------------------------------------------
# compute_mean_thermal_speed_2d
# ---------------------------------------------------------------------------


class TestComputeMeanThermalSpeed2D:
    def test_positive_and_increasing_with_temperature(self):
        v1 = compute_mean_thermal_speed_2d(1.0)
        v2 = compute_mean_thermal_speed_2d(4.0)
        assert v1 > 0.0
        assert v2 > v1  # higher T -> higher thermal speed

    def test_known_value(self):
        # v_rms = sqrt(2 * 1 * 1 / 1) = sqrt(2) ≈ 1.4142
        v = compute_mean_thermal_speed_2d(1.0, mass=1.0, k_b=1.0)
        assert v == pytest.approx(math.sqrt(2.0))

    def test_negative_temperature_returns_nan(self):
        assert math.isnan(compute_mean_thermal_speed_2d(-1.0))

    def test_zero_mass_returns_nan(self):
        assert math.isnan(compute_mean_thermal_speed_2d(1.0, mass=0.0))


# ---------------------------------------------------------------------------
# estimate_kinetic_viscosity_2d
# ---------------------------------------------------------------------------


class TestEstimateKineticViscosity2D:
    def test_positive_viscosity(self):
        eta = estimate_kinetic_viscosity_2d(
            density=2.0,
            mean_thermal_speed=1.5,
            mean_free_path=0.5,
            coefficient=0.5,
        )
        assert eta > 0.0
        assert eta == pytest.approx(0.5 * 2.0 * 1.5 * 0.5)

    def test_default_coefficient(self):
        eta = estimate_kinetic_viscosity_2d(
            density=2.0,
            mean_thermal_speed=1.0,
            mean_free_path=1.0,
        )
        assert eta == pytest.approx(1.0)  # 0.5 * 2 * 1 * 1

    def test_nan_density_returns_nan(self):
        assert math.isnan(
            estimate_kinetic_viscosity_2d(
                density=float("nan"),
                mean_thermal_speed=1.0,
                mean_free_path=1.0,
            )
        )

    def test_zero_density_returns_nan(self):
        assert math.isnan(
            estimate_kinetic_viscosity_2d(
                density=0.0,
                mean_thermal_speed=1.0,
                mean_free_path=1.0,
            )
        )

    def test_negative_mean_free_path_returns_nan(self):
        assert math.isnan(
            estimate_kinetic_viscosity_2d(
                density=2.0,
                mean_thermal_speed=1.0,
                mean_free_path=-0.5,
            )
        )


# ---------------------------------------------------------------------------
# compute_reynolds_number
# ---------------------------------------------------------------------------


class TestComputeReynoldsNumber:
    def test_positive_re(self):
        re = compute_reynolds_number(
            density=2.0,
            flow_speed=0.5,
            characteristic_length=3.0,
            dynamic_viscosity=1.0,
        )
        assert re > 0.0
        assert re == pytest.approx(2.0 * 0.5 * 3.0 / 1.0)

    def test_nan_viscosity_returns_nan(self):
        re = compute_reynolds_number(
            density=2.0,
            flow_speed=0.5,
            characteristic_length=3.0,
            dynamic_viscosity=float("nan"),
        )
        assert math.isnan(re)

    def test_zero_viscosity_returns_nan(self):
        re = compute_reynolds_number(
            density=2.0,
            flow_speed=0.5,
            characteristic_length=3.0,
            dynamic_viscosity=0.0,
        )
        assert math.isnan(re)

    def test_negative_viscosity_returns_nan(self):
        re = compute_reynolds_number(
            density=2.0,
            flow_speed=0.5,
            characteristic_length=3.0,
            dynamic_viscosity=-1.0,
        )
        assert math.isnan(re)

    def test_negative_flow_speed_returns_nan(self):
        re = compute_reynolds_number(
            density=2.0,
            flow_speed=-0.5,
            characteristic_length=3.0,
            dynamic_viscosity=1.0,
        )
        assert math.isnan(re)

    def test_zero_flow_speed_returns_nan(self):
        re = compute_reynolds_number(
            density=2.0,
            flow_speed=0.0,
            characteristic_length=3.0,
            dynamic_viscosity=1.0,
        )
        assert math.isnan(re)


# ---------------------------------------------------------------------------
# classify_reynolds_number
# ---------------------------------------------------------------------------


class TestClassifyReynoldsNumber:
    def test_undefined_for_nan(self):
        assert classify_reynolds_number(float("nan")) == "undefined"

    def test_undefined_for_inf(self):
        assert classify_reynolds_number(float("inf")) == "undefined"

    def test_creeping(self):
        assert classify_reynolds_number(0.5) == "creeping"

    def test_laminar_like(self):
        assert classify_reynolds_number(10.0) == "laminar-like"

    def test_transitional_like(self):
        assert classify_reynolds_number(500.0) == "transitional-like"

    def test_inertial_turbulent_like(self):
        assert classify_reynolds_number(5000.0) == "inertial/turbulent-like"

    def test_boundary_creeping(self):
        assert classify_reynolds_number(1.0) == "laminar-like"  # not < 1

    def test_boundary_laminar(self):
        assert classify_reynolds_number(100.0) == "transitional-like"  # not < 100

    def test_boundary_transitional(self):
        assert classify_reynolds_number(2000.0) == "inertial/turbulent-like"  # not < 2000