"""
Viscosity and Reynolds number estimation for a 2D hard-disk gas.

This module provides kinetic-theory estimates of dynamic viscosity and
Reynolds number for the 2D molecular dynamics model.  All quantities
are order-of-magnitude estimates and should not be compared directly
with real 3D gas properties.
"""

import math

import numpy as np


def compute_2d_number_density(
    num_particles: int,
    width: float,
    height: float,
) -> float:
    """
    Compute the 2D number density.

    n_2D = N / (width * height)

    Parameters
    ----------
    num_particles : int
        Number of particles N.
    width : float
        Channel width (x-direction).
    height : float
        Channel height (y-direction).

    Returns
    -------
    float
        Number density (particles per unit area).
    """
    area = width * height
    if area <= 0.0:
        return float("nan")
    return num_particles / area


def compute_2d_mass_density(
    num_particles: int,
    mass: float,
    width: float,
    height: float,
) -> float:
    """
    Compute the 2D mass (surface) density.

    rho_2D = N * m / (width * height)

    Parameters
    ----------
    num_particles : int
        Number of particles N.
    mass : float
        Particle mass.
    width : float
        Channel width (x-direction).
    height : float
        Channel height (y-direction).

    Returns
    -------
    float
        Mass density (mass per unit area).
    """
    area = width * height
    if area <= 0.0:
        return float("nan")
    return num_particles * mass / area


def compute_mean_thermal_speed_2d(
    temperature: float,
    mass: float = 1.0,
    k_b: float = 1.0,
) -> float:
    """
    Compute the 2D RMS thermal speed estimate.

    v_rms = sqrt(2 * k_B * T / m)

    This is a 2D RMS thermal speed estimate based on the equipartition
    theorem for a 2D ideal gas:  <K> = (1/2) * m * <v^2> = k_B * T,
    so <v^2> = 2 * k_B * T / m.

    Parameters
    ----------
    temperature : float
        Gas temperature (energy units, k_B = 1 by default).
    mass : float
        Particle mass (default: 1.0).
    k_b : float
        Boltzmann constant (default: 1.0).

    Returns
    -------
    float
        RMS thermal speed.
    """
    if temperature < 0.0 or mass <= 0.0:
        return float("nan")
    return math.sqrt(2.0 * k_b * temperature / mass)


def estimate_kinetic_viscosity_2d(
    density: float,
    mean_thermal_speed: float,
    mean_free_path: float,
    coefficient: float = 0.5,
) -> float:
    """
    Estimate the dynamic viscosity for a 2D hard-disk gas from kinetic theory.

    eta = C * rho_2D * v_rms * lambda_MD

    This is an order-of-magnitude kinetic-theory estimate for a 2D
    hard-disk gas, not a calibrated real-gas viscosity.

    Parameters
    ----------
    density : float
        2D mass density rho_2D.
    mean_thermal_speed : float
        RMS thermal speed v_rms.
    mean_free_path : float
        Mean free path lambda_MD measured from the simulation.
    coefficient : float
        Dimensionless prefactor C (default: 0.5).

    Returns
    -------
    float
        Estimated dynamic viscosity.  Returns NaN if any input is
        non-positive or NaN.
    """
    if not np.isfinite(density) or density <= 0.0:
        return float("nan")
    if not np.isfinite(mean_thermal_speed) or mean_thermal_speed <= 0.0:
        return float("nan")
    if not np.isfinite(mean_free_path) or mean_free_path <= 0.0:
        return float("nan")
    return coefficient * density * mean_thermal_speed * mean_free_path


def compute_reynolds_number(
    density: float,
    flow_speed: float,
    characteristic_length: float,
    dynamic_viscosity: float,
) -> float:
    """
    Compute the Reynolds number for the 2D channel flow.

    Re = rho_2D * |U| * R / eta

    where R is the characteristic length (channel half-height).

    Parameters
    ----------
    density : float
        2D mass density rho_2D.
    flow_speed : float
        Absolute streamwise flow speed |U|.
    characteristic_length : float
        Characteristic length scale (e.g. channel half-height).
    dynamic_viscosity : float
        Dynamic viscosity eta.

    Returns
    -------
    float
        Estimated Reynolds number.  Returns NaN if dynamic_viscosity
        is non-positive or NaN.
    """
    if not np.isfinite(dynamic_viscosity) or dynamic_viscosity <= 0.0:
        return float("nan")
    if not np.isfinite(density) or density <= 0.0:
        return float("nan")
    if not np.isfinite(flow_speed) or flow_speed <= 0.0:
        return float("nan")
    if not np.isfinite(characteristic_length) or characteristic_length <= 0.0:
        return float("nan")
    return density * abs(flow_speed) * characteristic_length / dynamic_viscosity


def classify_reynolds_number(re: float) -> str:
    """
    Classify the flow regime based on the estimated Reynolds number.

    In this small 2D MD model these labels are qualitative only.

    Parameters
    ----------
    re : float
        Estimated Reynolds number.

    Returns
    -------
    str
        One of: "undefined", "creeping", "laminar-like",
        "transitional-like", "inertial/turbulent-like".
    """
    if not np.isfinite(re):
        return "undefined"
    if re < 1.0:
        return "creeping"
    if re < 100.0:
        return "laminar-like"
    if re < 2000.0:
        return "transitional-like"
    return "inertial/turbulent-like"