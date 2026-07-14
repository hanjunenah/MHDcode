"""Doubly-periodic resistive Harris-sheet reconnection problem."""

import numpy as np


grid = {
    "nx": 128,
    "ny": 128,
    "lx": 25.6,
    "ly": 12.8,
}

solver = {
    "gamma": 5.0 / 3.0,
    "cfl": 0.35,
    "diffusion_cfl": 0.45,
    "resistivity": 5e-3,
    "t_end": 100.0,
    "max_steps": 200_000,
    "second_order": True,
    "limiter_theta": 1.5,
    "glm_speed_factor": 1.0,
    "glm_damping": 0.1,
    "rho_floor": 1.0e-8,
    "p_floor": 1.0e-8,
    # Zero selects all logical CPU cores; set one for a serial run.
    "threads": 0,
}

output = {
    "file": "output/harris.h5",
    "interval": 1,
}

b0 = 1.0
sheet_half_thickness = 0.2
n0 = 1
background_density = 1.0
# T*n0 = B0^2/2 gives exact total-pressure balance for the base sheet.
temperature = b0**2 / (2.0 * n0)
background_pressure = temperature * background_density
perturbation = 0.03


def _sheet_argument(y):
    ky = 2.0 * np.pi / grid["ly"]
    epsilon = 2.0 * np.pi * sheet_half_thickness / grid["ly"]
    return np.sin(ky * (y - 0.25 * grid["ly"])) / epsilon


def _base_bx(y):
    return b0 * np.tanh(_sheet_argument(y))


def _sheet_envelope(y):
    # sech^2 evaluated as 1-tanh^2 is stable even for very thin sheets.
    tanh_argument = np.tanh(_sheet_argument(y))
    return 1.0 - tanh_argument**2


def initial_state(x, y):
    """Return scalar or grid-shaped profiles for the fluid variables."""
    density = background_density + n0 * _sheet_envelope(y)
    return {
        "rho": density,
        "u": 0.0,
        "v": 0.0,
        "w": 0.0,
        "p": temperature * density,
    }


def magnetic_field(x, y):
    """Return Bx, By, Bz; the perturbation is a discrete periodic curl."""
    dx, dy = grid["lx"] / grid["nx"], grid["ly"] / grid["ny"]
    kx, ky = 2.0 * np.pi / grid["lx"], 4.0 * np.pi / grid["ly"]
    discrete_kx, discrete_ky = np.sin(kx * dx) / dx, np.sin(ky * dy) / dy
    az_scale = perturbation * b0 / kx
    # A_z = az_scale*cos(kx*x)*cos(ky*y), with B=(d_y A_z, -d_x A_z).
    bx = _base_bx(y) - az_scale * discrete_ky * np.cos(kx * x) * np.sin(ky * y)
    by = az_scale * discrete_kx * np.sin(kx * x) * np.cos(ky * y)
    return bx, by, 0.0
