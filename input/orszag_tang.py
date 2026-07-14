"""Standard doubly-periodic Orszag-Tang ideal-MHD vortex benchmark."""

import numpy as np


grid = {
    "nx": 256,
    "ny": 256,
    "lx": 1.0,
    "ly": 1.0,
}

solver = {
    "gamma": 5.0 / 3.0,
    "cfl": 0.35,
    "diffusion_cfl": 0.45,
    "resistivity": 0.0,
    "t_end": 0.5,
    "max_steps": 200_000,
    "second_order": True,
    "limiter_theta": 1.5,
    "glm_speed_factor": 1.0,
    "glm_damping": 0.1,
    "rho_floor": 1.0e-10,
    "p_floor": 1.0e-10,
    "threads": 0,
}

output = {
    "file": "output/orszag_tang.h5",
    "interval": 0.1,
}


gamma = solver["gamma"]
initial_density = 25.0 / (36.0 * np.pi)
initial_pressure = 5.0 / (12.0 * np.pi)
magnetic_amplitude = 1.0 / np.sqrt(4.0 * np.pi)


def initial_state(x, y):
    """The conventional normalization has initial sound speed c_s=1."""
    return {
        "rho": initial_density,
        "u": -np.sin(2.0 * np.pi * y),
        "v": np.sin(2.0 * np.pi * x),
        "w": 0.0,
        "p": initial_pressure,
    }


def magnetic_field(x, y):
    """A periodic, discretely divergence-free in-plane magnetic field."""
    bx = -magnetic_amplitude * np.sin(2.0 * np.pi * y)
    by = magnetic_amplitude * np.sin(4.0 * np.pi * x)
    return bx, by, 0.0
