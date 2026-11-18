"""A centered current antenna launching two shear-Alfven wave packets."""

import numpy as np


grid = {
    "nx": 128,
    "ny": 128,
    "lx": 8.0,
    "ly": 8.0,
}

solver = {
    "gamma": 5.0 / 3.0,
    "cfl": 0.35,
    "diffusion_cfl": 0.45,
    "resistivity": 0.0,
    "t_end": 10.0,
    "max_steps": 100_000,
    "second_order": True,
    "limiter_theta": 1.5,
    "glm_speed_factor": 1.0,
    "glm_damping": 0.1,
    "rho_floor": 1.0e-8,
    "p_floor": 1.0e-8,
    "threads": 0,
}

output = {
    "file": "output/alfven_antenna.h5",
    "interval": 0.1,
}


def initial_state(x, y):
    return {"rho": 1.0, "u": 0.0, "v": 0.0, "w": 0.0, "p": 0.2}


def magnetic_field(x, y):
    # With rho=1, this guide field gives the dimensionless Alfven speed v_A=1.
    return 1.0, 0.0, 0.0


antenna_amplitude = 3
antenna_width = 0.1
antenna_duration = 10.0


def centered_current_ribbon(x, y):
    """A periodic Gaussian Jz ribbon centered at x=Lx/2 and uniform in y."""
    distance = np.abs(x - 0.5 * grid["lx"])
    periodic_distance = np.minimum(distance, grid["lx"] - distance)
    distance = np.abs(y - 0.5 * grid["ly"])
    periodic_distance_y = np.minimum(distance, grid["ly"] - distance)
    return antenna_amplitude * np.exp(-0.5 * (periodic_distance / antenna_width) ** 2) \
        * np.exp(-0.5 * (periodic_distance_y / antenna_width) ** 2)


def smooth_pulse(t):
    """A compact sin^2 pulse with zero value and slope at both endpoints."""
    inside = (t >= 0.0) & (t <= antenna_duration)
    return np.where(inside, np.sin(5 * 2 * np.pi * t / antenna_duration), 0.0)
 

# Like SMILEI, each antenna selects one current component and combines spatial
# and temporal Python profiles.  Multiple antenna dictionaries may be supplied.
antennas = [
    {
        "field": "Jz",
        "space_profile": centered_current_ribbon,
        "time_profile": smooth_pulse,
    }
]

# Python functions are sampled before the adaptive C++ run and linearly
# interpolated at every Runge-Kutta stage.  Resolve the shortest time scale here.
antenna_sampling_interval = 0.01
