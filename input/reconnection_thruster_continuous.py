"""Continuously driven thruster with stationary plates and outflow boundaries.

This is the recommended thruster setup.  The historical periodic inputs are
``reconnection_thruster.py`` and ``reconnection_thruster_pulse.py``.

The larger 40 x 20 box preserves the reference cell spacing while moving the
outer boundaries away from the plates.  Outflow boundaries are transmissive,
not perfectly nonreflecting.  Compare larger boxes when interpreting late flow.

The antenna bodies are fixed, impenetrable slip structures with uniform
magnetic diffusivity eta=5e-3 in the body and plasma.  Magnetic field diffuses
through the bodies; this is not an ideal perfectly conducting wall, a no-slip
wall, or a thermal/material model.  Body density/pressure are fixed numerical
placeholders and are excluded from plasma diagnostics by the solver.

Run from MHDcode with::

    ~/anaconda3/bin/python run.py input/reconnection_thruster_continuous.py
"""

import numpy as np


grid = {"nx": 256, "ny": 128, "lx": 40.0, "ly": 20.0}

drive_omega = 0.05
drive_period = 2.0 * np.pi / drive_omega
startup_duration = drive_period / 4.0

solver = {
    "gamma": 5.0 / 3.0,
    "cfl": 0.25,
    "diffusion_cfl": 0.40,
    "resistivity": 5.0e-3,
    #"boundary_x": "outflow",
    #"boundary_y": "outflow",
    "t_end": 4.0 * drive_period,
    "max_steps": 5_000_000,
    "second_order": True,
    "limiter_theta": 1.3,
    "glm_speed_factor": 1.0,
    "glm_damping": 0.1,
    "rho_floor": 1.0e-8,
    "p_floor": 1.0e-8,
    "threads": 0,
}

output = {
    "file": "output/thruster/continuous/reconnection_thruster_continuous.h5",
    "interval": drive_period / 32.0,
}

background_density = 1.0
background_pressure = 1.0e-2


def initial_state(x, y):
    return {
        "rho": background_density,
        "u": 0.0,
        "v": 0.0,
        "w": 0.0,
        "p": background_pressure,
    }


def magnetic_field(x, y):
    return 0.0, 0.0, 0.0


# Positions use the same length normalization as the earlier Hall-MHD copy.
# Plate length, slope and throat gap are independent of the enlarged box size.
plate_x_min = 10.0
plate_x_max = plate_x_min + 20.0 / 3.0
throat_y = 10.0
throat_half_gap = 0.5
plate_slope = 16.0 / 3.0  # dx/dy; centerlines diverge toward increasing x.
plate_normal_thickness = 0.4
plate_half_thickness = plate_normal_thickness / 2.0
tip_taper_length = 0.4

# Peak current density before the temporal envelope; no spatial mean is
# subtracted.  The external circuit is outside this nonperiodic domain.
current_amplitude = 1.0


def _plate_normal_distances(x, y):
    """Signed perpendicular distances to the two plate centerlines."""
    axial_distance = np.asarray(x) - plate_x_min
    upper_centerline = throat_y + throat_half_gap + axial_distance / plate_slope
    lower_centerline = throat_y - throat_half_gap - axial_distance / plate_slope
    normal_scale = np.sqrt(1.0 + 1.0 / plate_slope**2)
    return (
        (np.asarray(y) - upper_centerline) / normal_scale,
        (np.asarray(y) - lower_centerline) / normal_scale,
    )


def solid_mask(x, y):
    """Two stationary strips of uniform normal thickness, with flat axial tips."""
    upper_distance, lower_distance = _plate_normal_distances(x, y)
    axial_window = (np.asarray(x) >= plate_x_min) & (np.asarray(x) <= plate_x_max)
    return axial_window & (
        (np.abs(upper_distance) <= plate_half_thickness)
        | (np.abs(lower_distance) <= plate_half_thickness)
    )


def plate_current(x, y):
    """Jz confined exactly to the solid, tapered to zero at all body edges."""
    upper_distance, lower_distance = _plate_normal_distances(x, y)

    def normal_taper(distance):
        inside = np.abs(distance) <= plate_half_thickness
        return np.where(
            inside,
            np.cos(0.5 * np.pi * distance / plate_half_thickness) ** 2,
            0.0,
        )

    distance_to_tip = np.minimum(
        np.asarray(x) - plate_x_min, plate_x_max - np.asarray(x)
    )
    tip_fraction = np.clip(distance_to_tip / tip_taper_length, 0.0, 1.0)
    axial_taper = np.sin(0.5 * np.pi * tip_fraction) ** 2
    profile = current_amplitude * axial_taper * np.maximum(
        normal_taper(upper_distance), normal_taper(lower_distance)
    )
    return np.where(solid_mask(x, y), profile, 0.0)


def continuous_drive(t):
    """Smoothly start a negative sine, then keep oscillating for the whole run."""
    time = np.asarray(t)
    startup_fraction = np.clip(time / startup_duration, 0.0, 1.0)
    envelope = np.sin(0.5 * np.pi * startup_fraction) ** 2
    return np.where(time >= 0.0, -envelope * np.sin(drive_omega * time), 0.0)


antennas = [
    {
        "field": "Jz",
        "space_profile": plate_current,
        "time_profile": continuous_drive,
    }
]

antenna_sampling_interval = drive_period / 64.0
