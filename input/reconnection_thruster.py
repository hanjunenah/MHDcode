"""Low-frequency thruster with powered, stationary current-carrying plates.

The antenna specifies the actual current density inside nonplasma plates,
independently of plasma resistivity. A constrained vector-potential update
couples the powered plates to the evolving plasma through Faraday's law.
Magnetic field is allowed inside the plates. Eta defaults to 1e-3 and applies
to the plasma; eta=0 also establishes plate field and permits plasma screening.

Run: ~/anaconda3/bin/python run.py input/reconnection_thruster.py
Set MHD_THRUSTER_NX / MHD_THRUSTER_NY to refine and MHD_THRUSTER_ETA to change
plasma resistivity. The fiducial default is padded (36x24, 384x256).
MHD_THRUSTER_DOMAIN=compact selects 24x12 at 256x128 with the same cell spacing. The absorbing
buffer is two units wide; MHD_THRUSTER_LX / MHD_THRUSTER_LY and
MHD_THRUSTER_BUFFER override the box and buffer width.
Plate dimensions stay fixed when the box changes. Default amplitude uses the
passive radiation budget and an explicit SI reference scale;
it is an engineering design case, not a validated hardware maximum.
MHD_THRUSTER_CURRENT_SCALE overrides the code amplitude, MHD_THRUSTER_OMEGA
sets angular frequency (0.005 with current scale 1 restores the old drive),
and MHD_THRUSTER_CYCLES
selects the run duration in drive periods. The cooling model does not add a
material resistivity or thermal equation to the powered current constraint.
The default 10 worker threads were fastest on this Mac at both supported
comparison grids. Set MHD_THRUSTER_THREADS or use --threads to override.
All results live under output/thruster/.
"""
import os
import numpy as np

domain_preset = os.environ.get("MHD_THRUSTER_DOMAIN", "padded")
domain_presets = {
    "compact": {"nx": 256, "ny": 128, "lx": 24.0, "ly": 12.0},
    "padded": {"nx": 384, "ny": 256, "lx": 36.0, "ly": 24.0},
}
if domain_preset not in domain_presets:
    raise ValueError("MHD_THRUSTER_DOMAIN must be compact or padded")
domain_grid = domain_presets[domain_preset]
grid = {"nx": int(os.environ.get("MHD_THRUSTER_NX", domain_grid["nx"])),
        "ny": int(os.environ.get("MHD_THRUSTER_NY", domain_grid["ny"])),
        "lx": float(os.environ.get("MHD_THRUSTER_LX", domain_grid["lx"])),
        "ly": float(os.environ.get("MHD_THRUSTER_LY", domain_grid["ly"]))}
# Fixed physical geometry in code units; domain presets only translate it.
plate_x_min = grid["lx"] / 4.0
plate_axial_length = 20.0 / 3.0
plate_x_max = plate_x_min + plate_axial_length
throat_y = grid["ly"] / 2.0
throat_half_gap = 0.5
plate_slope = 16.0 / 3.0
plate_normal_thickness = 0.4
plate_half_thickness = plate_normal_thickness / 2.0
tip_taper_length = 0.4

# Illustrative Hall-size normalization, not a Hall-thruster plasma model.
mu0_si = 4*np.pi*1e-7
atomic_mass_unit_kg = 1.66053906660e-27
stefan_boltzmann_si = 5.670374419e-8
physical_scaling = {
    "length_m": float(os.environ.get("MHD_THRUSTER_L0_M", "0.01")),
    "magnetic_field_t": float(os.environ.get("MHD_THRUSTER_B0_T", "0.02")),
    "ion_number_density_m3": float(os.environ.get("MHD_THRUSTER_N0_M3", "1e18")),
    "ion_mass_amu": 131.293,
    "out_of_plane_length_m": float(os.environ.get("MHD_THRUSTER_DEPTH_M", "0.1")),
}
if any(not np.isfinite(v) or v <= 0 for v in physical_scaling.values()):
    raise ValueError("SI reference scales must be finite and positive")
length_unit = physical_scaling["length_m"]
field_unit = physical_scaling["magnetic_field_t"]
mass_density_unit = (physical_scaling["ion_number_density_m3"]*
                     physical_scaling["ion_mass_amu"]*atomic_mass_unit_kg)
alfven_speed_unit = field_unit/np.sqrt(mu0_si*mass_density_unit)
current_density_unit_a_m2 = field_unit/(mu0_si*length_unit)
physical_scaling.update({
    "mass_density_kg_m3": mass_density_unit,
    "alfven_speed_m_s": alfven_speed_unit,
    "time_s": length_unit/alfven_speed_unit,
    "current_density_a_m2": current_density_unit_a_m2,
    "integrated_current_a": field_unit*length_unit/mu0_si,
})

# Passive solid-copper case in vacuum. Only the two OUTWARD broad faces
# radiate to the environment; no convection, mounting or inner-face credit.
# The default ceiling gives all available radiation to copper Ohmic loss.
# Plasma heat is NOT calculated by this MHD model: enter a measured/estimated
# load below before interpreting this as a hardware operating limit.
emissivity = float(os.environ.get("MHD_THRUSTER_EMISSIVITY", "0.05"))
max_copper_temperature_c = float(os.environ.get("MHD_THRUSTER_COPPER_MAX_C", "80"))
environment_temperature_k = float(os.environ.get("MHD_THRUSTER_ENVIRONMENT_K", "300"))
external_plate_heat_w = float(os.environ.get("MHD_THRUSTER_PLATE_HEAT_W", "0"))
max_copper_temperature_k = max_copper_temperature_c+273.15
if (not np.isfinite(emissivity) or not 0 < emissivity <= 1 or
    not np.isfinite(max_copper_temperature_k) or
    not np.isfinite(environment_temperature_k) or
    not 293.15 <= max_copper_temperature_k <= 473.15 or
    not 0 < environment_temperature_k < max_copper_temperature_k or
    not np.isfinite(external_plate_heat_w) or external_plate_heat_w < 0):
    raise ValueError("Invalid passive copper thermal parameters (temperature model supports 20–200 C)")
normal_scale = np.sqrt(1+1/plate_slope**2)
plate_length_m = plate_axial_length*normal_scale*length_unit
radiating_area_m2 = 2*plate_length_m*physical_scaling["out_of_plane_length_m"]
radiated_power_w = (emissivity*stefan_boltzmann_si*radiating_area_m2*
                    (max_copper_temperature_k**4-environment_temperature_k**4))
copper_resistivity_ohm_m = 1.7241e-8*(1+0.00393*(max_copper_temperature_c-20))
# Exact continuum integral of g(x,y)^2 for both cos²-normal/sin²-tip profiles.
# This keeps the thermal ceiling independent of numerical grid resolution.
profile_square_integral = (2*(3/8)*plate_normal_thickness*normal_scale*
                           (plate_axial_length-5*tip_taper_length/4))
unit_amplitude_copper_loss_w = (copper_resistivity_ohm_m*
    physical_scaling["out_of_plane_length_m"]*length_unit**2*
    current_density_unit_a_m2**2*profile_square_integral/2)
available_copper_power_w = radiated_power_w-external_plate_heat_w
if available_copper_power_w <= 0:
    raise ValueError("External plate heat exhausts the passive radiation budget")
hardware_peak_current_scale = np.sqrt(available_copper_power_w/unit_amplitude_copper_loss_w)
current_scale = float(os.environ.get("MHD_THRUSTER_CURRENT_SCALE", hardware_peak_current_scale))
if not np.isfinite(current_scale):
    raise ValueError("Current scale must be finite")
antenna_hardware = {
    "cooling": "passive radiation from bare solid-copper outer faces",
    "emissivity": emissivity,
    "copper_max_temperature_c": max_copper_temperature_c,
    "environment_temperature_k": environment_temperature_k,
    "external_plate_heat_w": external_plate_heat_w,
    "radiating_area_m2": radiating_area_m2,
    "radiated_power_w": radiated_power_w,
    "copper_resistivity_ohm_m": copper_resistivity_ohm_m,
    "continuum_profile_square_integral": profile_square_integral,
    "unit_amplitude_copper_loss_w": unit_amplitude_copper_loss_w,
    "peak_current_scale_ceiling": hardware_peak_current_scale,
    "selected_current_scale": current_scale,
    "within_selected_thermal_budget": bool(abs(current_scale) <= hardware_peak_current_scale*(1+1e-12)),
    "interpretation": "conditional radiation-only upper bound; external plasma heat and feed losses are not resolved",
}

# Deliberately slow test: with the passive current ceiling, the old-run
# conditional SP estimate is ~160–250, giving omega*tau_SP about 0.06–0.10.
# This is an adjustable design frequency, not feedback from eta alone.
drive_omega = float(os.environ.get("MHD_THRUSTER_OMEGA", "0.0004"))
if not np.isfinite(drive_omega) or drive_omega <= 0:
    raise ValueError("MHD_THRUSTER_OMEGA must be finite and positive")
drive_period = 2*np.pi/drive_omega
startup_duration = drive_period/4
antenna_hardware.update({
    "angular_frequency_code": drive_omega,
    "ordinary_frequency_hz": drive_omega/(2*np.pi*physical_scaling["time_s"]),
    "nominal_peak_current_density_a_m2": abs(current_scale)*current_density_unit_a_m2,
    "nominal_rms_current_density_a_m2": abs(current_scale)*current_density_unit_a_m2/np.sqrt(2),
    "mean_copper_loss_w": unit_amplitude_copper_loss_w*current_scale**2,
})
eta = float(os.environ.get("MHD_THRUSTER_ETA", "1e-3"))
run_cycles = float(os.environ.get("MHD_THRUSTER_CYCLES", "3"))
solver = {
    "gamma": 5/3, "cfl": 0.25, "diffusion_cfl": 0.4,
    "resistivity": eta, "antenna_model": "prescribed_current",
    "boundary_x": "absorbing", "boundary_y": "absorbing",
    "absorption_width": float(os.environ.get("MHD_THRUSTER_BUFFER", "2")),
    "absorption_rate": 0.5,
    "absorption_magnetic_diffusivity": 0.05,
    "t_end": run_cycles*drive_period, "max_steps": 5_000_000,
    "second_order": True, "limiter_theta": 1.3,
    "glm_speed_factor": 1.0, "glm_damping": 0.1,
    "rho_floor": 1e-8, "p_floor": 1e-8,
    "threads": int(os.environ.get("MHD_THRUSTER_THREADS", "10")),
}
output = {
    "file": f"output/thruster/prescribed_current/eta_{eta:g}/thruster_{grid['nx']}x{grid['ny']}_L{grid['lx']:g}x{grid['ly']:g}_J{current_scale:.8g}_omega{drive_omega:g}_cycles{run_cycles:g}.h5",
    "interval": drive_period/32,
}
boundary_ambient = {"rho": 1.0, "p": 0.01, "u": 0.0, "v": 0.0,
                    "w": 0.0, "Bx": 0.0, "By": 0.0, "Bz": 0.0}


def initial_state(x, y):
    return {"rho": 1.0, "p": 0.01, "u": 0.0, "v": 0.0, "w": 0.0}


def magnetic_field(x, y):
    return 0.0, 0.0, 0.0


# Peak current density before the temporal envelope; no spatial mean is
# subtracted.  The external circuit is outside this nonperiodic domain.
current_amplitude = current_scale


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
