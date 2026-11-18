#!/usr/bin/env python3
"""Interactive thruster-focused viewer for an mhd2d HDF5 result."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.widgets import Slider
from matplotlib.colors import LogNorm
from matplotlib.patches import Ellipse

from plot_results import cell_extent, load_plot_metadata, magnetic_flux_function, outline_absorption_layer, padded_limits


FIELD_NAMES = (
    "rho", "u", "v", "w", "p", "Bx", "By", "Bz",
    "Jz", "Jx_antenna", "Jy_antenna", "Jz_antenna",
)
DIAGNOSTIC_NAMES = (
    "total_energy", "magnetic_energy", "kinetic_energy", "thermal_energy", "antenna_power",
    "antenna_energy_injected", "max_abs_antenna_current",
)
VELOCITY_LABELS = {"u": "x", "v": "y", "w": "z"}
ANTENNA_LABELS = {
    "Jx_antenna": "x", "Jy_antenna": "y", "Jz_antenna": "z"
}


def electric_field_z(fields, resistivity=0.0, antenna_model="prescribed_current"):
    """Reconstruct cell-centered plasma Ez from the physical resistive Ohm law.

    Jz is the saved curl of the cell-centered magnetic field, not the antenna
    target current. Artificial boundary-buffer diffusion is excluded.
    """
    if not np.isfinite(resistivity) or resistivity < 0:
        raise ValueError("Resistivity must be finite and nonnegative")
    if antenna_model not in ("prescribed_current", "resistive_impressed"):
        raise ValueError("Unknown antenna model for electric-field reconstruction")
    current = np.asarray(fields["Jz"], dtype=float)
    if antenna_model == "resistive_impressed":
        current = current-np.asarray(fields["Jz_antenna"], dtype=float)
    return (np.asarray(fields["v"])*np.asarray(fields["Bx"])
            -np.asarray(fields["u"])*np.asarray(fields["By"])+resistivity*current)


def load_thruster_results(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as source:
        required = ("time", "fields", "diagnostics", "x", "y")
        if any(name not in source for name in required):
            raise ValueError(f"{path} is not an mhd2d result file")
        x, y, times = source["x"][:], source["y"][:], source["time"][:]
        missing_fields = [name for name in FIELD_NAMES if name not in source["fields"]]
        missing_diagnostics = [
            name for name in DIAGNOSTIC_NAMES if name not in source["diagnostics"]
        ]
        if missing_fields:
            raise ValueError(
                f"{path} has no antenna output fields: {', '.join(missing_fields)}"
            )
        if missing_diagnostics:
            raise ValueError(
                f"{path} has no antenna diagnostics: {', '.join(missing_diagnostics)}"
            )
        fields = {name: source[f"fields/{name}"][:] for name in FIELD_NAMES}
        if "prescribed_current/Az" in source:
            potential = source["prescribed_current/Az"][:]
            expected = (times.size, y.size + 1, x.size + 1)
            if potential.shape != expected:
                raise ValueError(
                    f"Native Az has shape {potential.shape}; expected {expected}"
                )
            # The solver evolves this vertex potential directly. Average its
            # four corners for contours on the existing cell-centered panels.
            fields["Az"] = 0.25 * (
                potential[:, :-1, :-1] + potential[:, :-1, 1:]
                + potential[:, 1:, :-1] + potential[:, 1:, 1:]
            )
        diagnostics = {
            name: source[f"diagnostics/{name}"][:] for name in DIAGNOSTIC_NAMES
        }
        for name in (
            "solid_constraint_energy", "absorption_energy_exchange", "boundary_energy_outflow",
            "plasma_energy", "current_drive_energy_correction", "native_magnetic_energy",
            "magnetic_averaging_energy", "max_current_constraint_error",
            "max_stage_current_constraint_error", "max_abs_ct_divB",
            "injection_energy", "injection_mass", "injection_momentum_x",
            "injection_momentum_y", "injection_momentum_z",
            "solid_force_x", "solid_force_y", "solid_force_z",
            "solid_impulse_x", "solid_impulse_y", "solid_impulse_z",
        ):
            if name in source["diagnostics"]:
                diagnostics[name] = source[f"diagnostics/{name}"][:]

    if times.size == 0:
        raise ValueError(f"{path} contains no output frames")
    field_shape = (times.size, y.size, x.size)
    for name, values in fields.items():
        if values.shape != field_shape:
            raise ValueError(f"Field {name} has shape {values.shape}; expected {field_shape}")
    for name, values in diagnostics.items():
        if values.shape != times.shape:
            raise ValueError(f"Diagnostic {name} is not aligned with output times")
    return x, y, times, fields, diagnostics


def strongest_component(fields: dict[str, np.ndarray], names: tuple[str, ...]) -> str:
    amplitudes = [float(np.max(np.abs(fields[name]))) for name in names]
    return names[int(np.argmax(amplitudes))]


def load_thruster_metadata(path: Path) -> dict:
    """Read the actual archived source and reference scales, not an input file."""
    metadata = load_plot_metadata(path)
    with h5py.File(path, "r") as source:
        metadata["resistivity"] = float(source.attrs.get("resistivity", 0.0))
        model = source.attrs.get("antenna_model", "resistive_impressed")
        metadata["antenna_model"] = model.decode() if isinstance(model, bytes) else str(model)
        metadata["physical_scaling"] = dict(source["physical_scaling"].attrs) if "physical_scaling" in source else {}
        hardware = source["antenna_hardware"].attrs if "antenna_hardware" in source else {}
        omega = hardware.get("angular_frequency_code")
        metadata["drive_omega"] = float(omega) if omega is not None else None
        injection = None
        if "plasma_injection" in source:
            group = source["plasma_injection"]
            injection = {name: group[name][:] for name in ("time", "amplitude", "mass_rate")}
            loading = source["plasma_loading"].attrs if "plasma_loading" in source else {}
            if all(name in loading for name in ("inlet_center_x", "inlet_center_y", "inlet_radius_x", "inlet_radius_y")):
                injection["center"] = (float(loading["inlet_center_x"]), float(loading["inlet_center_y"]))
                injection["radii"] = (float(loading["inlet_radius_x"]), float(loading["inlet_radius_y"]))
        metadata["injection"] = injection
    return metadata


def injection_history(x, y, times, injection):
    """Return feed rate and exact integral of the archived piecewise-linear source."""
    times = np.asarray(times, dtype=float)
    if injection is None:
        return {name: np.zeros_like(times) for name in ("amplitude", "mass_rate", "cumulative_mass")}
    sample_times = np.asarray(injection["time"], dtype=float)
    samples = np.asarray(injection["amplitude"], dtype=float)
    q = np.asarray(injection["mass_rate"], dtype=float)
    if q.shape != (len(y), len(x)) or not np.isfinite(q).all() or np.any(q < 0):
        raise ValueError("Archived injection mass_rate must be finite, nonnegative, and match the grid")
    if (sample_times.ndim != 1 or samples.shape != sample_times.shape or not len(samples)
            or not np.isfinite(sample_times).all() or not np.isfinite(samples).all()
            or sample_times[0] != 0 or np.any(np.diff(sample_times) <= 0) or np.any(samples < 0)):
        raise ValueError("Invalid archived injection times or amplitudes")
    if np.any(times < 0) or (len(samples) > 1 and times[-1] > sample_times[-1]+1e-9):
        raise ValueError("Injection samples do not cover the saved times")
    amplitude = np.interp(times, sample_times, samples)
    if len(samples) == 1:
        integrated = samples[0]*times
    else:
        dt = np.diff(sample_times)
        cumulative = np.r_[0.0, np.cumsum(0.5*(samples[1:]+samples[:-1])*dt)]
        interval = np.clip(np.searchsorted(sample_times, times, side="right")-1, 0, len(samples)-2)
        elapsed = times-sample_times[interval]
        slope = np.diff(samples)/dt
        integrated = cumulative[interval]+samples[interval]*elapsed+0.5*slope[interval]*elapsed**2
    dx, dy = float(x[1]-x[0]), float(y[1]-y[0])
    total_rate = float(np.sum(q)*dx*dy)
    return {"amplitude": amplitude, "mass_rate": total_rate*amplitude,
            "cumulative_mass": total_rate*integrated}


def display_units(physical_scaling=None, units="auto"):
    """Multipliers turn code quantities into the labeled plotting units."""
    if units not in ("auto", "code", "si"):
        raise ValueError("units must be auto, code, or si")
    s = physical_scaling or {}
    required = ("length_m", "magnetic_field_t", "mass_density_kg_m3", "time_s", "out_of_plane_length_m")
    complete = all(k in s and np.isfinite(float(s[k])) and float(s[k]) > 0 for k in required)
    if units == "si" and not complete:
        raise ValueError("SI display requires complete positive physical_scaling metadata; use --units code")
    if units == "code" or not complete:
        return dict(time=1.0, force=1.0, mass_rate=1.0, mass=1.0, source=1.0,
                    time_label="time (code)", force_label="thrust (code)",
                    mass_rate_label="mass / code time", mass_label="mass (code)",
                    source_label="density / code time", is_si=False)
    mass = float(s["mass_density_kg_m3"])*float(s["length_m"])**2*float(s["out_of_plane_length_m"])
    force = float(s["magnetic_field_t"])**2/(4*np.pi*1e-7)*float(s["length_m"])*float(s["out_of_plane_length_m"])
    return dict(time=float(s["time_s"])*1e6, force=force*1e3,
                mass_rate=mass/float(s["time_s"])*1e6, mass=mass*1e12,
                source=float(s["mass_density_kg_m3"])/float(s["time_s"]),
                time_label="time (µs)", force_label="thrust (mN)",
                mass_rate_label="mass flow (mg/s)", mass_label="injected mass (ng)",
                source_label="kg m⁻³ s⁻¹", is_si=True)


def specific_impulse_history(times, body_impulse, injected_mass, physical_scaling=None):
    """Signed feed-based Isp from matched impulse/mass windows, in seconds.

    Zero-feed windows and missing SI references are undefined, not infinite.
    The cumulative estimate measures the window starting at the first frame.
    """
    times = np.asarray(times, dtype=float)
    impulse = np.asarray(body_impulse, dtype=float)
    mass = np.asarray(injected_mass, dtype=float)
    if (times.ndim != 1 or not times.size or impulse.shape != times.shape or mass.shape != times.shape
            or not np.isfinite(times).all() or not np.isfinite(impulse).all() or not np.isfinite(mass).all()
            or np.any(np.diff(times) <= 0)):
        raise ValueError("Specific impulse requires finite, aligned impulse/mass histories and increasing times")
    tolerance = 32*np.finfo(float).eps*max(float(np.max(np.abs(mass))), np.finfo(float).tiny)
    if np.any(mass < -tolerance) or np.any(np.diff(mass) < -tolerance):
        raise ValueError("Cumulative injected mass must be nonnegative and nondecreasing")
    references = display_units(physical_scaling, "auto")
    factor = None
    interval = np.full(max(len(times)-1, 0), np.nan)
    cumulative = np.full(len(times), np.nan)
    if references["is_si"]:
        # mN * microseconds / ng = 1000 m/s. Divide by standard gravity.
        factor = references["force"]*references["time"]/references["mass"]*1e3/9.80665
        dm = np.diff(mass)
        np.divide(-np.diff(impulse)*factor, dm, out=interval, where=dm > tolerance)
        dm = mass-mass[0]
        np.divide(-(impulse-impulse[0])*factor, dm, out=cumulative, where=dm > tolerance)
    return {"interval": interval, "cumulative": cumulative, "factor": factor}


def make_figure(
    x, y, times, fields, diagnostics, filename, initial_frame, *,
    boundary_x="periodic", boundary_y="periodic", solid_mask=None,
    velocity_component="x", absorption_width=0.0,
    injection=None, physical_scaling=None, drive_omega=None, units="auto", zoom=False,
    resistivity=0.0, antenna_model="resistive_impressed",
):
    if velocity_component not in ("auto", "x", "y", "z"):
        raise ValueError("velocity_component must be auto, x, y, or z")
    times = np.asarray(times, dtype=float)
    if not len(times) or np.any(np.diff(times) <= 0):
        raise ValueError("Saved times must be nonempty and strictly increasing")
    scales = display_units(physical_scaling, units)
    history = injection_history(x, y, times, injection)
    q = np.zeros((len(y), len(x))) if injection is None else np.asarray(injection["mass_rate"])
    if solid_mask is not None and np.any(q[np.asarray(solid_mask, dtype=bool)] != 0):
        raise ValueError("Injection overlaps stationary solid cells")
    amount = diagnostics.get("injection_mass", history["cumulative_mass"])
    amount = np.asarray(amount, dtype=float)
    if amount.shape != times.shape:
        raise ValueError("injection_mass is not aligned with saved times")
    antenna_name = strongest_component(fields, ("Jx_antenna", "Jy_antenna", "Jz_antenna"))
    velocity_name = (strongest_component(fields, ("u", "v", "w"))
                     if velocity_component == "auto" else {"x": "u", "y": "v", "z": "w"}[velocity_component])
    mask = np.zeros(q.shape, dtype=bool) if solid_mask is None else np.asarray(solid_mask, dtype=bool)
    spatial_panels = {
        0: (rf"Antenna current $J_{{{ANTENNA_LABELS[antenna_name]}}}$ + field lines", fields[antenna_name]),
        1: (rf"Velocity $v_{VELOCITY_LABELS[velocity_name]}$", fields[velocity_name]),
        2: (rf"Momentum density $\rho v_{VELOCITY_LABELS[velocity_name]}$", fields["rho"]*fields[velocity_name]),
        3: (r"Absolute gas pressure $p$", fields["p"]),
        4: (r"Plasma $E_z$ (Ohm law, code)", electric_field_z(fields, resistivity, antenna_model)),
        5: (r"Absolute density $\rho$ (log scale)", fields["rho"]),
    }
    spatial_panels = {i: (title, np.ma.array(values, mask=np.broadcast_to(mask, values.shape))
                            if i != 0 else values) for i, (title, values) in spatial_panels.items()}
    flux = (fields["Az"] if "Az" in fields else
            magnetic_flux_function(fields["Bx"], fields["By"], x, y, boundary_x, boundary_y))
    extent = (*cell_extent(x), *cell_extent(y))
    figure, axes = plt.subplots(3, 3, figsize=(17, 12))
    figure.subplots_adjust(left=0.055, right=0.94, top=0.89, bottom=0.10, wspace=0.49, hspace=0.40)
    try:
        figure.canvas.manager.set_window_title("MHD2D thruster results")
    except AttributeError:
        pass
    images = {}
    for panel, (title, values) in spatial_panels.items():
        axis = axes.flat[panel]
        if panel == 5:
            positive = values.compressed()
            positive = positive[positive > 0]
            low, high = float(np.min(positive)), float(np.max(positive))
            if high <= low:
                low, high = low/1.01, high*1.01
            options = dict(cmap="viridis", norm=LogNorm(vmin=low, vmax=high))
        else:
            lower, upper = padded_limits(values, signed=panel != 3)
            options = dict(cmap="viridis" if panel == 3 else "RdBu_r", vmin=lower, vmax=upper)
        im = axis.imshow(values[initial_frame], origin="lower", extent=extent,
                         interpolation="nearest", aspect="equal", **options)
        axis.set(title=title, xlabel="x (code)", ylabel="y (code)")
        figure.colorbar(im, ax=axis, fraction=0.046, pad=0.04)
        images[panel] = im

    injection_axis = axes.flat[6]
    source_max = float(np.max(q))*max(float(np.max(history["amplitude"])), 1e-30)*scales["source"]
    injection_image = injection_axis.imshow(q*history["amplitude"][initial_frame]*scales["source"],
        origin="lower", extent=extent, interpolation="nearest", cmap="magma",
        vmin=0, vmax=source_max if source_max > 0 else 1, aspect="equal")
    injection_axis.set(title=r"Plasma injection $S_\rho(x,y,t)$", xlabel="x (code)", ylabel="y (code)")
    figure.colorbar(injection_image, ax=injection_axis, fraction=0.046, pad=0.04, label=scales["source_label"])
    injection_text = injection_axis.text(0.02, 0.98, "", transform=injection_axis.transAxes,
        va="top", fontsize=8, bbox=dict(facecolor="white", edgecolor="0.6", alpha=0.9))
    spatial_axes = [axes.flat[i] for i in (*spatial_panels, 6)]
    for axis in spatial_axes:
        if mask.any():
            axis.contour(x, y, mask.astype(float), levels=[0.5], colors="0.35", linewidths=0.9)
        if np.any(q > 0):
            if "center" in injection and "radii" in injection:
                axis.add_patch(Ellipse(injection["center"], 2*injection["radii"][0], 2*injection["radii"][1],
                    fill=False, edgecolor="#f8a523", linestyle="--", linewidth=1.3))
            else:
                axis.contour(x, y, (q > 0).astype(float), levels=[0.5], colors="#f8a523", linewidths=1.3, linestyles="--")
        outline_absorption_layer(axis, x, y, boundary_x, boundary_y, absorption_width)
    if zoom and (mask.any() or np.any(q > 0)):
        yy, xx = np.nonzero(mask if mask.any() else q > 0)
        xspan, yspan = max(float(x[xx].max()-x[xx].min()), 1.0), max(float(y[yy].max()-y[yy].min()), 1.0)
        xpad, ypad = max(1.0, 0.2*xspan), max(1.0, 0.35*yspan)
        for axis in spatial_axes:
            axis.set_xlim(max(extent[0], x[xx].min()-xpad), min(extent[1], x[xx].max()+xpad))
            axis.set_ylim(max(extent[2], y[yy].min()-ypad), min(extent[3], y[yy].max()+ypad))
    fmin, fmax = float(np.min(flux)), float(np.max(flux))
    flux_levels = np.linspace(fmin, fmax, 21) if fmax > fmin+1e-14 else None
    field_lines = None
    plot_time = times*scales["time"]
    thrust_axis = axes.flat[7]
    isp_axis = thrust_axis.twinx()
    isp_axis.patch.set_visible(False)
    isp_interval_line = isp_cumulative_line = None
    isp_history = None
    if "solid_impulse_x" in diagnostics:
        isp_history = specific_impulse_history(times, diagnostics["solid_impulse_x"], amount, physical_scaling)
        if np.isfinite(isp_history["interval"]).any():
            isp_interval_line, = isp_axis.plot(0.5*(plot_time[1:]+plot_time[:-1]), isp_history["interval"],
                color="tab:green", lw=0.9, ls=":", alpha=0.7, label="interval Isp")
            isp_cumulative_line, = isp_axis.plot(plot_time, isp_history["cumulative"],
                color="tab:green", lw=1.6, ls="--", label="cumulative Isp")
        else:
            message = ("Isp undefined: SI reference scales unavailable" if isp_history["factor"] is None
                       else "Feed-based Isp undefined: no injected mass")
            isp_axis.text(0.5, 0.5, message, ha="center", va="center", fontsize=9,
                          wrap=True, transform=isp_axis.transAxes)
    else:
        isp_axis.text(0.5, 0.5, "Isp undefined: body impulse unavailable", ha="center",
                      va="center", transform=isp_axis.transAxes)
    isp_axis.set_ylabel("apparent specific impulse (s)", color="tab:green", fontsize=9)
    isp_axis.tick_params(axis="y", colors="tab:green")
    force = diagnostics.get("solid_force_x")
    impulse = diagnostics.get("solid_impulse_x")
    instantaneous_line = interval_line = None
    if force is not None:
        instantaneous_line, = thrust_axis.plot(plot_time, -np.asarray(force)*scales["force"],
            color="tab:blue", alpha=0.5, lw=0.9, label="last RK step")
    if impulse is not None and len(times) > 1:
        impulse = np.asarray(impulse)
        means = -np.diff(impulse)/np.diff(times)
        interval_line, = thrust_axis.plot(0.5*(plot_time[1:]+plot_time[:-1]), means*scales["force"],
            color="navy", lw=1.1, label="output-interval mean")
        mean = -float(impulse[-1]-impulse[0])/float(times[-1]-times[0])
        thrust_axis.axhline(mean*scales["force"], color="0.4", ls=":", lw=1,
                           label=f"whole-run mean: {mean*scales['force']:.3g}")
        if drive_omega is not None and np.isfinite(drive_omega) and drive_omega > 0:
            period = 2*np.pi/drive_omega
            edges = times[0]+np.arange(int(np.floor((times[-1]-times[0])/period+1e-10))+1)*period
            if len(edges) > 1:
                boundary_impulse = np.interp(edges, times, impulse)
                cycle_thrust = -np.diff(boundary_impulse)/np.diff(edges)
                exact = all(np.min(abs(times-edge)) < 1e-9 for edge in edges)
                thrust_axis.stairs(cycle_thrust*scales["force"], edges*scales["time"], baseline=None,
                    color="tab:orange", lw=2, label="cycle mean" if exact else "cycle mean (interpolated)")
    if force is None and impulse is None:
        thrust_axis.text(0.5, 0.5, "Body-force diagnostics unavailable", ha="center", transform=thrust_axis.transAxes)
    thrust_axis.axhline(0, color="0.6", lw=0.7)
    thrust_marker = thrust_axis.axvline(plot_time[initial_frame], color="black", ls="--", lw=0.9)
    isp_marker = thrust_marker
    thrust_axis.set(title=r"Net thrust toward $-x$ and apparent $I_{\rm sp}$", xlabel=scales["time_label"], ylabel=scales["force_label"])
    thrust_axis.tick_params(axis="y", colors="navy")
    thrust_axis.grid(alpha=0.2)
    handles, labels = thrust_axis.get_legend_handles_labels()
    isp_handles, isp_labels = isp_axis.get_legend_handles_labels()
    if handles or isp_handles:
        thrust_axis.legend(handles+isp_handles, labels+isp_labels, fontsize=6.3, loc="lower left", ncol=2)
    feed_axis = axes.flat[8]
    mass_axis = feed_axis.twinx()
    feed_line, = feed_axis.plot(plot_time, history["mass_rate"]*scales["mass_rate"],
                               color="tab:green", lw=1.6, label="feed rate")
    mass_line, = mass_axis.plot(plot_time, amount*scales["mass"], color="tab:purple", lw=1.5,
        label="mass added (solver)" if "injection_mass" in diagnostics else "mass added (source integral)")
    feed_marker = feed_axis.axvline(plot_time[initial_frame], color="black", ls="--", lw=0.9)
    feed_axis.set(title="Plasma feed and cumulative amount", xlabel=scales["time_label"], ylabel=scales["mass_rate_label"])
    mass_axis.set_ylabel(scales["mass_label"], color="tab:purple")
    mass_axis.tick_params(axis="y", colors="tab:purple")
    feed_axis.tick_params(axis="y", colors="tab:green")
    feed_axis.grid(alpha=0.2)
    feed_axis.legend([feed_line, mass_line], [feed_line.get_label(), mass_line.get_label()], fontsize=7)
    peak_feed = float(np.max(history["mass_rate"]))*scales["mass_rate"]
    peak_mass = float(np.max(amount))*scales["mass"]
    feed_axis.set_ylim(0.0, 1.1*peak_feed if peak_feed > 0.0 else 1.0)
    mass_axis.set_ylim(0.0, 1.05*peak_mass if peak_mass > 0.0 else 1.0)
    if np.any(q > 0):
        row, col = np.nonzero(q > 0)
        center = injection.get("center", (float(np.sum(q*x[None, :])/q.sum()), float(np.sum(q*y[:, None])/q.sum())))
        position_text = f"Feed center: ({center[0]:g}, {center[1]:g}); {len(row)} cells"
    else:
        position_text = "No density injection in this run"
    title = figure.suptitle("")

    def show_frame(frame, *, draw=True):
        nonlocal field_lines
        for panel, im in images.items():
            im.set_data(spatial_panels[panel][1][frame])
        if field_lines is not None:
            field_lines.remove()
            field_lines = None
        if flux_levels is not None:
            field_lines = axes.flat[0].contour(x, y, flux[frame], levels=flux_levels,
                                              colors="black", linewidths=0.6, alpha=0.65)
        local = q*history["amplitude"][frame]*scales["source"]
        injection_image.set_data(local)
        rate_unit, mass_unit = ("mg/s", "ng") if scales["is_si"] else ("mass / code time", "code mass")
        injection_text.set_text(position_text+f"\npeak source: {float(np.max(local)):.4g} {scales['source_label']}"
            +f"\nfeed: {history['mass_rate'][frame]*scales['mass_rate']:.5g} {rate_unit}"
            +f"\nadded: {amount[frame]*scales['mass']:.5g} {mass_unit}")
        for marker in (isp_marker, thrust_marker, feed_marker):
            marker.set_xdata([plot_time[frame], plot_time[frame]])
        title.set_text(f"2-D MHD thruster: t={times[frame]:.6g} code"
            +(f" ({plot_time[frame]:.5g} µs)" if scales["is_si"] else "")
            +f"; frame {frame}/{len(times)-1}\n{filename}\n"
            +"Gray: fixed plates; orange dashed: plasma feed; T>0 is force toward −x")
        if draw:
            figure.canvas.draw_idle()

    show_frame(initial_frame)
    slider = None
    if len(times) > 1:
        slider_axis = figure.add_axes((0.16, 0.035, 0.68, 0.025))
        slider = Slider(slider_axis, "Time frame", 0, len(times)-1, valinit=initial_frame, valstep=1, valfmt="%0.0f")
        slider.on_changed(lambda value: show_frame(int(round(value))))

        def on_key(event):
            current = int(round(slider.val))
            if event.key in ("right", "up"):
                slider.set_val(min(current+1, len(times)-1))
            elif event.key in ("left", "down"):
                slider.set_val(max(current-1, 0))
            elif event.key == "home":
                slider.set_val(0)
            elif event.key == "end":
                slider.set_val(len(times)-1)
        figure.canvas.mpl_connect("key_press_event", on_key)
    figure._thruster_artists = dict(injection_image=injection_image, injection_text=injection_text,
        thrust_marker=thrust_marker, feed_marker=feed_marker, thrust_axis=thrust_axis,
        feed_axis=feed_axis, mass_axis=mass_axis, thrust_instantaneous_line=instantaneous_line,
        thrust_interval_line=interval_line, feed_line=feed_line, mass_line=mass_line)
    figure._thruster_artists.update(isp_axis=isp_axis, isp_marker=isp_marker,
        isp_interval_line=isp_interval_line, isp_cumulative_line=isp_cumulative_line,
        electric_field_axis=axes.flat[4], electric_field_image=images[4])
    figure._thruster_units = scales
    figure._thruster_show_frame = show_frame
    figure._thruster_frame_count = len(times)
    figure._time_slider = slider
    return figure, slider


def export_movie(figure, destination, fps=20):
    """Render every saved output frame with the same updates as the slider."""
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Movie frame rate must be finite and positive")
    if not FFMpegWriter.isAvailable():
        raise RuntimeError("Movie export requires ffmpeg on PATH")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    slider = getattr(figure, "_time_slider", None)
    old_frame = int(round(slider.val)) if slider is not None else 0
    if slider is not None:
        slider.ax.set_visible(False)
    writer = FFMpegWriter(fps=fps, codec="libx264",
                         extra_args=["-pix_fmt", "yuv420p", "-crf", "20", "-threads", "2",
                                     "-movflags", "+faststart"])
    count = figure._thruster_frame_count
    try:
        with writer.saving(figure, str(destination), dpi=100):
            for frame in range(count):
                figure._thruster_show_frame(frame, draw=False)
                writer.grab_frame()
                if frame % max(count//10, 1) == 0 or frame == count-1:
                    print(f"Movie: frame {frame+1}/{count}", flush=True)
    finally:
        if slider is not None:
            slider.ax.set_visible(True)
        figure._thruster_show_frame(old_frame)
    print(f"Saved movie {destination}", flush=True)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_file", type=Path, help="thruster HDF5 result file")
    parser.add_argument("--frame", type=int, default=-1, help="initial frame, negative counts from end (default: last)")
    parser.add_argument("--save", type=Path, help="save the selected frame instead of opening a window")
    parser.add_argument("--movie", type=Path, help="export all saved frames to an MP4 movie")
    parser.add_argument("--fps", type=float, default=20, help="movie frames per second (default: 20)")
    parser.add_argument("--velocity-component", choices=("auto", "x", "y", "z"), default="x")
    parser.add_argument("--units", choices=("auto", "code", "si"), default="auto", help="units for source/force/feed/time histories (auto uses saved SI metadata)")
    view = parser.add_mutually_exclusive_group()
    view.add_argument("--zoom", action="store_true", help="zoom the spatial panels to the plates")
    view.add_argument("--full-domain", dest="zoom", action="store_false", help="show the whole domain (default)")
    parser.set_defaults(zoom=False)
    return parser.parse_args()


def main():
    args = parse_arguments()
    if args.movie is not None:
        plt.switch_backend("Agg")
    print(f"Loading {args.result_file} ...")
    x, y, times, fields, diagnostics = load_thruster_results(args.result_file)
    frame = args.frame if args.frame >= 0 else len(times)+args.frame
    if not 0 <= frame < len(times):
        raise ValueError("Selected frame is outside the saved range")
    metadata = load_thruster_metadata(args.result_file)
    history = injection_history(x, y, times, metadata["injection"])
    print(f"Loaded {len(times)} frames, {len(x)}x{len(y)}, t={times[0]:g}..{times[-1]:g}")
    print(f"Feed rate range: {history['mass_rate'].min():.6g}..{history['mass_rate'].max():.6g} code mass/time")
    print(f"Total added mass: {diagnostics.get('injection_mass', history['cumulative_mass'])[-1]:.8g} code mass")
    if "solid_impulse_x" in diagnostics and len(times) > 1:
        mean = -(diagnostics["solid_impulse_x"][-1]-diagnostics["solid_impulse_x"][0])/(times[-1]-times[0])
        scales = display_units(metadata["physical_scaling"], args.units)
        print(f"Mean thrust toward -x: {mean:.8g} code; {mean*scales['force']:.6g} {scales['force_label']}")
    figure, slider = make_figure(x, y, times, fields, diagnostics, args.result_file.name, frame,
        **metadata, velocity_component=args.velocity_component,
        units=args.units, zoom=args.zoom)
    figure._time_slider = slider
    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.save, dpi=160)
        print(f"Saved {args.save}")
    if args.movie is not None:
        export_movie(figure, args.movie, args.fps)
    if args.save is None and args.movie is None:
        print("Use the slider or arrow keys; Home/End select endpoints. Orange dashed outlines show the plasma feed.")
        plt.show()


if __name__ == "__main__":
    main()
