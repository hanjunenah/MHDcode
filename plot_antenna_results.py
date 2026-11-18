#!/usr/bin/env python3
"""Interactive antenna-focused viewer for an mhd2d HDF5 result."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

from plot_results import cell_extent, load_plot_metadata, magnetic_flux_function, outline_absorption_layer, padded_limits


FIELD_NAMES = (
    "rho", "u", "v", "w", "p", "Bx", "By", "Bz", "divB", "psi",
    "Jx_antenna", "Jy_antenna", "Jz_antenna",
)
DIAGNOSTIC_NAMES = (
    "total_energy", "magnetic_energy", "kinetic_energy", "thermal_energy", "antenna_power",
    "antenna_energy_injected", "max_abs_antenna_current",
)
VELOCITY_LABELS = {"u": "x", "v": "y", "w": "z"}
ANTENNA_LABELS = {
    "Jx_antenna": "x", "Jy_antenna": "y", "Jz_antenna": "z"
}


def load_antenna_results(path: Path):
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
            # Use the matching face divergence, including for early files
            # whose stored divB used a second curl of averaged cell fields.
            dx, dy = float(x[1] - x[0]), float(y[1] - y[0])
            face_bx = np.diff(potential, axis=1) / dy
            face_by = -np.diff(potential, axis=2) / dx
            fields["divB"] = (np.diff(face_bx, axis=2) / dx
                              + np.diff(face_by, axis=1) / dy)
        diagnostics = {
            name: source[f"diagnostics/{name}"][:] for name in DIAGNOSTIC_NAMES
        }
        for name in (
            "solid_constraint_energy", "absorption_energy_exchange", "boundary_energy_outflow",
            "plasma_energy", "current_drive_energy_correction", "native_magnetic_energy",
            "magnetic_averaging_energy", "max_current_constraint_error",
            "max_stage_current_constraint_error", "max_abs_ct_divB",
            "injection_energy",
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


def strongest_perturbation(fields: dict[str, np.ndarray], names: tuple[str, ...]) -> str:
    amplitudes = [
        float(np.max(np.abs(fields[name] - fields[name][0]))) for name in names
    ]
    return names[int(np.argmax(amplitudes))]


def make_figure(
    x: np.ndarray,
    y: np.ndarray,
    times: np.ndarray,
    fields: dict[str, np.ndarray],
    diagnostics: dict[str, np.ndarray],
    filename: str,
    initial_frame: int,
    *,
    boundary_x: str = "periodic",
    boundary_y: str = "periodic",
    solid_mask: np.ndarray | None = None,
    velocity_component: str = "auto",
    absorption_width: float = 0.0,
    density_limit: float | None = None,
):
    if density_limit is not None and (not np.isfinite(density_limit) or density_limit <= 0.0):
        raise ValueError("density_limit must be finite and positive")
    prescribed_current = "current_drive_energy_correction" in diagnostics
    antenna_name = strongest_component(
        fields, ("Jx_antenna", "Jy_antenna", "Jz_antenna")
    )
    if velocity_component not in ("auto", "x", "y", "z"):
        raise ValueError("velocity_component must be auto, x, y, or z")
    velocity_name = (strongest_perturbation(fields, ("u", "v", "w"))
                     if velocity_component == "auto"
                     else {"x": "u", "y": "v", "z": "w"}[velocity_component])
    velocity_perturbation = fields[velocity_name] - fields[velocity_name][0]
    momentum_density = fields["rho"] * fields[velocity_name]
    _, antenna_y, antenna_x = np.unravel_index(
        int(np.argmax(np.abs(fields[antenna_name]))), fields[antenna_name].shape
    )
    antenna_time_profile = fields[antenna_name][:, antenna_y, antenna_x]

    spatial_panels = (
        (
            ("Prescribed" if prescribed_current else "Impressed")
            + rf" current $J_{{{ANTENNA_LABELS[antenna_name]},\,\rm ant}}$",
            fields[antenna_name],
        ),
        (
            rf"Velocity response $\delta v_{VELOCITY_LABELS[velocity_name]}$",
            velocity_perturbation,
        ),
        (
            rf"Momentum density $\rho v_{VELOCITY_LABELS[velocity_name]}$",
            momentum_density,
        ),
        (r"Pressure perturbation $\delta p$", fields["p"] - fields["p"][0]),
        (r"Density perturbation $\delta\rho$", fields["rho"] - fields["rho"][0]),
        (r"Magnetic divergence $\nabla\!\cdot\!\mathbf{B}$", fields["divB"]),
        (r"GLM field $\psi$", fields["psi"]),
    )
    if solid_mask is not None:
        spatial_panels = tuple(
            (title, np.ma.array(data, mask=np.broadcast_to(solid_mask, data.shape)) if panel in (1, 2, 3, 4) else data)
            for panel, (title, data) in enumerate(spatial_panels)
        )
    flux_function = (fields["Az"] if "Az" in fields else
                     magnetic_flux_function(fields["Bx"], fields["By"], x, y,
                                            boundary_x, boundary_y))
    extent = (*cell_extent(x), *cell_extent(y))

    figure, axes = plt.subplots(3, 3, figsize=(16.5, 13))
    figure.subplots_adjust(
        left=0.055, right=0.94, top=0.90, bottom=0.09, wspace=0.48, hspace=0.34
    )
    try:
        figure.canvas.manager.set_window_title("MHD2D antenna results")
    except AttributeError:
        pass

    images = []
    for panel, (axis, (panel_title, data)) in enumerate(zip(axes.flat[:7], spatial_panels)):
        lower, upper = padded_limits(data, signed=True)
        extend = "neither"
        if panel == 4:
            # Keep a single scale for the entire slider, excluding body
            # placeholders and clipping thin, dense sheaths near the plates.
            values = np.ma.asarray(data).compressed()
            finite = values[np.isfinite(values)]
            bound = (density_limit if density_limit is not None else
                     float(np.percentile(np.abs(finite), 98)) if finite.size else 0.0)
            if bound > 0.0:
                lower, upper = -bound, bound
            below, above = np.any(finite < lower), np.any(finite > upper)
            extend = "both" if below and above else "min" if below else "max" if above else "neither"
        image = axis.imshow(
            data[initial_frame], origin="lower", extent=extent,
            interpolation="nearest", cmap="RdBu_r", vmin=lower, vmax=upper,
            aspect="equal",
        )
        axis.set(title=panel_title, xlabel="x", ylabel="y")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, extend=extend)
        images.append(image)
        if solid_mask is not None and np.any(solid_mask):
            axis.contour(x, y, solid_mask.astype(float), levels=[0.5], colors="0.25", linewidths=0.8)
        outline_absorption_layer(axis, x, y, boundary_x, boundary_y, absorption_width)

    flux_min, flux_max = float(np.min(flux_function)), float(np.max(flux_function))
    if np.isclose(flux_min, flux_max):
        flux_min, flux_max = flux_min - 1.0, flux_max + 1.0
    flux_levels = np.linspace(flux_min, flux_max, 21)
    field_lines = None

    energy_axis = axes.flat[7]
    total_change = diagnostics["total_energy"] - diagnostics["total_energy"][0]
    magnetic_change = diagnostics["magnetic_energy"] - diagnostics["magnetic_energy"][0]
    kinetic_change = diagnostics["kinetic_energy"] - diagnostics["kinetic_energy"][0]
    thermal_change = diagnostics["thermal_energy"] - diagnostics["thermal_energy"][0]
    injected_energy = diagnostics["antenna_energy_injected"]
    expected_energy = injected_energy.copy()
    if prescribed_current:
        # This is the difference between the powered-body FV reservoir and
        # native J*Adot work. Adding solid_constraint_energy here would count
        # the native circuit work twice.
        expected_energy += diagnostics["current_drive_energy_correction"]
    elif "solid_constraint_energy" in diagnostics:
        expected_energy += diagnostics["solid_constraint_energy"]
    if "absorption_energy_exchange" in diagnostics:
        expected_energy += diagnostics["absorption_energy_exchange"]
    if "boundary_energy_outflow" in diagnostics:
        expected_energy -= diagnostics["boundary_energy_outflow"]
    if "injection_energy" in diagnostics:
        expected_energy += diagnostics["injection_energy"]
    balance_label = (r"$W_{\rm circuit}+C_{\rm drive}+W_{\rm abs}-E_{\rm out}$"
                     if prescribed_current and "absorbing" in (boundary_x, boundary_y) else
                     r"$W_{\rm circuit}+C_{\rm drive}-E_{\rm out}$"
                     if prescribed_current else
                     r"$W_{\rm ant}+W_{\rm body}+W_{\rm abs}-E_{\rm out}$"
                     if "absorbing" in (boundary_x, boundary_y) else
                     r"$W_{\rm ant}+W_{\rm body}-E_{\rm out}$"
                     if "solid_constraint_energy" in diagnostics or boundary_x != "periodic" or boundary_y != "periodic"
                     else r"$\int P_{\rm ant}\,dt$")
    if "injection_energy" in diagnostics and np.any(diagnostics["injection_energy"] != 0.0):
        balance_label += r" $+E_{\rm feed}$"
    energy_axis.plot(times, total_change, color="black", linewidth=2.0, label=r"$\Delta E$")
    energy_axis.plot(
        times, expected_energy, color="tab:green", linestyle="--", linewidth=1.8,
        label=balance_label,
    )
    energy_axis.plot(times, magnetic_change, color="tab:blue", label=r"$\Delta E_B$")
    energy_axis.plot(times, kinetic_change, color="tab:orange", label=r"$\Delta E_K$")
    energy_axis.plot(times, thermal_change, color="tab:red", label=r"$\Delta E_{\rm th}$")
    if prescribed_current:
        energy_axis.plot(times, injected_energy, color="tab:purple", linestyle=":",
                         label=r"$W_{\rm circuit}$: native electrical work")
        energy_axis.plot(times, diagnostics["current_drive_energy_correction"],
                         color="0.5", linestyle=":", label=r"$C_{\rm drive}$: discrete correction")
    energy_marker = energy_axis.axvline(times[initial_frame], color="0.4", linestyle="--")
    energy_axis.axhline(0.0, color="0.65", linewidth=0.8)
    energy_axis.set(
        title="Energy changes and external balance", xlabel="time",
        ylabel="domain energy change",
    )
    energy_axis.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    energy_axis.grid(alpha=0.25)
    energy_axis.legend(fontsize="small")

    drive_axis = axes.flat[8]
    power_axis = drive_axis.twinx()
    current_line = drive_axis.plot(
        times, antenna_time_profile, color="tab:blue",
        linewidth=2.0, label=rf"$J_{{{ANTENNA_LABELS[antenna_name]},\,\rm ant}}$ at source",
    )[0]
    power_line = power_axis.plot(
        times, diagnostics["antenna_power"], color="tab:red", linewidth=1.8,
        label=r"$P_{\rm ant}$",
    )[0]
    drive_marker = drive_axis.axvline(times[initial_frame], color="0.4", linestyle="--")
    drive_axis.axhline(0.0, color="tab:blue", linewidth=0.7, alpha=0.35)
    drive_axis.set(
        title="Antenna temporal profile and power", xlabel="time",
        ylabel="antenna current at source",
    )
    power_axis.set_ylabel("native circuit power" if prescribed_current else "antenna power", color="tab:red")
    power_axis.tick_params(axis="y", colors="tab:red")
    power_axis.axhline(0.0, color="tab:red", linewidth=0.7, alpha=0.35)
    drive_axis.grid(alpha=0.25)
    drive_axis.legend(
        [current_line, power_line], [current_line.get_label(), power_line.get_label()],
        fontsize="small", loc="best",
    )

    title = figure.suptitle("")

    def show_frame(frame: int) -> None:
        nonlocal field_lines
        for image, (_, data) in zip(images, spatial_panels):
            image.set_data(data[frame])
        if field_lines is not None:
            field_lines.remove()
        field_lines = axes.flat[0].contour(
            x, y, flux_function[frame], levels=flux_levels,
            colors="black", linewidths=0.65, alpha=0.7,
        )
        energy_marker.set_xdata([times[frame], times[frame]])
        drive_marker.set_xdata([times[frame], times[frame]])
        title.set_text(
            f"2-D MHD antenna: t = {times[frame]:.6g}   "
            f"(frame {frame}/{times.size - 1}, {filename})\n"
            f"Boundaries: x={boundary_x}, y={boundary_y}"
            + ("; gray outlines: stationary bodies" if solid_mask is not None and np.any(solid_mask) else "")
            + (f"; dotted lines: absorber width={absorption_width:g}" if "absorbing" in (boundary_x, boundary_y) else "")
            + (f"\nPowered-current constraint: max |Jz-Jz_target| = {diagnostics['max_current_constraint_error'][frame]:.3e}"
               if "max_current_constraint_error" in diagnostics else "")
        )
        figure.canvas.draw_idle()

    show_frame(initial_frame)
    slider = None
    if times.size > 1:
        slider_axis = figure.add_axes((0.16, 0.035, 0.68, 0.025))
        slider = Slider(
            slider_axis, "Time frame", 0, times.size - 1,
            valinit=initial_frame, valstep=1, valfmt="%0.0f",
        )
        slider.on_changed(lambda value: show_frame(int(round(value))))

        def on_key(event) -> None:
            current = int(round(slider.val))
            if event.key in ("right", "up"):
                slider.set_val(min(current + 1, times.size - 1))
            elif event.key in ("left", "down"):
                slider.set_val(max(current - 1, 0))
            elif event.key == "home":
                slider.set_val(0)
            elif event.key == "end":
                slider.set_val(times.size - 1)

        figure.canvas.mpl_connect("key_press_event", on_key)
    return figure, slider


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_file", type=Path, help="antenna HDF5 result file")
    parser.add_argument(
        "--frame", type=int,
        help="initial frame; negative values count from the end (default: peak antenna current)",
    )
    parser.add_argument("--save", type=Path, help="save the selected frame instead of opening a window")
    parser.add_argument("--velocity-component", choices=("auto", "x", "y", "z"), default="auto",
                        help="velocity and momentum-density component; use x for axial thruster flow")
    parser.add_argument("--density-limit", type=float,
                        help="symmetric density-perturbation color limit (default: 98th percentile of |delta rho| over all plasma frames)")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    print(f"Loading {arguments.result_file} ...")
    x, y, times, fields, diagnostics = load_antenna_results(arguments.result_file)
    if arguments.frame is None:
        frame = int(np.argmax(diagnostics["max_abs_antenna_current"]))
    else:
        frame = arguments.frame if arguments.frame >= 0 else times.size + arguments.frame
    if not 0 <= frame < times.size:
        raise ValueError(f"Frame {arguments.frame} is outside 0..{times.size - 1}")
    print(
        f"Loaded {times.size} frames on a {x.size} x {y.size} grid, "
        f"t={times[0]:g}..{times[-1]:g}; opening frame {frame}"
    )
    figure, slider = make_figure(
        x, y, times, fields, diagnostics, arguments.result_file.name, frame,
        **load_plot_metadata(arguments.result_file),
        velocity_component=arguments.velocity_component,
        density_limit=arguments.density_limit,
    )
    figure._time_slider = slider  # type: ignore[attr-defined]
    if arguments.save is not None:
        arguments.save.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(arguments.save, dpi=160)
        print(f"Saved {arguments.save}")
    else:
        print("Use the slider or arrow keys to change time; Home/End select the endpoints.")
        plt.show()


if __name__ == "__main__":
    main()
