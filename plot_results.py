#!/usr/bin/env python3
"""Interactive viewer for the single HDF5 file written by mhd2d."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider


REQUIRED_FIELDS = ("rho", "u", "v", "p", "Bx", "By", "Jz", "divB", "psi")
REQUIRED_DIAGNOSTICS = ("total_energy", "magnetic_energy", "thermal_energy")


def load_results(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as source:
        required_groups = ("time", "fields", "diagnostics", "x", "y")
        if any(name not in source for name in required_groups):
            raise ValueError(f"{path} is not an mhd2d result file")
        x, y, times = source["x"][:], source["y"][:], source["time"][:]
        missing_fields = [name for name in REQUIRED_FIELDS if name not in source["fields"]]
        missing_diagnostics = [
            name for name in REQUIRED_DIAGNOSTICS if name not in source["diagnostics"]
        ]
        if missing_fields:
            raise ValueError(f"{path} is missing fields: {', '.join(missing_fields)}")
        if missing_diagnostics:
            raise ValueError(f"{path} is missing diagnostics: {', '.join(missing_diagnostics)}")
        values = {name: source[f"fields/{name}"][:] for name in REQUIRED_FIELDS}
        energies = {
            name: source[f"diagnostics/{name}"][:] for name in REQUIRED_DIAGNOSTICS
        }
    if times.size == 0:
        raise ValueError(f"{path} contains no output frames")
    expected_shape = (times.size, y.size, x.size)
    for name, field in values.items():
        if field.shape != expected_shape:
            raise ValueError(f"Field {name} has shape {field.shape}; expected {expected_shape}")
    for name, history in energies.items():
        if history.shape != times.shape:
            raise ValueError(f"Diagnostic {name} is not aligned with output times")
    return x, y, times, values, energies


def magnetic_flux_function(
    bx: np.ndarray, by: np.ndarray, x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """Reconstruct A_z with Bx=dA_z/dy and By=-dA_z/dx.

    The inversion uses the central-difference symbols used by the solver.  It
    returns the least-squares solenoidal projection if a small div(B) is present.
    """
    nx, ny = x.size, y.size
    dx = float(np.median(np.diff(x))) if nx > 1 else 1.0
    dy = float(np.median(np.diff(y))) if ny > 1 else 1.0
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
    symbol_x = np.sin(kx * dx) / dx
    symbol_y = np.sin(ky * dy) / dy
    derivative_x, derivative_y = np.meshgrid(symbol_x, symbol_y)
    denominator = derivative_x**2 + derivative_y**2

    bx_hat = np.fft.fft2(bx, axes=(-2, -1))
    by_hat = np.fft.fft2(by, axes=(-2, -1))
    numerator = -1j * derivative_y * bx_hat + 1j * derivative_x * by_hat
    flux_hat = np.zeros_like(numerator)
    np.divide(numerator, denominator, out=flux_hat, where=denominator > 1.0e-24)
    flux = np.fft.ifft2(flux_hat, axes=(-2, -1)).real

    # The periodic inversion cannot determine a uniform field.  Add its linear
    # flux function so contours also represent problems with a nonzero mean B.
    x_grid, y_grid = np.meshgrid(x - x[0], y - y[0])
    mean_bx = np.mean(bx, axis=(-2, -1))
    mean_by = np.mean(by, axis=(-2, -1))
    flux += mean_bx[:, None, None] * y_grid - mean_by[:, None, None] * x_grid
    flux -= np.mean(flux, axis=(-2, -1), keepdims=True)
    return flux


def padded_limits(data: np.ndarray, signed: bool) -> tuple[float, float]:
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return (-1.0, 1.0)
    if signed:
        bound = float(np.max(np.abs(finite))) or 1.0
        return (-bound, bound)
    lower, upper = float(np.min(finite)), float(np.max(finite))
    if lower == upper:
        padding = max(abs(lower) * 0.01, 1.0e-12)
        return (lower - padding, upper + padding)
    return (lower, upper)


def reconnection_history(flux: np.ndarray, jz: np.ndarray, times: np.ndarray):
    """Return flux differences and d(flux difference)/dt for both Harris sheets."""
    ny = flux.shape[1]
    current_profile = np.mean(np.abs(jz[0]), axis=1)
    first = int(np.argmax(current_profile))
    indices = np.arange(ny)
    circular_distance = np.minimum(np.abs(indices - first), ny - np.abs(indices - first))
    candidates = np.where(circular_distance >= max(2, ny // 4), current_profile, -np.inf)
    second = int(np.argmax(candidates))
    sheet_indices = tuple(sorted((first, second)))
    reconnected_flux = np.stack(
        [np.ptp(flux[:, sheet, :], axis=1) for sheet in sheet_indices], axis=0
    )
    if times.size == 1:
        rates = np.zeros_like(reconnected_flux)
    else:
        edge_order = 2 if times.size >= 3 else 1
        rates = np.stack(
            [np.gradient(history, times, edge_order=edge_order) for history in reconnected_flux],
            axis=0,
        )
    return sheet_indices, reconnected_flux, rates


def cell_extent(coordinates: np.ndarray) -> tuple[float, float]:
    spacing = float(np.median(np.diff(coordinates))) if coordinates.size > 1 else 1.0
    return coordinates[0] - 0.5 * spacing, coordinates[-1] + 0.5 * spacing


def make_figure(x: np.ndarray, y: np.ndarray, times: np.ndarray,
                values: dict[str, np.ndarray], energies: dict[str, np.ndarray],
                filename: str, initial_frame: int):
    flux_function = magnetic_flux_function(values["Bx"], values["By"], x, y)
    spatial_panels = (
        (r"Out-of-plane current $J_z$", values["Jz"], "RdBu_r", True),
        (r"Vertical velocity $v_y$", values["v"], "RdBu_r", True),
        (r"Horizontal velocity $v_x$", values["u"], "RdBu_r", True),
        (r"Pressure $p$", values["p"], "magma", False),
        (r"Density $\rho$", values["rho"], "viridis", False),
        (r"Magnetic divergence $\nabla\!\cdot\!\mathbf{B}$", values["divB"], "RdBu_r", True),
        (r"GLM field $\psi$", values["psi"], "RdBu_r", True),
    )
    extent = (*cell_extent(x), *cell_extent(y))
    figure, axes = plt.subplots(3, 3, figsize=(16.5, 13))
    figure.subplots_adjust(left=0.055, right=0.97, top=0.92, bottom=0.09,
                           wspace=0.30, hspace=0.32)
    try:
        figure.canvas.manager.set_window_title("MHD2D results")
    except AttributeError:
        pass

    images = []
    for axis, (panel_title, data, color_map, signed) in zip(axes.flat[:7], spatial_panels):
        lower, upper = padded_limits(data, signed)
        image = axis.imshow(data[initial_frame], origin="lower", extent=extent,
                            interpolation="nearest", cmap=color_map, vmin=lower,
                            vmax=upper, aspect="equal")
        axis.set(title=panel_title, xlabel="x", ylabel="y")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        images.append(image)

    flux_min, flux_max = float(np.min(flux_function)), float(np.max(flux_function))
    if np.isclose(flux_min, flux_max):
        flux_min, flux_max = flux_min - 1.0, flux_max + 1.0
    flux_levels = np.linspace(flux_min, flux_max, 21)
    field_lines = None

    energy_axis = axes.flat[7]
    magnetic_energy = energies["magnetic_energy"]
    thermal_energy = energies["thermal_energy"]
    total_energy = energies["total_energy"]
    energy_axis.plot(times, total_energy, label="total", color="black", linewidth=2.0)
    energy_axis.plot(times, magnetic_energy, label="magnetic", color="tab:blue")
    energy_axis.plot(times, thermal_energy, label="thermal", color="tab:red")
    energy_marker = energy_axis.axvline(times[initial_frame], color="0.4", linestyle="--")
    energy_axis.set(title="Energy history", xlabel="time", ylabel="domain-integrated energy")
    energy_axis.grid(alpha=0.25)
    energy_axis.legend()

    sheet_indices, _, reconnection_rates = reconnection_history(
        flux_function, values["Jz"], times
    )
    reconnection_axis = axes.flat[8]
    mean_rate = np.mean(reconnection_rates, axis=0)
    for sheet_number, (sheet, rate) in enumerate(zip(sheet_indices, reconnection_rates), start=1):
        reconnection_axis.plot(
            times, rate, linestyle="--", linewidth=1.1, alpha=0.7,
            label=rf"sheet {sheet_number} ($y\approx {y[sheet]:.3g}$)",
        )
    reconnection_axis.plot(times, mean_rate, color="black", linewidth=2.0, label="sheet mean")
    reconnection_marker = reconnection_axis.axvline(
        times[initial_frame], color="0.4", linestyle="--"
    )
    reconnection_axis.axhline(0.0, color="0.6", linewidth=0.8)
    reconnection_axis.set(
        title=r"Reconnection rate $d\Delta A_z/dt$",
        xlabel="time",
        ylabel=r"$d[\max_x(A_z)-\min_x(A_z)]/dt$",
    )
    reconnection_axis.grid(alpha=0.25)
    reconnection_axis.legend(fontsize="small")
    title = figure.suptitle("")

    def show_frame(frame: int) -> None:
        nonlocal field_lines
        for image, (_, data, _, _) in zip(images, spatial_panels):
            image.set_data(data[frame])
        if field_lines is not None:
            field_lines.remove()
        field_lines = axes.flat[0].contour(
            x, y, flux_function[frame], levels=flux_levels,
            colors="black", linewidths=0.65, alpha=0.75,
        )
        energy_marker.set_xdata([times[frame], times[frame]])
        reconnection_marker.set_xdata([times[frame], times[frame]])
        title.set_text(
            f"2-D MHD: t = {times[frame]:.6g}   "
            f"(frame {frame}/{times.size - 1}, {filename})"
        )
        figure.canvas.draw_idle()

    show_frame(initial_frame)
    slider = None
    if times.size > 1:
        slider_axis = figure.add_axes((0.16, 0.035, 0.68, 0.025))
        slider = Slider(slider_axis, "Time frame", 0, times.size - 1,
                        valinit=initial_frame, valstep=1, valfmt="%0.0f")
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
    parser.add_argument("result_file", type=Path, help="HDF5 result file")
    parser.add_argument("--frame", type=int, default=0,
                        help="initial frame; negative values count from the end")
    parser.add_argument("--save", type=Path,
                        help="save the selected frame instead of opening a window")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    print(f"Loading {arguments.result_file} ...")
    x, y, times, values, energies = load_results(arguments.result_file)
    frame = arguments.frame if arguments.frame >= 0 else times.size + arguments.frame
    if not 0 <= frame < times.size:
        raise ValueError(f"Frame {arguments.frame} is outside 0..{times.size - 1}")
    print(f"Loaded {times.size} frames on a {x.size} x {y.size} grid, t={times[0]:g}..{times[-1]:g}")
    figure, slider = make_figure(
        x, y, times, values, energies, arguments.result_file.name, frame
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
