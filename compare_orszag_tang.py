#!/usr/bin/env python3
"""Compare an mhd2d Orszag-Tang result with the bundled Athena++ snapshot."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


FIELD_NAMES = ("rho", "p", "u", "v", "Bx", "By")
DIAGNOSTIC_NAMES = (
    "total_energy", "mass", "max_abs_divB", "l1_divB",
)
SLICE_LOCATIONS = (0.4277, 0.3125)


def load_result(path: Path, requested_time: float):
    if not path.is_file():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as source:
        times = source["time"][:]
        if times.size == 0:
            raise ValueError(f"{path} contains no frames")
        frame = int(np.argmin(np.abs(times - requested_time)))
        tolerance = 128.0 * np.finfo(float).eps * max(1.0, abs(requested_time))
        if abs(times[frame] - requested_time) > tolerance:
            raise ValueError(
                f"{path} has no frame at t={requested_time:g}; nearest is t={times[frame]:g}"
            )
        x, y = source["x"][:], source["y"][:]
        fields = {name: source[f"fields/{name}"][frame] for name in FIELD_NAMES}
        diagnostics = {
            name: source[f"diagnostics/{name}"][:] for name in DIAGNOSTIC_NAMES
        }
    return x, y, times, fields, diagnostics, frame


def load_reference(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as source:
        required = {"x", "y", "time", "commit", "method", *FIELD_NAMES}
        missing = required - set(source.files)
        if missing:
            raise ValueError(f"{path} is missing: {', '.join(sorted(missing))}")
        result = {name: np.array(source[name]) for name in required}
    return result


def periodic_slice(field: np.ndarray, y: np.ndarray, location: float) -> np.ndarray:
    """Linearly interpolate a cell-centered field to an exact periodic y."""
    if y.size < 2:
        return field[0]
    dy = float(np.median(np.diff(y)))
    position = (location - y[0]) / dy
    lower_unwrapped = int(np.floor(position))
    weight = position - lower_unwrapped
    lower = lower_unwrapped % y.size
    upper = (lower + 1) % y.size
    return (1.0 - weight) * field[lower] + weight * field[upper]


def comparison_metrics(actual: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    difference = actual - reference
    reference_rms = float(np.sqrt(np.mean(reference**2)))
    return {
        "l1": float(np.mean(np.abs(difference))),
        "relative_l2": float(np.sqrt(np.mean(difference**2)) / reference_rms),
        "correlation": float(np.corrcoef(actual.ravel(), reference.ravel())[0, 1]),
        "linf": float(np.max(np.abs(difference))),
    }


def print_report(
    result_path: Path,
    reference_path: Path,
    time: float,
    fields: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    y: np.ndarray,
) -> dict[str, dict[str, float]]:
    metrics = {
        name: comparison_metrics(fields[name], reference[name]) for name in FIELD_NAMES
    }
    print(f"Orszag-Tang comparison at t={time:g}")
    print(f"  result:    {result_path}")
    print(f"  reference: {reference_path}")
    print(f"  Athena++ commit: {reference['commit'].item()}")
    print(f"  method: {reference['method'].item()}")
    for name in FIELD_NAMES:
        values = metrics[name]
        print(
            f"  {name:>3}: L1={values['l1']:.6e}, "
            f"relative L2={values['relative_l2']:.6e}, "
            f"correlation={values['correlation']:.8f}"
        )
    for location in SLICE_LOCATIONS:
        actual_slice = periodic_slice(fields["p"], y, location)
        reference_slice = periodic_slice(reference["p"], reference["y"], location)
        values = comparison_metrics(actual_slice, reference_slice)
        print(
            f"  p(x, y={location:g}): L1={values['l1']:.6e}, "
            f"Linf={values['linf']:.6e}"
        )
    return metrics


def make_figure(
    x: np.ndarray,
    y: np.ndarray,
    times: np.ndarray,
    fields: dict[str, np.ndarray],
    diagnostics: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    metrics: dict[str, dict[str, float]],
    time: float,
):
    figure, axes = plt.subplots(2, 3, figsize=(16, 9.5))
    figure.subplots_adjust(
        left=0.06, right=0.96, top=0.88, bottom=0.08, wspace=0.27, hspace=0.31
    )
    extent = (0.0, 1.0, 0.0, 1.0)
    density_min = min(float(np.min(fields["rho"])), float(np.min(reference["rho"])))
    density_max = max(float(np.max(fields["rho"])), float(np.max(reference["rho"])))

    actual_image = axes[0, 0].imshow(
        fields["rho"], origin="lower", extent=extent, cmap="viridis",
        vmin=density_min, vmax=density_max, interpolation="nearest", aspect="equal",
    )
    axes[0, 0].set(title="mhd2d density", xlabel="x", ylabel="y")
    figure.colorbar(actual_image, ax=axes[0, 0], fraction=0.046, pad=0.04)

    reference_image = axes[0, 1].imshow(
        reference["rho"], origin="lower", extent=extent, cmap="viridis",
        vmin=density_min, vmax=density_max, interpolation="nearest", aspect="equal",
    )
    axes[0, 1].set(title="Athena++ density", xlabel="x", ylabel="y")
    figure.colorbar(reference_image, ax=axes[0, 1], fraction=0.046, pad=0.04)

    density_difference = fields["rho"] - reference["rho"]
    difference_limit = float(np.max(np.abs(density_difference))) or 1.0
    difference_image = axes[0, 2].imshow(
        density_difference, origin="lower", extent=extent, cmap="RdBu_r",
        vmin=-difference_limit, vmax=difference_limit, interpolation="nearest", aspect="equal",
    )
    axes[0, 2].set(
        title=(r"Density difference: mhd2d $-$ Athena++" + "\n" +
               rf"relative $L_2={100.0 * metrics['rho']['relative_l2']:.3f}\%$"),
        xlabel="x", ylabel="y",
    )
    figure.colorbar(difference_image, ax=axes[0, 2], fraction=0.046, pad=0.04)

    for axis, location in zip(axes[1, :2], SLICE_LOCATIONS):
        actual_slice = periodic_slice(fields["p"], y, location)
        reference_slice = periodic_slice(reference["p"], reference["y"], location)
        axis.plot(x, actual_slice, color="tab:blue", linewidth=1.8, label="mhd2d")
        axis.plot(
            reference["x"], reference_slice, color="black", linestyle="--",
            linewidth=1.3, label="Athena++",
        )
        axis.set(
            title=rf"Pressure slice at $y={location}$", xlabel="x", ylabel="pressure"
        )
        axis.grid(alpha=0.25)
        axis.legend()

    diagnostic_axis = axes[1, 2]
    energy_error = np.abs(diagnostics["total_energy"] / diagnostics["total_energy"][0] - 1.0)
    mass_error = np.abs(diagnostics["mass"] / diagnostics["mass"][0] - 1.0)
    dx = float(np.median(np.diff(x)))
    floor = np.finfo(float).eps
    diagnostic_axis.semilogy(times, np.maximum(energy_error, floor), label=r"$|E/E_0-1|$")
    diagnostic_axis.semilogy(times, np.maximum(mass_error, floor), label=r"$|M/M_0-1|$")
    diagnostic_axis.semilogy(
        times, np.maximum(dx * diagnostics["max_abs_divB"], floor),
        label=r"$\Delta x\,\max|\nabla\!\cdot\!B|$",
    )
    diagnostic_axis.semilogy(
        times, np.maximum(dx * diagnostics["l1_divB"], floor),
        label=r"$\Delta x\,\langle|\nabla\!\cdot\!B|\rangle$",
    )
    diagnostic_axis.set(
        title="Conservation and divergence", xlabel="time", ylabel="dimensionless measure"
    )
    diagnostic_axis.grid(alpha=0.25, which="both")
    diagnostic_axis.legend(fontsize="small")

    figure.suptitle(
        f"Orszag-Tang vortex at t={time:g}: 256² mhd2d versus 256² Athena++ HLLD"
    )
    return figure


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path, help="mhd2d Orszag-Tang HDF5 result")
    parser.add_argument(
        "--reference", type=Path,
        default=Path("reference/orszag_tang_athena_256.npz"),
        help="Athena++ reference NPZ",
    )
    parser.add_argument("--time", type=float, default=0.5, help="comparison time")
    parser.add_argument("--save", type=Path, help="save the comparison instead of showing it")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    x, y, times, fields, diagnostics, _ = load_result(arguments.result, arguments.time)
    reference = load_reference(arguments.reference)
    if fields["rho"].shape != reference["rho"].shape:
        raise ValueError(
            f"Grid mismatch: result {fields['rho'].shape}, reference {reference['rho'].shape}"
        )
    metrics = print_report(
        arguments.result, arguments.reference, arguments.time, fields, reference, y
    )
    figure = make_figure(
        x, y, times, fields, diagnostics, reference, metrics, arguments.time
    )
    if arguments.save is not None:
        arguments.save.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(arguments.save, dpi=170)
        print(f"Saved {arguments.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
