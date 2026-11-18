#!/usr/bin/env python3
"""Prepare a Python-defined MHD problem and launch the C++ solver."""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Mapping

import h5py
import numpy as np


GRID_KEYS = {"nx", "ny", "lx", "ly"}
SOLVER_DEFAULTS = {
    "gamma": 5.0 / 3.0,
    "cfl": 0.35,
    "diffusion_cfl": 0.45,
    "resistivity": 0.0,
    "antenna_model": "resistive_impressed",
    "t_end": 1.0,
    "max_steps": 200_000,
    "second_order": True,
    "limiter_theta": 1.5,
    "glm_speed_factor": 1.0,
    "glm_damping": 0.1,
    "rho_floor": 1.0e-8,
    "p_floor": 1.0e-8,
    "threads": 0,
    "boundary_x": "periodic",
    "boundary_y": "periodic",
    "absorption_width": 0.0,
    "absorption_rate": 0.5,
    "absorption_magnetic_diffusivity": 0.05,
}
OUTPUT_DEFAULTS = {"file": "output/results.h5", "interval": 0.1}
FLUID_KEYS = {"rho", "u", "v", "w", "p"}
ANTENNA_KEYS = {"field", "space_profile", "time_profile", "space_time_profile"}
ANTENNA_COMPONENT = {"Jx": 0, "Jy": 1, "Jz": 2}
BOUNDARY_AMBIENT_KEYS = FLUID_KEYS | {"Bx", "By", "Bz"}
OPTIONAL_METADATA_GROUPS = ("physical_scaling", "antenna_hardware")


def load_problem(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("mhd_user_problem", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load problem file {path}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.resolve().parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def checked_dictionary(name: str, supplied: object, allowed: set[str], defaults: dict | None = None) -> dict:
    if not isinstance(supplied, Mapping):
        raise TypeError(f"{name} must be a dictionary")
    unknown = set(supplied) - allowed
    if unknown:
        raise ValueError(f"Unknown {name} keys: {', '.join(sorted(unknown))}")
    result = {} if defaults is None else dict(defaults)
    result.update(supplied)
    return result


def checked_problem_metadata(problem: ModuleType) -> dict[str, dict]:
    """Validate scalar SI/design annotations before preparing or launching a run."""
    metadata = {}
    for name in OPTIONAL_METADATA_GROUPS:
        if not hasattr(problem, name):
            continue
        supplied = getattr(problem, name)
        if not isinstance(supplied, Mapping):
            raise TypeError(f"{name} must be a dictionary of scalar attributes")
        values = {}
        for key, value in supplied.items():
            if not isinstance(key, str) or not key.isascii() or not key.isidentifier():
                raise ValueError(f"{name} attribute names must be plain identifiers without '/'")
            label = f"{name}.{key}"
            if isinstance(value, (bool, np.bool_)):
                values[key] = bool(value)
            elif isinstance(value, str):
                if "\0" in value:
                    raise ValueError(f"{label} must not contain a null character")
                values[key] = value
            elif isinstance(value, (int, np.integer)):
                if not np.iinfo(np.int64).min <= int(value) <= np.iinfo(np.int64).max:
                    raise ValueError(f"{label} integer is outside the supported 64-bit range")
                values[key] = int(value)
            elif isinstance(value, (float, np.floating)):
                number = float(value)
                if not np.isfinite(number):
                    raise ValueError(f"{label} must be finite")
                values[key] = number
            else:
                raise TypeError(f"{label} must be a scalar number, boolean, or string")
        metadata[name] = values
    return metadata


def copy_problem_metadata(prepared: Path, output: Path) -> None:
    """Attach validated annotations to a closed, successfully completed result."""
    with h5py.File(prepared, "r") as source:
        names = [name for name in OPTIONAL_METADATA_GROUPS if name in source]
        if not names:
            return  # Leave legacy result files completely untouched.
        with h5py.File(output, "r+") as destination:
            for name in names:
                if name in destination:
                    raise ValueError(f"Solver output already contains metadata group {name}")
                source.copy(name, destination)


def profile_array(value: object, shape: tuple[int, int], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    try:
        result = np.array(np.broadcast_to(array, shape), dtype=np.float64, copy=True)
    except ValueError as error:
        raise ValueError(f"Profile {name} cannot be broadcast to grid shape {shape}") from error
    if not np.all(np.isfinite(result)):
        raise ValueError(f"Profile {name} contains NaN or infinity")
    return result


def vector_profile(value: object, shape: tuple[int, int], prefix: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    names = (f"{prefix}x", f"{prefix}y", f"{prefix}z")
    if isinstance(value, Mapping):
        missing = [name for name in names if name not in value]
        if missing:
            raise ValueError(f"{prefix} profile is missing: {', '.join(missing)}")
        components = tuple(value[name] for name in names)
    else:
        if not isinstance(value, (tuple, list)) or len(value) != 3:
            raise TypeError(f"{prefix} profile must return three components or a mapping")
        components = tuple(value)
    return tuple(profile_array(component, shape, name) for component, name in zip(components, names))  # type: ignore[return-value]


def evaluate_profile(profile: object, *coordinates: object) -> object:
    return profile(*coordinates) if callable(profile) else profile


def temporal_profile_array(profile: object, times: np.ndarray, name: str) -> np.ndarray:
    try:
        raw_values = evaluate_profile(profile, times)
        values = np.array(np.broadcast_to(np.asarray(raw_values, dtype=np.float64), times.shape), copy=True)
    except (TypeError, ValueError):
        if not callable(profile):
            raise ValueError(f"Temporal profile {name} cannot be broadcast to {times.shape}")
        values = np.asarray([profile(float(time)) for time in times], dtype=np.float64)
    if values.shape != times.shape or not np.all(np.isfinite(values)):
        raise ValueError(f"Temporal profile {name} must produce one finite value per sample")
    return values


def prepare_antennas(
    problem: ModuleType,
    destination: h5py.File,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    shape: tuple[int, int],
    t_end: float,
) -> None:
    """Sample Python antenna functions for interpolation by the C++ solver."""
    supplied = getattr(problem, "antennas", ())
    if supplied is None:
        supplied = ()
    if not isinstance(supplied, (tuple, list)):
        raise TypeError("antennas must be a list or tuple of antenna dictionaries")
    if not supplied:
        return

    sampling_interval = float(getattr(problem, "antenna_sampling_interval", 0.0))
    if not np.isfinite(sampling_interval) or sampling_interval <= 0.0:
        raise ValueError(
            "A positive antenna_sampling_interval is required when antennas are defined"
        )
    if t_end < 0.0:
        raise ValueError("solver t_end must be non-negative")
    number_of_intervals = max(1, int(np.ceil(t_end / sampling_interval)))
    times = np.linspace(0.0, t_end, number_of_intervals + 1) if t_end > 0.0 else np.array([0.0])
    currents = np.zeros((times.size, 3, *shape), dtype=np.float64)

    for antenna_index, raw_antenna in enumerate(supplied):
        antenna = checked_dictionary(
            f"antennas[{antenna_index}]", raw_antenna, ANTENNA_KEYS
        )
        field = antenna.get("field")
        if field not in ANTENNA_COMPONENT:
            raise ValueError(
                f"antennas[{antenna_index}].field must be Jx, Jy, or Jz"
            )
        component = ANTENNA_COMPONENT[field]
        has_space_time = "space_time_profile" in antenna
        if has_space_time and ("space_profile" in antenna or "time_profile" in antenna):
            raise ValueError(
                f"antennas[{antenna_index}] cannot combine space_time_profile with "
                "space_profile or time_profile"
            )
        if has_space_time:
            profile = antenna["space_time_profile"]
            for time_index, time in enumerate(times):
                values = evaluate_profile(profile, x_grid, y_grid, float(time))
                currents[time_index, component] += profile_array(
                    values, shape, f"antennas[{antenna_index}].space_time_profile"
                )
        else:
            if "space_profile" not in antenna:
                raise ValueError(
                    f"antennas[{antenna_index}] requires space_profile or space_time_profile"
                )
            spatial = profile_array(
                evaluate_profile(antenna["space_profile"], x_grid, y_grid),
                shape,
                f"antennas[{antenna_index}].space_profile",
            )
            temporal = temporal_profile_array(
                antenna.get("time_profile", 1.0),
                times,
                f"antennas[{antenna_index}].time_profile",
            )
            currents[:, component] += temporal[:, None, None] * spatial[None, :, :]

    antenna_group = destination.create_group("antenna")
    antenna_group.attrs["sampling_interval_requested"] = sampling_interval
    antenna_group.create_dataset("time", data=times)
    chunks = (1, min(64, shape[0]), min(64, shape[1]))
    for component, name in enumerate(("Jx", "Jy", "Jz")):
        antenna_group.create_dataset(
            name,
            data=currents[:, component],
            chunks=chunks,
            shuffle=True,
            compression="gzip",
            compression_opts=4,
        )
    actual_interval = 0.0 if times.size < 2 else float(np.max(np.diff(times)))
    print(
        f"Prepared {len(supplied)} antenna(s): {times.size} samples, "
        f"maximum interval {actual_interval:.6g}",
        flush=True,
    )


def periodic_curl(
    magnetic: tuple[np.ndarray, np.ndarray, np.ndarray], dx: float, dy: float,
    boundary_x: str = "periodic", boundary_y: str = "periodic",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Central curl, using wrapped or zero-gradient exterior ghost cells."""
    bx, by, bz = magnetic

    def derivative(array: np.ndarray, axis: int, spacing: float, boundary: str) -> np.ndarray:
        if boundary == "periodic":
            return (np.roll(array, -1, axis=axis) - np.roll(array, 1, axis=axis)) / (2.0 * spacing)
        padding = [(0, 0), (0, 0)]
        padding[axis] = (1, 1)
        padded = np.pad(array, padding, mode="edge")
        forward, backward = [slice(None)] * 2, [slice(None)] * 2
        forward[axis], backward[axis] = slice(2, None), slice(None, -2)
        return (padded[tuple(forward)] - padded[tuple(backward)]) / (2.0 * spacing)

    derivative_x = lambda array: derivative(array, 1, dx, boundary_x)
    derivative_y = lambda array: derivative(array, 0, dy, boundary_y)
    return derivative_y(bz), -derivative_x(bz), derivative_x(by) - derivative_y(bx)


def magnetic_field_from_current(
    current: tuple[np.ndarray, np.ndarray, np.ndarray], dx: float, dy: float,
    mean_field: tuple[float, float, float], tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Invert the periodic discrete curl with div(B)=0 using FFTs."""
    jx, jy, jz = current
    ny, nx = jx.shape
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
    # Symbols of the central difference used by the C++ Ampere diagnostic.
    sx = np.sin(kx * dx) / dx
    sy = np.sin(ky * dy) / dy
    symbol_x, symbol_y = np.meshgrid(sx, sy)
    wave_number_squared = symbol_x**2 + symbol_y**2
    invertible = wave_number_squared > 1.0e-24

    jx_hat, jy_hat, jz_hat = (np.fft.fft2(component) for component in current)
    bx_hat = np.zeros_like(jx_hat)
    by_hat = np.zeros_like(jy_hat)
    bz_hat = np.zeros_like(jz_hat)
    bx_hat[invertible] = 1j * symbol_y[invertible] * jz_hat[invertible] / wave_number_squared[invertible]
    by_hat[invertible] = -1j * symbol_x[invertible] * jz_hat[invertible] / wave_number_squared[invertible]
    bz_hat[invertible] = 1j * (
        symbol_x[invertible] * jy_hat[invertible] - symbol_y[invertible] * jx_hat[invertible]
    ) / wave_number_squared[invertible]
    bx_hat[0, 0] = mean_field[0] * nx * ny
    by_hat[0, 0] = mean_field[1] * nx * ny
    bz_hat[0, 0] = mean_field[2] * nx * ny
    magnetic = tuple(np.fft.ifft2(component).real for component in (bx_hat, by_hat, bz_hat))
    check_current_compatibility(magnetic, current, dx, dy, tolerance, "FFT-reconstructed B")
    return magnetic  # type: ignore[return-value]


def check_current_compatibility(
    magnetic: tuple[np.ndarray, np.ndarray, np.ndarray],
    prescribed: tuple[np.ndarray, np.ndarray, np.ndarray],
    dx: float,
    dy: float,
    tolerance: float,
    description: str,
    boundary_x: str = "periodic",
    boundary_y: str = "periodic",
) -> None:
    calculated = periodic_curl(magnetic, dx, dy, boundary_x, boundary_y)
    error = max(float(np.max(np.abs(actual - expected))) for actual, expected in zip(calculated, prescribed))
    scale = max(1.0, *(float(np.max(np.abs(component))) for component in prescribed))
    relative_error = error / scale
    if relative_error > tolerance:
        raise ValueError(
            f"The prescribed current is incompatible with {description}: "
            f"normalized max error {relative_error:.3e} exceeds tolerance {tolerance:.3e}"
        )
    print(f"Ampere compatibility check: normalized max error {relative_error:.3e}", flush=True)


def prepare_input(problem: ModuleType, filename: Path, t_end_override: float | None,
                  threads_override: int | None) -> Path:
    if not hasattr(problem, "grid") or not hasattr(problem, "initial_state"):
        raise AttributeError("Problem files must define grid and initial_state(x, y)")
    metadata = checked_problem_metadata(problem)
    grid = checked_dictionary("grid", problem.grid, GRID_KEYS)
    missing_grid = GRID_KEYS - set(grid)
    if missing_grid:
        raise ValueError(f"Missing grid keys: {', '.join(sorted(missing_grid))}")
    solver = checked_dictionary(
        "solver", getattr(problem, "solver", {}), set(SOLVER_DEFAULTS), SOLVER_DEFAULTS
    )
    output = checked_dictionary(
        "output", getattr(problem, "output", {}), set(OUTPUT_DEFAULTS), OUTPUT_DEFAULTS
    )
    if t_end_override is not None:
        solver["t_end"] = t_end_override
    if threads_override is not None:
        solver["threads"] = threads_override
    if solver["antenna_model"] not in ("resistive_impressed", "prescribed_current"):
        raise ValueError(
            "solver antenna_model must be 'resistive_impressed' or 'prescribed_current'"
        )
    for name in ("boundary_x", "boundary_y"):
        if solver[name] not in ("periodic", "outflow", "absorbing"):
            raise ValueError(f"solver {name} must be 'periodic', 'outflow', or 'absorbing'")
    absorbing = any(solver[name] == "absorbing" for name in ("boundary_x", "boundary_y"))
    for name in ("absorption_width", "absorption_rate", "absorption_magnetic_diffusivity"):
        solver[name] = float(solver[name])
        if not np.isfinite(solver[name]) or solver[name] < 0.0:
            raise ValueError(f"solver {name} must be finite and non-negative")
    if absorbing:
        if solver["absorption_width"] <= 0.0 or solver["absorption_rate"] <= 0.0:
            raise ValueError("Absorbing boundaries require positive absorption_width and absorption_rate")
        for axis in ("x", "y"):
            if solver[f"boundary_{axis}"] == "absorbing" and solver["absorption_width"] >= float(grid[f"l{axis}"]) / 2.0:
                raise ValueError(f"absorption_width must be smaller than half of the {axis} domain")
    ambient = None
    raw_ambient = getattr(problem, "boundary_ambient", None)
    if absorbing and raw_ambient is None:
        raise ValueError("Absorbing boundaries require an explicit boundary_ambient dictionary")
    if raw_ambient is not None:
        ambient = checked_dictionary("boundary_ambient", raw_ambient, BOUNDARY_AMBIENT_KEYS,
                                     {name: 0.0 for name in BOUNDARY_AMBIENT_KEYS - {"rho", "p"}})
        if "rho" not in ambient or "p" not in ambient:
            raise ValueError("boundary_ambient must provide rho and p")
        ambient = {name: float(value) for name, value in ambient.items()}
        if not all(np.isfinite(value) for value in ambient.values()):
            raise ValueError("boundary_ambient values must be finite")
        if ambient["rho"] < solver["rho_floor"] or ambient["p"] < solver["p_floor"]:
            raise ValueError("boundary_ambient rho and p must be positive and meet solver floors")

    nx, ny = int(grid["nx"]), int(grid["ny"])
    lx, ly = float(grid["lx"]), float(grid["ly"])
    if nx < 4 or ny < 4 or lx <= 0.0 or ly <= 0.0:
        raise ValueError("Grid dimensions must be positive with at least four cells per direction")
    dx, dy = lx / nx, ly / ny
    if absorbing and not any(
        solver[f"boundary_{axis}"] == "absorbing"
        and solver["absorption_width"] > spacing / 2.0
        for axis, spacing in (("x", dx), ("y", dy))
    ):
        raise ValueError("The absorption width must include at least one active cell center")
    x = (np.arange(nx) + 0.5) * dx
    y = (np.arange(ny) + 0.5) * dy
    x_grid, y_grid = np.meshgrid(x, y)
    shape = (ny, nx)

    raw_state = problem.initial_state(x_grid, y_grid)
    state = checked_dictionary("initial_state return value", raw_state, FLUID_KEYS)
    if "rho" not in state or "p" not in state:
        raise ValueError("initial_state must provide rho and p")
    fluid = {
        name: profile_array(state.get(name, 0.0), shape, name)
        for name in ("rho", "u", "v", "w", "p")
    }
    if np.min(fluid["rho"]) <= 0.0 or np.min(fluid["p"]) <= 0.0:
        raise ValueError("Initial rho and p must be positive")

    solid_function = getattr(problem, "solid_mask", None)
    solid = None
    if solid_function is not None:
        values = profile_array(evaluate_profile(solid_function, x_grid, y_grid), shape, "solid_mask")
        if not np.all((values == 0.0) | (values == 1.0)):
            raise ValueError("solid_mask must contain only booleans or 0/1 values")
        solid = values.astype(bool)
        if np.all(solid):
            raise ValueError("solid_mask must leave at least one plasma cell")
        if any(np.any(fluid[name][solid] != 0.0) for name in ("u", "v", "w")):
            raise ValueError("Initial velocity must be zero inside a stationary solid")
        if absorbing:
            layer = np.zeros(shape, dtype=bool)
            width = solver["absorption_width"]
            if solver["boundary_x"] == "absorbing":
                layer |= (x_grid < width) | (x_grid > lx - width)
            if solver["boundary_y"] == "absorbing":
                layer |= (y_grid < width) | (y_grid > ly - width)
            if np.any(layer & solid):
                raise ValueError("The absorption layer must not overlap stationary solid cells")

    magnetic_function = getattr(problem, "magnetic_field", None)
    current_function = getattr(problem, "current_density", None)
    tolerance = float(getattr(problem, "current_tolerance", 1.0e-8))
    if tolerance <= 0.0:
        raise ValueError("current_tolerance must be positive")
    magnetic = None if magnetic_function is None else vector_profile(
        magnetic_function(x_grid, y_grid), shape, "B"
    )
    current = None if current_function is None else vector_profile(
        current_function(x_grid, y_grid), shape, "J"
    )
    if magnetic is None and current is None:
        magnetic = tuple(np.zeros(shape) for _ in range(3))
    elif magnetic is None:
        if solver["boundary_x"] != "periodic" or solver["boundary_y"] != "periodic":
            raise ValueError(
                "FFT initialization from current_density requires periodic boundaries; "
                "provide magnetic_field(x, y) for a nonperiodic domain"
            )
        mean = tuple(float(value) for value in getattr(problem, "mean_magnetic_field", (0.0, 0.0, 0.0)))
        if len(mean) != 3:
            raise ValueError("mean_magnetic_field must have three values")
        magnetic = magnetic_field_from_current(current, dx, dy, mean, tolerance)  # type: ignore[arg-type]
    elif current is not None:
        check_current_compatibility(
            magnetic, current, dx, dy, tolerance, "prescribed B",
            solver["boundary_x"], solver["boundary_y"],
        )

    with h5py.File(filename, "w") as destination:
        for name, values in metadata.items():
            group = destination.create_group(name)
            for key, value in values.items():
                group.attrs[key] = value
        config_group = destination.create_group("config")
        config_values = {**grid, **solver, "output_interval": output["interval"]}
        for key, value in config_values.items():
            config_group.attrs[key] = int(value) if isinstance(value, (bool, np.bool_)) else value
        initial_group = destination.create_group("initial")
        for name, values in fluid.items():
            initial_group.create_dataset(name, data=values)
        for name, values in zip(("Bx", "By", "Bz"), magnetic):
            initial_group.create_dataset(name, data=values)
        if solid is not None and np.any(solid):
            solid_group = destination.create_group("solid")
            solid_group.create_dataset("mask", data=solid.astype(np.float64))
            solid_group.attrs["model"] = (
                "stationary slip wall with powered prescribed-current magnetic coupling"
                if solver["antenna_model"] == "prescribed_current" else
                "stationary slip wall with uniform-resistivity magnetic diffusion"
            )
        if ambient is not None:
            ambient_group = destination.create_group("boundary/ambient")
            for name, value in ambient.items():
                ambient_group.attrs[name] = value
        prepare_antennas(problem, destination, x_grid, y_grid, shape, float(solver["t_end"]))
    return Path(output["file"])


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("problem", type=Path, help="Python problem file")
    parser.add_argument("--executable", type=Path, default=Path("./mhd2d"), help="C++ solver executable")
    parser.add_argument("--output", type=Path, help="override output.file from the problem")
    parser.add_argument("--t-end", type=float, help="override solver t_end")
    parser.add_argument("--threads", type=int, help="override thread count (0 selects hardware concurrency)")
    parser.add_argument("--force", action="store_true", help="replace an existing output file")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if not arguments.problem.is_file():
        raise FileNotFoundError(arguments.problem)
    executable = arguments.executable.resolve()
    if not executable.is_file():
        raise FileNotFoundError(f"Solver executable {executable} does not exist; run make first")
    problem = load_problem(arguments.problem)
    with tempfile.TemporaryDirectory(prefix="mhd2d_") as temporary_directory:
        prepared = Path(temporary_directory) / "prepared_input.h5"
        configured_output = prepare_input(problem, prepared, arguments.t_end, arguments.threads)
        output = arguments.output if arguments.output is not None else configured_output
        if output.exists() and not arguments.force:
            raise FileExistsError(f"{output} already exists; use --force to replace it")
        output.parent.mkdir(parents=True, exist_ok=True)
        print(f"Launching {executable} -> {output}", flush=True)
        subprocess.run([str(executable), str(prepared), str(output)], check=True)
        copy_problem_metadata(prepared, output)


if __name__ == "__main__":
    main()
