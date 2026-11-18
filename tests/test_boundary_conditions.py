"""Integration checks for outer boundaries and a stationary resistive body.

Run with ``python -m unittest discover -s tests -v`` after building mhd2d.
MHD_EXECUTABLE selects a different solver. MHD_LEGACY_EXECUTABLE optionally
enables a bitwise comparison with the original periodic solver.
All inputs and outputs are created in a temporary directory.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType
import unittest

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run


class SolverTestCase(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="mhd_boundary_test_")
        self.directory = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def problem(self, name="test_problem"):
        problem = ModuleType(name)
        problem.grid = {"nx": 24, "ny": 16, "lx": 1.0, "ly": 0.5}
        problem.solver = {"t_end": 0.1, "threads": 1, "cfl": 0.3}
        problem.output = {"file": "unused.h5", "interval": 0.05}
        problem.initial_state = lambda x, y: {"rho": 1.0, "p": 0.3}
        problem.magnetic_field = lambda x, y: (0.0, 0.0, 0.0)
        return problem

    def prepare(self, problem, name="prepared"):
        path = self.directory / (name + ".h5")
        run.prepare_input(problem, path, None, None)
        return path

    def simulate(self, prepared, name="result", executable=None):
        executable = Path(executable or os.environ.get("MHD_EXECUTABLE", ROOT / "mhd2d"))
        if not executable.is_absolute():
            executable = ROOT / executable
        if not executable.is_file():
            self.skipTest(f"Solver executable is unavailable: {executable}")
        output = self.directory / (name + ".h5")
        result = subprocess.run(
            [str(executable), str(prepared), str(output)],
            cwd=ROOT, capture_output=True, text=True, timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        with h5py.File(output) as source:
            for group in ("fields", "diagnostics"):
                for dataset in source[group].values():
                    self.assertTrue(np.isfinite(dataset[:]).all(), dataset.name)
        return output

    @staticmethod
    def string_attribute(value):
        return value.decode() if isinstance(value, bytes) else str(value)


class BoundaryConditionTests(SolverTestCase):
    def test_uniform_moving_magnetized_state_for_each_axis_mode(self):
        expected = {"rho": 1.0, "u": 0.3, "v": 0.1, "w": -0.2, "p": 0.2,
                    "Bx": 0.4, "By": -0.2, "Bz": 0.3, "psi": 0.0}
        for boundary_x in ("periodic", "outflow"):
            for boundary_y in ("periodic", "outflow"):
                with self.subTest(boundary_x=boundary_x, boundary_y=boundary_y):
                    name = boundary_x + "_" + boundary_y
                    problem = self.problem(name)
                    problem.solver.update(boundary_x=boundary_x, boundary_y=boundary_y,
                                          resistivity=0.01)
                    problem.initial_state = lambda x, y: {
                        key: value for key, value in expected.items()
                        if key in ("rho", "u", "v", "w", "p")
                    }
                    problem.magnetic_field = lambda x, y: (0.4, -0.2, 0.3)
                    result = self.simulate(self.prepare(problem, name), name)
                    with h5py.File(result) as source:
                        for field, value in expected.items():
                            np.testing.assert_allclose(source["fields/" + field][:], value,
                                                       rtol=0, atol=2e-14)
                        for axis, mode in (("x", boundary_x), ("y", boundary_y)):
                            self.assertEqual(self.string_attribute(source.attrs["boundary_" + axis]), mode)

    def test_outgoing_contact_does_not_reenter_either_outflow_axis(self):
        for direction in ("x", "y"):
            for boundary in ("periodic", "outflow"):
                with self.subTest(direction=direction, boundary=boundary):
                    name = "contact_" + direction + "_" + boundary
                    problem = self.problem(name)
                    if direction == "x":
                        problem.grid.update(nx=96, ny=8, lx=1.0, ly=0.2)
                    else:
                        problem.grid.update(nx=8, ny=96, lx=0.2, ly=1.0)
                    problem.solver.update(t_end=0.5, **{"boundary_" + direction: boundary})
                    problem.output["interval"] = 0.5

                    def state(x, y):
                        coordinate = x if direction == "x" else y
                        return {"rho": 1 + 0.2 * np.exp(-0.5 * ((coordinate - 0.75) / 0.04)**2),
                                "u": float(direction == "x"), "v": float(direction == "y"), "p": 0.2}

                    problem.initial_state = state
                    result = self.simulate(self.prepare(problem, name), name)
                    with h5py.File(result) as source:
                        excess = source["fields/rho"][-1] - 1.0
                        left_half = source[direction][:] < 0.5
                        returned = excess[:, left_half] if direction == "x" else excess[left_half, :]
                        if boundary == "periodic":
                            self.assertGreater(float(returned.max()), 0.12)
                            mass = source["diagnostics/mass"][:]
                            np.testing.assert_allclose(mass, mass[0], rtol=2e-14)
                        else:
                            self.assertLess(float(returned.max()), 1e-7)
                            self.assertLess(float(excess.max()), 1e-5)
                            self.assertLess(source["diagnostics/mass"][-1], source["diagnostics/mass"][0])

    def test_prepared_input_without_boundary_attributes_remains_periodic(self):
        problem = self.problem()
        problem.initial_state = lambda x, y: {"rho": 1.0, "u": 0.1, "p": 0.3}
        prepared = self.prepare(problem)
        with h5py.File(prepared, "a") as source:
            del source["config"].attrs["boundary_x"]
            del source["config"].attrs["boundary_y"]
        result = self.simulate(prepared)
        with h5py.File(result) as source:
            self.assertEqual(self.string_attribute(source.attrs["boundary_x"]), "periodic")
            self.assertEqual(self.string_attribute(source.attrs["boundary_y"]), "periodic")
            np.testing.assert_allclose(source["fields/u"][:], 0.1, rtol=0, atol=2e-15)

    @unittest.skipUnless(os.environ.get("MHD_LEGACY_EXECUTABLE"), "Optional legacy solver was not specified")
    def test_unmasked_periodic_run_matches_legacy_solver_bitwise(self):
        problem = run.load_problem(ROOT / "input/orszag_tang.py")
        problem.grid.update(nx=24, ny=24)
        problem.solver.update(t_end=0.02, threads=1)
        problem.output["interval"] = 0.01
        prepared = self.prepare(problem)
        current = self.simulate(prepared, "current")
        legacy = self.simulate(prepared, "legacy", os.environ["MHD_LEGACY_EXECUTABLE"])
        with h5py.File(current) as new, h5py.File(legacy) as old:
            for group in ("fields", "diagnostics"):
                # Additional diagnostics in the current solver are expected.
                for name in old[group]:
                    np.testing.assert_array_equal(new[group + "/" + name][:], old[group + "/" + name][:])


class InputValidationTests(SolverTestCase):
    def test_invalid_outer_boundary_is_rejected(self):
        for axis in ("x", "y"):
            with self.subTest(axis=axis):
                problem = self.problem()
                problem.solver["boundary_" + axis] = "reflecting"
                with self.assertRaisesRegex(ValueError, "periodic.*outflow"):
                    self.prepare(problem)

    def test_fft_current_initialization_is_rejected_for_outflow(self):
        for axis in ("x", "y"):
            with self.subTest(axis=axis):
                problem = self.problem()
                del problem.magnetic_field
                problem.current_density = lambda x, y: (0.0, 0.0, 0.0)
                problem.solver["boundary_" + axis] = "outflow"
                with self.assertRaisesRegex(ValueError, "FFT.*periodic"):
                    self.prepare(problem)

    def test_invalid_or_fully_solid_mask_is_rejected(self):
        for value in (0.5, True):
            with self.subTest(value=value):
                problem = self.problem()
                problem.solid_mask = lambda x, y: value
                with self.assertRaisesRegex(ValueError, "solid_mask"):
                    self.prepare(problem)

    def test_initial_motion_inside_solid_is_rejected(self):
        problem = self.problem()
        problem.solid_mask = lambda x, y: (x > 0.4) & (x < 0.6)
        problem.initial_state = lambda x, y: {"rho": 1.0, "u": 0.1, "p": 0.3}
        with self.assertRaisesRegex(ValueError, "velocity.*stationary solid"):
            self.prepare(problem)


class SolidBoundaryTests(SolverTestCase):
    def test_stationary_slab_blocks_plasma_mass(self):
        problem = self.problem()
        problem.grid.update(nx=40, ny=12, lx=1.0, ly=0.3)
        problem.solid_mask = lambda x, y: (x >= 0.45) & (x <= 0.55)

        def state(x, y):
            solid = problem.solid_mask(x, y)
            velocity = np.where(x < 0.5, 0.1, -0.1)
            return {"rho": 1.0, "u": np.where(solid, 0.0, velocity), "p": 0.5}

        problem.initial_state = state
        result = self.simulate(self.prepare(problem))
        with h5py.File(result) as source:
            x, y = np.meshgrid(source["x"][:], source["y"][:])
            mask = problem.solid_mask(x, y)
            np.testing.assert_array_equal(source["solid/mask"][:], mask)
            for name in ("u", "v", "w"):
                np.testing.assert_array_equal(source["fields/" + name][:][:, mask], 0.0)
            mass = source["diagnostics/plasma_mass"][:]
            np.testing.assert_allclose(mass, mass[0], rtol=2e-13)
            np.testing.assert_array_equal(source["diagnostics/max_wall_mass_flux"][:], 0.0)
            np.testing.assert_allclose(source["fields/rho"][:][:, mask], 1.0, rtol=0, atol=0)
            np.testing.assert_allclose(source["fields/p"][:][:, mask], 0.5, rtol=0, atol=3e-15)
            # The adjacent plasma must respond to the wall rather than pass through it.
            self.assertGreater(float(np.max(source["fields/p"][-1][~mask])), 0.5 + 1e-4)

    def test_body_field_evolves_and_full_energy_budget_closes(self):
        problem = self.problem()
        problem.grid.update(nx=32, ny=12, lx=1.0, ly=0.3)
        problem.solver.update(t_end=0.12, resistivity=0.02,
                              boundary_x="outflow", boundary_y="periodic")
        problem.output["interval"] = 0.04
        problem.solid_mask = lambda x, y: (x > 0.4) & (x < 0.6)

        def state(x, y):
            solid = problem.solid_mask(x, y)
            return {"rho": 1.0, "u": np.where(solid, 0.0, 0.08 * (x - 0.5)), "p": 0.3}

        problem.initial_state = state
        problem.antennas = [{
            "field": "Jz",
            "space_profile": lambda x, y: 0.2 * np.sin(2 * np.pi * x) * problem.solid_mask(x, y),
            "time_profile": lambda t: np.sin(5 * t),
        }]
        problem.antenna_sampling_interval = 0.01
        result = self.simulate(self.prepare(problem))
        with h5py.File(result) as source:
            mask = source["solid/mask"][:].astype(bool)
            for name in ("u", "v", "w"):
                np.testing.assert_array_equal(source["fields/" + name][:][:, mask], 0.0)
            magnetic = sum(source["fields/" + name][-1]**2 for name in ("Bx", "By", "Bz"))
            self.assertGreater(float(magnetic[mask].max()), 1e-10)
            self.assertGreater(float(magnetic[~mask].max()), 1e-12)
            diagnostics = source["diagnostics"]
            self.assertGreater(float(diagnostics["boundary_energy_outflow"][-1]), 0.0)
            energy = diagnostics["total_energy"][:]
            expected = (energy[0] + diagnostics["antenna_energy_injected"][:]
                        + diagnostics["solid_constraint_energy"][:]
                        - diagnostics["boundary_energy_outflow"][:])
            np.testing.assert_allclose(energy, expected, rtol=0, atol=2e-11)
            mass = diagnostics["plasma_mass"][:]
            np.testing.assert_allclose(mass + diagnostics["boundary_mass_outflow"][:], mass[0],
                                       rtol=0, atol=2e-12)
            np.testing.assert_array_equal(diagnostics["max_wall_mass_flux"][:], 0.0)


if __name__ == "__main__":
    unittest.main()
