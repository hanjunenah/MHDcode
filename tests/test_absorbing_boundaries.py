"""Physical and conservation checks for the far-field absorbing layer.

Run after building with ``python -m unittest discover -s tests -v``.
MHD_EXECUTABLE can select a separately built candidate solver.
"""

from pathlib import Path
import os
import shlex
import shutil
import subprocess
import sys
import unittest

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from test_boundary_conditions import SolverTestCase


class AbsorbingTestCase(SolverTestCase):
    def absorbing_problem(self):
        problem = self.problem()
        problem.grid.update(nx=48, ny=12, lx=1.0, ly=0.25)
        problem.solver.update(
            boundary_x="absorbing", boundary_y="periodic",
            absorption_width=0.15, absorption_rate=10.0,
            absorption_magnetic_diffusivity=0.02,
        )
        problem.boundary_ambient = {
            "rho": 1.0, "p": 0.3, "u": 0.0, "v": 0.0, "w": 0.0,
            "Bx": 0.0, "By": 0.0, "Bz": 0.0,
        }
        return problem

    def assert_exchange_budgets(self, source, tolerance=3e-11):
        diagnostics = source["diagnostics"]
        energy = diagnostics["total_energy"][:]
        expected = (
            energy[0] + diagnostics["antenna_energy_injected"][:]
            + diagnostics["solid_constraint_energy"][:]
            + diagnostics["absorption_energy_exchange"][:]
            - diagnostics["boundary_energy_outflow"][:]
        )
        np.testing.assert_allclose(energy, expected, rtol=0, atol=tolerance)
        mass = diagnostics["plasma_mass"][:]
        expected_mass = (
            mass[0] + diagnostics["absorption_mass_exchange"][:]
            - diagnostics["boundary_mass_outflow"][:]
        )
        np.testing.assert_allclose(mass, expected_mass, rtol=0, atol=tolerance)


class AbsorbingBoundaryTests(AbsorbingTestCase):
    def test_uniform_ambient_magnetized_state_is_preserved(self):
        expected = {
            "rho": 1.0, "p": 0.3, "u": 0.12, "v": -0.03, "w": 0.02,
            "Bx": 0.4, "By": -0.2, "Bz": 0.1,
        }
        for modes in (("absorbing", "periodic"), ("periodic", "absorbing"),
                      ("absorbing", "absorbing")):
            with self.subTest(modes=modes):
                problem = self.absorbing_problem()
                problem.grid.update(nx=32, ny=24, lx=1.0, ly=0.6)
                problem.solver.update(boundary_x=modes[0], boundary_y=modes[1])
                problem.boundary_ambient = dict(expected)
                problem.initial_state = lambda x, y: {
                    name: expected[name] for name in ("rho", "p", "u", "v", "w")
                }
                problem.magnetic_field = lambda x, y: tuple(expected[name] for name in ("Bx", "By", "Bz"))
                name = "_".join(modes)
                result = self.simulate(self.prepare(problem, name), name)
                with h5py.File(result) as source:
                    for field, value in expected.items():
                        np.testing.assert_allclose(source["fields/" + field][:], value,
                                                   rtol=0, atol=3e-14)
                    np.testing.assert_allclose(source["fields/psi"][:], 0.0, atol=3e-14)
                    self.assert_exchange_budgets(source)
                    for axis, mode in zip(("x", "y"), modes):
                        self.assertEqual(self.string_attribute(source.attrs["boundary_" + axis]), mode)

    def test_normal_field_is_preserved_when_ambient_normal_field_differs(self):
        # Normal B is constrained by div B: neither the face state nor the
        # magnetic sponge may simply relax it toward the reservoir value.
        problem = self.absorbing_problem()
        problem.magnetic_field = lambda x, y: (0.4, 0.0, 0.0)
        result = self.simulate(self.prepare(problem))
        with h5py.File(result) as source:
            for field, value in (("Bx", 0.4), ("By", 0.0), ("Bz", 0.0),
                                 ("rho", 1.0), ("p", 0.3)):
                np.testing.assert_allclose(source["fields/" + field][:], value,
                                           rtol=0, atol=3e-14)
            np.testing.assert_allclose(source["fields/divB"][:], 0.0, atol=3e-14)
            self.assert_exchange_budgets(source)

    def test_layer_relaxes_only_its_outer_support_and_accounts_for_exchange(self):
        problem = self.absorbing_problem()
        problem.grid.update(nx=80, ny=8, lx=1.0, ly=0.1)
        problem.solver.update(t_end=1e-4, absorption_magnetic_diffusivity=0.0)
        problem.output["interval"] = 1e-4
        problem.initial_state = lambda x, y: {"rho": 1.1, "p": 0.35}
        result = self.simulate(self.prepare(problem))
        with h5py.File(result) as source:
            x = source["x"][:]
            center = (x > 0.35) & (x < 0.65)
            layer = (x > 0.04) & (x < 0.10)
            np.testing.assert_allclose(source["fields/rho"][-1, :, center], 1.1,
                                       rtol=0, atol=2e-14)
            np.testing.assert_allclose(source["fields/p"][-1, :, center], 0.35,
                                       rtol=0, atol=2e-14)
            self.assertLess(float(source["fields/rho"][-1, :, layer].mean()), 1.1 - 1e-7)
            self.assertLess(float(source["fields/p"][-1, :, layer].mean()), 0.35 - 1e-7)
            self.assertLess(float(source["diagnostics/absorption_mass_exchange"][-1]), 0.0)
            self.assertLess(float(source["diagnostics/absorption_energy_exchange"][-1]), 0.0)
            self.assertGreater(float(source["fields/rho"][:].min()), 0.0)
            self.assertGreater(float(source["fields/p"][:].min()), 0.0)
            self.assert_exchange_budgets(source)

    def test_body_remains_impermeable_and_stationary_with_absorbing_faces(self):
        problem = self.absorbing_problem()
        problem.grid.update(nx=64, ny=12, lx=1.0, ly=0.2)
        problem.solver.update(t_end=0.06, resistivity=0.01)
        problem.output["interval"] = 0.02
        problem.solid_mask = lambda x, y: (x > 0.35) & (x < 0.45)

        def initial(x, y):
            mask = problem.solid_mask(x, y)
            return {"rho": np.where(mask, 2.0, 1.1),
                    "p": np.where(mask, 0.7, 0.4),
                    "u": np.where(mask, 0.0, np.where(x < 0.4, 0.05, -0.05))}

        problem.initial_state = initial
        problem.magnetic_field = lambda x, y: (0.02, 0.0, 0.0)
        problem.antennas = [{
            "field": "Jz", "space_profile": lambda x, y: 0.1 * problem.solid_mask(x, y),
            "time_profile": lambda time: np.sin(20 * time),
        }]
        problem.antenna_sampling_interval = 0.01
        result = self.simulate(self.prepare(problem))
        with h5py.File(result) as source:
            mask = source["solid/mask"][:].astype(bool)
            for field in ("u", "v", "w"):
                np.testing.assert_array_equal(source["fields/" + field][:][:, mask], 0.0)
            np.testing.assert_array_equal(source["fields/rho"][:][:, mask], 2.0)
            np.testing.assert_allclose(source["fields/p"][:][:, mask], 0.7, rtol=0, atol=1e-14)
            np.testing.assert_array_equal(source["diagnostics/max_wall_mass_flux"][:], 0.0)
            self.assert_exchange_budgets(source)


class AbsorbingInputTests(AbsorbingTestCase):
    def test_solver_rechecks_raw_input_before_running(self):
        executable = Path(os.environ.get("MHD_EXECUTABLE", ROOT / "mhd2d"))
        if not executable.is_absolute():
            executable = ROOT / executable
        if not executable.is_file():
            self.skipTest(f"Solver executable is unavailable: {executable}")
        for condition in ("missing_ambient", "negative_rate", "wide_layer", "solid_overlap"):
            with self.subTest(condition=condition):
                prepared = self.prepare(self.absorbing_problem(), condition)
                with h5py.File(prepared, "a") as source:
                    if condition == "missing_ambient":
                        del source["boundary"]
                    elif condition == "negative_rate":
                        source["config"].attrs["absorption_rate"] = -1.0
                    elif condition == "wide_layer":
                        source["config"].attrs["absorption_width"] = 0.5
                    else:
                        mask = np.zeros((12, 48))
                        mask[:, 0] = 1.0
                        source.require_group("solid").create_dataset("mask", data=mask)
                result = subprocess.run([str(executable), str(prepared),
                                         str(self.directory / (condition + "_result.h5"))],
                                        capture_output=True, text=True, timeout=10)
                self.assertNotEqual(result.returncode, 0)
                self.assertRegex(result.stdout + result.stderr,
                                 "ambient|absorp|layer|solid|boundary|width")

    def test_ambient_state_is_explicit_and_physical(self):
        problem = self.absorbing_problem()
        del problem.boundary_ambient
        with self.assertRaisesRegex(ValueError, "ambient"):
            self.prepare(problem)
        for name, value in (("rho", 0.0), ("p", -0.1), ("u", np.inf)):
            with self.subTest(name=name):
                problem = self.absorbing_problem()
                problem.boundary_ambient[name] = value
                with self.assertRaisesRegex((ValueError, TypeError), "ambient|finite|positive|floor"):
                    self.prepare(problem)

    def test_ambient_optional_components_default_to_zero_and_unknown_keys_fail(self):
        problem = self.absorbing_problem()
        problem.boundary_ambient = {"rho": 1.0, "p": 0.3}
        prepared = self.prepare(problem)
        with h5py.File(prepared) as source:
            ambient = source["boundary/ambient"].attrs
            for name in ("u", "v", "w", "Bx", "By", "Bz"):
                self.assertEqual(float(ambient[name]), 0.0)
        problem.boundary_ambient["temperature"] = 0.3
        with self.assertRaisesRegex(ValueError, "ambient|unknown|unexpected"):
            self.prepare(problem)

    def test_invalid_layer_parameters_are_rejected(self):
        for name, value in (("absorption_width", 0.0), ("absorption_width", -0.1),
                            ("absorption_width", 1e-9),
                            ("absorption_rate", 0.0), ("absorption_rate", -1.0),
                            ("absorption_magnetic_diffusivity", -0.1),
                            ("absorption_rate", np.nan)):
            with self.subTest(name=name, value=value):
                problem = self.absorbing_problem()
                problem.solver[name] = value
                with self.assertRaisesRegex(ValueError, "absorption|finite"):
                    self.prepare(problem)

    def test_layers_cannot_overlap_bodies_or_cover_half_an_axis(self):
        problem = self.absorbing_problem()
        problem.solid_mask = lambda x, y: x < 0.1
        with self.assertRaisesRegex(ValueError, "solid|body|overlap|absorp"):
            self.prepare(problem)
        problem = self.absorbing_problem()
        problem.solver["absorption_width"] = 0.5
        with self.assertRaisesRegex(ValueError, "width|half|absorp"):
            self.prepare(problem)


class MagneticAbsorbingOperatorTests(SolverTestCase):
    def test_actual_split_operator_preserves_divergence_and_removes_only_magnetic_energy(self):
        # Include the production operator in an isolated test translation unit;
        # no test-only command or access hook is added to the solver binary.
        compiler = os.environ.get("MHD_TEST_CXX") or shutil.which("h5c++")
        if compiler:
            command = shlex.split(compiler)
            flags = []
        else:
            compiler = shutil.which(os.environ.get("CXX", "c++"))
            if compiler is None or shutil.which("pkg-config") is None:
                self.skipTest("A C++ compiler and HDF5 development headers are required")
            command = [compiler]
            flags = shlex.split(subprocess.check_output(
                ["pkg-config", "--cflags", "--libs", "hdf5"], text=True))
        executable = self.directory / "magnetic_absorption_operator"
        source = ROOT / "tests/magnetic_absorption_operator.cpp"
        compilation = subprocess.run(command + ["-O1", "-std=c++17", str(source),
                                                  "-pthread", "-o", str(executable)] + flags,
                                     cwd=self.directory, capture_output=True, text=True, timeout=60)
        self.assertEqual(compilation.returncode, 0, compilation.stdout + compilation.stderr)
        result = subprocess.run([str(executable)], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
