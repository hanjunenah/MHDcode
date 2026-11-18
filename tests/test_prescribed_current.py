"""Prescribed plate-current tests, distinct from the legacy Ohmic antenna.

Run after building with ``python -m unittest discover -s tests -v``.  The
operator test includes the production solver directly; integration fixtures
use temporary inputs and outputs and never modify research run products.
"""

from pathlib import Path
import os
import shlex
import shutil
import subprocess
import unittest

import h5py
import numpy as np

from test_boundary_conditions import ROOT, SolverTestCase


class PrescribedCurrentTestCase(SolverTestCase):
    def powered_slab(self, eta=0.0):
        problem = self.problem("powered_slab")
        problem.grid.update(nx=32, ny=24, lx=1.0, ly=0.75)
        problem.solver.update(
            antenna_model="prescribed_current", resistivity=eta,
            t_end=0.04, boundary_x="outflow", boundary_y="outflow",
        )
        problem.output["interval"] = 0.02
        problem.solid_mask = lambda x, y: (
            (x >= 0.3125) & (x < 0.6875) & (y >= 0.21875) & (y < 0.53125)
        )
        problem.initial_state = lambda x, y: {"rho": 1.0, "p": 0.3}

        def current(x, y):
            coordinate = np.clip((x - 0.3125) / 0.375, 0.0, 1.0)
            transverse = np.clip((y - 0.21875) / 0.3125, 0.0, 1.0)
            return np.where(problem.solid_mask(x, y),
                            0.2 * np.sin(np.pi * coordinate)**2
                            * np.sin(np.pi * transverse)**2, 0.0)

        problem.antennas = [{"field": "Jz", "space_profile": current,
                             "time_profile": lambda t: t / 0.04}]
        problem.antenna_sampling_interval = 0.01
        return problem

    def assert_energy_budget(self, source, tolerance=2e-11):
        diagnostics = source["diagnostics"]
        energy = diagnostics["total_energy"][:]
        expected = (
            energy[0] + diagnostics["antenna_energy_injected"][:]
            + diagnostics["current_drive_energy_correction"][:]
            + diagnostics["absorption_energy_exchange"][:]
            - diagnostics["boundary_energy_outflow"][:]
        )
        np.testing.assert_allclose(energy, expected, rtol=0, atol=tolerance)
        np.testing.assert_allclose(
            diagnostics["current_drive_energy_correction"][:],
            diagnostics["solid_constraint_energy"][:] - diagnostics["antenna_energy_injected"][:],
            rtol=0, atol=tolerance,
        )


class PrescribedCurrentIntegrationTests(PrescribedCurrentTestCase):
    def test_actual_plate_current_is_prescribed_for_zero_and_finite_plasma_eta(self):
        # Check the staggered curl(B), not only the stored antenna command.  The old
        # eta*curl(J_ant) source fails the eta=0 part of this fixture.
        for eta in (0.0, 0.02):
            with self.subTest(eta=eta):
                name = "eta_" + str(eta)
                problem = self.powered_slab(eta)
                result = self.simulate(self.prepare(problem, name), name)
                with h5py.File(result) as source:
                    self.assertEqual(self.string_attribute(source.attrs["antenna_model"]),
                                     "prescribed_current")
                    mask = source["solid/mask"][:].astype(bool)
                    nodes = source["prescribed_current"]
                    powered = nodes["powered_mask"][:].astype(bool)
                    actual = nodes["Jz"][:][:, powered]
                    commanded = nodes["Jz_target"][:][:, powered]
                    np.testing.assert_allclose(actual, commanded, rtol=0, atol=2e-9)
                    self.assertGreater(float(np.abs(commanded[-1]).max()), 0.1)
                    # Check the authoritative curl independently from Az,
                    # rather than accepting a current diagnostic copied from
                    # the command table.
                    potential = nodes["Az"][:]
                    dx = float(nodes["x"][1] - nodes["x"][0])
                    dy = float(nodes["y"][1] - nodes["y"][0])
                    curl = -(potential[:, 1:-1, 2:] - 2*potential[:, 1:-1, 1:-1]
                             + potential[:, 1:-1, :-2]) / dx**2
                    curl -= (potential[:, 2:, 1:-1] - 2*potential[:, 1:-1, 1:-1]
                             + potential[:, :-2, 1:-1]) / dy**2
                    interior_powered = powered[1:-1, 1:-1]
                    np.testing.assert_allclose(curl[:, interior_powered],
                                               commanded, rtol=0, atol=2e-9)
                    face_bx = np.diff(potential, axis=1) / dy
                    face_by = -np.diff(potential, axis=2) / dx
                    cell_bx = 0.5 * (face_bx[:, :, 1:] + face_bx[:, :, :-1])
                    cell_by = 0.5 * (face_by[:, 1:, :] + face_by[:, :-1, :])
                    np.testing.assert_allclose(source["fields/Bx"][:], cell_bx,
                                               rtol=0, atol=2e-13)
                    np.testing.assert_allclose(source["fields/By"][:], cell_by,
                                               rtol=0, atol=2e-13)
                    divergence = np.diff(face_bx, axis=2) / dx + np.diff(face_by, axis=1) / dy
                    np.testing.assert_allclose(divergence, 0.0, rtol=0, atol=2e-13)
                    np.testing.assert_allclose(source["fields/divB"][:], divergence,
                                               rtol=0, atol=2e-13)
                    for diagnostic in ("max_current_constraint_error", "max_stage_current_constraint_error"):
                        self.assertLess(float(np.abs(source["diagnostics/" + diagnostic][:]).max()), 2e-9)
                    self.assertLess(float(np.abs(source["diagnostics/max_abs_ct_divB"][:]).max()), 2e-13)
                    magnetic = sum(source["fields/" + name][-1]**2
                                   for name in ("Bx", "By", "Bz"))
                    self.assertGreater(float(magnetic[mask].max()), 1e-6)
                    for field in ("u", "v", "w"):
                        np.testing.assert_array_equal(source["fields/" + field][:][:, mask], 0.0)
                    np.testing.assert_array_equal(source["fields/rho"][:][:, mask], 1.0)
                    np.testing.assert_allclose(source["fields/p"][:][:, mask], 0.3,
                                               rtol=0, atol=2e-14)
                    self.assertGreater(float(source["fields/rho"][:].min()), 0.0)
                    self.assertGreater(float(source["fields/p"][:].min()), 0.0)
                    np.testing.assert_array_equal(source["diagnostics/max_wall_mass_flux"][:], 0.0)
                    self.assertGreater(float(source["diagnostics/antenna_energy_injected"][-1]), 0.0)
                    self.assert_energy_budget(source)

    def test_finite_resistivity_allows_plasma_response_without_remote_field_overwrite(self):
        problem = self.powered_slab(0.02)
        result = self.simulate(self.prepare(problem))
        with h5py.File(result) as source:
            mask = source["solid/mask"][:].astype(bool)
            magnetic = sum(source["fields/" + name][-1]**2
                           for name in ("Bx", "By", "Bz"))
            self.assertGreater(float(magnetic[~mask].max()), 1e-12)
            # A vacuum Poisson solve followed by a global B overwrite would
            # immediately magnetize these remote initially field-free cells.
            remote = (source["x"][:] < 0.125) | (source["x"][:] > 0.875)
            self.assertLess(float(magnetic[:, remote].max()), 1e-12)
            self.assert_energy_budget(source)

    def test_absorbing_magnetic_layer_preserves_the_prescribed_current(self):
        problem = self.powered_slab(0.02)
        problem.solver.update(
            boundary_x="absorbing", boundary_y="absorbing",
            absorption_width=0.125, absorption_rate=2.0,
            absorption_magnetic_diffusivity=0.05, t_end=0.25,
        )
        problem.output["interval"] = 0.05
        problem.antennas[0]["time_profile"] = lambda t: np.minimum(t / 0.04, 1.0)
        problem.boundary_ambient = {"rho": 1.0, "p": 0.3, "u": 0.0, "v": 0.0,
                                    "w": 0.0, "Bx": 0.0, "By": 0.0, "Bz": 0.0}
        result = self.simulate(self.prepare(problem))
        with h5py.File(result) as source:
            nodes = source["prescribed_current"]
            powered = nodes["powered_mask"][:].astype(bool)
            np.testing.assert_allclose(nodes["Jz"][:][:, powered],
                                       nodes["Jz_target"][:][:, powered], rtol=0, atol=2e-9)
            self.assertLess(float(source["diagnostics/max_stage_current_constraint_error"][:].max()), 2e-9)
            self.assertGreater(float(np.abs(source["diagnostics/absorption_energy_exchange"][:]).max()),
                               1e-12)
            self.assert_energy_budget(source)

    @unittest.skipUnless(os.environ.get("MHD_PRESCRIBED_LEGACY_EXECUTABLE"),
                         "Optional pre-change solver was not specified")
    def test_legacy_ohmic_antenna_remains_bitwise_unchanged(self):
        problem = self.powered_slab(0.02)
        del problem.solver["antenna_model"]
        problem.magnetic_field = lambda x, y: (0.1, 0.04, 0.02)
        prepared = self.prepare(problem)
        current = self.simulate(prepared, "current")
        legacy = self.simulate(prepared, "legacy",
                               os.environ["MHD_PRESCRIBED_LEGACY_EXECUTABLE"])
        with h5py.File(current) as new, h5py.File(legacy) as old:
            np.testing.assert_array_equal(new["time"][:], old["time"][:])
            for group in ("fields", "diagnostics"):
                for name in old[group]:
                    np.testing.assert_array_equal(new[group + "/" + name][:],
                                                  old[group + "/" + name][:])


class PrescribedCurrentInputTests(PrescribedCurrentTestCase):
    def test_mode_is_validated_and_saved_in_prepared_input(self):
        problem = self.powered_slab()
        prepared = self.prepare(problem)
        with h5py.File(prepared) as source:
            self.assertEqual(self.string_attribute(source["config"].attrs["antenna_model"]),
                             "prescribed_current")
        problem.solver["antenna_model"] = "unknown_model"
        with self.assertRaisesRegex(ValueError, "antenna_model|prescribed_current"):
            self.prepare(problem, "invalid_mode")

    def test_planar_powered_body_requirements_are_enforced(self):
        executable = Path(os.environ.get("MHD_EXECUTABLE", ROOT / "mhd2d"))
        if not executable.is_absolute():
            executable = ROOT / executable
        if not executable.is_file():
            self.skipTest(f"Solver executable is unavailable: {executable}")
        for invalid in ("missing_solid", "in_plane_current", "outside_body",
                        "touching_outer_boundary", "nonplanar_field", "periodic_domain"):
            with self.subTest(invalid=invalid):
                problem = self.powered_slab()
                if invalid == "missing_solid":
                    mask = problem.solid_mask
                    problem.antennas[0]["space_profile"] = lambda x, y: 0.2 * mask(x, y)
                    del problem.solid_mask
                elif invalid == "in_plane_current":
                    problem.antennas[0]["field"] = "Jx"
                elif invalid == "outside_body":
                    problem.antennas[0]["space_profile"] = 0.2
                elif invalid == "touching_outer_boundary":
                    problem.solid_mask = lambda x, y: (x < 0.2) & (y > 0.2) & (y < 0.55)
                    problem.antennas[0]["space_profile"] = lambda x, y: 0.2 * problem.solid_mask(x, y)
                elif invalid == "nonplanar_field":
                    problem.magnetic_field = lambda x, y: (0.0, 0.0, 0.1)
                else:
                    problem.solver["boundary_y"] = "periodic"
                try:
                    prepared = self.prepare(problem, invalid)
                except ValueError as error:
                    self.assertRegex(str(error), "solid|plate|body|Jz|planar|current|periodic|boundary")
                    continue
                result = subprocess.run(
                    [str(executable), str(prepared), str(self.directory / (invalid + "_out.h5"))],
                    cwd=ROOT, capture_output=True, text=True, timeout=10,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertRegex(result.stdout + result.stderr,
                                 "solid|plate|body|Jz|planar|current|periodic|boundary")


class PrescribedCurrentPlotTests(SolverTestCase):
    def test_native_potential_is_averaged_and_used_for_every_contour_frame(self):
        from unittest.mock import patch
        from matplotlib.axes import Axes
        from test_plot_boundaries import PlotMetadataTests, antenna, results, plt
        fixture = PlotMetadataTests()
        fixture.directory = self.directory
        path, _ = fixture.write_output()
        with h5py.File(path, "a") as source:
            times = source["time"][:]
            xn = np.linspace(0.0, 1.0, source["x"].size + 1)
            yn = np.linspace(0.0, 1.0, source["y"].size + 1)
            xx, yy = np.meshgrid(xn, yn)
            potential = np.stack([(1 + time) * np.sin(2*np.pi*xx) * np.cos(np.pi*yy)
                                  + 0.3*yy for time in times])
            source["prescribed_current/Az"] = potential
            source["fields/divB"][:] = 7.0  # Early files used a mismatched cell stencil.
        expected = 0.25 * (potential[:, :-1, :-1] + potential[:, :-1, 1:]
                           + potential[:, 1:, :-1] + potential[:, 1:, 1:])
        x, y, times, fields, diagnostics = antenna.load_antenna_results(path)
        np.testing.assert_array_equal(fields["Az"], expected)
        dx, dy = float(x[1] - x[0]), float(y[1] - y[0])
        native_divergence = (np.diff(np.diff(potential, axis=1) / dy, axis=2) / dx
                             - np.diff(np.diff(potential, axis=2) / dx, axis=1) / dy)
        np.testing.assert_array_equal(fields["divB"], native_divergence)
        np.testing.assert_allclose(fields["divB"], 0.0, rtol=0, atol=2e-13)
        contour_frames = []
        original_contour = Axes.contour

        def record_contour(axis, *args, **kwargs):
            contour_frames.append(np.asarray(args[2]).copy())
            return original_contour(axis, *args, **kwargs)

        with patch.object(antenna, "magnetic_flux_function",
                          side_effect=AssertionError("Native Az must bypass field reconstruction")):
            with patch.object(Axes, "contour", record_contour):
                figure, slider = antenna.make_figure(
                    x, y, times, fields, diagnostics, path.name, 0,
                    **results.load_plot_metadata(path))
                self.addCleanup(plt.close, figure)
                np.testing.assert_array_equal(contour_frames[-1], expected[0])
                for frame in (2, 0, 1):
                    slider.set_val(frame)
                    np.testing.assert_array_equal(contour_frames[-1], expected[frame])

    def test_legacy_output_keeps_magnetic_field_reconstruction(self):
        from unittest.mock import patch
        from test_plot_boundaries import PlotMetadataTests, antenna, results, plt
        fixture = PlotMetadataTests()
        fixture.directory = self.directory
        path, _ = fixture.write_output(modern=False)
        x, y, times, fields, diagnostics = antenna.load_antenna_results(path)
        self.assertNotIn("Az", fields)
        with patch.object(antenna, "magnetic_flux_function",
                          wraps=antenna.magnetic_flux_function) as reconstruction:
            figure, _ = antenna.make_figure(
                x, y, times, fields, diagnostics, path.name, 0,
                **results.load_plot_metadata(path))
            self.addCleanup(plt.close, figure)
            reconstruction.assert_called_once()
            arguments = reconstruction.call_args.args
            np.testing.assert_array_equal(arguments[0], fields["Bx"])
            np.testing.assert_array_equal(arguments[1], fields["By"])
            self.assertEqual(arguments[-2:], ("periodic", "periodic"))

    def test_circuit_work_and_correction_are_not_double_counted_by_slider(self):
        from test_plot_boundaries import PlotMetadataTests, antenna, results, plt
        fixture = PlotMetadataTests()
        fixture.directory = self.directory
        path, mask = fixture.write_output()
        with h5py.File(path, "a") as source:
            diagnostics = source["diagnostics"]
            circuit = diagnostics["antenna_energy_injected"][:]
            body = diagnostics["solid_constraint_energy"][:]
            diagnostics["current_drive_energy_correction"] = body - circuit
            diagnostics["max_current_constraint_error"] = [0.0, 2e-13, 3e-13]
            diagnostics["total_energy"][:] = (3.0 + body
                                                - diagnostics["boundary_energy_outflow"][:])
        x, y, times, fields, diagnostics = antenna.load_antenna_results(path)
        figure, slider = antenna.make_figure(x, y, times, fields, diagnostics, path.name, 0,
                                             **results.load_plot_metadata(path))
        self.addCleanup(plt.close, figure)
        balance = next(line for line in figure.axes[7].lines if line.get_color() == "tab:green")
        np.testing.assert_allclose(balance.get_ydata(),
                                   diagnostics["total_energy"] - diagnostics["total_energy"][0],
                                   rtol=0, atol=5e-16)
        for frame in (2, 0, 1):
            slider.set_val(frame)
            self.assertIn(f"frame {frame}/2", figure._suptitle.get_text())
            self.assertIn("constraint", figure._suptitle.get_text().lower())
            np.testing.assert_array_equal(
                np.ma.getmaskarray(figure.axes[2].images[0].get_array()), mask)


class PrescribedCurrentOperatorTests(SolverTestCase):
    def test_constraint_faraday_and_stage_energy_accounting(self):
        compiler = os.environ.get("MHD_TEST_CXX") or shutil.which("h5c++")
        if compiler:
            command, flags = shlex.split(compiler), []
        else:
            compiler = shutil.which(os.environ.get("CXX", "c++"))
            if compiler is None or shutil.which("pkg-config") is None:
                self.skipTest("A C++ compiler and HDF5 development headers are required")
            command = [compiler]
            flags = shlex.split(subprocess.check_output(
                ["pkg-config", "--cflags", "--libs", "hdf5"], text=True))
        executable = self.directory / "prescribed_current_operator"
        source = ROOT / "tests/prescribed_current_operator.cpp"
        compilation = subprocess.run(
            command + ["-O1", "-std=c++17", str(source), "-pthread", "-o", str(executable)] + flags,
            cwd=self.directory, capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(compilation.returncode, 0, compilation.stdout + compilation.stderr)
        result = subprocess.run([str(executable)], capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
