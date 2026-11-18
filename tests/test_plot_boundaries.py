"""Analytic checks for nonperiodic flux contours and body-aware plot data."""

from pathlib import Path
import sys
import tempfile
import unittest

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import plot_antenna_results as antenna
import plot_results as results


class MagneticFluxTests(unittest.TestCase):
    def test_cosine_transform_roundtrips_real_and_complex_on_any_axis(self):
        generator = np.random.default_rng(71)
        for shape in ((3, 5, 8), (1, 4, 1)):
            real = generator.normal(size=shape)
            for values in (real, real + 1j * generator.normal(size=shape)):
                for axis in (0, 1, 2, -1):
                    with self.subTest(shape=shape, complex=np.iscomplexobj(values), axis=axis):
                        transformed = results._cosine_transform(values, axis)
                        restored = results._cosine_transform(transformed, axis, inverse=True)
                        np.testing.assert_allclose(restored, values, rtol=2e-14, atol=2e-14)

    def test_open_bilinear_flux_is_exact_up_to_constant(self):
        x = (np.arange(17) + 0.5) * 0.13
        y = (np.arange(11) + 0.5) * 0.17
        xx, yy = np.meshgrid(x, y)
        expected = xx * yy
        expected -= expected.mean()
        bx = np.stack((xx, -2 * xx))
        by = np.stack((-yy, 2 * yy))
        actual = results.magnetic_flux_function(bx, by, x, y, "outflow", "outflow")
        np.testing.assert_allclose(actual, np.stack((expected, -2 * expected)), rtol=0, atol=5e-13)

    def test_constant_fields_work_for_each_boundary_combination(self):
        x = (np.arange(13) + 0.5) * 0.12
        y = (np.arange(9) + 0.5) * 0.21
        xx, yy = np.meshgrid(x, y)
        expected = 0.3 * yy + 0.2 * xx
        expected -= expected.mean()
        bx = np.full((1, y.size, x.size), 0.3)
        by = np.full_like(bx, -0.2)
        for boundary_x in ("periodic", "outflow"):
            for boundary_y in ("periodic", "outflow"):
                with self.subTest(boundary_x=boundary_x, boundary_y=boundary_y):
                    actual = results.magnetic_flux_function(bx, by, x, y, boundary_x, boundary_y)
                    np.testing.assert_allclose(actual[0], expected, rtol=0, atol=5e-14)

    def test_mixed_boundaries_recover_periodic_edge_gradient_modes(self):
        # These analytic modes exactly satisfy the edge-averaged gradient
        # equations used by the open-boundary least-squares reconstruction.
        x = (np.arange(24) + 0.5) / 24
        y = (np.arange(18) + 0.5) / 18
        xx, yy = np.meshgrid(x, y)
        for periodic_axis in ("x", "y"):
            if periodic_axis == "x":
                wave = 2 * np.pi
                expected = np.cos(wave * xx) * yy
                bx = np.cos(wave * xx)
                by = 2 * np.tan(wave / (2 * x.size)) * x.size * np.sin(wave * xx) * yy
                boundaries = ("periodic", "outflow")
            else:
                wave = 2 * np.pi
                expected = xx * np.cos(wave * yy)
                bx = -2 * np.tan(wave / (2 * y.size)) * y.size * xx * np.sin(wave * yy)
                by = -np.cos(wave * yy)
                boundaries = ("outflow", "periodic")
            expected -= expected.mean()
            with self.subTest(periodic_axis=periodic_axis):
                actual = results.magnetic_flux_function(bx[None], by[None], x, y, *boundaries)
                np.testing.assert_allclose(actual[0], expected, rtol=0, atol=3e-13)

    def test_mixed_continuous_field_error_decreases_quadratically(self):
        errors = []
        for n in (24, 48):
            x = (np.arange(n) + 0.5) / n
            y = (np.arange(n) + 0.5) / n
            xx, yy = np.meshgrid(x, y)
            expected = np.cos(2 * np.pi * xx) * yy
            bx = np.cos(2 * np.pi * xx)
            by = 2 * np.pi * np.sin(2 * np.pi * xx) * yy
            actual = results.magnetic_flux_function(bx[None], by[None], x, y, "periodic", "outflow")[0]
            errors.append(float(np.max(np.abs(actual - (expected - expected.mean())))))
        self.assertLess(errors[1], errors[0] / 3.5)

    def test_periodic_default_retains_discrete_curl_inverse(self):
        nx, ny = 24, 18
        dx, dy = 1 / nx, 1 / ny
        x, y = (np.arange(nx) + 0.5) * dx, (np.arange(ny) + 0.5) * dy
        xx, yy = np.meshgrid(x, y)
        kx, ky = 4 * np.pi, 2 * np.pi
        az = np.cos(kx * xx) * np.cos(ky * yy)
        bx = -np.sin(ky * dy) / dy * np.cos(kx * xx) * np.sin(ky * yy) + 0.3
        by = np.sin(kx * dx) / dx * np.sin(kx * xx) * np.cos(ky * yy) - 0.2
        expected = az + 0.3 * yy + 0.2 * xx
        expected -= expected.mean()
        actual = results.magnetic_flux_function(bx[None], by[None], x, y)
        explicit = results.magnetic_flux_function(bx[None], by[None], x, y, "periodic", "periodic")
        np.testing.assert_array_equal(actual, explicit)
        np.testing.assert_allclose(actual[0], expected, rtol=0, atol=3e-13)


class PlotMetadataTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="mhd_plot_boundary_test_")
        self.directory = Path(temporary.name)
        self.addCleanup(temporary.cleanup)

    def write_output(self, modern=True):
        path = self.directory / ("modern.h5" if modern else "legacy.h5")
        x, y = (np.arange(8) + 0.5) / 8, (np.arange(6) + 0.5) / 6
        xx, yy = np.meshgrid(x, y)
        times = np.array([0.0, 0.5, 1.0])
        mask = (xx > 0.35) & (xx < 0.65)
        shape = (times.size, y.size, x.size)
        fields = {name: np.zeros(shape) for name in set(antenna.FIELD_NAMES) | {"Jx", "Jy", "Jz"}}
        for frame, time in enumerate(times):
            fields["rho"][frame] = np.where(mask, 1.0, 1.0 + 0.01 * time * xx)
            fields["p"][frame] = np.where(mask, 0.5, 0.3 + 0.02 * time * yy)
            fields["u"][frame] = np.where(mask, 0.0, 0.03 * time * xx)
            fields["v"][frame] = np.where(mask, 0.0, 0.01 * time * yy)
            fields["Bx"][frame] = (1.0 + 0.2 * time) * xx
            fields["By"][frame] = -(1.0 + 0.2 * time) * yy
            fields["Jz_antenna"][frame] = (1 + time) * 0.2 * mask
        antenna_work = np.array([0.0, 0.2, 0.5])
        body_work = np.array([0.0, 0.03, 0.08])
        energy_out = np.array([0.0, 0.1, 0.25])
        total = 3 + antenna_work + body_work - energy_out
        diagnostics = {
            "total_energy": total, "magnetic_energy": np.array([0.1, 0.15, 0.25]),
            "kinetic_energy": np.array([0.0, 0.02, 0.05]),
            "antenna_power": np.array([0.1, 0.2, 0.4]),
            "antenna_energy_injected": antenna_work,
            "max_abs_antenna_current": np.array([0.2, 0.3, 0.4]),
        }
        diagnostics["thermal_energy"] = total - diagnostics["magnetic_energy"] - diagnostics["kinetic_energy"]
        if modern:
            diagnostics.update(solid_constraint_energy=body_work, boundary_energy_outflow=energy_out,
                               plasma_energy=total - 0.5)
        with h5py.File(path, "w") as destination:
            destination["x"], destination["y"], destination["time"] = x, y, times
            for name, data in fields.items():
                destination["fields/" + name] = data
            for name, data in diagnostics.items():
                destination["diagnostics/" + name] = data
            if modern:
                destination.attrs["boundary_x"] = "outflow"
                destination.attrs["boundary_y"] = np.bytes_("outflow")
                destination["solid/mask"] = mask.astype(float)
        return path, mask

    def test_legacy_metadata_defaults_and_optional_diagnostics(self):
        path, _ = self.write_output(modern=False)
        metadata = results.load_plot_metadata(path)
        self.assertEqual(metadata["boundary_x"], "periodic")
        self.assertEqual(metadata["boundary_y"], "periodic")
        self.assertIsNone(metadata["solid_mask"])
        *_, diagnostics = antenna.load_antenna_results(path)
        self.assertNotIn("solid_constraint_energy", diagnostics)
        self.assertNotIn("boundary_energy_outflow", diagnostics)

    def test_new_metadata_and_optional_diagnostics_are_loaded(self):
        path, mask = self.write_output()
        metadata = results.load_plot_metadata(path)
        self.assertEqual(metadata["boundary_x"], "outflow")
        self.assertEqual(metadata["boundary_y"], "outflow")
        np.testing.assert_array_equal(metadata["solid_mask"], mask)
        *_, diagnostics = antenna.load_antenna_results(path)
        for name in ("solid_constraint_energy", "boundary_energy_outflow", "plasma_energy"):
            self.assertIn(name, diagnostics)

    def test_optional_diagnostic_must_match_time_axis(self):
        path, _ = self.write_output()
        with h5py.File(path, "a") as source:
            del source["diagnostics/boundary_energy_outflow"]
            source["diagnostics/boundary_energy_outflow"] = [0.0, 0.1]
        with self.assertRaisesRegex(ValueError, "not aligned"):
            antenna.load_antenna_results(path)

    def test_antenna_viewer_energy_balance_and_body_masks_survive_slider(self):
        path, mask = self.write_output()
        x, y, times, fields, diagnostics = antenna.load_antenna_results(path)
        figure, slider = antenna.make_figure(x, y, times, fields, diagnostics, path.name, 1,
                                            **results.load_plot_metadata(path))
        self.addCleanup(plt.close, figure)
        expected = (diagnostics["antenna_energy_injected"] + diagnostics["solid_constraint_energy"]
                    - diagnostics["boundary_energy_outflow"])
        energy_axis = figure.axes[7]
        balance = next(line for line in energy_axis.lines if line.get_color() == "tab:green")
        np.testing.assert_array_equal(balance.get_ydata(), expected)
        self.assertIn("W_", balance.get_label())
        for frame in (2, 0, 1):
            slider.set_val(frame)
            for panel in (1, 2, 3, 4):
                np.testing.assert_array_equal(np.ma.getmaskarray(figure.axes[panel].images[0].get_array()), mask)
            momentum = figure.axes[2].images[0].get_array()
            np.testing.assert_allclose(momentum.compressed(), (fields["rho"][frame] * fields["u"][frame])[~mask])
            # Impressed current remains visible inside the material.
            self.assertFalse(np.ma.getmaskarray(figure.axes[0].images[0].get_array()).any())
            self.assertIn(f"frame {frame}/2", figure._suptitle.get_text())
            self.assertIn("x=outflow, y=outflow", figure._suptitle.get_text())

    def test_generic_viewer_hides_body_placeholder_fluid_state(self):
        path, mask = self.write_output()
        x, y, times, fields, diagnostics = results.load_results(path)
        figure, slider = results.make_figure(x, y, times, fields, diagnostics, path.name, 1,
                                            **results.load_plot_metadata(path))
        self.addCleanup(plt.close, figure)
        for frame in (1, 2):
            slider.set_val(frame)
            for panel in (1, 2, 3, 4):
                np.testing.assert_array_equal(np.ma.getmaskarray(figure.axes[panel].images[0].get_array()), mask)
            self.assertIn("x=outflow, y=outflow", figure._suptitle.get_text())
        # The Harris-sheet reconnection diagnostic must not be assigned to
        # an arbitrary open-domain magnetic field.
        self.assertEqual(figure.axes[8].get_title(), "Global magnetic-flux range")


if __name__ == "__main__":
    unittest.main()
