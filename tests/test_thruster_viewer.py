"""Thruster source, impulse and slider checks using small synthetic results."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import plot_thruster_results as thruster


class ThrusterViewerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="mhd_thruster_viewer_")
        self.directory = Path(temporary.name)
        self.addCleanup(temporary.cleanup)
        self.x = (np.arange(8)+0.5)*0.25
        self.y = (np.arange(6)+0.5)*0.25
        self.times = np.array([0.0, 0.4, 1.0])
        xx, yy = np.meshgrid(self.x, self.y)
        self.solid = (xx > 0.5) & (xx < 1.0) & (yy < 0.5)
        self.source = np.where((xx > 1.1) & (xx < 1.6) & (yy > 0.5) & (yy < 1.0), 2.0, 0.0)
        # Four source cells, area 1/16 each: integral(q)=1/2.
        self.source_times = np.array([0.0, 0.25, 0.5, 1.0])
        self.amplitudes = np.array([0.0, 1.0, 0.0, 1.0])
        self.sampled_amplitude = np.array([0.0, 0.4, 1.0])
        self.expected_rate = np.array([0.0, 0.2, 0.5])
        # Integration must include the temporal knots between output frames.
        self.expected_mass = np.array([0.0, 0.115, 0.25])
        self.resistivity = 0.07
        self.scaling = {
            "length_m": 0.01,
            "magnetic_field_t": 0.02,
            "out_of_plane_length_m": 0.1,
            "mass_density_kg_m3": 2.180171556711138e-7,
            "alfven_speed_m_s": 38210.23678362279,
            "time_s": 2.617099720325753e-7,
        }

    def injection(self):
        return {"time": self.source_times, "amplitude": self.amplitudes,
                "mass_rate": self.source}

    def write_result(self, *, feeding=True, si=False):
        path = self.directory / ("feed" if feeding else "no_feed")
        path = path.with_suffix(".h5")
        xx, yy = np.meshgrid(self.x, self.y)
        shape = (self.times.size, self.y.size, self.x.size)
        names = ("rho", "u", "v", "w", "p", "Bx", "By", "Bz", "divB", "psi", "Jz",
                 "Jx_antenna", "Jy_antenna", "Jz_antenna")
        fields = {name: np.zeros(shape) for name in names}
        initial_density = 0.02+0.8*(xx > 1.0)
        for frame, time in enumerate(self.times):
            fields["rho"][frame] = np.where(self.solid, 1.0, initial_density+0.03*time*self.source)
            fields["p"][frame] = np.where(self.solid, 0.3, 0.3+0.01*time*yy)
            # Nonzero initial velocity and pressure distinguish absolute
            # fields from the formerly plotted responses/perturbations.
            fields["u"][frame] = np.where(self.solid, 0.0, 0.03+0.04*time*xx)
            fields["v"][frame] = np.where(self.solid, 0.0, -0.02+0.015*time*yy)
            fields["Bx"][frame] = 0.05*xx
            fields["By"][frame] = -0.05*yy
            fields["Jz_antenna"][frame] = (1.0+time)*0.2*self.solid
            # Actual curl(B) current also exists in plasma; antenna current
            # alone cannot reconstruct the resistive electric field there.
            fields["Jz"][frame] = np.where(self.solid, (1.0+time)*0.2,
                                             0.17+0.09*time*xx-0.03*yy)
        diagnostics = {
            "total_energy": np.array([2.0, 2.2, 2.5]),
            "magnetic_energy": np.array([0.1, 0.15, 0.2]),
            "kinetic_energy": np.array([0.0, 0.05, 0.1]),
            "thermal_energy": np.array([1.9, 2.0, 2.2]),
            "antenna_power": np.array([0.1, 0.2, 0.3]),
            "antenna_energy_injected": np.array([0.0, 0.2, 0.5]),
            "max_abs_antenna_current": np.array([0.2, 0.28, 0.4]),
            # Deliberately not reconstructible by integrating sparse forces.
            "solid_force_x": np.array([0.0, -10.0, 2.0]),
            "solid_impulse_x": np.array([0.0, -0.12, -0.3]),
        }
        with h5py.File(path, "w") as f:
            f["x"], f["y"], f["time"] = self.x, self.y, self.times
            f.attrs.update(boundary_x="outflow", boundary_y="outflow",
                           absorption_width=0.0, nx=len(self.x), ny=len(self.y), lx=2.0, ly=1.5,
                           plasma_injection_enabled=int(feeding),
                           resistivity=self.resistivity, antenna_model="prescribed_current")
            f["solid/mask"] = self.solid.astype(float)
            for name, values in fields.items():
                f["fields/"+name] = values
            if feeding:
                diagnostics["injection_mass"] = self.expected_mass
                diagnostics["injection_energy"] = 0.015*self.expected_mass
                for name, values in self.injection().items():
                    f["plasma_injection/"+name] = values
                for name in ("u", "v", "w"):
                    f["plasma_injection/"+name] = np.zeros_like(self.source)
                f["plasma_injection/specific_internal_energy"] = np.full_like(self.source, 0.015)
            for name, values in diagnostics.items():
                f["diagnostics/"+name] = values
            if si:
                f.create_group("physical_scaling")
                for name, value in self.scaling.items():
                    f["physical_scaling"].attrs[name] = value
            f.create_group("antenna_hardware")
            f["antenna_hardware"].attrs["angular_frequency_code"] = 2*np.pi
        return path

    def figure(self, path, *, units="code"):
        x, y, times, fields, diagnostics = thruster.load_thruster_results(path)
        metadata = thruster.load_thruster_metadata(path)
        fig, slider = thruster.make_figure(x, y, times, fields, diagnostics, path.name, 0,
                                         units=units, zoom=False, **metadata)
        self.addCleanup(plt.close, fig)
        return fig, slider, fields, diagnostics

    def test_source_history_integrates_interior_knots_between_sparse_outputs(self):
        history = thruster.injection_history(self.x, self.y, self.times, self.injection())
        np.testing.assert_allclose(history["amplitude"], self.sampled_amplitude, rtol=0, atol=1e-15)
        np.testing.assert_allclose(history["mass_rate"], self.expected_rate, rtol=0, atol=1e-15)
        np.testing.assert_allclose(history["cumulative_mass"], self.expected_mass, rtol=0, atol=1e-15)

    def test_absent_source_has_zero_rate_and_cumulative_mass(self):
        history = thruster.injection_history(self.x, self.y, self.times, None)
        for name in ("amplitude", "mass_rate", "cumulative_mass"):
            np.testing.assert_array_equal(history[name], np.zeros_like(self.times))

    def test_loader_preserves_injection_and_force_diagnostics(self):
        path = self.write_result()
        _, _, _, _, diagnostics = thruster.load_thruster_results(path)
        metadata = thruster.load_thruster_metadata(path)
        np.testing.assert_array_equal(metadata["injection"]["mass_rate"], self.source)
        np.testing.assert_array_equal(metadata["injection"]["time"], self.source_times)
        np.testing.assert_array_equal(diagnostics["injection_mass"], self.expected_mass)
        np.testing.assert_array_equal(diagnostics["solid_impulse_x"], [0, -0.12, -0.3])
        self.assertAlmostEqual(metadata["drive_omega"], 2*np.pi)
        self.assertEqual(metadata["resistivity"], self.resistivity)
        self.assertEqual(metadata["antenna_model"], "prescribed_current")
        with h5py.File(path) as f:
            np.testing.assert_array_equal(thruster.load_thruster_results(path)[3]["Jz"],
                                          f["fields/Jz"][:])

    def test_slider_tracks_source_amount_times_and_absolute_fields(self):
        fig, slider, fields, _ = self.figure(self.write_result())
        artists = fig._thruster_artists
        initial_text = artists["injection_text"].get_text()
        electric_limits = artists["electric_field_image"].get_clim()
        np.testing.assert_allclose(artists["feed_line"].get_ydata(), self.expected_rate, atol=1e-15)
        np.testing.assert_allclose(artists["mass_line"].get_ydata(), self.expected_mass, atol=1e-15)
        for frame in (1, 2, 0):
            slider.set_val(frame)
            image = artists["injection_image"].get_array()
            np.testing.assert_allclose(np.ma.filled(image, 0.0),
                self.source*self.sampled_amplitude[frame], atol=1e-15)
            for key in ("thrust_marker", "feed_marker", "isp_marker"):
                np.testing.assert_allclose(artists[key].get_xdata(), [self.times[frame]]*2, atol=1e-15)
            for panel in (1, 2, 3, 4, 5):
                image = fig.axes[panel].images[0].get_array()
                np.testing.assert_array_equal(np.ma.getmaskarray(image), self.solid)
            for panel, field in ((1, "u"), (3, "p"), (5, "rho")):
                image = fig.axes[panel].images[0].get_array()
                np.testing.assert_allclose(image.compressed(), fields[field][frame][~self.solid], atol=1e-15)
            expected_ez = (fields["v"][frame]*fields["Bx"][frame]
                           -fields["u"][frame]*fields["By"][frame]
                           +self.resistivity*fields["Jz"][frame])
            np.testing.assert_allclose(artists["electric_field_image"].get_array().compressed(),
                                       expected_ez[~self.solid], atol=1e-15)
            self.assertEqual(artists["electric_field_image"].get_clim(), electric_limits)
            self.assertIs(artists["electric_field_axis"], fig.axes[4])
            if frame == 1:
                self.assertNotEqual(artists["injection_text"].get_text(), initial_text)

    def test_signed_force_and_interval_means_use_distinct_stored_diagnostics(self):
        fig, _, _, _ = self.figure(self.write_result())
        a = fig._thruster_artists
        np.testing.assert_array_equal(a["thrust_instantaneous_line"].get_ydata(), [0, 10, -2])
        np.testing.assert_allclose(a["thrust_interval_line"].get_xdata(), [0.2, 0.7], atol=1e-15)
        np.testing.assert_allclose(a["thrust_interval_line"].get_ydata(), [0.3, 0.3], atol=1e-15)

    def test_no_feed_older_result_still_opens_and_shows_zero_source(self):
        path = self.write_result(feeding=False, si=True)
        metadata = thruster.load_thruster_metadata(path)
        self.assertIsNone(metadata["injection"])
        fig, slider, _, diagnostics = self.figure(path)
        self.assertNotIn("injection_mass", diagnostics)
        slider.set_val(2)
        a = fig._thruster_artists
        np.testing.assert_array_equal(a["feed_line"].get_ydata(), 0)
        np.testing.assert_array_equal(a["mass_line"].get_ydata(), 0)
        np.testing.assert_array_equal(np.ma.filled(a["injection_image"].get_array(), 0), 0)
        for key in ("isp_interval_line", "isp_cumulative_line"):
            if a[key] is not None:
                self.assertFalse(np.isfinite(a[key].get_ydata()).any())
        self.assertTrue(any("undefined" in text.get_text().lower() for text in a["isp_axis"].texts))

    def isp_factor(self):
        s = self.scaling
        force_unit = s["magnetic_field_t"]**2/(4*np.pi*1e-7)*s["length_m"]*s["out_of_plane_length_m"]
        mass_unit = s["mass_density_kg_m3"]*s["length_m"]**2*s["out_of_plane_length_m"]
        return force_unit*s["time_s"]/(mass_unit*9.80665)

    def test_specific_impulse_uses_matched_integrated_impulse_and_mass_windows(self):
        t = np.array([0.0, 1.0, 2.0, 3.0])
        impulse = np.array([0.0, -0.2, -0.3, -0.3])
        mass = np.array([0.0, 0.1, 0.1, 0.2])
        history = thruster.specific_impulse_history(t, impulse, mass, self.scaling)
        factor = self.isp_factor()
        self.assertAlmostEqual(history["factor"], factor)
        np.testing.assert_allclose(history["interval"][[0, 2]], np.array([2.0, 0.0])*factor)
        self.assertTrue(np.isnan(history["interval"][1]))  # thrust persists while feed is off
        self.assertTrue(np.isnan(history["cumulative"][0]))
        np.testing.assert_allclose(history["cumulative"][1:], np.array([2.0, 3.0, 1.5])*factor)
        self.assertFalse(np.isinf(history["interval"]).any())

    def test_specific_impulse_without_reference_scales_is_unavailable(self):
        history = thruster.specific_impulse_history(self.times, np.array([0.0, -0.12, -0.3]),
                                                   self.expected_mass, {})
        self.assertIsNone(history["factor"])
        self.assertTrue(np.isnan(history["interval"]).all())
        self.assertTrue(np.isnan(history["cumulative"]).all())

    def test_specific_impulse_shares_thrust_panel_and_uses_integrated_impulse(self):
        # Even with code units on the other histories, Isp is in seconds
        # when physical reference scales are available.
        fig, slider, _, _ = self.figure(self.write_result(si=True), units="code")
        a = fig._thruster_artists
        self.assertIsNot(a["isp_axis"], a["electric_field_axis"])
        self.assertTrue(a["isp_axis"].get_shared_x_axes().joined(a["isp_axis"], a["thrust_axis"]))
        np.testing.assert_allclose(a["isp_axis"].get_position().bounds,
                                   a["thrust_axis"].get_position().bounds)
        self.assertIn("(s)", a["isp_axis"].get_ylabel())
        self.assertIs(a["isp_interval_line"].axes, a["isp_axis"])
        self.assertIs(a["isp_marker"], a["thrust_marker"])
        factor = self.isp_factor()
        np.testing.assert_allclose(a["isp_interval_line"].get_xdata(), [0.2, 0.7])
        np.testing.assert_allclose(a["isp_interval_line"].get_ydata(),
                                   np.array([0.12/0.115, 0.18/0.135])*factor)
        cumulative = np.asarray(a["isp_cumulative_line"].get_ydata())
        self.assertTrue(np.isnan(cumulative[0]))
        np.testing.assert_allclose(cumulative[1:], np.array([0.12/0.115, 0.3/0.25])*factor)
        slider.set_val(1)
        np.testing.assert_array_equal(a["isp_marker"].get_xdata(), [0.4, 0.4])

    def test_electric_field_uses_ohm_signs_and_actual_model_current(self):
        fields = {
            "u": np.array([[2., -1.], [.5, 0.]]),
            "v": np.array([[-3., 2.], [1., 0.]]),
            "Bx": np.array([[4., -.2], [2., 0.]]),
            "By": np.array([[5., 6.], [-2., 0.]]),
            "Jz": np.array([[7., 8.], [-4., 1.]]),
            "Jz_antenna": np.array([[1., 3.], [-2., 1.]]),
        }
        np.testing.assert_allclose(thruster.electric_field_z(fields, 0.),
                                   [[-22., 5.6], [3., 0.]], atol=1e-15)
        np.testing.assert_allclose(thruster.electric_field_z(fields, .2, "prescribed_current"),
                                   [[-20.6, 7.2], [2.2, .2]], atol=1e-15)
        np.testing.assert_allclose(thruster.electric_field_z(fields, .2, "resistive_impressed"),
                                   [[-20.8, 6.6], [2.6, 0.]], atol=1e-15)

    def test_movie_renders_every_frame_and_restores_slider_view(self):
        fig, slider, fields, _ = self.figure(self.write_result(si=True))
        slider.set_val(1)
        artists = fig._thruster_artists
        rendered = []
        def capture():
            self.assertFalse(slider.ax.get_visible())
            rendered.append(artists["electric_field_image"].get_array().copy())
        with patch.object(thruster, "FFMpegWriter") as writer:
            writer.isAvailable.return_value = True
            writer.return_value.grab_frame.side_effect = capture
            thruster.export_movie(fig, self.directory/"movie"/"run.mp4", fps=12.5)
            self.assertEqual(writer.call_args.kwargs["fps"], 12.5)
            self.assertEqual(writer.return_value.grab_frame.call_count, len(self.times))
        self.assertEqual(len(rendered), len(self.times))
        for frame, image in enumerate(rendered):
            expected = (fields["v"][frame]*fields["Bx"][frame]
                        -fields["u"][frame]*fields["By"][frame]
                        +self.resistivity*fields["Jz"][frame])
            np.testing.assert_array_equal(np.ma.getmaskarray(image), self.solid)
            np.testing.assert_allclose(image.compressed(), expected[~self.solid], atol=1e-15)
        self.assertTrue(slider.ax.get_visible())
        self.assertEqual(slider.val, 1)
        np.testing.assert_array_equal(artists["electric_field_image"].get_array(), rendered[1])
        with self.assertRaisesRegex(ValueError, "frame rate"):
            thruster.export_movie(fig, self.directory/"invalid.mp4", fps=0)

    def test_si_display_uses_depth_aware_force_mass_flow_and_source_units(self):
        fig, slider, _, _ = self.figure(self.write_result(si=True), units="si")
        s = self.scaling
        force = s["magnetic_field_t"]**2/(4*np.pi*1e-7)*s["length_m"]*s["out_of_plane_length_m"]*1e3
        rate = s["mass_density_kg_m3"]*s["alfven_speed_m_s"]*s["length_m"]*s["out_of_plane_length_m"]*1e6
        mass = s["mass_density_kg_m3"]*s["length_m"]**2*s["out_of_plane_length_m"]*1e12
        time = s["time_s"]*1e6
        source = s["mass_density_kg_m3"]/s["time_s"]
        a = fig._thruster_artists
        np.testing.assert_allclose(a["thrust_instantaneous_line"].get_ydata(), np.array([0, 10, -2])*force)
        np.testing.assert_allclose(a["thrust_interval_line"].get_ydata(), np.array([0.3, 0.3])*force)
        np.testing.assert_allclose(a["feed_line"].get_ydata(), self.expected_rate*rate)
        np.testing.assert_allclose(a["mass_line"].get_ydata(), self.expected_mass*mass)
        np.testing.assert_allclose(a["feed_line"].get_xdata(), self.times*time)
        slider.set_val(1)
        np.testing.assert_allclose(a["thrust_marker"].get_xdata(), [0.4*time]*2)
        np.testing.assert_allclose(a["isp_marker"].get_xdata(), [0.4*time]*2)
        actual_source = np.ma.filled(a["injection_image"].get_array(), 0)
        np.testing.assert_allclose(actual_source, self.source*0.4*source)


if __name__ == "__main__":
    unittest.main()
