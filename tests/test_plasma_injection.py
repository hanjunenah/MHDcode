"""Conservative optional plasma-source tests; all run products are temporary.

The source is physical mass addition, not a density reset: it supplies the
specified momentum and total energy, and mixing contributes thermal energy.
"""
from pathlib import Path
import os
import subprocess
import unittest

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from test_boundary_conditions import ROOT, SolverTestCase
from test_prescribed_current import PrescribedCurrentTestCase
import plot_antenna_results as antenna


INJECTION_DIAGNOSTICS = (
    "injection_mass", "injection_momentum_x", "injection_momentum_y",
    "injection_momentum_z", "injection_energy",
)


class PlasmaInjectionTests(SolverTestCase):
    def uniform_source_problem(self):
        problem = self.problem("uniform_injection")
        problem.grid.update(nx=12, ny=8)
        problem.solver.update(t_end=0.12, cfl=0.25)
        problem.output["interval"] = 0.03
        problem.initial_state = lambda x, y: {
            "rho": 1.1, "p": 0.4, "u": 0.2, "v": -0.1, "w": 0.05,
        }
        problem.plasma_injection = {
            "mass_rate": 0.7, "u": -0.3, "v": 0.25, "w": -0.2,
            "specific_internal_energy": 0.6,
        }
        return problem

    def check_uniform_solution(self, problem, time_integral):
        prepared = self.prepare(problem)
        result = self.simulate(prepared)
        with h5py.File(result) as f:
            t = f["time"][:]
            added_density = 0.7*time_integral(t)
            initial_velocity = np.array([0.2, -0.1, 0.05])
            injected_velocity = np.array([-0.3, 0.25, -0.2])
            density = 1.1+added_density
            momentum = 1.1*initial_velocity[:, None]+injected_velocity[:, None]*added_density
            energy = (0.4/(5/3-1)+0.5*1.1*np.dot(initial_velocity, initial_velocity)
                      +added_density*(0.6+0.5*np.dot(injected_velocity, injected_velocity)))
            pressure = (5/3-1)*(energy-0.5*np.sum(momentum**2, axis=0)/density)
            expected = {"rho": density, "p": pressure,
                        **{k: momentum[i]/density for i, k in enumerate(("u", "v", "w"))}}
            for name, value in expected.items():
                np.testing.assert_allclose(f["fields/"+name][:],
                    np.broadcast_to(value[:, None, None], f["fields/"+name].shape),
                    rtol=0, atol=3e-12)
            area = problem.grid["lx"]*problem.grid["ly"]
            d = f["diagnostics"]
            np.testing.assert_allclose(d["injection_mass"][:], area*added_density,
                                       rtol=0, atol=2e-13)
            for i, axis in enumerate("xyz"):
                np.testing.assert_allclose(d["injection_momentum_"+axis][:],
                    area*added_density*injected_velocity[i], rtol=0, atol=2e-13)
            np.testing.assert_allclose(d["injection_energy"][:],
                area*added_density*(0.6+0.5*np.dot(injected_velocity, injected_velocity)),
                rtol=0, atol=2e-13)
            # Fresh mass moving differently from the original fluid mixes:
            # its kinetic energy cannot simply be discarded or counted twice.
            np.testing.assert_allclose(d["total_energy"][:]-d["total_energy"][0],
                                       d["injection_energy"][:], rtol=0, atol=2e-12)
        with h5py.File(prepared) as before, h5py.File(result) as after:
            for name, dataset in before["plasma_injection"].items():
                np.testing.assert_array_equal(after["plasma_injection/"+name][:], dataset[:])

    def test_uniform_addition_preserves_injected_mass_momentum_and_energy(self):
        self.check_uniform_solution(self.uniform_source_problem(), lambda t: t)

    def test_time_dependent_linear_source_is_integrated_at_rk_stages(self):
        problem = self.uniform_source_problem()
        problem.plasma_injection.update(time_profile=lambda t: 0.3+0.8*t,
                                        sampling_interval=0.025)
        self.check_uniform_solution(problem, lambda t: 0.3*t+0.4*t*t)

    def test_pulse_knots_are_not_crossed_by_source_time_quadrature(self):
        problem = self.uniform_source_problem()
        problem.plasma_injection.update(
            time_profile=lambda t: np.maximum(0.0, 1.0-abs(t-0.06)/0.06),
            sampling_interval=0.02,
        )

        def integrated_pulse(t):
            return np.where(t <= 0.06, t*t/0.12,
                            0.03+(t-0.06)-(t-0.06)**2/0.12)

        self.check_uniform_solution(problem, integrated_pulse)

    def test_localized_source_conserves_global_budgets(self):
        problem = self.problem("localized_injection")
        problem.solver["t_end"] = 0.08
        problem.output["interval"] = 0.02
        problem.plasma_injection = {
            "mass_rate": lambda x, y: 0.8*np.exp(-((x-0.4)/0.1)**2-((y-0.3)/0.06)**2),
            "u": lambda x, y: 0.2+0.1*y,
            "v": -0.1, "w": 0.04,
            "specific_internal_energy": lambda x, y: 0.5+0.1*x,
            "time_profile": 0.7,
        }
        prepared = self.prepare(problem)
        with h5py.File(prepared) as f:
            g = f["plasma_injection"]
            q = g["mass_rate"][:]
            vel = [g[k][:] for k in ("u", "v", "w")]
            e = g["specific_internal_energy"][:]
            cell_area = problem.grid["lx"]*problem.grid["ly"]/(q.size)
            rates = {
                "mass": float(np.sum(q)*cell_area*0.7),
                "momentum_x": float(np.sum(q*vel[0])*cell_area*0.7),
                "momentum_y": float(np.sum(q*vel[1])*cell_area*0.7),
                "momentum_z": float(np.sum(q*vel[2])*cell_area*0.7),
                "total_energy": float(np.sum(q*(e+0.5*sum(v*v for v in vel)))*cell_area*0.7),
            }
        result = self.simulate(prepared)
        with h5py.File(result) as f:
            t = f["time"][:]
            d = f["diagnostics"]
            for name, rate in rates.items():
                source_name = "injection_energy" if name == "total_energy" else "injection_"+name
                np.testing.assert_allclose(d[source_name][:], rate*t, rtol=0, atol=2e-13)
                np.testing.assert_allclose(d[name][:]-d[name][0], rate*t,
                                           rtol=0, atol=2e-12)
            self.assertGreater(float(np.ptp(f["fields/rho"][-1])), 1e-3)

    def test_absent_source_and_explicit_zero_source_have_identical_fluid(self):
        problem = self.problem("no_injection")
        absent = self.simulate(self.prepare(problem, "absent"), "absent_result")
        problem.plasma_injection = {"mass_rate": 0.0, "specific_internal_energy": 0.0}
        zero = self.simulate(self.prepare(problem, "zero"), "zero_result")
        with h5py.File(absent) as a, h5py.File(zero) as b:
            self.assertNotIn("plasma_injection", a)
            for name in a["fields"]:
                np.testing.assert_array_equal(a["fields/"+name][:], b["fields/"+name][:])
            for name in INJECTION_DIAGNOSTICS:
                np.testing.assert_array_equal(a["diagnostics/"+name][:], 0.0)
                np.testing.assert_array_equal(b["diagnostics/"+name][:], 0.0)

    def test_viewer_energy_balance_includes_plasma_feed_energy(self):
        problem = self.uniform_source_problem()
        result = self.simulate(self.prepare(problem))
        x, y, times, fields, diagnostics = antenna.load_antenna_results(result)
        self.assertIn("injection_energy", diagnostics)
        self.assertGreater(float(diagnostics["injection_energy"][-1]), 1e-3)
        figure, slider = antenna.make_figure(
            x, y, times, fields, diagnostics, result.name, len(times)-1,
        )
        self.addCleanup(plt.close, figure)
        energy_axis = next(axis for axis in figure.axes
                           if axis.get_title() == "Energy changes and external balance")
        balance_lines = [line for line in energy_axis.lines if "feed" in line.get_label()]
        self.assertEqual(len(balance_lines), 1)
        balance = balance_lines[0]
        np.testing.assert_array_equal(balance.get_xdata(), times)
        np.testing.assert_allclose(balance.get_ydata(),
            diagnostics["total_energy"]-diagnostics["total_energy"][0],
            rtol=0, atol=2e-12)
        # With no antenna, fields or open faces the source is the complete
        # external energy balance, so omitting it leaves a visibly wrong curve.
        np.testing.assert_array_equal(balance.get_ydata(), diagnostics["injection_energy"])


class PoweredPlasmaInjectionTests(PrescribedCurrentTestCase):
    def test_out_of_plane_injection_is_rejected_by_planar_powered_model(self):
        problem = self.powered_slab(0.02)
        problem.plasma_injection = {
            "mass_rate": lambda x, y: np.where(x < 0.2, 0.8, 0.0),
            "w": 0.1, "specific_internal_energy": 0.4,
        }
        with self.assertRaisesRegex(ValueError, "planar|prescribed|w"):
            self.prepare(problem, "nonplanar_python")
        problem.plasma_injection["w"] = 0.0
        prepared = self.prepare(problem, "nonplanar_hdf5")
        with h5py.File(prepared, "a") as f:
            f["plasma_injection/w"][0, 0] = 0.1
        executable = Path(os.environ.get("MHD_EXECUTABLE", ROOT / "mhd2d")).resolve()
        if not executable.is_file():
            self.skipTest("Build the solver for direct-HDF5 validation")
        result = subprocess.run([str(executable), str(prepared), str(self.directory/"nonplanar_result.h5")],
            cwd=ROOT, capture_output=True, text=True, timeout=20)
        self.assertNotEqual(result.returncode, 0, result.stdout+result.stderr)

    def test_powered_body_with_injection_closes_mass_and_energy_ledgers(self):
        problem = self.powered_slab(0.02)
        problem.plasma_injection = {
            "mass_rate": lambda x, y: np.where((x > 0.05) & (x < 0.2)
                & (abs(y-0.375) < 0.08), 0.8, 0.0),
            "u": 0.07, "v": 0.0, "w": 0.0, "specific_internal_energy": 0.4,
        }
        prepared = self.prepare(problem)
        result = self.simulate(prepared)
        with h5py.File(result) as f:
            d = f["diagnostics"]
            expected_energy = (d["total_energy"][0]+d["antenna_energy_injected"][:]
                +d["current_drive_energy_correction"][:]+d["injection_energy"][:]
                +d["absorption_energy_exchange"][:]-d["boundary_energy_outflow"][:])
            expected_mass = (d["plasma_mass"][0]+d["injection_mass"][:]
                +d["absorption_mass_exchange"][:]-d["boundary_mass_outflow"][:])
            np.testing.assert_allclose(d["total_energy"][:], expected_energy, rtol=0, atol=3e-11)
            np.testing.assert_allclose(d["plasma_mass"][:], expected_mass, rtol=0, atol=3e-12)
            self.assertGreater(float(d["injection_mass"][-1]), 1e-4)
            np.testing.assert_array_equal(d["max_wall_mass_flux"][:], 0.0)
            self.assertLess(float(d["max_stage_current_constraint_error"][:].max()), 2e-9)
            body = f["solid/mask"][:].astype(bool)
            for name in ("u", "v", "w"):
                np.testing.assert_array_equal(f["fields/"+name][:][:, body], 0.0)


class PlasmaInjectionValidationTests(SolverTestCase):
    def test_required_source_parameters_are_rejected_when_missing(self):
        for name in ("mass_rate", "specific_internal_energy"):
            with self.subTest(missing=name):
                problem = self.problem()
                problem.plasma_injection = {"mass_rate": 0.2, "specific_internal_energy": 0.4}
                del problem.plasma_injection[name]
                with self.assertRaisesRegex(ValueError, name):
                    self.prepare(problem, "missing_"+name)

    def test_invalid_profiles_or_temporal_controls_are_rejected(self):
        invalid = [
            {"mass_rate": -0.1}, {"mass_rate": np.nan},
            {"mass_rate": np.ones((2, 2))}, {"specific_internal_energy": 0.0},
            {"specific_internal_energy": -0.1}, {"specific_internal_energy": np.inf},
            {"u": np.nan}, {"time_profile": -0.1},
            {"time_profile": lambda t: np.nan, "sampling_interval": 0.01},
            {"time_profile": lambda t: -np.ones_like(t), "sampling_interval": 0.01},
            {"time_profile": lambda t: t},
            {"time_profile": lambda t: t, "sampling_interval": -0.1},
        ]
        for i, replacement in enumerate(invalid):
            with self.subTest(replacement=replacement):
                problem = self.problem()
                problem.plasma_injection = {"mass_rate": 0.2, "specific_internal_energy": 0.4}
                problem.plasma_injection.update(replacement)
                with self.assertRaises((ValueError, TypeError)):
                    self.prepare(problem, f"invalid_{i}")

    def test_nonzero_injection_into_stationary_solid_is_rejected(self):
        problem = self.problem()
        problem.solid_mask = lambda x, y: (x > 0.4) & (x < 0.6)
        problem.plasma_injection = {"mass_rate": 0.2, "specific_internal_energy": 0.4}
        with self.assertRaisesRegex(ValueError, "solid"):
            self.prepare(problem)

    def test_direct_hdf5_input_is_independently_validated(self):
        executable = Path(os.environ.get("MHD_EXECUTABLE", ROOT / "mhd2d")).resolve()
        if not executable.is_file():
            self.skipTest("Build the solver for direct-HDF5 validation")
        problem = self.problem()
        problem.plasma_injection = {"mass_rate": 0.2, "specific_internal_energy": 0.4,
            "time_profile": lambda t: 1+t, "sampling_interval": 0.02}
        cases = ("mass_rate", "specific_internal_energy", "amplitude", "time", "time_coverage",
                 "u", "solid_overlap")
        for name in cases:
            with self.subTest(invalid=name):
                if name == "solid_overlap":
                    problem.solid_mask = lambda x, y: (x > 0.4) & (x < 0.6)
                    problem.plasma_injection["mass_rate"] = lambda x, y: np.where(problem.solid_mask(x, y), 0., 0.2)
                prepared = self.prepare(problem, name)
                with h5py.File(prepared, "a") as f:
                    g = f["plasma_injection"]
                    if name == "amplitude":
                        g[name][0] = -1.0
                    elif name == "time":
                        g[name][1] = g[name][0]
                    elif name == "time_coverage":
                        g["time"][:] *= 0.5
                    elif name == "u":
                        g[name][0, 0] = np.nan
                    elif name == "solid_overlap":
                        mask = f["solid/mask"][:].astype(bool)
                        q = g["mass_rate"][:]
                        q[mask] = 0.2
                        g["mass_rate"][:] = q
                    elif name == "specific_internal_energy":
                        g[name][0, 0] = 0.0
                    else:
                        g[name][0, 0] = -1.0
                result = subprocess.run([str(executable), str(prepared), str(self.directory/(name+"_result.h5"))],
                    cwd=ROOT, capture_output=True, text=True, timeout=20)
                self.assertNotEqual(result.returncode, 0, name+result.stdout+result.stderr)


if __name__ == "__main__":
    unittest.main()
