"""Separate the sponge reservoir from its exterior Riemann state.

These small integrations check transmission, legacy compatibility and the
native magnetic constraint. They do not establish a nonreflecting boundary.
"""
from pathlib import Path
import os
import subprocess

import h5py
import numpy as np

from test_absorbing_boundaries import AbsorbingTestCase
from test_boundary_conditions import ROOT
from test_prescribed_current import PrescribedCurrentTestCase


class AbsorptionExteriorTests(AbsorbingTestCase):
    def test_invalid_exterior_option_is_rejected_in_python(self):
        for value in ("radiating", "", None, 1):
            with self.subTest(value=value):
                problem = self.absorbing_problem()
                problem.solver["absorption_exterior"] = value
                with self.assertRaises((ValueError, TypeError)):
                    self.prepare(problem, "bad_python")

    def test_solver_rechecks_invalid_exterior_option_in_raw_input(self):
        problem = self.absorbing_problem()
        prepared = self.prepare(problem)
        with h5py.File(prepared, "a") as f:
            f["config"].attrs["absorption_exterior"] = "radiating"
        executable = Path(os.environ.get("MHD_EXECUTABLE", ROOT / "mhd2d")).resolve()
        result = subprocess.run([str(executable), str(prepared), str(self.directory/"bad_result.h5")],
            cwd=ROOT, capture_output=True, text=True, timeout=20)
        self.assertNotEqual(result.returncode, 0)
        self.assertRegex(result.stderr, "exterior|ambient|outflow")

    def test_omitted_option_preserves_legacy_ambient_solution_bitwise(self):
        problem = self.absorbing_problem()
        problem.solver.update(t_end=0.025)
        problem.output["interval"] = 0.0125
        problem.initial_state = lambda x, y: {
            "rho": 1+0.1*np.exp(-((x-0.85)/0.07)**2), "p": 0.32, "u": 0.08,
        }
        legacy = self.prepare(problem, "legacy_prepared")
        with h5py.File(legacy, "a") as f:
            if "absorption_exterior" in f["config"].attrs:
                del f["config"].attrs["absorption_exterior"]
        problem.solver["absorption_exterior"] = "ambient"
        explicit = self.prepare(problem, "ambient_prepared")
        legacy = self.simulate(legacy, "legacy")
        explicit = self.simulate(explicit, "ambient")
        with h5py.File(legacy) as a, h5py.File(explicit) as b:
            np.testing.assert_array_equal(a["time"][:], b["time"][:])
            for group in ("fields", "diagnostics"):
                for name in a[group]:
                    np.testing.assert_array_equal(a[group+"/"+name][:], b[group+"/"+name][:])
            self.assertEqual(self.string_attribute(b.attrs["absorption_exterior"]), "ambient")

    def test_uniform_matched_ambient_is_preserved_with_outflow_exterior(self):
        expected = {"rho": 1.0, "p": 0.3, "u": 0.12, "v": -0.03, "w": 0.02,
                    "Bx": 0.4, "By": -0.2, "Bz": 0.1}
        for modes in (("absorbing", "periodic"), ("absorbing", "absorbing")):
            with self.subTest(modes=modes):
                problem = self.absorbing_problem()
                problem.grid.update(nx=24, ny=16, lx=1.0, ly=0.6)
                problem.solver.update(boundary_x=modes[0], boundary_y=modes[1],
                    absorption_exterior="outflow", absorption_rate=0.02,
                    absorption_magnetic_diffusivity=0.005)
                problem.initial_state = lambda x, y: {
                    k: v for k, v in expected.items() if k in ("rho", "p", "u", "v", "w")}
                problem.magnetic_field = lambda x, y: (0.4, -0.2, 0.1)
                problem.boundary_ambient = dict(expected)
                name = "uniform_"+"_".join(modes)
                result = self.simulate(self.prepare(problem, name), name)
                with h5py.File(result) as f:
                    for key, value in expected.items():
                        np.testing.assert_allclose(f["fields/"+key][:], value, rtol=0, atol=3e-14)
                    self.assert_exchange_budgets(f)
                    self.assertEqual(self.string_attribute(f.attrs["absorption_exterior"]), "outflow")

    def test_normal_field_survives_mismatched_ambient_with_outflow_exterior(self):
        problem = self.absorbing_problem()
        problem.solver.update(absorption_exterior="outflow", absorption_rate=0.02,
                              absorption_magnetic_diffusivity=0.005)
        problem.magnetic_field = lambda x, y: (0.4, 0.0, 0.0)
        # The ambient has Bx=0. A constant normal field cannot be damped by
        # a curl-form sponge or reset by the exterior state.
        result = self.simulate(self.prepare(problem))
        with h5py.File(result) as f:
            np.testing.assert_allclose(f["fields/Bx"][:], 0.4, rtol=0, atol=3e-14)
            np.testing.assert_allclose(f["fields/divB"][:], 0.0, rtol=0, atol=3e-14)
            self.assert_exchange_budgets(f)

    def test_subfast_exterior_outflow_transmits_flux_without_ambient_reset(self):
        # A subfast right-going stream reaches the right face, while the left
        # face is at rest. At this short time the internal transition cannot
        # influence either exterior face. Tiny sponge rate isolates face BCs.
        paths = {}
        for label in ("pure_outflow", "outflow", "ambient"):
            problem = self.absorbing_problem()
            problem.grid.update(nx=48, ny=8, lx=1.0, ly=0.25)
            problem.solver.update(t_end=0.002, absorption_rate=1e-8,
                absorption_magnetic_diffusivity=0.0,
                absorption_exterior="ambient" if label == "pure_outflow" else label)
            if label == "pure_outflow":
                problem.solver["boundary_x"] = "outflow"
            problem.initial_state = lambda x, y: {
                "rho": 0.01, "p": 0.01, "u": np.where(x > 0.6, 0.1, 0.0)}
            problem.boundary_ambient = {"rho": 0.01, "p": 0.01}
            problem.output["interval"] = 0.002
            paths[label] = self.simulate(self.prepare(problem, label), label)
        with h5py.File(paths["pure_outflow"]) as pure, h5py.File(paths["outflow"]) as out, h5py.File(paths["ambient"]) as ambient:
            expected_mass_out = 0.01*0.1*0.25*0.002
            np.testing.assert_allclose(pure["diagnostics/boundary_mass_outflow"][-1], expected_mass_out,
                                       rtol=0, atol=2e-14)
            np.testing.assert_allclose(out["diagnostics/boundary_mass_outflow"][:],
                                       pure["diagnostics/boundary_mass_outflow"][:], rtol=0, atol=2e-14)
            for name in ("rho", "p", "u"):
                np.testing.assert_allclose(out["fields/"+name][-1], pure["fields/"+name][-1],
                                           rtol=0, atol=2e-10)
            self.assertLess(float(ambient["diagnostics/boundary_mass_outflow"][-1]),
                            0.8*expected_mass_out)
            self.assertGreater(float(np.max(abs(ambient["fields/u"][-1, :, -1]-0.1))), 1e-4)
            self.assert_exchange_budgets(out)
            self.assert_exchange_budgets(ambient)


class PoweredAbsorptionExteriorTests(PrescribedCurrentTestCase):
    def test_powered_body_constraint_and_budgets_with_gentle_outflow_sponge(self):
        problem = self.powered_slab(0.02)
        problem.solver.update(boundary_x="absorbing", boundary_y="absorbing",
            absorption_exterior="outflow", absorption_width=0.125,
            absorption_rate=0.02, absorption_magnetic_diffusivity=0.005, t_end=0.08)
        problem.output["interval"] = 0.02
        problem.antennas[0]["time_profile"] = lambda t: np.minimum(t/0.04, 1.0)
        problem.magnetic_field = lambda x, y: (0.1, 0.0, 0.0)
        problem.boundary_ambient = {"rho": 1.0, "p": 0.3}
        result = self.simulate(self.prepare(problem))
        with h5py.File(result) as f:
            d = f["diagnostics"]
            nodes = f["prescribed_current"]
            powered = nodes["powered_mask"][:].astype(bool)
            np.testing.assert_allclose(nodes["Jz"][:][:, powered], nodes["Jz_target"][:][:, powered],
                                       rtol=0, atol=2e-9)
            self.assertLess(float(np.max(d["max_stage_current_constraint_error"][:])), 2e-9)
            self.assertLess(float(np.max(d["max_abs_ct_divB"][:])), 2e-13)
            body = f["solid/mask"][:].astype(bool)
            for name in ("u", "v", "w"):
                np.testing.assert_array_equal(f["fields/"+name][:][:, body], 0.0)
            np.testing.assert_array_equal(d["max_wall_mass_flux"][:], 0.0)
            np.testing.assert_allclose(d["plasma_mass"][:],
                d["plasma_mass"][0]+d["absorption_mass_exchange"][:]-d["boundary_mass_outflow"][:],
                rtol=0, atol=3e-12)
            self.assert_energy_budget(f)
