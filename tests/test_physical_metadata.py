"""Keep optional physical/design annotations with completed solver outputs."""
from pathlib import Path
import os
import subprocess
import sys
import unittest
from unittest.mock import patch

import h5py
import numpy as np

from test_boundary_conditions import ROOT, SolverTestCase
import run


class PhysicalMetadataTests(SolverTestCase):
    def clean_environment(self):
        return {key: value for key, value in os.environ.items()
                if not key.startswith('MHD_THRUSTER_')}

    def test_canonical_si_and_passive_budget_survive_preparation_and_cli(self):
        environment = self.clean_environment()
        with patch.dict(os.environ, environment, clear=True):
            problem = run.load_problem(ROOT / 'input/reconnection_thruster.py')
            prepared = self.directory / 'canonical_prepared.h5'
            run.prepare_input(problem, prepared, 0.0, 1)
        # Fixed independent SI checks catch missing or differently scaled units.
        self.assertAlmostEqual(problem.physical_scaling['length_m'], 0.01)
        self.assertAlmostEqual(problem.physical_scaling['magnetic_field_t'], 0.02)
        self.assertAlmostEqual(problem.physical_scaling['current_density_a_m2'],
                               1591549.4309189534, places=7)
        self.assertAlmostEqual(problem.antenna_hardware['radiated_power_w'],
                               0.2866829471872197, places=12)
        self.assertGreater(problem.current_scale, 0.74)
        self.assertLess(problem.current_scale, 0.76)
        executable = Path(os.environ.get('MHD_EXECUTABLE', ROOT / 'mhd2d')).resolve()
        if not executable.is_file():
            self.skipTest('Build the solver for the CLI metadata test')
        output = self.directory / 'canonical_output.h5'
        result = subprocess.run(
            [sys.executable, str(ROOT / 'run.py'), str(ROOT / 'input/reconnection_thruster.py'),
             '--executable', str(executable), '--t-end', '0', '--threads', '1', '--output', str(output)],
            cwd=ROOT, env=environment, capture_output=True, text=True, timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        with h5py.File(prepared) as before, h5py.File(output) as after:
            for name in run.OPTIONAL_METADATA_GROUPS:
                self.assertEqual(set(before[name].attrs), set(getattr(problem, name)))
                self.assertEqual(set(after[name].attrs), set(before[name].attrs))
                for key, value in before[name].attrs.items():
                    self.assertEqual(after[name].attrs[key], value, name + '/' + key)
            self.assertTrue(after['antenna_hardware'].attrs['within_selected_thermal_budget'])
            self.assertIn('passive', after['antenna_hardware'].attrs['cooling'])
            np.testing.assert_array_equal(after['time'][:], [0.0])

    def test_invalid_metadata_is_rejected_before_prepared_file_is_created(self):
        invalid = [
            ('not a mapping', TypeError),
            ({'nested': {'a': 1}}, TypeError),
            ({'array': [1, 2]}, TypeError),
            ({'value': np.nan}, ValueError),
            ({'value': np.inf}, ValueError),
            ({'bad/path': 1}, ValueError),
            ({'': 1}, ValueError),
            ({3: 1}, ValueError),
            ({'string': 'bad\0text'}, ValueError),
            ({'too_large': 2**100}, ValueError),
        ]
        for group in run.OPTIONAL_METADATA_GROUPS:
            for index, (value, error) in enumerate(invalid):
                with self.subTest(group=group, value=value):
                    problem = self.problem()
                    setattr(problem, group, value)
                    target = self.directory / f'{group}_{index}.h5'
                    with self.assertRaises(error):
                        run.prepare_input(problem, target, None, None)
                    self.assertFalse(target.exists())

    def test_problem_without_optional_groups_keeps_result_bytes_unchanged(self):
        prepared = self.prepare(self.problem())
        output = self.directory / 'legacy_output.h5'
        with h5py.File(output, 'w') as destination:
            destination.create_dataset('sentinel', data=np.arange(6).reshape(2, 3))
        before = output.read_bytes()
        run.copy_problem_metadata(prepared, output)
        self.assertEqual(output.read_bytes(), before)
        with h5py.File(prepared) as source:
            self.assertTrue(all(name not in source for name in run.OPTIONAL_METADATA_GROUPS))

    def test_failed_solver_output_is_not_annotated_as_a_completed_result(self):
        problem = self.directory / 'problem.py'
        problem.write_text("grid={'nx':4,'ny':4,'lx':1.,'ly':1.}\n"
                           "solver={'t_end':0.}\n"
                           "physical_scaling={'length_m':0.01}\n"
                           "antenna_hardware={'cooling':'passive','valid':True}\n"
                           "def initial_state(x,y): return {'rho':1.,'p':1.}\n")
        failed = self.directory / 'failed_solver.py'
        failed.write_text(f'#!{sys.executable}\n'
                          'import sys,h5py\n'
                          'with h5py.File(sys.argv[2],"w") as f: f.create_dataset("partial",data=[1.])\n'
                          'raise SystemExit(7)\n')
        failed.chmod(0o755)
        output = self.directory / 'partial.h5'
        result = subprocess.run([sys.executable, str(ROOT/'run.py'), str(problem),
                                 '--executable', str(failed), '--output', str(output)],
                                cwd=ROOT, capture_output=True, text=True, timeout=20)
        self.assertNotEqual(result.returncode, 0)
        with h5py.File(output) as source:
            self.assertIn('partial', source)
            self.assertTrue(all(name not in source for name in run.OPTIONAL_METADATA_GROUPS))


if __name__ == '__main__':
    unittest.main()
