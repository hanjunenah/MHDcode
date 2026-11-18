"""Keep optional physical/design annotations with completed solver outputs."""
from pathlib import Path
import os
import subprocess
import sys
import unittest

import h5py
import numpy as np

from test_boundary_conditions import ROOT, SolverTestCase
import run


class PhysicalMetadataTests(SolverTestCase):
    def clean_environment(self):
        return {key: value for key, value in os.environ.items()
                if not key.startswith('MHD_THRUSTER_')}

    def test_optional_metadata_survives_preparation_and_cli(self):
        environment = self.clean_environment()
        metadata = {
            'physical_scaling': {'length_m': 0.4, 'magnetic_field_t': 0.3},
            'antenna_hardware': {'description': 'metadata fixture without an antenna',
                                 'enabled': False, 'sequence': 3},
            'plasma_loading': {'description': 'uniform stationary fluid',
                               'ongoing_injection': False},
        }
        problem_path = self.directory / 'annotated_problem.py'
        problem_path.write_text(
            "grid={'nx':4,'ny':4,'lx':1.,'ly':1.}\n"
            "solver={'t_end':0.,'threads':1}\n"
            "def initial_state(x,y): return {'rho':1.,'p':1.}\n"
            "def magnetic_field(x,y): return 0.,0.,0.\n"
            + ''.join(f'{name}={values!r}\n' for name, values in metadata.items())
        )
        problem = run.load_problem(problem_path)
        prepared = self.directory / 'annotated_prepared.h5'
        run.prepare_input(problem, prepared, 0.0, 1)
        executable = Path(os.environ.get('MHD_EXECUTABLE', ROOT / 'mhd2d')).resolve()
        if not executable.is_file():
            self.skipTest('Build the solver for the CLI metadata test')
        output = self.directory / 'annotated_output.h5'
        result = subprocess.run(
            [sys.executable, str(ROOT / 'run.py'), str(problem_path),
             '--executable', str(executable), '--t-end', '0', '--threads', '1', '--output', str(output)],
            cwd=ROOT, env=environment, capture_output=True, text=True, timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        with h5py.File(prepared) as before, h5py.File(output) as after:
            for name in run.OPTIONAL_METADATA_GROUPS:
                self.assertEqual(set(before[name].attrs), set(metadata[name]))
                self.assertEqual(set(after[name].attrs), set(before[name].attrs))
                for key, value in metadata[name].items():
                    self.assertEqual(before[name].attrs[key], value, name + '/' + key)
                    self.assertEqual(after[name].attrs[key], value, name + '/' + key)
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
