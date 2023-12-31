import unittest
import pathlib
import os
import os.path
from textwrap import dedent

from dax.util.output import *

from test.environment import CI_ENABLED
from test.artiq import master, client_submit


class OutputTestCase(unittest.TestCase):

    def test_temp_dir(self):
        cwd = os.getcwd()
        with temp_dir():
            self.assertNotEqual(cwd, os.getcwd(), 'CWD was not changed in temp_dir() context')
        self.assertEqual(cwd, os.getcwd(), 'CWD was not restored after temp_dir() context')

    def test_base_name(self):
        with temp_dir():
            base = get_base_path(None)
            self.assertIsInstance(base, pathlib.Path)
            self.assertTrue(str(base).startswith(os.getcwd()))

    @unittest.skipUnless(CI_ENABLED, 'Not in a CI environment, skipping slow test')
    def test_experiment_cwd(self):
        with master() as (path, process):
            with open('experiment.py', mode='w') as file:
                file.write(dedent("""
                import pathlib
                from artiq.experiment import EnvExperiment
                class BasePathExperiment(EnvExperiment):
                    def run(self):
                        print(pathlib.Path().absolute())
                """))

            self.assertEqual(client_submit(os.path.join(path, 'experiment.py')), 0, 'Unexpected RID returned')
            line = process.stdout.readline().rstrip()
            self.assertTrue(line.split(':print:')[-1].startswith(os.path.join(path, 'results')),
                            'CWD of experiment does not match expected location for dax.util.output.get_base_path()')

    def test_base_file_name_generator(self):
        with temp_dir():
            gen = FileNameGenerator(None)
            base_gen = BaseFileNameGenerator()
            self.assertIn(base_gen('bar'), gen('bar'))
            self.assertIn(base_gen('bar', 'pdf'), gen('bar', 'pdf'))
            self.assertEqual('bar.pdf', base_gen('bar', 'pdf'))
            self.assertEqual(base_gen('bar'), base_gen('bar'), 'Generated different file names, expected the same')

            with self.assertRaises(ValueError, msg='Empty name did not raise'):
                base_gen('')

    def test_base_file_name_generator_base_path(self):
        base_gen = BaseFileNameGenerator()
        base_path = os.path.join('some', 'path')
        base_gen_path = BaseFileNameGenerator(base_path)
        self.assertIn(base_gen('bar'), base_gen_path('bar'))
        self.assertEqual(os.path.join(base_path, 'bar'), base_gen_path('bar'))

    def test_file_name_generator(self):
        with temp_dir():
            gen = FileNameGenerator(None)
            self.assertIn('bar', gen('bar'))
            self.assertIn('bar.pdf', gen('bar', 'pdf'))

            with self.assertRaises(ValueError, msg='Empty name did not raise'):
                gen('')

    def test_file_name_generator_unique(self):
        with temp_dir():
            gen = FileNameGenerator(None)

            for _ in range(5):
                n = gen('bar', 'pdf')
                p = pathlib.Path(n)
                self.assertFalse(p.exists(), 'Generator returned an existing file')

                # Create file for testing
                p.touch()

            with self.assertRaises(ValueError, msg='Empty name did not raise'):
                gen('')

    def test_file_name(self):
        with temp_dir():
            self.assertIn('bar', get_file_name(None, 'bar'))
            self.assertIn('bar.pdf', get_file_name(None, 'bar', 'pdf'))
