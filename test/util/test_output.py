import unittest
import pathlib
import os

from dax.util.output import *


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

    def test_file_name_generator(self):
        with temp_dir():
            gen = get_file_name_generator(None)
            self.assertIn('bar', gen('bar'))
            self.assertIn('bar.pdf', gen('bar', 'pdf'))

            with self.assertRaises(ValueError, msg='Empty name did not raise'):
                gen('')

    def test_file_name_generator_unique(self):
        with temp_dir():
            gen = get_file_name_generator(None)

            for _ in range(5):
                n = gen('bar', 'pdf')
                p = pathlib.Path(n)
                self.assertFalse(p.exists(), 'Generator returned an existing file')

                # Create file for testing
                p.touch()

            with self.assertRaises(ValueError, msg='Empty name did not raise'):
                gen('')

    def test_dummy_file_name_generator(self):
        with temp_dir():
            gen = get_file_name_generator(None)
            self.assertIn(dummy_file_name_generator('bar'), gen('bar'))
            self.assertIn(dummy_file_name_generator('bar', 'pdf'), gen('bar', 'pdf'))

            with self.assertRaises(ValueError, msg='Empty name did not raise'):
                dummy_file_name_generator('')

    def test_file_name(self):
        with temp_dir():
            self.assertIn('bar', get_file_name(None, 'bar'))
            self.assertIn('bar.pdf', get_file_name(None, 'bar', 'pdf'))


if __name__ == '__main__':
    unittest.main()
