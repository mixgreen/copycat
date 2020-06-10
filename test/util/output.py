import unittest
import pathlib
import os

from dax.util.output import temp_dir


class OutputTestCase(unittest.TestCase):

    def test_temp_dir(self):
        cwd = os.getcwd()
        with temp_dir():
            self.assertNotEqual(cwd, os.getcwd(), 'CWD was not changed in temp_dir() context')
        self.assertEqual(cwd, os.getcwd(), 'CWD was not restored after temp_dir() context')

    def test_base_name(self):
        from dax.util.output import get_base_path
        with temp_dir():
            base = get_base_path(None)
            self.assertIsInstance(base, pathlib.Path)
            self.assertTrue(str(base).startswith(os.getcwd()))

    def test_file_name_generator(self):
        from dax.util.output import get_file_name_generator
        with temp_dir():
            gen = get_file_name_generator(None)
            self.assertIn('bar', gen('bar'))
            self.assertIn('bar.pdf', gen('bar', 'pdf'))

    def test_file_name(self):
        from dax.util.output import get_file_name
        with temp_dir():
            self.assertIn('bar', get_file_name(None, 'bar'))
            self.assertIn('bar.pdf', get_file_name(None, 'bar', 'pdf'))


if __name__ == '__main__':
    unittest.main()
