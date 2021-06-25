import unittest
import os.path

import dax


class PackageTestCase(unittest.TestCase):

    def test_dax_dir(self):
        self.assertIsInstance(dax.__dax_dir__, str, '__dax_dir__ has an unexpected type')
        self.assertTrue(os.path.exists(dax.__dax_dir__), '__dax_dir__ path does not exist')
        self.assertTrue(os.path.isdir(dax.__dax_dir__), '__dax_dir__ path is not a directory')
        self.assertEqual(dax.__dax_dir__, os.path.normpath(dax.__dax_dir__),
                         '__dax_dir__ path was not normalized')

    def test_version(self):
        self.assertIsInstance(dax.__version__, str, '__version__ has an unexpected type')
        self.assertTrue(len(dax.__version__) > 0, '__version__ is empty')
        self.assertNotIn('_', dax.__version__, '__version__ contains an underscore')
