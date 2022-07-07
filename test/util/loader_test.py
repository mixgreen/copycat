import unittest
import shutil
import os
import os.path

import dax.util.output
import dax.util.loader


def _write_test_file(path):
    file_name = os.path.join(path, 'foo.py')
    with open(file_name, 'w') as f:
        f.write("FOO = 5")
    return file_name


class LoaderTestCase(unittest.TestCase):

    def _test_load_module(self, file_name):
        m = dax.util.loader.load_module(file_name, prefix='loader_test_case_')
        self.assertEqual(m.FOO, 5)

    def test_load_module(self):
        with dax.util.output.temp_dir() as t:
            file_name = _write_test_file(t)
            self._test_load_module(file_name)

    def test_load_archive(self):
        formats = shutil.get_archive_formats()
        if not formats:
            raise unittest.SkipTest('No archive formats available')
        archive_format, _ = formats[0]

        with dax.util.output.temp_dir() as t:
            file_name = _write_test_file(t)
            shutil.make_archive(os.path.join(t, f'archive.{archive_format}'), archive_format, root_dir=t)
            self._test_load_module(file_name)
