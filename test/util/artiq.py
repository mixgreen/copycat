import unittest

from artiq.language.core import rpc, portable, kernel, host_only
import artiq.experiment

import dax.util.artiq
import dax.util.output

__all__ = ['ArtiqTestCase']


class ArtiqTestCase(unittest.TestCase):

    def test_get_managers(self):
        # Create an experiment object using the helper get_managers() function
        self.assertIsInstance(artiq.experiment.EnvExperiment(dax.util.artiq.get_managers()),
                              artiq.experiment.HasEnvironment)

    def test_get_managers_dataset_db(self):
        with dax.util.output.temp_dir():
            dataset_db = 'dataset_db.pyon'
            key = 'foo'
            value = 99

            with open(dataset_db, mode='x') as f:
                # Write pyon file
                f.write(f'{{\n    "{key}": {value}\n}}')

            # Create environment
            env = artiq.experiment.EnvExperiment(dax.util.artiq.get_managers(dataset_db=dataset_db))
            self.assertEqual(env.get_dataset(key), value, 'Retrieved dataset did not match earlier set value')

    def test_is_kernel(self):
        self.assertFalse(dax.util.artiq.is_kernel(self._undecorated_func),
                         'Undecorated function wrongly marked as a kernel function')
        self.assertFalse(dax.util.artiq.is_kernel(self._rpc_func),
                         'RPC function wrongly marked as a kernel function')
        self.assertFalse(dax.util.artiq.is_kernel(self._portable_func),
                         'Portable function wrongly marked as a kernel function')
        self.assertTrue(dax.util.artiq.is_kernel(self._kernel_func),
                        'Kernel function not marked as a kernel function')
        self.assertFalse(dax.util.artiq.is_kernel(self._host_only_func),
                         'Host only function wrongly marked as a kernel function')

    def test_is_portable(self):
        self.assertFalse(dax.util.artiq.is_portable(self._undecorated_func),
                         'Undecorated function wrongly marked as a portable function')
        self.assertFalse(dax.util.artiq.is_portable(self._rpc_func),
                         'RPC function wrongly marked as a portable function')
        self.assertTrue(dax.util.artiq.is_portable(self._portable_func),
                        'Portable function not marked as a portable function')
        self.assertFalse(dax.util.artiq.is_portable(self._kernel_func),
                         'Kernel function wrongly marked as a portable function')
        self.assertFalse(dax.util.artiq.is_portable(self._host_only_func),
                         'Host only function wrongly marked as a portable function')

    def test_is_host_only(self):
        self.assertFalse(dax.util.artiq.is_host_only(self._undecorated_func),
                         'Undecorated function wrongly marked as a host only function')
        self.assertFalse(dax.util.artiq.is_host_only(self._rpc_func),
                         'RPC function wrongly marked as a host only function')
        self.assertFalse(dax.util.artiq.is_host_only(self._portable_func),
                         'Portable function wrongly marked as a host only function')
        self.assertFalse(dax.util.artiq.is_host_only(self._kernel_func),
                         'Kernel function wrongly marked as a host only function')
        self.assertTrue(dax.util.artiq.is_host_only(self._host_only_func),
                        'Host only function not marked as a host only function')

    """Functions used for tests"""

    def _undecorated_func(self):
        pass

    @rpc
    def _rpc_func(self):
        pass

    @portable
    def _portable_func(self):
        pass

    @kernel
    def _kernel_func(self):
        pass

    @host_only
    def _host_only_func(self):
        pass


if __name__ == '__main__':
    unittest.main()
