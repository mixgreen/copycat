import unittest
import unittest.mock
import inspect

from artiq.language.core import rpc, portable, host_only

import dax.interfaces.data_context


class DataContextInstance(dax.interfaces.data_context.DataContextInterface):
    """This should be a correct implementation of the data context interface."""

    @rpc(flags={'async'})
    def open(self):
        pass

    @rpc(flags={'async'})
    def close(self):
        pass

    @portable
    def __enter__(self):
        self.open()

    @portable
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @host_only
    def get_raw(self):
        return [[[]]]


class DataContextInterfaceTestCase(unittest.TestCase):
    HOST_ONLY_FN = {'get_raw'}
    NOT_HOST_ONLY_FN = {n for n, _ in inspect.getmembers(DataContextInstance, inspect.isfunction)
                        if not n.startswith('_') or n in {'__enter__', '__exit__'}} - HOST_ONLY_FN

    def test_validate_interface(self):
        interface = DataContextInstance()
        self.assertTrue(dax.interfaces.data_context.validate_interface(interface))

    def _validate_functions(self, fn_names, valid_fn):
        all_fn = {self._dummy_fn, self._portable_fn, self._rpc_fn, self._host_only_fn}
        interface = DataContextInstance()

        for fn in fn_names:
            with self.subTest(fn=fn):
                # Make sure the interface is valid
                dax.interfaces.data_context.validate_interface(interface)

                for replacement_fn in valid_fn:
                    with unittest.mock.patch.object(interface, fn, replacement_fn):
                        dax.interfaces.data_context.validate_interface(interface)

                for replacement_fn in all_fn - valid_fn:
                    with unittest.mock.patch.object(interface, fn, replacement_fn):
                        # Patch interface and verify the validation fails
                        with self.assertRaises(TypeError, msg='Validate did not raise'):
                            dax.interfaces.data_context.validate_interface(interface)

    def test_validate_not_host_only_fn(self):
        self.assertGreater(len(self.NOT_HOST_ONLY_FN), 2, 'Not enough functions were found')
        self._validate_functions(self.NOT_HOST_ONLY_FN, {self._dummy_fn, self._portable_fn, self._rpc_fn})

    def test_validate_host_only_fn(self):
        self._validate_functions(self.HOST_ONLY_FN, {self._host_only_fn})

    """Helper functions"""

    def _dummy_fn(self):
        pass

    @portable
    def _portable_fn(self):
        pass

    @rpc
    def _rpc_fn(self):
        pass

    @host_only
    def _host_only_fn(self):
        """Dummy function used for testing."""
        pass


if __name__ == '__main__':
    unittest.main()
