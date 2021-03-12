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
    def __exit__(self, exc_type, exc_val, exc_tb):  # noqa: ATQ306
        self.close()


class DataContextInterfaceTestCase(unittest.TestCase):
    def test_valid_interface(self):
        interface = DataContextInstance()
        self.assertTrue(dax.interfaces.data_context.validate_interface(interface))

    def _validate_functions(self, fn_names):
        interface = DataContextInstance()

        for fn in fn_names:
            with self.subTest(fn=fn):
                # Make sure the interface is valid
                dax.interfaces.data_context.validate_interface(interface)

                for replacement_fn in [self._dummy_fn, self._portable_fn, self._rpc_fn]:
                    with unittest.mock.patch.object(interface, fn, replacement_fn):
                        dax.interfaces.data_context.validate_interface(interface)

                with unittest.mock.patch.object(interface, fn, self._host_only_fn):
                    # Patch interface and verify the validation fails
                    with self.assertRaises(AssertionError, msg='Validate did not raise'):
                        dax.interfaces.data_context.validate_interface(interface)

    def test_validate_not_host_only_fn(self):
        not_host_only_fn = [n for n, _ in inspect.getmembers(DataContextInstance, inspect.isfunction)
                            if not n.startswith('_') or n in {'__enter__', '__exit__'}]
        self.assertGreater(len(not_host_only_fn), 2, 'Not enough functions were found')
        self._validate_functions(not_host_only_fn)

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
