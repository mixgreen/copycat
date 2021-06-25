import unittest
import copy
import logging
import typing
import inspect

import artiq.coredevice.core

import dax.sim.coredevice.core

from dax.sim.device import DaxSimDevice
from dax.sim.signal import set_signal_manager, NullSignalManager
from dax.sim.ddb import enable_dax_sim
from dax.util.artiq import get_managers, is_kernel, is_portable

from test.environment import CI_ENABLED

__all__ = ['CoredeviceCompileTestCase']


class _SkipCompilerTest(RuntimeError):
    pass


@unittest.skipUnless(CI_ENABLED, 'Not in a CI environment, skipping compilation test')
class CoredeviceCompileTestCase(unittest.TestCase):
    DEVICE_CLASS: type
    """The device class to test."""
    DEVICE_KWARGS: typing.Dict[str, typing.Any] = {}
    """Keyword arguments to instantiate the device class."""
    FN_ARGS: typing.Dict[str, typing.Union[typing.Tuple[typing.Any, ...], typing.List[typing.Any]]] = {}
    """Function positional arguments."""
    FN_KWARGS: typing.Dict[str, typing.Dict[str, typing.Any]] = {}
    """Function keyword arguments."""
    FN_EXCLUDE: typing.Set[str] = set()
    """Excluded functions."""
    FN_EXCEPTIONS: typing.Dict[str, type] = {}
    """Expected exceptions when executing specific functions (defaults to ``NotImplementedError``)."""

    DEVICE_DB = {
        'core': {
            'type': 'local',
            'module': 'artiq.coredevice.core',
            'class': 'Core',
            'arguments': {'host': None, 'ref_period': 1e-9},
            'sim_args': {'compile': True}  # Enable compilation
        },
    }

    def setUp(self) -> None:
        set_signal_manager(NullSignalManager())
        self.managers = get_managers(enable_dax_sim(copy.deepcopy(self.DEVICE_DB), enable=True,
                                                    logging_level=logging.WARNING, moninj_service=False, output='null'))

    def tearDown(self) -> None:
        self.managers.close()

    def test_compile_functions(self):
        assert isinstance(self.DEVICE_CLASS, type), 'DEVICE_CLASS must be a type'
        assert isinstance(self.DEVICE_KWARGS, dict), 'DEVICE_KWARGS must be a dict'
        assert isinstance(self.FN_ARGS, dict), 'FN_ARGS must be a dict'
        assert isinstance(self.FN_KWARGS, dict), 'FN_KWARGS must be a dict'
        assert isinstance(self.FN_EXCLUDE, set), 'FN_EXCLUDE must be a set'
        assert isinstance(self.FN_EXCEPTIONS, dict), 'FN_EXCEPTIONS must be a dict'

        try:
            # Create device
            key = 'core' if issubclass(self.DEVICE_CLASS, dax.sim.coredevice.core.Core) else self.DEVICE_CLASS.__name__
            device = self.DEVICE_CLASS(self.managers.device_mgr, _key=key, **self.DEVICE_KWARGS)
            self.assertIsInstance(device, DaxSimDevice)

            # Get function lists
            fn_list = [(n, f) for n, f in inspect.getmembers(device, inspect.ismethod)
                       if (is_kernel(f) or is_portable(f)) and n not in self.FN_EXCLUDE]
            # Verify list is not empty
            self.assertGreater(len(fn_list), 0, 'No kernel functions were found')

            for n, f in fn_list:
                args = self.FN_ARGS.get(n, ())
                kwargs = self.FN_KWARGS.get(n, {})
                expected_exception = self.FN_EXCEPTIONS.get(n, NotImplementedError)

                with self.subTest(function=n, args=args, kwargs=kwargs, expected_exception=expected_exception):
                    try:
                        f(*args, **kwargs)  # This will cause compilation of the kernel function
                    except expected_exception:
                        # Ignore expected exception
                        pass
                    except artiq.coredevice.core.CompileError as e:
                        if "name 'NotImplementedError' is not bound to anything" in str(e):
                            # Ignore compile errors due to NotImplementedError
                            pass
                        else:
                            raise
                    except FileNotFoundError as e:
                        # Break out of all loops and skip test
                        raise _SkipCompilerTest(e)

        except _SkipCompilerTest as error:
            # NOTE: compilation only works from the Nix shell/Conda env
            self.skipTest(f'Skipping compiler test: {error}')
