import unittest
import copy
import logging
import typing
import inspect

import artiq.coredevice.core

from dax.sim.device import DaxSimDevice
from dax.sim.signal import set_signal_manager, NullSignalManager
from dax.sim.ddb import enable_dax_sim
from dax.util.artiq import get_managers, is_kernel, is_portable

__all__ = ['CompileTestCase']


class _SkipCompilerTest(RuntimeError):
    pass


class CompileTestCase(unittest.TestCase):
    """Test case for compilation of simulated coredevice drivers."""

    DEVICE_CLASS: typing.ClassVar[type]
    """The device class to test."""
    DEVICE_KWARGS: typing.ClassVar[typing.Dict[str, typing.Any]] = {}
    """Keyword arguments to instantiate the device class."""
    FN_ARGS: typing.ClassVar[typing.Dict[str, typing.Union[typing.Tuple[typing.Any, ...], typing.List[typing.Any]]]]
    FN_ARGS = {}
    """Function positional arguments (presence forces function testing)."""
    FN_KWARGS: typing.ClassVar[typing.Dict[str, typing.Dict[str, typing.Any]]] = {}
    """Function keyword arguments (presence forces function testing)."""
    FN_EXCLUDE: typing.ClassVar[typing.Set[str]] = set()
    """Excluded functions."""
    SIM_DEVICE: typing.ClassVar[bool] = True
    """:const:`True` if the device tested is a DAX.sim simulation driver."""

    DEVICE_DB: typing.ClassVar[typing.Dict[str, typing.Any]] = {
        'core': {
            'type': 'local',
            'module': 'artiq.coredevice.core',
            'class': 'Core',
            'arguments': {'host': None, 'ref_period': 1e-9},
            'sim_args': {'compile': True}  # Enable compilation
        },
    }

    @classmethod
    def setUpClass(cls) -> None:
        assert isinstance(cls.DEVICE_CLASS, type), 'DEVICE_CLASS must be a type'
        assert isinstance(cls.DEVICE_KWARGS, dict), 'DEVICE_KWARGS must be a dict'
        assert isinstance(cls.FN_ARGS, dict), 'FN_ARGS must be a dict'
        assert isinstance(cls.FN_KWARGS, dict), 'FN_KWARGS must be a dict'
        assert isinstance(cls.FN_EXCLUDE, set), 'FN_EXCLUDE must be a set'
        assert isinstance(cls.SIM_DEVICE, bool), 'SIM_DEVICE must be a bool'

    def setUp(self) -> None:
        ddb = copy.deepcopy(self.DEVICE_DB)
        ddb.update(CompileTestCase.DEVICE_DB)  # Always override the core device

        set_signal_manager(NullSignalManager())
        self.managers = get_managers(enable_dax_sim(ddb, enable=True, logging_level=logging.WARNING,
                                                    moninj_service=False, output='null'))

    def tearDown(self) -> None:
        self.managers.close()

    def test_compile_functions(self) -> None:
        try:
            # Add extra kwargs for testing
            test_kwargs = dict(_key=self.DEVICE_CLASS.__name__.lower()) if self.SIM_DEVICE else {}
            # Create device
            device = self.DEVICE_CLASS(self.managers.device_mgr, **self.DEVICE_KWARGS, **test_kwargs)
            if self.SIM_DEVICE:
                self.assertIsInstance(device, DaxSimDevice)

            # Get function lists
            fn_list = [(n, f) for n, f in inspect.getmembers(device, inspect.ismethod)
                       if (is_kernel(f) or is_portable(f)) and n not in self.FN_EXCLUDE]
            # Verify list is not empty
            self.assertGreater(len(fn_list), 0, 'No kernel functions were found')

            for n, f in fn_list:
                args = self.FN_ARGS.get(n, ())
                kwargs = self.FN_KWARGS.get(n, {})

                with self.subTest(function=n, args=args, kwargs=kwargs):
                    try:
                        # Compile the function, do not run it
                        device.core.compile(f, args, kwargs)
                    except artiq.coredevice.core.CompileError as e:
                        err_msg = "name 'NotImplementedError' is not bound to anything"
                        if err_msg in str(e) and n not in self.FN_ARGS and n not in self.FN_KWARGS:
                            pass  # Ignore compile errors due to NotImplementedError
                        else:
                            raise
                    except FileNotFoundError as e:
                        # Break out of all loops and skip test
                        raise _SkipCompilerTest(e)

        except _SkipCompilerTest as error:
            # NOTE: compilation only works from the Nix shell/Conda env
            self.skipTest(f'Skipping compiler test: {error}')
