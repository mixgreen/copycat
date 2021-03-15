import unittest
import copy
import logging
import typing
import inspect

import dax.sim.coredevice.core

from dax.sim.device import DaxSimDevice
from dax.sim.signal import set_signal_manager, NullSignalManager
from dax.sim.ddb import enable_dax_sim
from dax.util.artiq import get_managers, is_kernel


class CoredeviceCompileTestCase(unittest.TestCase):
    class_list: typing.List[typing.Tuple[type, typing.Dict[str, typing.Any],
                                         typing.Dict[str, typing.Dict[str, typing.Any]]]] = [
        # Class, class arguments, function names with additional kwargs
        (dax.sim.coredevice.core.Core, {'host': None, 'ref_period': 1e-9, 'compile': True}, {
            'wait_until_mu': {'cursor_mu': 1000},
            'get_rtio_destination_status': {'destination': 0},
        }),
    ]

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
        error = None

        for device_class, device_kwargs, fn_args in self.class_list:
            if error:
                break

            with self.subTest(device_class=device_class):
                # Create device
                key = 'core' if issubclass(device_class, dax.sim.coredevice.core.Core) else device_class.__name__
                device = device_class(self.managers.device_mgr, _key=key, **device_kwargs)
                self.assertIsInstance(device, DaxSimDevice)

                # Get function lists
                fn_list = [(n, f) for n, f in inspect.getmembers(device, inspect.ismethod) if is_kernel(f)]
                # Verify list is not empty
                self.assertGreater(len(fn_list), 0, 'No kernel functions were found')

                for n, f in fn_list:
                    if error:
                        break

                    with self.subTest(function=n):
                        try:
                            kwargs = fn_args.get(n, {})
                            f(**kwargs)  # This will cause compilation of the kernel function
                        except FileNotFoundError as e:
                            # Break out of all loops and skip test
                            error = e

        if error:
            # NOTE: compilation only works from the Nix shell/Conda env
            self.skipTest(f'Skipping compiler test: {error}')
