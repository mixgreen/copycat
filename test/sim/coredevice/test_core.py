import unittest
import unittest.mock
import logging
import copy
import numpy as np

from artiq.language.core import kernel, rpc, portable, host_only
from artiq.language.types import TInt32, TFloat
from artiq.coredevice.core import CompileError

import dax.sim.coredevice.core
from dax.sim.signal import set_signal_manager, NullSignalManager
from dax.sim.ddb import enable_dax_sim
from dax.util.artiq import get_managers

import test.sim.coredevice.compile_testcase as compile_testcase


class BaseCoreTestCase(unittest.TestCase):
    _DEVICE_DB = {
        'core': {
            'type': 'local',
            'module': 'artiq.coredevice.core',
            'class': 'Core',
            'arguments': {'host': None, 'ref_period': 1e-9},
        },
    }

    def setUp(self):
        self.core_arguments = self._DEVICE_DB['core']['arguments']
        self.core_arguments.update(self._DEVICE_DB['core'].get('sim_args', {}))
        set_signal_manager(NullSignalManager())
        self.managers = get_managers(enable_dax_sim(copy.deepcopy(self._DEVICE_DB), enable=True,
                                                    logging_level=logging.WARNING, moninj_service=False, output='null'))

    def tearDown(self) -> None:
        # Close managers
        self.managers.close()

    def test_constructor_signature(self):
        # Should be able to construct base core without arguments
        self.assertIsNotNone(dax.sim.coredevice.core.BaseCore())


class CoreTestCase(BaseCoreTestCase):

    def test_constructor_signature(self):
        # Make sure the signature is as expected
        with self.assertRaises(TypeError, msg='Core class constructor did not match expected signature'):
            # Not adding _key
            dax.sim.coredevice.core.Core(dmgr=self.managers.device_mgr, **self.core_arguments)

        # Test with correct arguments
        dax.sim.coredevice.core.Core(dmgr=self.managers.device_mgr, _key='core', **self.core_arguments)

    def test_compile(self):
        core = dax.sim.coredevice.core.Core(dmgr=self.managers.device_mgr, _key='core', **self.core_arguments)
        compile_flag = self.core_arguments.get('compile', False)
        bad_kernels = [self._bad_kernel_fn_0, self._bad_kernel_fn_1, self._bad_kernel_fn_2, self._bad_kernel_fn_3]

        with unittest.mock.patch.object(self, 'core', core, create=True):
            if compile_flag:
                with unittest.mock.patch.object(core._compiler, 'compile', autospec=True) as mock_method:
                    # Call kernel function
                    self._kernel_fn()
                    # Verify if compile function was called exactly once
                    self.assertEqual(mock_method.call_count, compile_flag)

                try:
                    # Call the kernel function without patching the compile function, causing an actual compilation
                    self._kernel_fn()
                except FileNotFoundError as e:
                    # NOTE: compilation only works from the Nix shell/Conda env, otherwise the linker can not be found
                    self.skipTest(f'Skipping compiler test: {e}')

                for fn in bad_kernels:
                    with self.assertRaises(CompileError, msg='No compile error raised for bad kernel function'):
                        # Call kernel function
                        fn()
            else:
                self.assertIsNone(core._compiler)
                for fn in [self._kernel_fn] + bad_kernels:
                    # Call all functions, should simulate fine because we do not compile
                    fn()

    @kernel
    def _kernel_fn(self):
        self._other_kernel_fn()
        self._portable_fn()
        self._rpc_fn()
        self._async_rpc_fn()

    @kernel
    def _bad_kernel_fn_0(self):
        self._host_only_fn()

    @kernel
    def _bad_kernel_fn_1(self):
        self.__non_existing_attribute = 0

    @kernel
    def _bad_kernel_fn_2(self):
        # noinspection PyTypeChecker
        self._portable_fn(0.1)

    @kernel
    def _bad_kernel_fn_3(self) -> TFloat:
        return np.int32(3)

    @kernel
    def _other_kernel_fn(self):
        pass

    @portable
    def _portable_fn(self, e: TInt32 = 1) -> TInt32:
        return e + 1

    def _rpc_fn(self):
        pass

    @rpc(flags={'async'})
    def _async_rpc_fn(self):
        pass

    @host_only
    def _host_only_fn(self):
        pass


class CoreCompileTestCase(CoreTestCase):
    _DEVICE_DB = {
        'core': {
            'type': 'local',
            'module': 'artiq.coredevice.core',
            'class': 'Core',
            'arguments': {'host': None, 'ref_period': 1e-9},
            'sim_args': {'compile': True},
        },
    }


class CoreCoredeviceCompileTestCase(compile_testcase.CoredeviceCompileTestCase):
    DEVICE_CLASS = dax.sim.coredevice.core.Core
    DEVICE_KWARGS = {'host': None, 'ref_period': 1e-9, 'compile': True}
    FN_KWARGS = {
        'wait_until_mu': {'cursor_mu': 1000},
        'get_rtio_destination_status': {'destination': 0},
        'mu_to_seconds': {'mu': 100},
        'seconds_to_mu': {'seconds': 1.0},
    }


if __name__ == '__main__':
    unittest.main()
