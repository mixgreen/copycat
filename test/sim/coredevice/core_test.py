import unittest
import unittest.mock
import logging
import copy
import numpy as np

from artiq.language.core import kernel, rpc, portable, host_only, now_mu, kernel_from_string
from artiq.language.types import TInt32, TFloat
from artiq.coredevice.core import CompileError

import dax.sim.coredevice.core
from dax.sim.signal import set_signal_manager, NullSignalManager
from dax.sim.ddb import enable_dax_sim
from dax.util.artiq import get_managers

import test.sim.coredevice._compile_testcase as compile_testcase


class _BaseTestCase(unittest.TestCase):
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


class BaseCoreTestCase(_BaseTestCase):

    def test_constructor_signature(self):
        # Should be able to construct base core without arguments
        self.assertIsNotNone(dax.sim.coredevice.core.BaseCore())

    def test_default_reset(self):
        core = dax.sim.coredevice.core.BaseCore()
        t = now_mu()
        core.reset()
        self.assertEqual(now_mu() - t, dax.sim.coredevice.core.BaseCore.DEFAULT_RESET_TIME_MU)

    def test_variable_reset(self):
        for reset_mu in [0, 125000, 200000]:
            core = dax.sim.coredevice.core.BaseCore(reset_mu=reset_mu)
            t = now_mu()
            core.reset()
            self.assertEqual(now_mu() - t, reset_mu)

    def test_default_break_realtime(self):
        core = dax.sim.coredevice.core.BaseCore()
        t = now_mu()
        core.break_realtime()
        self.assertEqual(now_mu() - t, dax.sim.coredevice.core.BaseCore.DEFAULT_RESET_TIME_MU)

    def test_variable_break_realtime(self):
        for break_realtime_mu in [0, 125000, 200000]:
            core = dax.sim.coredevice.core.BaseCore(break_realtime_mu=break_realtime_mu)
            t = now_mu()
            core.break_realtime()
            self.assertEqual(now_mu() - t, break_realtime_mu)


class CoreTestCase(_BaseTestCase):

    def test_constructor_signature(self):
        # Make sure the signature is as expected
        with self.assertRaises(TypeError, msg='Core class constructor did not match expected signature'):
            # Not adding _key
            dax.sim.coredevice.core.Core(dmgr=self.managers.device_mgr, **self.core_arguments)

        # Test with correct arguments
        dax.sim.coredevice.core.Core(dmgr=self.managers.device_mgr, _key='core', **self.core_arguments)

    def test_level(self):
        core = dax.sim.coredevice.core.Core(dmgr=self.managers.device_mgr, _key='core', **self.core_arguments)

        with unittest.mock.patch.object(self, 'core', core, create=True):
            try:
                self.assertEqual(core._level, 0)
                self._kernel_fn()
                self.assertEqual(core._level, 0)
                with self.assertRaises(RuntimeError):
                    self._raise_exception_kernel()
                self.assertEqual(core._level, 0)
            except FileNotFoundError as e:
                # NOTE: compilation only works from the Nix shell/Conda env, otherwise the linker can not be found
                self.skipTest(f'Skipping compiler test: {e}')

    @kernel
    def _raise_exception_kernel(self):
        raise RuntimeError

    def test_compile(self):
        core = dax.sim.coredevice.core.Core(dmgr=self.managers.device_mgr, _key='core', **self.core_arguments)
        compile_flag = self.core_arguments.get('compile', False)
        bad_kernels = [self._bad_kernel_fn_0, self._bad_kernel_fn_1, self._bad_kernel_fn_2, self._bad_kernel_fn_3]

        with unittest.mock.patch.object(self, 'core', core, create=True):
            if compile_flag:
                with unittest.mock.patch.object(core._compiler, 'compile', autospec=True,
                                                return_value=(None, [], None, None)) as mock_method:
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
        self._kernel_str_fn()
        self._portable_fn()
        self._portable_str_fn()
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

    _kernel_str_fn = kernel_from_string(['self'], "pass", kernel)

    _portable_str_fn = kernel_from_string(['self'], "pass", portable)


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
