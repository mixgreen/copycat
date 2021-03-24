import typing
import unittest

from dax.experiment import *
import dax.base.system
from dax.sim import enable_dax_sim
from dax.util.artiq import get_managers

import dax.modules.beam_manager
import dax.modules.cpld_init
import dax.modules.led
import dax.modules.rpc_benchmark
import dax.modules.rtio_benchmark
import dax.modules.time_resolved_context


class _TestSystem(DaxSystem):
    SYS_ID = 'unittest_system'
    SYS_VER = 0


class BuildModuleTestCase(unittest.TestCase):
    """Test case that builds and initializes modules as a basic test."""

    _MODULES = {
        dax.modules.beam_manager.BeamManager: ((), dict(num_beams=2)),
        dax.modules.cpld_init.CpldInitModule: ((), {}),
        # HistogramContext not tested here due to system interface requirements
        dax.modules.led.LedModule: (('led0',), {}),
        dax.modules.rpc_benchmark.RpcBenchmarkModule: ((), {}),
        dax.modules.rtio_benchmark.RtioBenchmarkModule: ((), dict(ttl_out='ttl0')),
        dax.modules.rtio_benchmark.RtioLoopBenchmarkModule: ((), dict(ttl_out='ttl0', ttl_in='ttl1')),
        # SafetyContext not tested here due to build argument requirements
        dax.modules.time_resolved_context.TimeResolvedContext: ((), {}),
    }
    """List of module types and kwargs."""

    def setUp(self) -> None:
        self.managers = get_managers(enable_dax_sim(ddb=_DEVICE_DB, enable=True, logging_level=30,
                                                    output='null', moninj_service=False))

    def tearDown(self) -> None:
        # Close managers
        self.managers.close()

    def test_build_module(self):
        for module_type, (module_args, module_kwargs) in self._MODULES.items():
            with self.subTest(module_type=module_type.__name__):
                class _WrappedTestSystem(_TestSystem):
                    def build(self, *args: typing.Any, **kwargs: typing.Any) -> None:
                        super(_WrappedTestSystem, self).build()
                        self.module = module_type(self, module_type.__name__, *args, **kwargs)

                # Create system
                system = _WrappedTestSystem(self.managers, *module_args, **module_kwargs)
                self.assertIsInstance(system, DaxSystem)

                # Initialize system
                self.assertIsNone(system.dax_init())

                # Test module kernel invariants
                for m in system.registry.get_module_list():
                    self._test_kernel_invariants(m)

    def _test_kernel_invariants(self, component: dax.base.system.DaxHasSystem):
        # Test kernel invariants of this component
        for k in component.kernel_invariants:
            self.assertTrue(hasattr(component, k), f'Name "{k}" of "{component.get_system_key()}" was marked '
                                                   f'kernel invariant, but this attribute does not exist')


_DEVICE_DB = {
    # Core device
    'core': {
        'type': 'local',
        'module': 'artiq.coredevice.core',
        'class': 'Core',
        'arguments': {'host': None, 'ref_period': 1e-9}
    },
    'core_cache': {
        'type': 'local',
        'module': 'artiq.coredevice.cache',
        'class': 'CoreCache'
    },
    'core_dma': {
        'type': 'local',
        'module': 'artiq.coredevice.dma',
        'class': 'CoreDMA'
    },

    # Generic TTL
    'ttl0': {
        'type': 'local',
        'module': 'artiq.coredevice.ttl',
        'class': 'TTLInOut',
        'arguments': {'channel': 0},
    },
    'ttl1': {
        'type': 'local',
        'module': 'artiq.coredevice.ttl',
        'class': 'TTLInOut',
        'arguments': {'channel': 1},
    },

    # LED
    'led0': {
        'type': 'local',
        'module': 'artiq.coredevice.ttl',
        'class': 'TTLOut',
        'arguments': {'channel': 2},
    },

}

if __name__ == '__main__':
    unittest.main()
