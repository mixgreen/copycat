import typing
import unittest

from dax.experiment import *
from dax.sim import enable_dax_sim
from dax.util.output import temp_dir
from dax.util.artiq_helpers import get_manager_or_parent

import dax.modules.cpld_init
import dax.modules.led
import dax.modules.rpc_benchmark
import dax.modules.rtio_benchmark


class _TestSystem(DaxSystem):
    SYS_ID = 'unittest_system'
    SYS_VER = 0


class BuildModuleTestCase(unittest.TestCase):
    """Test case that builds and initializes modules as a basic test."""

    _MODULES = {
        dax.modules.cpld_init.CpldInitModule: {},
        dax.modules.led.LedModule: {},
        dax.modules.rpc_benchmark.RpcBenchmarkModule: {},
        dax.modules.rtio_benchmark.RtioBenchmarkModule: dict(ttl_out='ttl0'),
        dax.modules.rtio_benchmark.RtioLoopBenchmarkModule: dict(ttl_out='ttl0', ttl_in='ttl1'),
    }
    """List of module types and kwargs."""

    def test_build_module(self):
        with temp_dir():
            for module_type, module_kwargs in self._MODULES.items():
                with self.subTest(module_type=module_type.__name__):
                    class _WrappedTestSystem(_TestSystem):
                        def build(self, *args: typing.Any, **kwargs: typing.Any) -> None:
                            super(_WrappedTestSystem, self).build()
                            self.module = module_type(self, module_type.__name__, *args, **module_kwargs)

                    # Create system
                    manager = get_manager_or_parent(
                        enable_dax_sim(enable=True, ddb=_device_db, logging_level=30, moninj_service=False))
                    system = _WrappedTestSystem(manager, **module_kwargs)
                    self.assertIsInstance(system, DaxSystem)
                    # Initialize system
                    self.assertIsNone(system.dax_init())


_device_db = {
    # Core device
    'core': {
        'type': 'local',
        'module': 'artiq.coredevice.core',
        'class': 'Core',
        'arguments': {'host': '0.0.0.0', 'ref_period': 1e-9}
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
}

if __name__ == '__main__':
    unittest.main()