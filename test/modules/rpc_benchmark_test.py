import typing

from dax.experiment import *
from dax.modules.rpc_benchmark import RpcBenchmarkModule
import dax.sim.test_case


class _TestSystem(DaxSystem):
    SYS_ID = 'unittest_system'
    SYS_VER = 0

    def build(self, **kwargs) -> None:  # type: ignore[override]
        super(_TestSystem, self).build()
        self.rpc = RpcBenchmarkModule(self, 'rpc_bench')


class RpcModuleTestCase(dax.sim.test_case.PeekTestCase):
    _DEVICE_DB: typing.Dict[str, typing.Any] = {
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
    }

    def setUp(self) -> None:
        self.s = self.construct_env(_TestSystem, device_db=self._DEVICE_DB)

    def test_benchmark_latency(self):
        self.s.rpc.benchmark_latency(1)
        self.s.rpc.benchmark_latency(10)
        with self.assertRaises(ValueError, msg='No exception raised for invalid parameter'):
            self.s.rpc.benchmark_latency(0)

    def test_benchmark_async_throughput(self):
        self.s.rpc.benchmark_async_throughput(10, 10)
        with self.assertRaises(ValueError, msg='No exception raised for invalid parameter'):
            self.s.rpc.benchmark_async_throughput(0, 1)
        with self.assertRaises(ValueError, msg='No exception raised for invalid parameter'):
            self.s.rpc.benchmark_async_throughput(1, 0)
