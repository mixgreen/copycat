import unittest
import typing
import numpy as np

from dax.experiment import *
from dax.modules.rtio_benchmark import RtioBenchmarkModule, RtioLoopBenchmarkModule
import dax.sim.test_case


class _TestSystem(DaxSystem):
    SYS_ID = 'unittest_system'
    SYS_VER = 0

    def build(self, **kwargs) -> None:  # type: ignore
        super(_TestSystem, self).build()
        self.rtio = RtioBenchmarkModule(self, 'rtio', ttl_out='ttl_out', **kwargs)


class _LoopTestSystem(DaxSystem):
    SYS_ID = 'unittest_system'
    SYS_VER = 0

    def build(self, **kwargs) -> None:  # type: ignore
        super(_LoopTestSystem, self).build()
        self.rtio = RtioLoopBenchmarkModule(self, 'rtio', ttl_out='ttl_out', ttl_in='ttl_in', **kwargs)


class RtioBenchmarkModuleTestCase(dax.sim.test_case.PeekTestCase):
    _DEVICE_DB: typing.Dict[str, typing.Any] = {
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
        'ttl_out': {
            'type': 'local',
            'module': 'artiq.coredevice.ttl',
            'class': 'TTLInOut'
        },
        'ttl_in': {
            'type': 'local',
            'module': 'artiq.coredevice.ttl',
            'class': 'TTLInOut'
        },
    }

    def _construct_env(self, **kwargs):
        return self.construct_env(_TestSystem, device_db=self._DEVICE_DB, build_kwargs=kwargs)

    def test_dax_init(self):
        s = self._construct_env(init_kernel=True)
        self.expect(s.rtio.ttl_out, 'state', 'x')
        self.expect(s.rtio.ttl_out, 'direction', 'x')
        s.dax_init()
        self.expect(s.rtio.ttl_out, 'state', 'x')
        self.expect(s.rtio.ttl_out, 'direction', 1)

    def test_dax_init_disabled(self):
        s = self._construct_env(init_kernel=False)
        self.expect(s.rtio.ttl_out, 'state', 'x')
        self.expect(s.rtio.ttl_out, 'direction', 'x')
        s.dax_init()
        self.expect(s.rtio.ttl_out, 'state', 'x')
        self.expect(s.rtio.ttl_out, 'direction', 'x')
        s.rtio.init_kernel()
        self.expect(s.rtio.ttl_out, 'state', 'x')
        self.expect(s.rtio.ttl_out, 'direction', 1)

    def test_manual_init(self):
        s = self._construct_env(init_kernel=False)
        self.expect(s.rtio.ttl_out, 'state', 'x')
        self.expect(s.rtio.ttl_out, 'direction', 'x')
        s.rtio.init()
        self.expect(s.rtio.ttl_out, 'state', 'x')
        self.expect(s.rtio.ttl_out, 'direction', 'x')
        s.rtio.init_kernel()
        self.expect(s.rtio.ttl_out, 'state', 'x')
        self.expect(s.rtio.ttl_out, 'direction', 1)

    def test_manual_init_2(self):
        s = self._construct_env(init_kernel=True)
        self.expect(s.rtio.ttl_out, 'state', 'x')
        self.expect(s.rtio.ttl_out, 'direction', 'x')
        s.rtio.init(force=True)
        self.expect(s.rtio.ttl_out, 'state', 'x')
        self.expect(s.rtio.ttl_out, 'direction', 1)

    def test_manual_init_force(self):
        s = self._construct_env(init_kernel=False)
        self.expect(s.rtio.ttl_out, 'state', 'x')
        self.expect(s.rtio.ttl_out, 'direction', 'x')
        s.rtio.init(force=True)
        self.expect(s.rtio.ttl_out, 'state', 'x')
        self.expect(s.rtio.ttl_out, 'direction', 1)

    def test_event_throughput(self):
        s = self._construct_env()
        s.dax_init()

        with self.assertRaises(RuntimeWarning, msg='Did not raise expected RuntimeWarning in simulation'):
            s.rtio.benchmark_event_throughput(np.arange(200 * ns, 600 * ns, 10 * ns), 5, 1000, 5)

    def test_dma_throughput(self):
        s = self._construct_env()
        s.dax_init()

        with self.assertRaises(RuntimeWarning, msg='Did not raise expected RuntimeWarning in simulation'):
            s.rtio.benchmark_dma_throughput(np.arange(200 * us, 600 * us, 10 * us), 5, 1000, 5)

    def test_latency_core_rtio(self):
        s = self._construct_env()
        s.dax_init()

        with self.assertRaises(RuntimeWarning, msg='Did not raise expected RuntimeWarning in simulation'):
            s.rtio.benchmark_latency_core_rtio(1 * ms, 100 * ms, 1 * ms, 5, 5)


class RtioLoopBenchmarkModuleTestCase(RtioBenchmarkModuleTestCase):

    def _construct_env(self, **kwargs):
        return self.construct_env(_LoopTestSystem, device_db=self._DEVICE_DB, build_kwargs=kwargs)

    def test_dax_init(self):
        s = self._construct_env(init_kernel=True)
        self.expect(s.rtio.ttl_out, 'state', 'x')
        self.expect(s.rtio.ttl_out, 'direction', 'x')
        self.expect(s.rtio.ttl_in, 'state', 'x')
        self.expect(s.rtio.ttl_in, 'sensitivity', 'x')
        self.expect(s.rtio.ttl_in, 'direction', 'x')
        s.dax_init()
        self.expect(s.rtio.ttl_out, 'state', 'x')
        self.expect(s.rtio.ttl_out, 'direction', 1)
        self.expect(s.rtio.ttl_in, 'state', 'z')
        self.expect(s.rtio.ttl_in, 'sensitivity', 0)
        self.expect(s.rtio.ttl_in, 'direction', 0)

    def test_dax_init_disabled(self):
        s = self._construct_env(init_kernel=False)
        self.expect(s.rtio.ttl_out, 'state', 'x')
        self.expect(s.rtio.ttl_out, 'direction', 'x')
        self.expect(s.rtio.ttl_in, 'state', 'x')
        self.expect(s.rtio.ttl_in, 'sensitivity', 'x')
        self.expect(s.rtio.ttl_in, 'direction', 'x')
        s.dax_init()
        self.expect(s.rtio.ttl_out, 'state', 'x')
        self.expect(s.rtio.ttl_out, 'direction', 'x')
        self.expect(s.rtio.ttl_in, 'state', 'x')
        self.expect(s.rtio.ttl_in, 'sensitivity', 'x')
        self.expect(s.rtio.ttl_in, 'direction', 'x')
        s.rtio.init_kernel()
        self.expect(s.rtio.ttl_out, 'state', 'x')
        self.expect(s.rtio.ttl_out, 'direction', 1)
        self.expect(s.rtio.ttl_in, 'state', 'z')
        self.expect(s.rtio.ttl_in, 'sensitivity', 0)
        self.expect(s.rtio.ttl_in, 'direction', 0)

    def test_manual_init(self):
        s = self._construct_env(init_kernel=False)
        self.expect(s.rtio.ttl_out, 'state', 'x')
        self.expect(s.rtio.ttl_out, 'direction', 'x')
        self.expect(s.rtio.ttl_in, 'state', 'x')
        self.expect(s.rtio.ttl_in, 'sensitivity', 'x')
        self.expect(s.rtio.ttl_in, 'direction', 'x')
        s.rtio.init()
        self.expect(s.rtio.ttl_out, 'state', 'x')
        self.expect(s.rtio.ttl_out, 'direction', 'x')
        self.expect(s.rtio.ttl_in, 'state', 'x')
        self.expect(s.rtio.ttl_in, 'sensitivity', 'x')
        self.expect(s.rtio.ttl_in, 'direction', 'x')
        s.rtio.init_kernel()
        self.expect(s.rtio.ttl_out, 'state', 'x')
        self.expect(s.rtio.ttl_out, 'direction', 1)
        self.expect(s.rtio.ttl_in, 'state', 'z')
        self.expect(s.rtio.ttl_in, 'sensitivity', 0)
        self.expect(s.rtio.ttl_in, 'direction', 0)

    def test_manual_init_2(self):
        s = self._construct_env(init_kernel=True)
        self.expect(s.rtio.ttl_out, 'state', 'x')
        self.expect(s.rtio.ttl_out, 'direction', 'x')
        self.expect(s.rtio.ttl_in, 'state', 'x')
        self.expect(s.rtio.ttl_in, 'sensitivity', 'x')
        self.expect(s.rtio.ttl_in, 'direction', 'x')
        s.rtio.init(force=True)
        self.expect(s.rtio.ttl_out, 'state', 'x')
        self.expect(s.rtio.ttl_out, 'direction', 1)
        self.expect(s.rtio.ttl_in, 'state', 'z')
        self.expect(s.rtio.ttl_in, 'sensitivity', 0)
        self.expect(s.rtio.ttl_in, 'direction', 0)

    def test_manual_init_force(self):
        s = self._construct_env(init_kernel=False)
        self.expect(s.rtio.ttl_out, 'state', 'x')
        self.expect(s.rtio.ttl_out, 'direction', 'x')
        self.expect(s.rtio.ttl_in, 'state', 'x')
        self.expect(s.rtio.ttl_in, 'sensitivity', 'x')
        self.expect(s.rtio.ttl_in, 'direction', 'x')
        s.rtio.init(force=True)
        self.expect(s.rtio.ttl_out, 'state', 'x')
        self.expect(s.rtio.ttl_out, 'direction', 1)
        self.expect(s.rtio.ttl_in, 'state', 'z')
        self.expect(s.rtio.ttl_in, 'sensitivity', 0)
        self.expect(s.rtio.ttl_in, 'direction', 0)

    def test_loop_connection(self):
        s = self._construct_env()
        self.assertFalse(s.rtio.test_loop_connection())

    def test_input_buffer_size(self):
        s = self._construct_env()
        s.dax_init()

        with self.assertRaises(RuntimeError,
                               msg='Did not raise expected RuntimeError in simulation (loop not connected)'):
            s.rtio.benchmark_input_buffer_size(1, 2048)

    def test_latency_rtio_core(self):
        s = self._construct_env()
        s.dax_init()

        with self.assertRaises(RuntimeError,
                               msg='Did not raise expected RuntimeError in simulation (loop not connected)'):
            s.rtio.benchmark_latency_rtio_core(100)

    def test_latency_rtt(self):
        s = self._construct_env()
        s.dax_init()

        with self.assertRaises(RuntimeError,
                               msg='Did not raise expected RuntimeError in simulation (loop not connected)'):
            s.rtio.benchmark_latency_rtt(1 * ms, 100 * ms, 1 * ms, 5, 5)


if __name__ == '__main__':
    unittest.main()
