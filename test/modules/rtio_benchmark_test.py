import typing
import numpy as np

from dax.experiment import *
from dax.modules.rtio_benchmark import RtioBenchmarkModule, RtioLoopBenchmarkModule, RtioBenchmarkError
import dax.sim.test_case


class _TestSystem(DaxSystem):
    SYS_ID = 'unittest_system'
    SYS_VER = 0
    CORE_LOG_KEY = None
    DAX_INFLUX_DB_KEY = None

    def build(self, **kwargs) -> None:  # type: ignore[override]
        super(_TestSystem, self).build()
        self.rtio = RtioBenchmarkModule(self, 'rtio', ttl_out='ttl_out', **kwargs)


class _LoopTestSystem(DaxSystem):
    SYS_ID = 'unittest_system'
    SYS_VER = 0
    CORE_LOG_KEY = None
    DAX_INFLUX_DB_KEY = None

    def build(self, **kwargs) -> None:  # type: ignore[override]
        super(_LoopTestSystem, self).build()
        self.rtio = RtioLoopBenchmarkModule(self, 'rtio', ttl_out='ttl_out', ttl_in='ttl_in', **kwargs)


class RtioBenchmarkModuleTtlOutTestCase(dax.sim.test_case.PeekTestCase):
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
        'ttl_out': {
            'type': 'local',
            'module': 'artiq.coredevice.ttl',
            'class': 'TTLOut'  # Output pin only
        },
    }

    def _construct_env(self, **kwargs):
        return self.construct_env(_TestSystem, device_db=self._DEVICE_DB, build_kwargs=kwargs)

    def test_dax_init(self):
        # Verify that the module builds and initializes correctly
        s = self._construct_env(init_kernel=True)
        self.expect(s.rtio.ttl_out, 'state', 'x')
        s.dax_init()
        self.expect(s.rtio.ttl_out, 'state', 'x')

    def test_event_throughput(self):
        s = self._construct_env()
        s.dax_init()

        with self.assertRaises(RtioBenchmarkError, msg='Did not raise expected RtioBenchmarkError in simulation'):
            s.rtio.benchmark_event_throughput(np.arange(200 * ns, 500 * ns, 10 * ns), 5, 100, 5)

    def test_dma_throughput(self):
        s = self._construct_env()
        s.dax_init()

        with self.assertRaises(RtioBenchmarkError, msg='Did not raise expected RtioBenchmarkError in simulation'):
            s.rtio.benchmark_dma_throughput(np.arange(200 * us, 500 * us, 10 * us), 5, 100, 5)

    def test_latency_core_rtio(self):
        s = self._construct_env()
        s.dax_init()

        with self.assertRaises(RtioBenchmarkError, msg='Did not raise expected RtioBenchmarkError in simulation'):
            s.rtio.benchmark_latency_core_rtio(1 * ms, 50 * ms, 1 * ms, 5, 5)

    def test_burst_slow(self):
        s = self._construct_env(dma=False)

        # Inject burst size and period
        s.rtio.set_dataset_sys(s.rtio.EVENT_PERIOD_KEY, 500 * ns)
        s.rtio.set_dataset_sys(s.rtio.EVENT_BURST_KEY, 100)

        # Initialize
        s.dax_init()

        # Slow burst possible
        s.rtio.burst()
        s.rtio.burst_slow()
        self.assertEqual(s.rtio.burst, s.rtio.burst_slow, 'Burst function did not default to slow burst')
        with self.assertRaises(AttributeError, msg='Did not raise attribute error for burst DMA'):
            # Not possible since DMA was disabled
            s.rtio.burst_dma()

    def test_burst_dma(self):
        s = self._construct_env(dma=True)

        # Inject burst size and period
        s.rtio.set_dataset_sys(s.rtio.EVENT_PERIOD_KEY, 500 * ns)
        s.rtio.set_dataset_sys(s.rtio.EVENT_BURST_KEY, 100)

        # Initialize
        s.dax_init()

        # Verify DMA is still enabled
        self.assertTrue(s.rtio._dma_enabled, 'DMA enabled flag was not set correctly')

        # Burst possible
        s.rtio.burst()
        s.rtio.burst_slow()
        s.rtio.burst_dma()
        self.assertEqual(s.rtio.burst, s.rtio.burst_dma, 'Burst function did not default to DMA burst')


class RtioBenchmarkModuleTestCase(RtioBenchmarkModuleTtlOutTestCase):
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
        'ttl_out': {
            'type': 'local',
            'module': 'artiq.coredevice.ttl',
            'class': 'TTLInOut'
        },
        'ttl_in': {
            'type': 'local',
            'module': 'artiq.coredevice.ttl',
            'class': 'TTLInOut',
            'sim_args': {'input_freq': 0.0}  # Zero input frequency, loop is disconnected
        },
        'ttl_in_connected': {
            'type': 'local',
            'module': 'artiq.coredevice.ttl',
            'class': 'TTLInOut',
            'sim_args': {'input_freq': 1e7}
        },
    }

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


class RtioLoopBenchmarkModuleTestCase(RtioBenchmarkModuleTestCase):

    def _construct_env(self, *, loop_connected=False, **kwargs):
        ddb = self._DEVICE_DB.copy()
        if loop_connected:
            ddb['ttl_in'] = 'ttl_in_connected'
        return self.construct_env(_LoopTestSystem, device_db=ddb, build_kwargs=kwargs)

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

    def test_loop_connection_disconnected(self):
        s = self._construct_env()
        self.assertFalse(s.rtio.test_loop_connection())

    def test_loop_connection_connected(self):
        s = self._construct_env(loop_connected=True)
        self.assertTrue(s.rtio.test_loop_connection())

    def test_input_buffer_size_disconnected(self):
        s = self._construct_env()
        s.dax_init()

        with self.assertRaises(RtioBenchmarkError,
                               msg='Did not raise expected RuntimeError in simulation (loop not connected)'):
            s.rtio.benchmark_input_buffer_size(1, 32)

    def test_input_buffer_size_connected(self):
        s = self._construct_env(loop_connected=True)
        s.dax_init()

        with self.assertRaises(RtioBenchmarkError,
                               msg='Did not raise expected RtioBenchmarkError in simulation '
                                   '(could not determine buffer size)'):
            s.rtio.benchmark_input_buffer_size(1, 32)

    def test_latency_rtio_core_disconnected(self):
        s = self._construct_env()
        s.dax_init()

        with self.assertRaises(RtioBenchmarkError,
                               msg='Did not raise expected RuntimeError in simulation (loop not connected)'):
            s.rtio.benchmark_latency_rtio_core(10)

    def test_latency_rtio_core_connected(self):
        s = self._construct_env(loop_connected=True)
        s.dax_init()

        # Function works correctly as it receives input, but data is random
        s.rtio.benchmark_latency_rtio_core(10)

    def test_latency_rtt_disconnected(self):
        s = self._construct_env()
        s.dax_init()

        with self.assertRaises(RtioBenchmarkError,
                               msg='Did not raise expected RuntimeError in simulation (loop not connected)'):
            s.rtio.benchmark_latency_rtt(1 * ms, 200 * ms, 10 * ms, 5, 5)

    def test_latency_rtt_connected(self):
        s = self._construct_env(loop_connected=True)
        s.dax_init()

        with self.assertRaises(RtioBenchmarkError,
                               msg='Did not raise expected RtioBenchmarkError in simulation '
                                   '(no underflow exceptions raised)'):
            s.rtio.benchmark_latency_rtt(1 * ms, 200 * ms, 10 * ms, 5, 5)
