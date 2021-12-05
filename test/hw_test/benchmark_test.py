import typing
import logging

from dax.experiment import *
from dax.modules.rtio_benchmark import RtioLoopBenchmarkModule
from dax.modules.rpc_benchmark import RpcBenchmarkModule

import test.hw_test


class _CiTestSystem(DaxSystem):
    SYS_ID = 'ci_test_system'
    SYS_VER = 0
    CORE_LOG_KEY = None
    DAX_INFLUX_DB_KEY = None

    def build(self, *args: typing.Any, **kwargs: typing.Any) -> None:
        super(_CiTestSystem, self).build(*args, **kwargs)
        self.rtio = RtioLoopBenchmarkModule(self, 'rtio', ttl_out='loop_out', ttl_in='loop_in')
        self.rpc = RpcBenchmarkModule(self, 'rpc')
        self.update_kernel_invariants('rtio', 'rpc')


class _BenchmarkTestCase(test.hw_test.HardwareTestCase):
    def run_benchmark(self, env_class, *args, **kwargs):
        # Construct env
        env = self.construct_env(env_class(_CiTestSystem), *args, **kwargs)
        # Configure logger
        env.logger.setLevel(logging.INFO)

        # Run
        env.prepare()
        try:
            env.run()
        except RuntimeError as e:
            self.skipTest(str(e))
        else:
            env.analyze()


class RtioTestCase(_BenchmarkTestCase):
    def test_event_throughput(self):
        from dax.clients.rtio_benchmark import RtioBenchmarkEventThroughput
        self.run_benchmark(RtioBenchmarkEventThroughput)

    def test_dma_throughput(self):
        from dax.clients.rtio_benchmark import RtioBenchmarkDmaThroughput
        self.run_benchmark(RtioBenchmarkDmaThroughput)

    def test_latency_core_rtio(self):
        from dax.clients.rtio_benchmark import RtioBenchmarkLatencyCoreRtio
        self.run_benchmark(RtioBenchmarkLatencyCoreRtio)

    def test_input_buffer_size(self):
        from dax.clients.rtio_benchmark import RtioBenchmarkInputBufferSize
        self.run_benchmark(RtioBenchmarkInputBufferSize)

    def test_latency_rtio_core(self):
        from dax.clients.rtio_benchmark import RtioBenchmarkLatencyRtioCore
        self.run_benchmark(RtioBenchmarkLatencyRtioCore)

    def test_latency_rtt(self):
        from dax.clients.rtio_benchmark import RtioBenchmarkLatencyRtt
        self.run_benchmark(RtioBenchmarkLatencyRtt)


class RpcTestCase(_BenchmarkTestCase):
    def test_latency(self):
        from dax.clients.rpc_benchmark import RpcBenchmarkLatency
        self.run_benchmark(RpcBenchmarkLatency)

    def test_async_throughput(self):
        from dax.clients.rpc_benchmark import RpcBenchmarkAsyncThroughput
        self.run_benchmark(RpcBenchmarkAsyncThroughput)
