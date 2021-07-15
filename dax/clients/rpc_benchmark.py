from dax.experiment import *
from dax.modules.rpc_benchmark import RpcBenchmarkModule
import dax.util.units

__all__ = ['RpcBenchmarkLatency', 'RpcBenchmarkAsyncThroughput']


@dax_client_factory
class RpcBenchmarkLatency(DaxClient, Experiment):
    """RPC latency benchmark."""

    DAX_INIT = False
    """Disable DAX init."""

    def build(self) -> None:  # type: ignore[override]
        # Arguments
        self.num_samples = self.get_argument('num_samples', NumberValue(100, min=1, step=1, ndecimals=0))

        # Obtain RPC benchmark module
        self.rpc_bench = self.registry.find_module(RpcBenchmarkModule)

        # Update kernel invariants
        self.update_kernel_invariants('num_samples', 'rpc_bench')

    def run(self) -> None:
        self.rpc_bench.benchmark_latency(num_samples=self.num_samples)

    def analyze(self) -> None:
        # Report result
        hch = dax.util.units.time_to_str(self.rpc_bench.get_dataset_sys(self.rpc_bench.LATENCY_HOST_CORE_HOST_KEY))
        chc = dax.util.units.time_to_str(self.rpc_bench.get_dataset_sys(self.rpc_bench.LATENCY_CORE_HOST_CORE_KEY))
        chc_async = dax.util.units.time_to_str(
            self.rpc_bench.get_dataset_sys(self.rpc_bench.LATENCY_CORE_HOST_CORE_ASYNC_KEY))
        self.logger.info(f'Host-core-host latency (includes runtime compilation) is {hch}')
        self.logger.info(f'Core-host-core latency is {chc}')
        self.logger.info(f'Core-host-core async latency is {chc_async}')


@dax_client_factory
class RpcBenchmarkAsyncThroughput(DaxClient, Experiment):
    """Async RPC throughput benchmark."""

    DAX_INIT = False
    """Disable DAX init."""

    def build(self) -> None:  # type: ignore[override]
        # Arguments
        self.num_events = self.get_argument('num_events', NumberValue(1000, min=1, step=1, ndecimals=0))
        self.num_samples = self.get_argument('num_samples', NumberValue(100, min=1, step=1, ndecimals=0))

        # Obtain RPC benchmark module
        self.rpc_bench = self.registry.find_module(RpcBenchmarkModule)

        # Update kernel invariants
        self.update_kernel_invariants('num_events', 'num_samples', 'rpc_bench')

    def run(self) -> None:
        self.rpc_bench.benchmark_async_throughput(num_events=self.num_events,
                                                  num_samples=self.num_samples)

    def analyze(self) -> None:
        # Report result
        period = self.rpc_bench.get_dataset_sys(self.rpc_bench.ASYNC_REQUEST_PERIOD_KEY)
        frequency = 1.0 / period
        self.logger.info(f'RPC async request throughput is '
                         f'{dax.util.units.freq_to_str(frequency)} '
                         f'({dax.util.units.time_to_str(period)} period)')
