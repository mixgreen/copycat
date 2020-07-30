from dax.experiment import *
from dax.modules.rpc_benchmark import RpcBenchmarkModule
import dax.util.units

__all__ = ['RpcBenchmarkLatency']


@dax_client_factory
class RpcBenchmarkLatency(DaxClient, EnvExperiment):
    """RPC latency benchmark."""

    DAX_INIT: bool = False
    """Disable DAX init."""

    def build(self) -> None:  # type: ignore
        # Arguments
        self.num_samples = self.get_argument('num_samples', NumberValue(100, min=1, step=1, ndecimals=0))

        # Obtain RTIO benchmark module
        self.rpc_bench = self.registry.find_module(RpcBenchmarkModule)

        # Update kernel invariants
        self.update_kernel_invariants('num_samples', 'rpc_bench')

    def run(self) -> None:
        self.rpc_bench.benchmark_latency(self.num_samples)

    def analyze(self) -> None:
        # Report result
        hch = dax.util.units.time_to_str(self.rpc_bench.get_dataset_sys(self.rpc_bench.LATENCY_HOST_CORE_HOST_KEY))
        chc = dax.util.units.time_to_str(self.rpc_bench.get_dataset_sys(self.rpc_bench.LATENCY_CORE_HOST_CORE_KEY))
        chc_async = dax.util.units.time_to_str(
            self.rpc_bench.get_dataset_sys(self.rpc_bench.LATENCY_CORE_HOST_CORE_ASYNC_KEY))
        self.logger.info(f'Host-core-host latency (includes runtime compilation) is {hch}')
        self.logger.info(f'Core-host-core latency is {chc}')
        self.logger.info(f'Core-host-core async latency is {chc_async}')
