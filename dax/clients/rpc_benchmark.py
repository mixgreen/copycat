import numpy as np

from dax.modules.rpc_benchmark import *
import dax.util.units


@dax_client_factory
class RpcBenchmarkLatency(DaxClient, EnvExperiment):
    """RPC latency benchmark."""

    def build(self):
        # Arguments
        number_kwargs = {'scale': 1, 'step': 1, 'ndecimals': 0}
        self.setattr_argument('num_samples', NumberValue(100, min=1, **number_kwargs))

        # Obtain RTIO benchmark module
        self.rpc_bench = self.registry.search_module(RpcBenchmarkModule)

    def run(self):
        self.success = self.rpc_bench.benchmark_latency(self.num_samples)

    def analyze(self):
        if self.success:
            # Report result
            hch = dax.util.units.time_to_str(self.rpc_bench.get_dataset_sys(self.rpc_bench.LATENCY_HOST_CORE_HOST_KEY))
            chc = dax.util.units.time_to_str(self.rpc_bench.get_dataset_sys(self.rpc_bench.LATENCY_CORE_HOST_CORE_KEY))
            chc_async = dax.util.units.time_to_str(
                self.rpc_bench.get_dataset_sys(self.rpc_bench.LATENCY_CORE_HOST_CORE_ASYNC_KEY))
            self.logger.info('Host-core-host latency (includes runtime compilation) is {:s}'.format(hch))
            self.logger.info('Core-host-core latency is {:s}'.format(chc))
            self.logger.info('Core-host-core async latency is {:s}'.format(chc_async))
