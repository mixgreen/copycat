import numpy as np
import timeit

from dax.experiment import *


class RpcBenchmarkModule(DaxModule):
    """Module to benchmark RPC performance."""

    # System keys
    LATENCY_HOST_CORE_HOST_KEY = 'latency_host_core_host'
    LATENCY_CORE_HOST_CORE_KEY = 'latency_core_host_core'
    LATENCY_CORE_HOST_CORE_ASYNC_KEY = 'latency_core_host_core_async'

    def init(self):
        # Load parameters
        self.setattr_dataset_sys(self.LATENCY_HOST_CORE_HOST_KEY)
        self.setattr_dataset_sys(self.LATENCY_CORE_HOST_CORE_KEY)
        self.setattr_dataset_sys(self.LATENCY_CORE_HOST_CORE_ASYNC_KEY)

    def post_init(self):
        pass

    """Benchmark RPC latency"""

    def benchmark_latency(self, num_samples):
        # Convert types of arguments
        num_samples = np.int32(num_samples)

        # Check arguments
        if not num_samples > 0:
            msg = 'Number of samples must be larger than 0'
            self.logger.error(msg)
            raise ValueError(msg)

        # Store input values in dataset
        self.set_dataset('num_samples', num_samples)

        # Obtain total execution time of host-core-host RPC calls
        time = timeit.timeit(self._empty_kernel, number=num_samples)
        # Store average time per call (in ARTIQ notation)
        self.set_dataset_sys(self.LATENCY_HOST_CORE_HOST_KEY, time / num_samples)

        # Obtain total time of core-host-core RPC calls
        time_mu = self._benchmark_core_host_core(num_samples)
        # Convert total time and store average time per call
        time = self.core.mu_to_seconds(time_mu)
        self.set_dataset_sys(self.LATENCY_CORE_HOST_CORE_KEY, time / num_samples)

        # Obtain total time of core-host-core async RPC calls
        time_mu = self._benchmark_core_host_core_async(num_samples)
        # Convert total time and store average time per call
        time = self.core.mu_to_seconds(time_mu)
        self.set_dataset_sys(self.LATENCY_CORE_HOST_CORE_ASYNC_KEY, time / num_samples)

    @kernel
    def _empty_kernel(self):
        # Just break realtime to have minimal computation
        self.core.break_realtime()

    @kernel
    def _benchmark_core_host_core(self, num_samples):
        # Reset core
        self.core.reset()

        # Accumulated execution time
        t_total = np.int64(0)

        for _ in range(num_samples):
            # Store starting time
            t_start = self.core.get_rtio_counter_mu()
            # Call host function
            self._empty_host_function()
            # Register end time
            t_end = self.core.get_rtio_counter_mu()

            # Accumulate execution time
            t_total += t_end - t_start

        # Return accumulated total execution time
        return t_total

    def _empty_host_function(self):
        pass

    @kernel
    def _benchmark_core_host_core_async(self, num_samples):
        # Reset core
        self.core.reset()

        # Accumulated execution time
        t_total = np.int64(0)

        for i in range(num_samples):
            # Store starting time
            t_start = self.core.get_rtio_counter_mu()
            # Call async host function
            self._empty_host_function_async()
            # Register end time
            t_end = self.core.get_rtio_counter_mu()

            # Accumulate execution time
            t_total += t_end - t_start

            if (i & 0xF) == 0xF:
                # Sync call to other function just to make sure we do not accumulate too much async calls
                self._empty_host_function()

        # Return accumulated total execution time
        return t_total

    @rpc(flags={"async"})
    def _empty_host_function_async(self):
        pass
