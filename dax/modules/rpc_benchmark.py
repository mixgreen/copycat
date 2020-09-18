import numpy as np
import timeit

from dax.experiment import *

__all__ = ['RpcBenchmarkModule']


class RpcBenchmarkModule(DaxModule):
    """Module to benchmark RPC performance."""

    # System keys
    LATENCY_HOST_CORE_HOST_KEY: str = 'latency_host_core_host'
    LATENCY_CORE_HOST_CORE_KEY: str = 'latency_core_host_core'
    LATENCY_CORE_HOST_CORE_ASYNC_KEY: str = 'latency_core_host_core_async'
    ASYNC_REQUEST_PERIOD_KEY: str = 'async_request_period'

    def build(self) -> None:  # type: ignore
        """Build the RPC benchmark module."""
        pass

    def init(self) -> None:
        pass

    def post_init(self) -> None:
        pass

    """Benchmark RPC latency"""

    @host_only
    def benchmark_latency(self, num_samples: int) -> None:
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
    def _empty_kernel(self):  # type: () -> None
        # Just break realtime to have minimal computation
        self.core.break_realtime()

    @kernel
    def _benchmark_core_host_core(self, num_samples: TInt32) -> TInt64:
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

    @kernel
    def _benchmark_core_host_core_async(self, num_samples: TInt32) -> TInt64:
        # Reset core
        self.core.reset()

        # Accumulated execution time
        t_total = np.int64(0)

        for i in range(num_samples):
            # Reset the core (not required, but just to be sure)
            self.core.reset()
            # Call a sync RPC function to clear any RPC buffers
            self._empty_host_function()

            # Store starting time
            t_start = self.core.get_rtio_counter_mu()
            # Call async host function
            self._empty_host_function_async()
            # Register end time
            t_stop = self.core.get_rtio_counter_mu()

            # Accumulate execution time
            t_total += t_stop - t_start

        # Return accumulated total execution time
        return t_total

    def _empty_host_function(self):
        pass

    @rpc(flags={"async"})
    def _empty_host_function_async(self):  # type: () -> None
        pass

    """Benchmark async RPC throughput"""

    @host_only
    def benchmark_async_throughput(self, num_events: int, num_samples: int) -> None:
        assert isinstance(num_events, (int, np.integer)), 'Number of events must be of type int'
        assert isinstance(num_samples, (int, np.integer)), 'Number of samples must be of type int'

        # Check arguments
        if not num_events > 0:
            msg = 'Number of events must be larger than 0'
            self.logger.error(msg)
            raise ValueError(msg)
        if not num_samples > 0:
            msg = 'Number of samples must be larger than 0'
            self.logger.error(msg)
            raise ValueError(msg)

        # Store input values in dataset
        self.set_dataset('num_events', num_events)
        self.set_dataset('num_samples', num_samples)

        # Obtain total time of async throughput benchmark
        time_mu = self._benchmark_async_throughput(num_events, num_samples)
        # Convert total time and store average time per call
        time = self.core.mu_to_seconds(time_mu)
        self.set_dataset_sys(self.ASYNC_REQUEST_PERIOD_KEY, time / (num_events * num_samples))

    @kernel
    def _benchmark_async_throughput(self, num_events: TInt32, num_samples: TInt32) -> TInt64:
        # Total time
        t_total = np.int64(0)

        for _ in range(num_samples):
            # Reset the core (not required, but just to be sure)
            self.core.reset()
            # Call a sync RPC function to clear any RPC buffers
            self._empty_host_function()

            # Obtain the current time
            t_start = self.core.get_rtio_counter_mu()
            for _ in range(num_events):
                # Call the async RPC function
                self._empty_host_function_async()
            # Obtain the stop time
            t_stop = self.core.get_rtio_counter_mu()

            # Add difference to total time
            t_total += t_stop - t_start

        # Return accumulated total execution time
        return t_total
