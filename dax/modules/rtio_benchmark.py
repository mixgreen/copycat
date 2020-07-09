import typing
import numpy as np

import artiq.coredevice.ttl  # type: ignore

from dax.experiment import *
import dax.util.units

__all__ = ['RtioBenchmarkModule', 'RtioLoopBenchmarkModule']


class RtioBenchmarkModule(DaxModule):
    """Module to benchmark the RTIO output system."""

    # System keys
    EVENT_PERIOD_KEY: str = 'event_period'
    EVENT_BURST_KEY: str = 'event_burst'
    DMA_EVENT_PERIOD_KEY: str = 'dma_event_period'
    LATENCY_CORE_RTIO_KEY: str = 'latency_core_rtio'

    # Unique DMA tags
    DMA_BURST: str = 'rtio_benchmark_burst'

    def build(self, *,  # type: ignore
              ttl_out: str, dma: bool = False, max_burst: int = 10000, init: bool = True) -> None:
        """Build the RTIO benchmark module.

        :param ttl_out: Key of the TTLInOut device to use
        :param dma: Enable the DMA features of this module
        :param max_burst: The maximum burst size
        :param init: Enable initialization of this module
        """
        assert isinstance(dma, bool), 'DMA flag should be of type bool'
        assert isinstance(max_burst, int), 'Max burst should be of type int'
        assert isinstance(init, bool), 'Initialization flag must be of type bool'

        # Store attributes
        self._dma_enabled: bool = dma
        self._max_burst: int = max(max_burst, 0)
        self._init_flag: bool = init
        self.logger.debug(f'Init flag: {self._init_flag}')
        self.update_kernel_invariants('_dma_enabled', 'DMA_BURST', '_max_burst')

        # TTL output device
        self.ttl_out = self.get_device(ttl_out, artiq.coredevice.ttl.TTLInOut)
        self.update_kernel_invariants('ttl_out')

    def init(self) -> None:
        # Load parameters
        self.setattr_dataset_sys(self.EVENT_PERIOD_KEY)
        self.setattr_dataset_sys(self.EVENT_BURST_KEY)
        self.setattr_dataset_sys(self.DMA_EVENT_PERIOD_KEY)
        self.setattr_dataset_sys(self.LATENCY_CORE_RTIO_KEY)

        # Update DMA enabled flag
        self._dma_enabled = self._dma_enabled and self.hasattr(self.EVENT_PERIOD_KEY, self.EVENT_BURST_KEY)
        self.logger.debug(f'DMA enabled: {self._dma_enabled}')

        if self.hasattr(self.EVENT_BURST_KEY):
            # Limit event burst size
            self.event_burst_size = np.int32(min(self.event_burst, self._max_burst))  # type: ignore[attr-defined]
            self.update_kernel_invariants('event_burst_size')
            self.logger.debug(f'Event burst size set to: {self.event_burst_size:d}')

        if self._dma_enabled:
            # Assign DMA burst as default burst
            self.burst = self.burst_dma  # type: ignore[assignment]
        else:
            # Assign slow burst as default burst
            self.burst = self.burst_slow  # type: ignore[assignment]
            # Disable DMA recording during initialization
            self._record_dma_burst = self._nop  # type: ignore[assignment]

        if self._init_flag:
            # Call the init kernel function
            self.init_kernel()

    @kernel
    def init_kernel(self):  # type: () -> None
        """Kernel function to initialize this module.

        This function is called automatically during initialization unless the user configured otherwise.
        In that case, this function has to be called manually.
        """
        # Reset the core
        self.core.reset()

        # Set direction of pin
        self.ttl_out.output()

        # Wait until event is submitted
        self.core.wait_until_mu(now_mu())

        # Record DMA burst
        self._record_dma_burst()

    @kernel
    def _record_dma_burst(self):  # type: () -> None
        # Record the DMA burst trace
        with self.core_dma.record(self.DMA_BURST):
            for _ in range(self.event_burst_size):
                delay(self.event_period / 2)  # type: ignore[attr-defined]
                self.ttl_out.on()
                delay(self.event_period / 2)  # type: ignore[attr-defined]
                self.ttl_out.off()

    @kernel
    def _nop(self):  # type: () -> None
        """Empty function."""
        pass

    def post_init(self) -> None:
        if self._dma_enabled:
            # Obtain DMA handle
            self.burst_dma_handle = self.core_dma.get_handle(self.DMA_BURST)
            self.update_kernel_invariants('burst_dma_handle')

    """Module functionality"""

    @kernel
    def burst(self):  # type: () -> None
        """Burst using DMA if enabled, otherwise fallback on slow burst."""
        # This function is a placeholder that will be assigned during initialization
        raise RuntimeError('Module was not initialized')

    @kernel
    def burst_slow(self):  # type: () -> None
        """Burst by spawning events one by one."""
        for _ in range(self.event_burst_size):
            delay(self.event_period * 2)  # type: ignore[attr-defined]
            self.ttl_out.on()
            delay(self.event_period * 2)  # type: ignore[attr-defined]
            self.ttl_out.off()

    @kernel
    def burst_dma(self):  # type: () -> None
        """Burst by DMA handle playback."""
        self.core_dma.playback_handle(self.burst_dma_handle)

    """Benchmark event throughput"""

    def benchmark_event_throughput(self, period_scan: typing.Union[typing.List[float], np.ndarray],
                                   num_samples: int, num_events: int, no_underflow_cutoff: int) -> None:
        # Convert types of arguments
        num_samples = np.int32(num_samples)
        num_events = np.int32(num_events)
        no_underflow_cutoff = np.int32(no_underflow_cutoff)

        # Check arguments
        if not num_samples > 0:
            msg = 'Number of samples must be larger than 0'
            self.logger.error(msg)
            raise ValueError(msg)
        if not num_events > 0:
            msg = 'Number of events must be larger than 0'
            self.logger.error(msg)
            raise ValueError(msg)
        if not no_underflow_cutoff > 0:
            msg = 'No underflow cutoff must be larger than 0'
            self.logger.error(msg)
            raise ValueError(msg)

        # Sort scan (in-place)
        period_scan.sort()

        # Store input values in dataset
        self.set_dataset('period_scan', period_scan)
        self.set_dataset('num_samples', num_samples)
        self.set_dataset('num_events', num_events)
        self.set_dataset('no_underflow_cutoff', no_underflow_cutoff)

        # Run kernel
        self._benchmark_event_throughput(period_scan, num_samples, num_events, no_underflow_cutoff)

        # Get results
        no_underflow_count = self.get_dataset('no_underflow_count')
        underflow_flag = self.get_dataset('underflow_flag')
        last_period = self.get_dataset('last_period')

        # Process results directly (next experiment might need these values)
        if no_underflow_count == 0:
            # Last data point was an underflow, assuming all data points raised an underflow
            msg = 'Could not determine event throughput: All data points raised an underflow exception'
            self.logger.warning(msg)
            raise RuntimeWarning(msg)
        elif not underflow_flag:
            # No underflow occurred
            msg = 'Could not determine event throughput: No data points raised an underflow exception'
            self.logger.warning(msg)
            raise RuntimeWarning(msg)
        else:
            # Store result in system dataset
            self.set_dataset_sys(self.EVENT_PERIOD_KEY, last_period)

    @kernel
    def _benchmark_event_throughput(self, period_scan, num_samples: TInt32, num_events: TInt32,
                                    no_underflow_cutoff: TInt32):
        # Storage for last period
        last_period = 0.0
        # Count of last period without underflow
        no_underflow_count = np.int32(0)
        # A flag to mark if at least one underflow happened
        underflow_flag = False

        # Iterate over scan
        for current_period in period_scan:
            try:
                # Convert time and start spawning events
                self._spawn_events(current_period, num_samples, num_events)
            except RTIOUnderflow:
                # Set underflow flag
                underflow_flag = True
                # Reset counter
                no_underflow_count = 0
            else:
                if no_underflow_count == 0:
                    # Store the period that works
                    last_period = current_period

                # Increment counter
                no_underflow_count += 1

                if no_underflow_count >= no_underflow_cutoff:
                    # Cutoff reached, stop testing
                    break

        # Store results
        self.set_dataset('no_underflow_count', no_underflow_count)
        self.set_dataset('underflow_flag', underflow_flag)
        self.set_dataset('last_period', last_period)

    @kernel
    def _spawn_events(self, period: TFloat, num_samples: TInt32, num_events: TInt32):
        # Convert period to machine units
        period_mu = self.core.seconds_to_mu(period)
        # Scale number of events
        num_events >>= 1

        # Iterate over number of samples
        for _ in range(num_samples):
            # RTIO reset
            self.core.reset()
            self.ttl_out.off()

            # Spawn events, could throw RTIOUnderflow
            for _ in range(num_events):
                delay_mu(period_mu)
                self.ttl_out.on()
                delay_mu(period_mu)
                self.ttl_out.off()

            # RTIO sync
            self.core.wait_until_mu(now_mu())

    """Benchmark event burst"""

    def benchmark_event_burst(self, num_events_min: int, num_events_max: int, num_events_step: int,
                              num_samples: int, period_step: float,
                              no_underflow_cutoff: int, num_step_cutoff: int) -> None:
        # Convert types of arguments
        num_events_min = np.int32(num_events_min)
        num_events_max = np.int32(num_events_max)
        num_events_step = np.int32(num_events_step)
        num_samples = np.int32(num_samples)
        period_step = float(period_step)
        no_underflow_cutoff = np.int32(no_underflow_cutoff)
        num_step_cutoff = np.int32(num_step_cutoff)

        # Check arguments
        if not num_events_min > 0:
            msg = 'Minimum number of events must be larger then 0'
            self.logger.error(msg)
            raise ValueError(msg)
        if not num_events_max > 0:
            msg = 'Minimum number of events must be larger then 0'
            self.logger.error(msg)
            raise ValueError(msg)
        if not num_events_step > 0:
            msg = 'Number of events step must be larger then 0'
            self.logger.error(msg)
            raise ValueError(msg)
        if num_events_min > num_events_max:
            msg = 'Minimum number of events must be smaller than maximum number of events'
            self.logger.error(msg)
            raise ValueError(msg)
        if not num_samples > 0:
            msg = 'Number of samples must be larger than 0'
            self.logger.error(msg)
            raise ValueError(msg)
        if not period_step > 0.0:
            msg = 'Period step must be larger than 0'
            self.logger.error(msg)
            raise ValueError(msg)
        if not no_underflow_cutoff > 0:
            msg = 'No underflow cutoff must be larger than 0'
            self.logger.error(msg)
            raise ValueError(msg)

        # Store input values in dataset
        self.set_dataset('num_events_min', num_events_min)
        self.set_dataset('num_events_max', num_events_max)
        self.set_dataset('num_events_step', num_events_step)
        self.set_dataset('num_samples', num_samples)
        self.set_dataset('period_step', period_step)
        self.set_dataset('no_underflow_cutoff', no_underflow_cutoff)
        self.set_dataset('num_step_cutoff', num_step_cutoff)

        # Run kernel
        self._benchmark_event_burst(num_events_min, num_events_max, num_events_step, num_samples, period_step,
                                    no_underflow_cutoff, num_step_cutoff)

        # Get results
        no_underflow_count = self.get_dataset('no_underflow_count')
        underflow_flag = self.get_dataset('underflow_flag')
        last_num_events = self.get_dataset('last_num_events')

        # Process results directly (next experiment might need these values)
        if no_underflow_count == 0:
            msg = 'Could not determine event burst size: All data points raised an underflow exception'
            self.logger.warning(msg)
            raise RuntimeWarning(msg)
        elif not underflow_flag:
            msg = 'Could not determine event burst size: No data points raised an underflow exception'
            self.logger.warning(msg)
            raise RuntimeWarning(msg)
        else:
            # Store result in system dataset
            self.set_dataset_sys(self.EVENT_BURST_KEY, last_num_events)

    @rpc(flags={"async"})
    def _message_current_period(self, current_period):  # type: (float) -> None
        # Message current period
        self.logger.info(f'Using period {dax.util.units.time_to_str(current_period):s}')

    @kernel
    def _benchmark_event_burst(self, num_events_min: TInt32, num_events_max: TInt32, num_events_step: TInt32,
                               num_samples: TInt32, period_step: TFloat,
                               no_underflow_cutoff: TInt32, num_step_cutoff: TInt32):
        # Storage for last number of events
        last_num_events = np.int32(0)
        # Count of last number of events without underflow
        no_underflow_count = np.int32(0)
        # A flag to mark if at least one underflow happened
        underflow_flag = False
        # Current period
        current_period = self.event_period  # type: ignore[attr-defined]

        while num_step_cutoff > 0:
            # Reset variables
            last_num_events = np.int32(0)
            underflow_flag = False
            no_underflow_count = np.int32(0)

            # Message current period
            self._message_current_period(current_period)

            # Iterate over scan from max to min (manual iteration for better performance on large range)
            num_events = num_events_max
            while num_events > num_events_min:
                try:
                    # Spawn events
                    self._spawn_events(current_period, num_samples, num_events)
                except RTIOUnderflow:
                    # Set underflow flag
                    underflow_flag = True
                    # Reset no underflow counter
                    no_underflow_count = 0
                else:
                    if no_underflow_count == 0:
                        # Store the number that works
                        last_num_events = num_events

                    # Increment counter
                    no_underflow_count += 1

                    if no_underflow_count >= no_underflow_cutoff:
                        # No underflow detected and cutoff reached, stop testing
                        break

                # Manual update of iteration values
                num_events -= num_events_step

            if not underflow_flag:
                # No underflow events occurred, reducing period
                current_period -= period_step
                num_step_cutoff -= 1
            elif no_underflow_count == 0:
                # All points had an underflow event, increasing period
                current_period += period_step
                num_step_cutoff -= 1
            else:
                break  # Underflow events happened and threshold was found, stop testing

        # Store results in dataset
        self.set_dataset('no_underflow_count', no_underflow_count)
        self.set_dataset('underflow_flag', underflow_flag)
        self.set_dataset('last_num_events', last_num_events)
        self.set_dataset('last_period', current_period)
        self.set_dataset('last_num_step_cutoff', num_step_cutoff)

    """Benchmark DMA throughput"""

    def benchmark_dma_throughput(self, period_scan: typing.Union[typing.List[float], np.ndarray],
                                 num_samples: int, num_events: int, no_underflow_cutoff: int) -> None:
        # Convert types of arguments
        num_samples = np.int32(num_samples)
        num_events = np.int32(num_events)
        no_underflow_cutoff = np.int32(no_underflow_cutoff)

        # Check arguments
        if not num_samples > 0:
            msg = 'Number of samples must be larger than 0'
            self.logger.error(msg)
            raise ValueError(msg)
        if not num_events > 0:
            msg = 'Number of events must be larger than 0'
            self.logger.error(msg)
            raise ValueError(msg)
        if not no_underflow_cutoff > 0:
            msg = 'No underflow cutoff must be larger than 0'
            self.logger.error(msg)
            raise ValueError(msg)

        # Sort scan (in-place)
        period_scan.sort()

        # Store input values in dataset
        self.set_dataset('period_scan', period_scan)
        self.set_dataset('num_samples', num_samples)
        self.set_dataset('num_events', num_events)
        self.set_dataset('no_underflow_cutoff', no_underflow_cutoff)

        # Run kernel
        self._benchmark_dma_throughput(period_scan, num_samples, num_events, no_underflow_cutoff)

        # Get results
        no_underflow_count = self.get_dataset('no_underflow_count')
        underflow_flag = self.get_dataset('underflow_flag')
        last_period = self.get_dataset('last_period')

        # Process results directly (next experiment might need these values)
        if no_underflow_count == 0:
            # Last data point was an underflow, assuming all data points raised an underflow
            msg = 'Could not determine DMA throughput: All data points raised an underflow exception'
            self.logger.warning(msg)
            raise RuntimeWarning(msg)
        elif not underflow_flag:
            # No underflow occurred
            msg = 'Could not determine DMA throughput: No data points raised an underflow exception'
            self.logger.warning(msg)
            raise RuntimeWarning(msg)
        else:
            # Store result in system dataset
            self.set_dataset_sys(self.DMA_EVENT_PERIOD_KEY, last_period)

    @kernel
    def _benchmark_dma_throughput(self, period_scan, num_samples: TInt32, num_events: TInt32,
                                  no_underflow_cutoff: TInt32):
        # Storage for last period
        last_period = 0.0
        # Count of last period without underflow
        no_underflow_count = np.int32(0)
        # A flag to mark if at least one underflow happened
        underflow_flag = False

        # Record DMA traces
        dma_name_on = 'rtio_benchmark_dma_throughput_on'
        dma_name_off = 'rtio_benchmark_dma_throughput_off'
        with self.core_dma.record(dma_name_on):
            self.ttl_out.on()
        with self.core_dma.record(dma_name_off):
            self.ttl_out.off()

        # Obtain handles
        dma_handle_on = self.core_dma.get_handle(dma_name_on)
        dma_handle_off = self.core_dma.get_handle(dma_name_off)

        # Iterate over scan
        for current_period in period_scan:
            try:
                # Convert time and start spawning events
                self._spawn_dma_events(current_period, num_samples, num_events, dma_handle_on, dma_handle_off)
            except RTIOUnderflow:
                # Set underflow flag
                underflow_flag = True
                # Reset counter
                no_underflow_count = 0
            else:
                if no_underflow_count == 0:
                    # Store the period that works
                    last_period = current_period

                # Increment counter
                no_underflow_count += 1

                if no_underflow_count >= no_underflow_cutoff:
                    # Cutoff reached, stop testing
                    break

        # Erase DMA traces
        self.core_dma.erase(dma_name_on)
        self.core_dma.erase(dma_name_off)

        # Store results
        self.set_dataset('no_underflow_count', no_underflow_count)
        self.set_dataset('underflow_flag', underflow_flag)
        self.set_dataset('last_period', last_period)

    @kernel
    def _spawn_dma_events(self, period: TFloat, num_samples: TInt32, num_events: TInt32,
                          dma_handle_on, dma_handle_off):
        # Convert period to machine units
        period_mu = self.core.seconds_to_mu(period)
        # Scale number of events
        num_events >>= 1

        # Iterate over number of samples
        for _ in range(num_samples):
            # RTIO reset
            self.core.reset()
            self.ttl_out.off()

            # Spawn events, could throw RTIOUnderflow
            for _ in range(num_events):
                delay_mu(period_mu)
                self.core_dma.playback_handle(dma_handle_on)
                delay_mu(period_mu)
                self.core_dma.playback_handle(dma_handle_off)

            # RTIO sync
            self.core.wait_until_mu(now_mu())

    """Benchmark latency core-RTIO"""

    def benchmark_latency_core_rtio(self, latency_min: float, latency_max: float, latency_step: float,
                                    num_samples: int, no_underflow_cutoff: int) -> None:
        # Convert types of arguments
        latency_min = float(latency_min)
        latency_max = float(latency_max)
        latency_step = float(latency_step)
        num_samples = np.int32(num_samples)
        no_underflow_cutoff = np.int32(no_underflow_cutoff)

        # Check arguments
        if not latency_min > 0.0:
            msg = 'Minimum latency must be larger than 0.0'
            self.logger.error(msg)
            raise ValueError(msg)
        if not latency_max > 0.0:
            msg = 'Maximum latency must be larger than 0.0'
            self.logger.error(msg)
            raise ValueError(msg)
        if not latency_step > 0.0:
            msg = 'Latency step must be larger than 0.0'
            self.logger.error(msg)
            raise ValueError(msg)
        if latency_min > latency_max:
            msg = 'Minimum latency must be smaller than maximum latency'
            self.logger.error(msg)
            raise ValueError(msg)
        if not num_samples > 0:
            msg = 'Number of samples must be larger than 0'
            self.logger.error(msg)
            raise ValueError(msg)
        if not no_underflow_cutoff > 0:
            msg = 'No underflow cutoff must be larger than 0'
            self.logger.error(msg)
            raise ValueError(msg)

        # Store input values in dataset
        self.set_dataset('latency_min', latency_min)
        self.set_dataset('latency_max', latency_max)
        self.set_dataset('latency_step', latency_step)
        self.set_dataset('num_samples', num_samples)
        self.set_dataset('no_underflow_cutoff', no_underflow_cutoff)

        # Run kernel
        self._benchmark_latency_core_rtio(latency_min, latency_max, latency_step, num_samples, no_underflow_cutoff)

        # Get results
        no_underflow_count = self.get_dataset('no_underflow_count')
        underflow_flag = self.get_dataset('underflow_flag')
        last_latency = self.get_dataset('last_latency')

        # Process results directly (next experiment might need these values)
        if no_underflow_count == 0:
            # Last data point was an underflow, assuming all data points raised an underflow
            msg = 'Could not determine core-RTIO latency: All data points raised an underflow exception'
            self.logger.warning(msg)
            raise RuntimeWarning(msg)
        elif not underflow_flag:
            # No underflow occurred
            msg = 'Could not determine core-RTIO latency: No data points raised an underflow exception'
            self.logger.warning(msg)
            raise RuntimeWarning(msg)
        else:
            # Store result in system dataset
            self.set_dataset_sys(self.LATENCY_CORE_RTIO_KEY, last_latency)

    @kernel
    def _benchmark_latency_core_rtio(self, latency_min: TFloat, latency_max: TFloat, latency_step: TFloat,
                                     num_samples: TInt32, no_underflow_cutoff: TInt32):
        # Storage for last latency
        last_latency = 0.0
        # Count of last latency without underflow
        no_underflow_count = np.int32(0)
        # A flag to mark if at least one underflow happened
        underflow_flag = False

        # Reset core
        self.core.reset()

        # Iterate over range from max to min (manual iteration for better performance on large range)
        current_latency = latency_min
        while current_latency < latency_max:
            # Convert current latency to machine units
            current_latency_mu = self.core.seconds_to_mu(current_latency)

            try:
                for _ in range(num_samples):
                    # Break realtime to get some slack
                    self.core.break_realtime()  # break_realtime() performs better than reset() in this scenario

                    # Store time zero
                    t_zero = now_mu()
                    # Prepare cursor for event
                    delay_mu(current_latency_mu)

                    # Reduce the slack to zero by waiting
                    self.core.wait_until_mu(t_zero)
                    # Try to schedule an event
                    self.ttl_out.off()  # Could raise RTIOUnderFlow

            except RTIOUnderflow:
                # Set underflow flag
                underflow_flag = True
                # Reset counter
                no_underflow_count = 0

            else:
                if no_underflow_count == 0:
                    # Store the latency that works
                    last_latency = current_latency

                # Increment counter
                no_underflow_count += 1

                if no_underflow_count >= no_underflow_cutoff:
                    # Cutoff reached, stop testing
                    break

            current_latency += latency_step

        # Store results
        self.set_dataset('no_underflow_count', no_underflow_count)
        self.set_dataset('underflow_flag', underflow_flag)
        self.set_dataset('last_latency', last_latency)


class RtioLoopBenchmarkModule(RtioBenchmarkModule):
    """Module to benchmark the RTIO system with a looped connection."""

    # System keys
    INPUT_BUFFER_SIZE_KEY: str = 'input_buffer_size'
    LATENCY_RTIO_RTIO_KEY: str = 'latency_rtio_rtio'
    LATENCY_RTIO_CORE_KEY: str = 'latency_rtio_core'
    LATENCY_RTT_KEY: str = 'latency_rtt'  # Round-trip-time from RTIO input to RTIO output

    # Fixed edge delay time
    EDGE_DELAY: float = 1 * us

    def build(self, *, ttl_in: str, **kwargs: typing.Any) -> None:  # type: ignore
        """Build the RTIO loop benchmark module.

        :param ttl_in: Key of the TTLInOut device to use as input
        :param kwargs: Keyword arguments for the :class:`RtioBenchmarkModule` parent
        """
        # Call super
        super(RtioLoopBenchmarkModule, self).build(**kwargs)
        # TTL input device
        self.ttl_in = self.get_device(ttl_in, artiq.coredevice.ttl.TTLInOut)

        # Add edge delay to kernel invariants
        self.update_kernel_invariants('EDGE_DELAY', 'ttl_in')

    def init(self) -> None:
        # Log edge delay setting
        self.logger.debug(f'Edge delay set to: {dax.util.units.time_to_str(self.EDGE_DELAY):s}')

        # Load parameters
        self.setattr_dataset_sys(self.INPUT_BUFFER_SIZE_KEY)
        self.setattr_dataset_sys(self.LATENCY_RTIO_RTIO_KEY)
        self.setattr_dataset_sys(self.LATENCY_RTIO_CORE_KEY)
        self.setattr_dataset_sys(self.LATENCY_RTT_KEY)

        # Call super
        super(RtioLoopBenchmarkModule, self).init()

    @kernel
    def init_kernel(self):  # type: () -> None
        """Kernel function to initialize this module.

        This function is called automatically during initialization unless the user configured otherwise.
        In that case, this function has to be called manually.
        """
        # Reset the core
        self.core.reset()

        # Set direction of pins
        self.ttl_out.output()
        self.ttl_in.input()

        # Wait until event is submitted
        self.core.wait_until_mu(now_mu())

        # Record DMA burst
        self._record_dma_burst()

    @kernel
    def test_loop_connection(self, retry: TInt32 = np.int32(1)):
        """True if a loop connection was detected."""

        for _ in range(retry):
            # Reset core
            self.core.reset()

            # Turn output off
            self.ttl_out.off()
            delay(self.EDGE_DELAY)  # Guarantee a delay between off and on

            # Turn output on
            self.ttl_out.on()
            # Get the timestamp when the RTIO core detects the input event
            t_rtio = self.ttl_in.timestamp_mu(self.ttl_in.gate_rising(self.EDGE_DELAY))

            if t_rtio != -1:
                # Loop connection was confirmed
                return True

        # No connection was detected
        return False

    """Benchmark input buffer size"""

    def benchmark_input_buffer_size(self, min_events: int, max_events: int) -> None:
        # Convert types of arguments
        min_events = np.int32(min_events)
        max_events = np.int32(max_events)

        # Check arguments
        if not min_events > 0:
            msg = 'Minimum number of events must be larger than 0'
            self.logger.error(msg)
            raise ValueError(msg)
        if not max_events > 0:
            msg = 'Maximum number of events must be larger than 0'
            self.logger.error(msg)
            raise ValueError(msg)
        if min_events > max_events:
            msg = 'Minimum number of events must be smaller than maximum number of events'
            self.logger.error(msg)
            raise ValueError(msg)

        # Store input values in dataset
        self.set_dataset('min_events', min_events)
        self.set_dataset('max_events', max_events)

        # Test loop connection
        if not self.test_loop_connection():
            msg = 'Could not determine input buffer size: Loop not connected'
            self.logger.error(msg)
            raise RuntimeError(msg)

        # Call the kernel
        self._benchmark_input_buffer_size(min_events, max_events)

        # Get results
        num_events = self.get_dataset('num_events')
        buffer_overflow = self.get_dataset('buffer_overflow')

        if not buffer_overflow:
            # No buffer overflow, so we did not found the limit
            msg = 'Could not determine input buffer size: No overflow occurred'
            self.logger.warning(msg)
            raise RuntimeWarning(msg)
        else:
            # Process results directly (next experiment might need these values)
            self.set_dataset_sys(self.INPUT_BUFFER_SIZE_KEY, num_events)

    @kernel
    def _benchmark_input_buffer_size(self, min_events: TInt32, max_events: TInt32):
        # Counter for number of events posted
        num_events = np.int32(min_events)
        # Flag for overflow
        buffer_overflow = False

        try:
            while num_events <= max_events:
                # Reset core
                self.core.reset()

                # Start sampling
                for _ in range(num_events):
                    # Take one sample
                    self.ttl_in.sample_input()
                    delay(self.EDGE_DELAY)

                # Wait until all samples are taken
                self.core.wait_until_mu(now_mu())

                # Call sample_get() to verify if the queue is full
                self.ttl_in.sample_get()  # Exception could be raised here

                # Increment counter
                num_events += 1

        except RTIOOverflow:
            # Flag that an overflow occurred
            buffer_overflow = True

        finally:
            # Reset the core to clear the buffers
            self.core.reset()

        # Store values at a non-critical time
        self.set_dataset('num_events', num_events - 1)  # Minus one since last value was the last legal num_events
        self.set_dataset('buffer_overflow', buffer_overflow)

    """Benchmark latency RTIO-core"""

    def benchmark_latency_rtio_core(self, num_samples: int) -> None:
        # Convert types of arguments
        num_samples = np.int32(num_samples)

        # Check arguments
        if not num_samples > 0:
            msg = 'Number of samples must be larger than 0'
            self.logger.error(msg)
            raise ValueError(msg)

        # Store input values in dataset
        self.set_dataset('num_samples', num_samples)

        # Test loop connection
        if not self.test_loop_connection():
            msg = 'Could not determine RTIO-core latency: Loop not connected'
            self.logger.error(msg)
            raise RuntimeError(msg)

        # Prepare datasets for results
        self.set_dataset('t_zero', [])
        self.set_dataset('t_rtio', [])
        self.set_dataset('t_return', [])

        # Call the kernel
        self._benchmark_latency_rtio_core(num_samples)

        # Get results (mu)
        t_zero = self.get_dataset('t_zero')
        t_rtio = self.get_dataset('t_rtio')
        t_return = self.get_dataset('t_return')

        if any(t == -1 for t in t_rtio):  # type: ignore[union-attr]
            # One or more tests did not return a timestamp, test failed
            msg = 'Could not determine RTIO-core latency: One or more tests did not return a valid timestamp'
            self.logger.warning(msg)
            raise RuntimeWarning(msg)
        else:
            # Convert values to times
            t_zero = np.array([self.core.mu_to_seconds(t) for t in t_zero])  # type: ignore[union-attr]
            t_rtio = np.array([self.core.mu_to_seconds(t) for t in t_rtio])  # type: ignore[union-attr]
            t_return = np.array([self.core.mu_to_seconds(t) for t in t_return])  # type: ignore[union-attr]

            # Process results directly (next experiment might need these values)
            rtio_rtio = (t_rtio - t_zero).mean()
            rtio_core = (t_return - t_zero).mean()
            self.set_dataset_sys(self.LATENCY_RTIO_RTIO_KEY, rtio_rtio)
            self.set_dataset_sys(self.LATENCY_RTIO_CORE_KEY, rtio_core)

    @kernel
    def _benchmark_latency_rtio_core(self, num_samples: TInt32):
        # Reset core
        self.core.reset()

        for _ in range(num_samples):
            # Guarantee a healthy amount of slack to start the measurement
            self.core.break_realtime()

            # Turn output off
            self.ttl_out.off()
            delay(self.EDGE_DELAY)  # Guarantee a delay between off and on

            # Save time zero
            t_zero = now_mu()
            # Turn output on
            self.ttl_out.on()
            # Get the timestamp when the RTIO core detects the input event
            t_rtio = self.ttl_in.timestamp_mu(self.ttl_in.gate_rising(self.EDGE_DELAY))
            # Get the timestamp (of the RTIO core) when the RISC core reads the input event (return time)
            t_return = self.core.get_rtio_counter_mu()  # Returns an upper bound

            # Store values at a non-critical time
            self.append_to_dataset('t_zero', t_zero)
            self.append_to_dataset('t_rtio', t_rtio)
            self.append_to_dataset('t_return', t_return)

    """Benchmark RTT RTIO-core-RTIO"""

    def benchmark_latency_rtt(self, latency_min: float, latency_max: float, latency_step: float,
                              num_samples: int, no_underflow_cutoff: int) -> None:
        # Convert types of arguments
        latency_min = float(latency_min)
        latency_max = float(latency_max)
        latency_step = float(latency_step)
        num_samples = np.int32(num_samples)
        no_underflow_cutoff = np.int32(no_underflow_cutoff)

        # Check arguments
        if not latency_min > 0.0:
            msg = 'Minimum latency must be larger than 0.0'
            self.logger.error(msg)
            raise ValueError(msg)
        if not latency_max > 0.0:
            msg = 'Maximum latency must be larger than 0.0'
            self.logger.error(msg)
            raise ValueError(msg)
        if not latency_step > 0.0:
            msg = 'Latency step must be larger than 0.0'
            self.logger.error(msg)
            raise ValueError(msg)
        if latency_min > latency_max:
            msg = 'Minimum latency must be smaller than maximum latency'
            self.logger.error(msg)
            raise ValueError(msg)
        if not num_samples > 0:
            msg = 'Number of samples must be larger than 0'
            self.logger.error(msg)
            raise ValueError(msg)
        if not no_underflow_cutoff > 0:
            msg = 'No underflow cutoff must be larger than 0'
            self.logger.error(msg)
            raise ValueError(msg)

        # Store input values in dataset
        self.set_dataset('latency_min', latency_min)
        self.set_dataset('latency_max', latency_max)
        self.set_dataset('latency_step', latency_step)
        self.set_dataset('num_samples', num_samples)
        self.set_dataset('no_underflow_cutoff', no_underflow_cutoff)

        # Test loop connection
        if not self.test_loop_connection():
            msg = 'Could not determine RTT: Loop not connected'
            self.logger.error(msg)
            raise RuntimeError(msg)

        # Run kernel
        self._benchmark_latency_rtt(latency_min, latency_max, latency_step, num_samples, no_underflow_cutoff)

        # Get results
        no_underflow_count = self.get_dataset('no_underflow_count')
        underflow_flag = self.get_dataset('underflow_flag')
        last_latency = self.get_dataset('last_latency')

        # Process results directly (next experiment might need these values)
        if no_underflow_count == 0:
            # Last data point was an underflow, assuming all data points raised an underflow
            msg = 'Could not determine RTT: All data points raised an underflow exception'
            self.logger.warning(msg)
            raise RuntimeWarning(msg)
        elif not underflow_flag:
            # No underflow occurred
            msg = 'Could not determine RTT: No data points raised an underflow exception'
            self.logger.warning(msg)
            raise RuntimeWarning(msg)
        else:
            # Store result in system dataset
            self.set_dataset_sys(self.LATENCY_RTT_KEY, last_latency)

    @kernel
    def _benchmark_latency_rtt(self, latency_min: TFloat, latency_max: TFloat, latency_step: TFloat,
                               num_samples: TInt32, no_underflow_cutoff: TInt32):
        # Storage for last latency
        last_latency = 0.0
        # Count of last latency without underflow
        no_underflow_count = np.int32(0)
        # A flag to mark if at least one underflow happened
        underflow_flag = False

        # Reset core
        self.core.reset()

        # Iterate over scan from min to max (manual iteration for better performance on large range)
        current_latency = latency_min
        while current_latency < latency_max:
            # Convert current latency to machine units
            current_latency_mu = self.core.seconds_to_mu(current_latency)

            try:
                for _ in range(num_samples):
                    # Guarantee a healthy amount of slack to start the measurement
                    self.core.break_realtime()

                    # Turn output off
                    self.ttl_out.off()
                    delay(self.EDGE_DELAY)  # Guarantee a delay between off and on

                    # Save time zero
                    t_zero = now_mu()
                    # Turn output on (schedule event to respond on)
                    self.ttl_out.on()
                    # Order the RTIO core to detect a rising edge (moves cursor to end of detection window)
                    t_window = self.ttl_in.gate_rising(self.EDGE_DELAY)
                    # Set the cursor at time zero + current latency (prepare for scheduling feedback event)
                    at_mu(t_zero + current_latency_mu)

                    # Wait for the timestamp when the RTIO core detects the input event
                    self.ttl_in.timestamp_mu(t_window)
                    # Schedule the event at time zero + current latency
                    self.ttl_out.off()  # Could raise RTIOUnderflow

            except RTIOUnderflow:
                # Set underflow flag
                underflow_flag = True
                # Reset counter
                no_underflow_count = 0

            else:
                if no_underflow_count == 0:
                    # Store the latency that works
                    last_latency = current_latency

                # Increment counter
                no_underflow_count += 1

                if no_underflow_count >= no_underflow_cutoff:
                    # Cutoff reached, stop testing
                    break

            current_latency += latency_step

        # Store results
        self.set_dataset('no_underflow_count', no_underflow_count)
        self.set_dataset('underflow_flag', underflow_flag)
        self.set_dataset('last_latency', last_latency)
