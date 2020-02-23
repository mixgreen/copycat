import numpy as np

import artiq.coredevice.ttl

from dax.base import *
import dax.util.units


class RtioBenchmarkModule(DaxModule):
    """Module to benchmark the RTIO output system."""

    # System keys
    EVENT_PERIOD_KEY = 'event_period'
    EVENT_BURST_KEY = 'event_burst'
    DMA_EVENT_PERIOD_KEY = 'dma_event_period'
    LATENCY_CORE_RTIO_KEY = 'latency_core_rtio'

    # Unique DMA tags
    DMA_BURST = 'rtio_benchmark_burst'

    def build(self, ttl_out):
        # TTL output device
        self.setattr_device(ttl_out, 'ttl_out', (artiq.coredevice.ttl.TTLOut, artiq.coredevice.ttl.TTLInOut))

    def load(self):
        # Load parameters
        self.setattr_dataset_sys(self.EVENT_PERIOD_KEY)
        self.setattr_dataset_sys(self.EVENT_BURST_KEY)
        self.setattr_dataset_sys(self.DMA_EVENT_PERIOD_KEY)
        self.setattr_dataset_sys(self.LATENCY_CORE_RTIO_KEY)

        # Cap burst size to prevent too long DMA recordings, resulting in a connection timeout
        self.event_burst = min(self.event_burst, 40000)

    @kernel
    def init(self):
        # Break realtime to get some slack
        self.core.break_realtime()

        # Set TTL direction
        self.ttl_out.output()

        with self.core_dma.record(self.DMA_BURST):
            # Record the DMA burst trace
            for _ in range(self.event_burst // 2):
                delay(self.event_period / 2)
                self.ttl_out.on()
                delay(self.event_period / 2)
                self.ttl_out.off()

    def config(self):
        # Obtain DMA handle
        self.burst_dma_handle = self.core_dma.get_handle(self.DMA_BURST)
        self.update_kernel_invariants('burst_dma_handle')

    """Module functionality"""

    @kernel
    def on(self):
        self.ttl_out.on()

    @kernel
    def off(self):
        self.ttl_out.off()

    @kernel
    def pulse(self, duration):
        self.ttl_out.pulse(duration)

    @kernel
    def pulse_mu(self, duration):
        self.ttl_out.pulse_mu(duration)

    @kernel
    def burst(self):
        for _ in range(self.event_burst * 16):
            delay(self.event_period * 2)
            self.ttl_out.on()
            delay(self.event_period * 2)
            self.ttl_out.off()

    @kernel
    def burst_dma(self):
        for _ in range(128):
            self.core_dma.playback_handle(self.burst_dma_handle)

    """Benchmark throughput"""

    def benchmark_throughput(self, period_scan, num_samples, num_events, no_underflow_cutoff):
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
        self._benchmark_throughput(period_scan, num_samples, num_events, no_underflow_cutoff)

        # Get results
        no_underflow_count = self.get_dataset('no_underflow_count')
        underflow_flag = self.get_dataset('underflow_flag')
        last_period = self.get_dataset('last_period')

        # Process results directly (next experiment might need these values)
        if no_underflow_count == 0:
            # Last data point was an underflow, assuming all data points raised an underflow
            self.logger.warning('Could not determine throughput: All data points raised an underflow exception')
            return False
        elif not underflow_flag:
            # No underflow occurred
            self.logger.warning('Could not determine throughput: No data points raised an underflow exception')
            return False
        else:
            # Store result in system dataset
            self.set_dataset_sys(self.EVENT_PERIOD_KEY, last_period)
            return True

    @kernel
    def _benchmark_throughput(self, period_scan, num_samples, num_events, no_underflow_cutoff):
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
    def _spawn_events(self, period, num_samples, num_events):
        # Convert period to machine units
        period_mu = self.core.seconds_to_mu(period)
        # Scale number of events
        num_events //= 2

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

    """Benchmark throughput burst"""

    def benchmark_throughput_burst(self, num_events_min, num_events_max, num_events_step, num_samples, period_step,
                                   no_underflow_cutoff, num_step_cutoff):
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

        # Get current period
        current_period = self.get_dataset_sys(self.EVENT_PERIOD_KEY)

        # Store input values in dataset
        self.set_dataset('num_events_min', num_events_min)
        self.set_dataset('num_events_max', num_events_max)
        self.set_dataset('num_events_step', num_events_step)
        self.set_dataset('num_samples', num_samples)
        self.set_dataset('period_step', period_step)
        self.set_dataset('no_underflow_cutoff', no_underflow_cutoff)
        self.set_dataset('num_step_cutoff', num_step_cutoff)

        # Message starting period
        self._message_current_period(current_period)

        # Run kernel
        self._benchmark_throughput_burst(num_events_min, num_events_max, num_events_step, num_samples, current_period,
                                         period_step, no_underflow_cutoff, num_step_cutoff)

        # Get results
        no_underflow_count = self.get_dataset('no_underflow_count')
        underflow_flag = self.get_dataset('underflow_flag')
        last_num_events = self.get_dataset('last_num_events')

        # Process results directly (next experiment might need these values)
        if no_underflow_count == 0:
            self.logger.warning('Could not determine throughput burst: All data points raised an underflow exception')
            return False
        elif not underflow_flag:
            self.logger.warning('Could not determine throughput burst: No data points raised an underflow exception')
            return False
        else:
            # Store result in system dataset
            self.set_dataset_sys(self.EVENT_BURST_KEY, last_num_events)
            return True

    @rpc(flags={"async"})
    def _message_current_period(self, current_period):
        # Message current period
        self.logger.info('Using period {:s}'.format(dax.util.units.time_to_str(current_period)))

    @kernel
    def _benchmark_throughput_burst(self, num_events_min, num_events_max, num_events_step, num_samples, current_period,
                                    period_step, no_underflow_cutoff, num_step_cutoff):
        # Storage for last number of events
        last_num_events = np.int32(0)
        # Count of last number of events without underflow
        no_underflow_count = np.int32(0)
        # A flag to mark if at least one underflow happened
        underflow_flag = False

        while num_step_cutoff > 0:
            # Reset variables
            last_num_events = np.int32(0)
            underflow_flag = False
            no_underflow_count = np.int32(0)

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
                self._message_current_period(current_period)
            elif no_underflow_count == 0:
                # All points had an underflow event, increasing period
                current_period += period_step
                num_step_cutoff -= 1
                self._message_current_period(current_period)
            else:
                break  # Underflow events happened and threshold was found, stop testing

        # Store results in dataset
        self.set_dataset('no_underflow_count', no_underflow_count)
        self.set_dataset('underflow_flag', underflow_flag)
        self.set_dataset('last_num_events', last_num_events)
        self.set_dataset('last_period', current_period)
        self.set_dataset('last_num_step_cutoff', num_step_cutoff)

    """Benchmark DMA throughput"""

    def benchmark_dma_throughput(self, period_scan, num_samples, num_events, no_underflow_cutoff):
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
            self.logger.warning('Could not determine DMA throughput: All data points raised an underflow exception')
            return False
        elif not underflow_flag:
            # No underflow occurred
            self.logger.warning('Could not determine DMA throughput: No data points raised an underflow exception')
            return False
        else:
            # Store result in system dataset
            self.set_dataset_sys(self.DMA_EVENT_PERIOD_KEY, last_period)
            return True

    @kernel
    def _benchmark_dma_throughput(self, period_scan, num_samples, num_events, no_underflow_cutoff):
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
    def _spawn_dma_events(self, period, num_samples, num_events, dma_handle_on, dma_handle_off):
        # Convert period to machine units
        period_mu = self.core.seconds_to_mu(period)
        # Scale number of events
        num_events //= 2

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

    def benchmark_latency_core_rtio(self, latency_min, latency_max, latency_step, num_samples, no_underflow_cutoff):
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
            self.logger.warning('Could not determine core-RTIO latency: All data points raised an underflow exception')
            return False
        elif not underflow_flag:
            # No underflow occurred
            self.logger.warning('Could not determine core-RTIO latency: No data points raised an underflow exception')
            return False
        else:
            # Store result in system dataset
            self.set_dataset_sys(self.LATENCY_CORE_RTIO_KEY, last_latency)
            return True

    @kernel
    def _benchmark_latency_core_rtio(self, latency_min, latency_max, latency_step, num_samples, no_underflow_cutoff):
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
                    self.core.break_realtime()
                    # Reduce the slack to zero by waiting
                    self.core.wait_until_mu(now_mu())

                    # Insert latency and try to schedule an event
                    delay_mu(current_latency_mu)
                    self.ttl_out.off()

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
    INPUT_BUFFER_SIZE_KEY = 'input_buffer_size'
    LATENCY_RTIO_RTIO_KEY = 'latency_rtio_rtio'
    LATENCY_RTIO_CORE_KEY = 'latency_rtio_core'
    LATENCY_RTT_KEY = 'latency_rtt'  # Round-trip-time from RTIO input to RTIO output

    # Fixed edge delay time
    EDGE_DELAY = 1 * us

    def build(self, ttl_out, ttl_in):
        # Call super
        super(RtioLoopBenchmarkModule, self).build(ttl_out)
        # TTL input device
        self.setattr_device(ttl_in, 'ttl_in', artiq.coredevice.ttl.TTLInOut)

        # Add edge delay to kernel invariants
        self.update_kernel_invariants('EDGE_DELAY')

    def load(self):
        # Call super
        super(RtioLoopBenchmarkModule, self).load()

        # Load parameters
        self.setattr_dataset_sys(self.INPUT_BUFFER_SIZE_KEY)
        self.setattr_dataset_sys(self.LATENCY_RTIO_RTIO_KEY)
        self.setattr_dataset_sys(self.LATENCY_RTIO_CORE_KEY)
        self.setattr_dataset_sys(self.LATENCY_RTT_KEY)

    @kernel
    def init(self):
        # Call super (not using MRO/super because it is incompatible with the compiler, call parent function directly)
        RtioBenchmarkModule.init(self)

        # Break realtime to get some slack
        self.core.break_realtime()

        # Set TTL direction
        self.ttl_in.input()

    @kernel
    def _test_loop_connection(self, detection_window, retry=np.int32(1)):
        # Reset core
        self.core.reset()

        # Test if we can confirm the loop connection
        for _ in range(retry):
            # Guarantee a healthy amount of slack
            self.core.break_realtime()

            # Turn output off
            self.ttl_out.off()
            delay(self.EDGE_DELAY)  # Guarantee a delay between off and on

            # Turn output on
            self.ttl_out.on()
            # Get the timestamp when the RTIO core detects the input event
            t_rtio = self.ttl_in.timestamp_mu(self.ttl_in.gate_rising(detection_window))

            if t_rtio != -1:
                # Loop connection was confirmed
                return True

        # No connection was detected
        return False

    """Benchmark input buffer size"""

    def benchmark_input_buffer_size(self, max_events):
        # Convert types of arguments
        max_events = np.int32(max_events)

        # Check arguments
        if not max_events > 0:
            msg = 'Maximum number of events must be larger than 0'
            self.logger.error(msg)
            raise ValueError(msg)

        # Store input values in dataset
        self.set_dataset('max_events', max_events)

        # Test loop connection
        if not self._test_loop_connection(detection_window):
            self.logger.error('Could not determine input buffer size: Loop not connected')
            return False

        # Call the kernel
        self._benchmark_input_buffer_size(max_events)

        # Get results
        num_events = self.get_dataset('num_events')
        buffer_overflow = self.get_dataset('buffer_overflow')

        if not buffer_overflow:
            # No buffer overflow, so we did not found the limit
            self.logger.warning('Could not determine input buffer size: No overflow occurred')
            return False
        else:
            # Process results directly (next experiment might need these values)
            self.set_dataset_sys(self.INPUT_BUFFER_SIZE_KEY, num_events)
            return True

    @kernel
    def _benchmark_input_buffer_size(self, max_events):
        # Calculate detection window
        detection_window = self.EDGE_DELAY * (max_events + 2)

        # Counter for number of events posted
        num_events = np.int32(0)
        # Flag for overflow
        buffer_overflow = False

        # Reset core
        self.core.reset()
        # Save time zero
        t_zero = now_mu()
        # Start detection
        self.ttl_in.gate_rising(detection_window)
        # Move back to t_zero to generate events during detection window
        at_mu(t_zero)

        try:
            # Start generating events
            while num_events < max_events:
                # Generate event
                self.ttl_out.off()
                delay(self.EDGE_DELAY / 2)
                self.ttl_out.on()
                delay(self.EDGE_DELAY / 2)

                # Increment counter
                num_events += 1

        except RTIOOverflow:
            # Flag that an overflow occurred
            buffer_overflow = True

        # Store values at a non-critical time
        self.append_to_dataset('num_events', num_events)
        self.append_to_dataset('buffer_overflow', buffer_overflow)

    """Benchmark latency RTIO-core"""

    def benchmark_latency_rtio_core(self, num_samples, detection_window):
        # Convert types of arguments
        num_samples = np.int32(num_samples)
        detection_window = float(detection_window)

        # Check arguments
        if not num_samples > 0:
            msg = 'Number of samples must be larger than 0'
            self.logger.error(msg)
            raise ValueError(msg)
        if not detection_window > 0.0:
            msg = 'Detection window must be larger than 0.0'
            self.logger.error(msg)
            raise ValueError(msg)

        # Store input values in dataset
        self.set_dataset('num_samples', num_samples)
        self.set_dataset('detection_window', detection_window)

        # Test loop connection
        if not self._test_loop_connection(detection_window):
            self.logger.error('Could not determine RTIO-core latency: Loop not connected')
            return False

        # Prepare datasets for results
        self.set_dataset('t_zero', [])
        self.set_dataset('t_rtio', [])
        self.set_dataset('t_return', [])

        # Call the kernel
        self._benchmark_latency_rtio_core(num_samples, detection_window)

        # Get results
        t_zero = np.array(self.get_dataset('t_zero'))
        t_rtio = np.array(self.get_dataset('t_rtio'))
        t_return = np.array(self.get_dataset('t_return'))

        if any(t == -1 for t in t_rtio):
            # One or more tests did not return a timestamp, test failed
            self.logger.warning(
                'Could not determine RTIO-core latency: One or more tests did not return a valid timestamp')
            return False
        else:
            # Process results directly (next experiment might need these values)
            rtio_rtio = (t_rtio - t_zero).mean()
            rtio_core = (t_return - t_zero).mean()
            self.set_dataset_sys(self.LATENCY_RTIO_RTIO_KEY, rtio_rtio)
            self.set_dataset_sys(self.LATENCY_RTIO_CORE_KEY, rtio_core)
            return True

    @kernel
    def _benchmark_latency_rtio_core(self, num_samples, detection_window):
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
            t_rtio = self.ttl_in.timestamp_mu(self.ttl_in.gate_rising(detection_window))
            # Get the timestamp (of the RTIO core) when the RISC core reads the input event (return time)
            t_return = self.core.get_rtio_counter_mu()  # Returns a lower bound

            # Store values at a non-critical time
            self.append_to_dataset('t_zero', t_zero)
            self.append_to_dataset('t_rtio', t_rtio)
            self.append_to_dataset('t_return', t_return)

    """Benchmark RTT RTIO-core-RTIO"""

    def benchmark_latency_rtt(self, latency_min, latency_max, latency_step, num_samples, detection_window,
                              no_underflow_cutoff):
        # Convert types of arguments
        latency_min = float(latency_min)
        latency_max = float(latency_max)
        latency_step = float(latency_step)
        num_samples = np.int32(num_samples)
        detection_window = float(detection_window)
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
        if not detection_window > 0.0:
            msg = 'Detection window must be larger than 0.0'
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
        self.set_dataset('detection_window', detection_window)
        self.set_dataset('no_underflow_cutoff', no_underflow_cutoff)

        # Test loop connection
        if not self._test_loop_connection(detection_window):
            self.logger.error('Could not determine RTT: Loop not connected')
            return False

        # Run kernel
        self._benchmark_latency_rtt(latency_min, latency_max, latency_step, num_samples, detection_window,
                                    no_underflow_cutoff)

        # Get results
        no_underflow_count = self.get_dataset('no_underflow_count')
        underflow_flag = self.get_dataset('underflow_flag')
        last_latency = self.get_dataset('last_latency')

        # Process results directly (next experiment might need these values)
        if no_underflow_count == 0:
            # Last data point was an underflow, assuming all data points raised an underflow
            self.logger.warning('Could not determine RTT: All data points raised an underflow exception')
            return False
        elif not underflow_flag:
            # No underflow occurred
            self.logger.warning('Could not determine RTT: No data points raised an underflow exception')
            return False
        else:
            # Store result in system dataset
            self.set_dataset_sys(self.LATENCY_RTT_KEY, last_latency)
            return True

    @kernel
    def _benchmark_latency_rtt(self, latency_min, latency_max, latency_step, num_samples, detection_window,
                               no_underflow_cutoff):
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
                    t_window = self.ttl_in.gate_rising(detection_window)
                    # Set the cursor at time zero + current latency (prepare for scheduling feedback event)
                    at_mu(t_zero + current_latency_mu)
                    # Wait for the timestamp when the RTIO core detects the input event
                    self.ttl_in.timestamp_mu(t_window)
                    # Schedule the event at time zero + current latency
                    self.ttl_out.off()

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
