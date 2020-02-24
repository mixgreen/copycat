import numpy as np

from dax.modules.rtio_benchmark import *
import dax.util.units


@dax_client_factory
class RtioBenchmarkThroughput(DaxClient, EnvExperiment):
    """RTIO throughput benchmark."""

    def build(self):
        # Arguments
        time_kwargs = {'unit': 'ns', 'scale': 1, 'step': 1, 'ndecimals': 0}
        number_kwargs = {'scale': 1, 'step': 1, 'ndecimals': 0}
        self.setattr_argument('period_min', NumberValue(100, min=1, **time_kwargs))
        self.setattr_argument('period_max', NumberValue(1000, min=1, **time_kwargs))
        self.setattr_argument('period_step', NumberValue(1, min=1, **time_kwargs))
        self.setattr_argument('num_samples', NumberValue(5, min=1, **number_kwargs))
        self.setattr_argument('num_events', NumberValue(500000, min=1000, **number_kwargs))
        self.setattr_argument('no_underflow_cutoff', NumberValue(5, min=1, **number_kwargs))

        # Obtain RTIO benchmark module
        self.rtio_bench = self.registry.search_module(RtioBenchmarkModule)

    def prepare(self):
        # Additional check if arguments are valid
        if self.period_min > self.period_max:
            msg = 'Minimum period must be smaller than maximum period'
            self.logger.error(msg)
            raise ValueError(msg)

        # Create scan object
        self.period_scan = np.arange(self.period_min, self.period_max, self.period_step, dtype=np.float)
        self.period_scan *= ns
        self.update_kernel_invariants('period_scan')

    def run(self):
        self.success = self.rtio_bench.benchmark_throughput(self.period_scan, self.num_samples, self.num_events,
                                                            self.no_underflow_cutoff)

    def analyze(self):
        if self.success:
            # Report result
            last_period = self.rtio_bench.get_dataset_sys(self.rtio_bench.EVENT_PERIOD_KEY)
            throughput = dax.util.units.freq_to_str(1.0 / last_period)
            period = '{:s} period'.format(dax.util.units.time_to_str(last_period))
            self.logger.info('RTIO event throughput is {:s} ({:s})'.format(throughput, period))


@dax_client_factory
class RtioBenchmarkThroughputBurst(DaxClient, EnvExperiment):
    """RTIO throughput burst benchmark."""

    def build(self):
        # Arguments
        number_kwargs = {'scale': 1, 'step': 1, 'ndecimals': 0}
        time_kwargs = {'unit': 'ns', 'scale': 1, 'step': 1, 'ndecimals': 0}
        self.setattr_argument('num_events_min', NumberValue(1000, min=1, **number_kwargs))
        self.setattr_argument('num_events_max', NumberValue(500000, min=1, **number_kwargs))
        self.setattr_argument('num_events_step', NumberValue(1000, min=1, **number_kwargs))
        self.setattr_argument('num_samples', NumberValue(5, min=1, **number_kwargs))
        self.setattr_argument('period_step', NumberValue(1, min=1, **time_kwargs))
        self.setattr_argument('no_underflow_cutoff', NumberValue(5, min=1, **number_kwargs))
        self.setattr_argument('num_step_cutoff', NumberValue(5, min=0, **number_kwargs))

        # Obtain RTIO benchmark module
        self.rtio_bench = self.registry.search_module(RtioBenchmarkModule)

    def prepare(self):
        # Scale period step
        self.period_step *= ns

    def run(self):
        self.success = self.rtio_bench.benchmark_throughput_burst(self.num_events_min, self.num_events_max,
                                                                  self.num_events_step, self.num_samples,
                                                                  self.period_step,
                                                                  self.no_underflow_cutoff, self.num_step_cutoff)

    def analyze(self):
        if self.success:
            # Report result
            last_num_events = self.rtio_bench.get_dataset_sys(self.rtio_bench.EVENT_BURST_KEY)
            self.logger.info('RTIO event burst size is {:d}'.format(last_num_events))


@dax_client_factory
class RtioBenchmarkDmaThroughput(DaxClient, EnvExperiment):
    """RTIO DMA throughput benchmark."""

    def build(self):
        # Arguments
        time_kwargs = {'unit': 'ns', 'scale': 1, 'step': 1, 'ndecimals': 0}
        number_kwargs = {'scale': 1, 'step': 1, 'ndecimals': 0}
        self.setattr_argument('period_min', NumberValue(200, min=1, **time_kwargs))
        self.setattr_argument('period_max', NumberValue(1000, min=1, **time_kwargs))
        self.setattr_argument('period_step', NumberValue(1, min=1, **time_kwargs))
        self.setattr_argument('num_samples', NumberValue(5, min=1, **number_kwargs))
        self.setattr_argument('num_events', NumberValue(500000, min=1000, **number_kwargs))
        self.setattr_argument('no_underflow_cutoff', NumberValue(5, min=1, **number_kwargs))

        # Obtain RTIO benchmark module
        self.rtio_bench = self.registry.search_module(RtioBenchmarkModule)

    def prepare(self):
        # Additional check if arguments are valid
        if self.period_min > self.period_max:
            msg = 'Minimum period must be smaller than maximum period'
            self.logger.error(msg)
            raise ValueError(msg)

        # Create scan object
        self.period_scan = np.arange(self.period_min, self.period_max, self.period_step, dtype=np.float)
        self.period_scan *= ns
        self.update_kernel_invariants('period_scan')

    def run(self):
        self.success = self.rtio_bench.benchmark_dma_throughput(self.period_scan, self.num_samples, self.num_events,
                                                                self.no_underflow_cutoff)

    def analyze(self):
        if self.success:
            # Report result
            last_period = self.rtio_bench.get_dataset_sys(self.rtio_bench.DMA_EVENT_PERIOD_KEY)
            throughput = dax.util.units.freq_to_str(1.0 / last_period)
            period = '{:s} period'.format(dax.util.units.time_to_str(last_period))
            self.logger.info('RTIO DMA event throughput is {:s} ({:s})'.format(throughput, period))


@dax_client_factory
class RtioBenchmarkLatencyCoreRtio(DaxClient, EnvExperiment):
    """Core-RTIO latency benchmark."""

    def build(self):
        # Arguments
        time_kwargs = {'unit': 'ns', 'scale': 1, 'step': 1, 'ndecimals': 0}
        number_kwargs = {'scale': 1, 'step': 1, 'ndecimals': 0}
        self.setattr_argument('latency_min', NumberValue(600, min=1, **time_kwargs))
        self.setattr_argument('latency_max', NumberValue(1300, min=1, **time_kwargs))
        self.setattr_argument('latency_step', NumberValue(1, min=1, **time_kwargs))
        self.setattr_argument('num_samples', NumberValue(5, min=1, **number_kwargs))
        self.setattr_argument('no_underflow_cutoff', NumberValue(5, min=1, **number_kwargs))

        # Obtain RTIO benchmark module
        self.rtio_bench = self.registry.search_module(RtioBenchmarkModule)

    def prepare(self):
        # Scale latencies
        self.latency_min *= ns
        self.latency_max *= ns
        self.latency_step *= ns

    def run(self):
        self.success = self.rtio_bench.benchmark_latency_core_rtio(self.latency_min, self.latency_max,
                                                                   self.latency_step,
                                                                   self.num_samples, self.no_underflow_cutoff)

    def analyze(self):
        if self.success:
            # Report result
            core_rtio = dax.util.units.time_to_str(
                self.rtio_bench.get_dataset_sys(self.rtio_bench.LATENCY_CORE_RTIO_KEY))
            self.logger.info('Core-RTIO latency is {:s}'.format(core_rtio))


@dax_client_factory
class RtioBenchmarkInputBufferSize(DaxClient, EnvExperiment):
    """RTIO input buffer size benchmark."""

    def build(self):
        # Arguments
        number_kwargs = {'scale': 1, 'step': 1, 'ndecimals': 0}
        self.setattr_argument('max_events', NumberValue(500, min=1, **number_kwargs))

        # Obtain RTIO benchmark module
        self.rtio_bench = self.registry.search_module(RtioLoopBenchmarkModule)

    def run(self):
        self.success = self.rtio_bench.benchmark_input_buffer_size(self.max_events)

    def analyze(self):
        if self.success:
            # Report result
            input_buffer_size = self.rtio_bench.get_dataset_sys(self.rtio_bench.INPUT_BUFFER_SIZE_KEY)
            self.logger.info('RTIO input buffer size is {:d}'.format(input_buffer_size))


@dax_client_factory
class RtioBenchmarkLatencyRtioCore(DaxClient, EnvExperiment):
    """RTIO-core and RTIO-RTIO latency benchmark."""

    def build(self):
        # Arguments
        number_kwargs = {'scale': 1, 'step': 1, 'ndecimals': 0}
        time_kwargs = {'unit': 'ns', 'scale': 1, 'step': 1, 'ndecimals': 0}
        self.setattr_argument('num_samples', NumberValue(100, min=1, **number_kwargs))
        self.setattr_argument('detection_window', NumberValue(1000, min=1, **time_kwargs))

        # Obtain RTIO benchmark module
        self.rtio_bench = self.registry.search_module(RtioLoopBenchmarkModule)

    def prepare(self):
        # Scale detection window
        self.detection_window *= ns

    def run(self):
        self.success = self.rtio_bench.benchmark_latency_rtio_core(self.num_samples, self.detection_window)

    def analyze(self):
        if self.success:
            # Report result
            rtio_core = dax.util.units.time_to_str(
                self.rtio_bench.get_dataset_sys(self.rtio_bench.LATENCY_RTIO_CORE_KEY))
            rtio_rtio = dax.util.units.time_to_str(
                self.rtio_bench.get_dataset_sys(self.rtio_bench.LATENCY_RTIO_RTIO_KEY))
            self.logger.info('RTIO-core latency is {:s}'.format(rtio_core))
            self.logger.info('RTIO-RTIO latency is {:s}'.format(rtio_rtio))


@dax_client_factory
class RtioBenchmarkLatencyRtt(DaxClient, EnvExperiment):
    """RTT RTIO-core-RTIO latency benchmark."""

    def build(self):
        # Arguments
        time_kwargs = {'unit': 'ns', 'scale': 1, 'step': 1, 'ndecimals': 0}
        number_kwargs = {'scale': 1, 'step': 1, 'ndecimals': 0}
        self.setattr_argument('latency_min', NumberValue(600, min=1, **time_kwargs))
        self.setattr_argument('latency_max', NumberValue(1300, min=1, **time_kwargs))
        self.setattr_argument('latency_step', NumberValue(1, min=1, **time_kwargs))
        self.setattr_argument('num_samples', NumberValue(5, min=1, **number_kwargs))
        self.setattr_argument('detection_window', NumberValue(1000, min=1, **time_kwargs))
        self.setattr_argument('no_underflow_cutoff', NumberValue(5, min=1, **number_kwargs))

        # Obtain RTIO benchmark module
        self.rtio_bench = self.registry.search_module(RtioLoopBenchmarkModule)

    def prepare(self):
        # Scale latencies
        self.latency_min *= ns
        self.latency_max *= ns
        self.latency_step *= ns
        # Scale detection window
        self.detection_window *= ns

    def run(self):
        self.success = self.rtio_bench.benchmark_latency_rtt(self.latency_min, self.latency_max, self.latency_step,
                                                             self.num_samples, self.detection_window,
                                                             self.no_underflow_cutoff)

    def analyze(self):
        if self.success:
            # Report result
            rtt = dax.util.units.time_to_str(self.rtio_bench.get_dataset_sys(self.rtio_bench.LATENCY_RTT_KEY))
            self.logger.info('RTIO RTT is {:s}'.format(rtt))
