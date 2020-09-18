import numpy as np

from dax.experiment import *
from dax.modules.rtio_benchmark import RtioBenchmarkModule, RtioLoopBenchmarkModule
import dax.util.units

__all__ = ['RtioBenchmarkEventThroughput', 'RtioBenchmarkEventBurst', 'RtioBenchmarkDmaThroughput',
           'RtioBenchmarkLatencyCoreRtio', 'RtioBenchmarkInputBufferSize',
           'RtioBenchmarkLatencyRtioCore', 'RtioBenchmarkLatencyRtt']


@dax_client_factory
class RtioBenchmarkEventThroughput(DaxClient, EnvExperiment):
    """RTIO event throughput benchmark."""

    def build(self):
        # Arguments
        time_kwargs = {'unit': 'ns', 'scale': 1, 'step': 1, 'ndecimals': 0}
        number_kwargs = {'scale': 1, 'step': 1, 'ndecimals': 0}
        self.setattr_argument('period_min', NumberValue(200, min=1, **time_kwargs))
        self.setattr_argument('period_max', NumberValue(1000, min=1, **time_kwargs))
        self.setattr_argument('period_step', NumberValue(1, min=1, **time_kwargs))
        self.setattr_argument('num_samples', NumberValue(5, min=1, **number_kwargs))
        self.setattr_argument('num_events', NumberValue(100000, min=1000, **number_kwargs))
        self.setattr_argument('no_underflow_cutoff', NumberValue(5, min=1, **number_kwargs))

        # Obtain RTIO benchmark module
        self.rtio_bench = self.registry.find_module(RtioBenchmarkModule)

    def prepare(self):
        # Additional check if arguments are valid
        if self.period_min > self.period_max:
            msg = 'Minimum period must be smaller than maximum period'
            self.logger.error(msg)
            raise ValueError(msg)

        # Create scan object
        self.period_scan = np.arange(self.period_min, self.period_max, self.period_step, dtype=float)
        self.period_scan *= ns
        self.update_kernel_invariants('period_scan')

    def run(self):
        self.rtio_bench.benchmark_event_throughput(period_scan=self.period_scan,
                                                   num_samples=self.num_samples,
                                                   num_events=self.num_events,
                                                   no_underflow_cutoff=self.no_underflow_cutoff)

    def analyze(self):
        # Report result
        last_period = self.rtio_bench.get_dataset_sys(self.rtio_bench.EVENT_PERIOD_KEY)
        throughput = dax.util.units.freq_to_str(1.0 / last_period)
        period = f'{dax.util.units.time_to_str(last_period)} period'
        self.logger.info(f'RTIO event throughput is {throughput} ({period})')


@dax_client_factory
class RtioBenchmarkEventBurst(DaxClient, EnvExperiment):
    """RTIO event burst benchmark."""

    def build(self):
        # Arguments
        number_kwargs = {'scale': 1, 'step': 1, 'ndecimals': 0}
        time_kwargs = {'unit': 'ns', 'scale': 1, 'step': 1, 'ndecimals': 0}
        self.setattr_argument('num_events_min', NumberValue(1000, min=1, **number_kwargs))
        self.setattr_argument('num_events_max', NumberValue(100000, min=1, **number_kwargs))
        self.setattr_argument('num_events_step', NumberValue(100, min=1, **number_kwargs))
        self.setattr_argument('num_samples', NumberValue(5, min=1, **number_kwargs))
        self.setattr_argument('period_step', NumberValue(1, min=1, **time_kwargs))
        self.setattr_argument('no_underflow_cutoff', NumberValue(5, min=1, **number_kwargs))
        self.setattr_argument('num_step_cutoff', NumberValue(5, min=0, **number_kwargs))

        # Obtain RTIO benchmark module
        self.rtio_bench = self.registry.find_module(RtioBenchmarkModule)

    def prepare(self):
        # Scale period step
        self.period_step *= ns

    def run(self):
        self.rtio_bench.benchmark_event_burst(num_events_min=self.num_events_min,
                                              num_events_max=self.num_events_max,
                                              num_events_step=self.num_events_step,
                                              num_samples=self.num_samples,
                                              period_step=self.period_step,
                                              no_underflow_cutoff=self.no_underflow_cutoff,
                                              num_step_cutoff=self.num_step_cutoff)

    def analyze(self):
        # Report result
        last_num_events = self.rtio_bench.get_dataset_sys(self.rtio_bench.EVENT_BURST_KEY)
        self.logger.info(f'RTIO event burst size is {last_num_events}')


@dax_client_factory
class RtioBenchmarkDmaThroughput(DaxClient, EnvExperiment):
    """RTIO DMA throughput benchmark."""

    def build(self):
        # Arguments
        time_kwargs = {'unit': 'ns', 'scale': 1, 'step': 1, 'ndecimals': 0}
        number_kwargs = {'scale': 1, 'step': 1, 'ndecimals': 0}
        self.setattr_argument('period_min', NumberValue(1000, min=1, **time_kwargs))
        self.setattr_argument('period_max', NumberValue(5000, min=1, **time_kwargs))
        self.setattr_argument('period_step', NumberValue(100, min=1, **time_kwargs))
        self.setattr_argument('num_samples', NumberValue(5, min=1, **number_kwargs))
        self.setattr_argument('num_events', NumberValue(100000, min=1000, **number_kwargs))
        self.setattr_argument('no_underflow_cutoff', NumberValue(5, min=1, **number_kwargs))

        # Obtain RTIO benchmark module
        self.rtio_bench = self.registry.find_module(RtioBenchmarkModule)

    def prepare(self):
        # Additional check if arguments are valid
        if self.period_min > self.period_max:
            msg = 'Minimum period must be smaller than maximum period'
            self.logger.error(msg)
            raise ValueError(msg)

        # Create scan object
        self.period_scan = np.arange(self.period_min, self.period_max, self.period_step, dtype=float)
        self.period_scan *= ns
        self.update_kernel_invariants('period_scan')

    def run(self):
        self.rtio_bench.benchmark_dma_throughput(period_scan=self.period_scan,
                                                 num_samples=self.num_samples,
                                                 num_events=self.num_events,
                                                 no_underflow_cutoff=self.no_underflow_cutoff)

    def analyze(self):
        # Report result
        last_period = self.rtio_bench.get_dataset_sys(self.rtio_bench.DMA_EVENT_PERIOD_KEY)
        throughput = dax.util.units.freq_to_str(1.0 / last_period)
        period = f'{dax.util.units.time_to_str(last_period)} period'
        self.logger.info(f'RTIO DMA event throughput is {throughput} ({period})')


@dax_client_factory
class RtioBenchmarkLatencyCoreRtio(DaxClient, EnvExperiment):
    """Core-RTIO latency benchmark."""

    def build(self):
        # Arguments
        time_kwargs = {'unit': 'ns', 'scale': 1, 'step': 1, 'ndecimals': 0}
        number_kwargs = {'scale': 1, 'step': 1, 'ndecimals': 0}
        self.setattr_argument('latency_min', NumberValue(500, min=1, **time_kwargs))
        self.setattr_argument('latency_max', NumberValue(2000, min=1, **time_kwargs))
        self.setattr_argument('latency_step', NumberValue(10, min=1, **time_kwargs))
        self.setattr_argument('num_samples', NumberValue(5, min=1, **number_kwargs))
        self.setattr_argument('no_underflow_cutoff', NumberValue(5, min=1, **number_kwargs))

        # Obtain RTIO benchmark module
        self.rtio_bench = self.registry.find_module(RtioBenchmarkModule)

    def prepare(self):
        # Scale latencies
        self.latency_min *= ns
        self.latency_max *= ns
        self.latency_step *= ns

    def run(self):
        self.rtio_bench.benchmark_latency_core_rtio(latency_min=self.latency_min,
                                                    latency_max=self.latency_max,
                                                    latency_step=self.latency_step,
                                                    num_samples=self.num_samples,
                                                    no_underflow_cutoff=self.no_underflow_cutoff)

    def analyze(self):
        # Report result
        core_rtio = dax.util.units.time_to_str(self.rtio_bench.get_dataset_sys(self.rtio_bench.LATENCY_CORE_RTIO_KEY))
        self.logger.info(f'Core-RTIO latency is {core_rtio}')


@dax_client_factory
class RtioBenchmarkInputBufferSize(DaxClient, EnvExperiment):
    """RTIO input buffer size benchmark."""

    def build(self):
        # Arguments
        number_kwargs = {'scale': 1, 'step': 1, 'ndecimals': 0}
        self.setattr_argument('min_events', NumberValue(1, min=1, **number_kwargs))
        self.setattr_argument('max_events', NumberValue(5000, min=1, **number_kwargs))

        # Obtain RTIO benchmark module
        self.rtio_bench = self.registry.find_module(RtioLoopBenchmarkModule)

    def prepare(self):
        # Additional check if arguments are valid
        if self.min_events > self.max_events:
            msg = 'Minimum number of events must be smaller than maximum number of events'
            self.logger.error(msg)
            raise ValueError(msg)

    def run(self):
        self.rtio_bench.benchmark_input_buffer_size(min_events=self.min_events,
                                                    max_events=self.max_events)

    def analyze(self):
        # Report result
        input_buffer_size = self.rtio_bench.get_dataset_sys(self.rtio_bench.INPUT_BUFFER_SIZE_KEY)
        self.logger.info(f'RTIO input buffer size is {input_buffer_size}')


@dax_client_factory
class RtioBenchmarkLatencyRtioCore(DaxClient, EnvExperiment):
    """RTIO-core and RTIO-RTIO latency benchmark."""

    def build(self):
        # Arguments
        number_kwargs = {'scale': 1, 'step': 1, 'ndecimals': 0}
        self.setattr_argument('num_samples', NumberValue(100, min=1, **number_kwargs))

        # Obtain RTIO benchmark module
        self.rtio_bench = self.registry.find_module(RtioLoopBenchmarkModule)

    def run(self):
        self.rtio_bench.benchmark_latency_rtio_core(num_samples=self.num_samples)

    def analyze(self):
        # Report result
        rtio_rtio = dax.util.units.time_to_str(self.rtio_bench.get_dataset_sys(self.rtio_bench.LATENCY_RTIO_RTIO_KEY))
        rtio_core = dax.util.units.time_to_str(self.rtio_bench.get_dataset_sys(self.rtio_bench.LATENCY_RTIO_CORE_KEY))
        self.logger.info(f'RTIO-RTIO latency is {rtio_rtio}')
        self.logger.info(f'RTIO-core latency is {rtio_core}')


@dax_client_factory
class RtioBenchmarkLatencyRtt(DaxClient, EnvExperiment):
    """RTT RTIO-core-RTIO latency benchmark."""

    def build(self):
        # Arguments
        time_kwargs = {'unit': 'ns', 'scale': 1, 'step': 1, 'ndecimals': 0}
        number_kwargs = {'scale': 1, 'step': 1, 'ndecimals': 0}
        self.setattr_argument('latency_min', NumberValue(1000, min=1, **time_kwargs))
        self.setattr_argument('latency_max', NumberValue(3000, min=1, **time_kwargs))
        self.setattr_argument('latency_step', NumberValue(1, min=1, **time_kwargs))
        self.setattr_argument('num_samples', NumberValue(5, min=1, **number_kwargs))
        self.setattr_argument('no_underflow_cutoff', NumberValue(5, min=1, **number_kwargs))

        # Obtain RTIO benchmark module
        self.rtio_bench = self.registry.find_module(RtioLoopBenchmarkModule)

    def prepare(self):
        # Scale latencies
        self.latency_min *= ns
        self.latency_max *= ns
        self.latency_step *= ns

    def run(self):
        self.rtio_bench.benchmark_latency_rtt(latency_min=self.latency_min,
                                              latency_max=self.latency_max,
                                              latency_step=self.latency_step,
                                              num_samples=self.num_samples,
                                              no_underflow_cutoff=self.no_underflow_cutoff)

    def analyze(self):
        # Report result
        rtt = dax.util.units.time_to_str(self.rtio_bench.get_dataset_sys(self.rtio_bench.LATENCY_RTT_KEY))
        self.logger.info(f'RTIO RTT is {rtt}')
