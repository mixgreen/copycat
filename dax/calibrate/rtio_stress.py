import numpy as np

from dax.modules.rtio_stress import *
import dax.util.units


@dax_calibration_factory(RtioStressModule)
class CalibrateRtioThroughput(DaxCalibration, EnvExperiment):
    """RTIO throughput calibration (benchmark)."""

    def build(self):
        # Arguments
        period_kwargs = {'unit': 'ns', 'scale': 1, 'step': 1, 'ndecimals': 0}
        number_kwargs = {'scale': 1, 'step': 1, 'ndecimals': 0}
        self.setattr_argument('period_min', NumberValue(200, min=1, **period_kwargs))
        self.setattr_argument('period_max', NumberValue(1000, min=1, **period_kwargs))
        self.setattr_argument('period_step', NumberValue(1, min=1, **period_kwargs))
        self.setattr_argument('num_samples', NumberValue(5, min=1, **number_kwargs))
        self.setattr_argument('num_events', NumberValue(500000, min=1000, **number_kwargs))
        self.setattr_argument('no_underflow_cutoff', NumberValue(5, min=1, **number_kwargs))

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
        self.module.calibrate_throughput(self.period_scan, self.num_samples, self.num_events, self.no_underflow_cutoff)

    def analyze(self):
        # Report result
        last_period = self.module.get_dataset_sys(self.module.THROUGHPUT_PERIOD_KEY)
        throughput = dax.util.units.freq_to_str(1.0 / last_period)
        period = '{:s} period'.format(dax.util.units.time_to_str(last_period))
        self.logger.info('RTIO event throughput is {:s} ({:s})'.format(throughput, period))


@dax_calibration_factory(RtioStressModule)
class CalibrateRtioThroughputBurst(DaxCalibration, EnvExperiment):
    """RTIO throughput burst calibration (benchmark)."""

    def build(self):
        # Arguments
        number_kwargs = {'scale': 1, 'step': 1, 'ndecimals': 0}
        period_kwargs = {'unit': 'ns', 'scale': 1, 'step': 1, 'ndecimals': 0}
        self.setattr_argument('num_events_min', NumberValue(1000, min=1, **number_kwargs))
        self.setattr_argument('num_events_max', NumberValue(500000, min=1, **number_kwargs))
        self.setattr_argument('num_events_step', NumberValue(1000, min=1, **number_kwargs))
        self.setattr_argument('num_samples', NumberValue(5, min=1, **number_kwargs))
        self.setattr_argument('period_step', NumberValue(1, min=1, **period_kwargs))
        self.setattr_argument('no_underflow_cutoff', NumberValue(5, min=1, **number_kwargs))
        self.setattr_argument('num_step_cutoff', NumberValue(5, min=0, **number_kwargs))

    def prepare(self):
        # Additional check if arguments are valid
        if self.num_events_min > self.num_events_max:
            msg = 'Minimum number of events must be smaller than maximum number of events'
            self.logger.error(msg)
            raise ValueError(msg)

        # Scale period step
        self.period_step *= ns
        # Number of events scan
        self.num_event_scan = list(range(self.num_events_max, self.num_events_min, -self.num_events_step))
        self.update_kernel_invariants('num_event_scan')

    def run(self):
        self.module.calibrate_throughput_burst(self.num_event_scan, self.num_samples, self.period_step,
                                               self.no_underflow_cutoff, self.num_step_cutoff)

    def analyze(self):
        # Report result
        last_num_events = self.module.get_dataset_sys(self.module.THROUGHPUT_BURST_KEY)
        self.logger.info('RTIO event burst size is {:d}'.format(last_num_events))


@dax_calibration_factory(RtioStressModule)
class CalibrateRtioLatencyRtioCore(DaxCalibration, EnvExperiment):
    """RTIO-core latency calibration (benchmark)."""

    def build(self):
        # Arguments
        number_kwargs = {'scale': 1, 'step': 1, 'ndecimals': 0}
        time_kwargs = {'unit': 'ns', 'scale': 1, 'step': 1, 'ndecimals': 0}
        self.setattr_argument('num_samples', NumberValue(100, min=1, **number_kwargs))
        self.setattr_argument('detection_window', NumberValue(1000, min=1, **time_kwargs))

    def prepare(self):
        # Scale detection window
        self.detection_window *= ns

    def run(self):
        self.module.calibrate_latency_rtio_core(self.num_samples, self.detection_window)

    def analyze(self):
        # Report result
        rtio_rtio = dax.util.units.time_to_str(self.module.get_dataset_sys(self.module.LATENCY_RTIO_RTIO))
        rtio_core = dax.util.units.time_to_str(self.module.get_dataset_sys(self.module.LATENCY_RTIO_CORE))
        self.logger.info('RTIO-RTIO latency is {:s}'.format(rtio_rtio))
        self.logger.info('RTIO-core latency is {:s}'.format(rtio_core))
