import collections

from dax.experiment import *
from dax.interfaces.detection import DetectionInterface
from dax.util.ccb import get_ccb_tool

__all__ = ['PmtMonitor']


@dax_client_factory
class PmtMonitor(DaxClient, EnvExperiment):
    """PMT monitor utility to monitor a single PMT channel."""

    APPLET_NAME: str = 'pmt_monitor'
    """Name of the applet in the dashboard."""
    APPLET_GROUP: str = 'dax'
    """Group of the applet."""

    DEFAULT_DATASET: str = 'plot.dax.pmt_monitor_count'
    """Default dataset for output."""

    COUNT_SCALES = collections.OrderedDict(GHz=GHz, MHz=MHz, kHz=kHz, Hz=Hz, mHz=mHz)
    """Scales that can be used for the Y-axis."""

    def build(self) -> None:  # type: ignore
        # Obtain the detection interface
        self.detection = self.registry.find_interface(DetectionInterface)  # type: ignore[misc]
        # Obtain the PMT array
        self.pmt_array = self.detection.get_pmt_array()
        self.update_kernel_invariants('detection', 'pmt_array')
        self.logger.debug('Found PMT array with {:d} channel(s)'.format(len(self.pmt_array)))

        # Get the scheduler and CCB tool
        self.scheduler = self.get_device('scheduler')
        self.ccb = get_ccb_tool(self)
        self.update_kernel_invariants('scheduler')

        # Get max for PMT channel argument
        pmt_channel_max = len(self.pmt_array) - 1
        assert pmt_channel_max >= 0, 'PMT array can not be empty'

        # Arguments
        self.detection_window = self.get_argument("PMT detection window size",
                                                  NumberValue(default=100 * ms, unit="ms", min=0.0))
        self.detection_delay = self.get_argument("PMT detection delay",
                                                 NumberValue(default=10 * ms, unit="ms", min=0.0),
                                                 tooltip="Delay before starting detection")
        self.count_scale_label = self.get_argument("PMT count scale",
                                                   EnumerationValue(list(self.COUNT_SCALES), default='kHz'),
                                                   tooltip='Scaling factor for the PMT counts')
        self.pmt_channel = self.get_argument("PMT channel",
                                             NumberValue(default=0, step=1, min=0, max=pmt_channel_max, ndecimals=0))

        # Dataset related arguments
        self.reset_data = self.get_argument("Reset data", BooleanValue(default=True), group='Dataset')
        self.sliding_window = self.get_argument("Data window size",
                                                NumberValue(default=120 * s, unit='s', min=0, ndecimals=0, step=60),
                                                group='Dataset',
                                                tooltip='Data window size (use 0 for infinite window size)')
        self.dataset_key = self.get_argument("Dataset key", StringValue(default=self.DEFAULT_DATASET),
                                             group='Dataset')

        # Applet specific arguments
        self.create_applet = self.get_argument("Create applet", BooleanValue(default=True), group='Applet')
        self.applet_update_delay = self.get_argument("Applet update delay",
                                                     NumberValue(default=0.1 * s, unit='s', min=0.0),
                                                     group='Applet')
        self.applet_auto_close = self.get_argument("Close applet automatically", BooleanValue(default=False),
                                                   group='Applet')

        # Update kernel invariants
        self.update_kernel_invariants("detection_window", "detection_delay", "pmt_channel")

    def prepare(self) -> None:
        if self.sliding_window > 0 and self.detection_window > 0.0:
            # Convert window size to dataset size
            self.sliding_window = int(self.sliding_window / self.detection_window)
            self.logger.debug('Window size set to {:d}'.format(self.sliding_window))

        # Convert count scale
        self.count_scale = self.COUNT_SCALES[self.count_scale_label]

    def run(self) -> None:
        # NOTE: there is no dax_init() in this experiment!

        # Initial value is reset to an empty list or try to obtain the previous value defaulting to an empty list
        init_value = [] if self.reset_data else self.get_dataset(self.dataset_key, default=[], archive=False)
        self.logger.debug('Appending to previous data' if init_value else 'Starting with empty list')

        # Set the result datasets to the correct mode
        self.set_dataset(self.dataset_key, init_value, broadcast=True, archive=False)

        if self.create_applet:
            # Use the CCB to create an applet
            y_label = 'Counts per second ({:s})'.format(self.count_scale_label)
            self.ccb.plot_xy(self.APPLET_NAME, self.dataset_key, group=self.APPLET_GROUP,
                             sliding_window=self.sliding_window,
                             x_label='Sample', y_label=y_label, update_delay=self.applet_update_delay)

        try:
            # Only stop when termination is requested
            while True:
                # Monitor
                self.monitor()

                # To pause, close communications and call the pause function
                self.core.comm.close()
                self.scheduler.pause()  # Can raise a TerminationRequested exception

        except TerminationRequested:
            # Experiment was terminated, gracefully end the experiment
            self.logger.debug('Terminated gracefully')

        finally:
            if self.applet_auto_close:
                # Disable the applet
                self.ccb.disable_applet(self.APPLET_NAME, self.APPLET_GROUP)

    @kernel
    def monitor(self):  # type: () -> None
        # Reset the core
        self.core.reset()

        while True:
            # Check for pause condition and return if true
            if self.scheduler.check_pause():
                return

            # Guarantee slack
            self.core.break_realtime()

            # Insert delay
            delay(self.detection_delay)

            # Perform detection and get count
            self.pmt_array[self.pmt_channel].gate_rising(self.detection_window)
            count = self.pmt_array[self.pmt_channel].fetch_count()

            # Store obtained count
            self.store_count(count)

    @rpc(flags={'async'})
    def store_count(self, count):  # type: (int) -> None
        # Calculate value to store
        value = count / self.detection_window / self.count_scale

        # Append data to datasets
        self.append_to_dataset(self.dataset_key, value)
