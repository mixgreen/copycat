import numpy as np

from dax.experiment import *
from dax.interfaces.detection import DetectionInterface
from dax.util.ccb import get_ccb_tool

__all__ = ['PmtMonitor']


@dax_client_factory
class PmtMonitor(DaxClient, EnvExperiment):
    """PMT monitor utility to monitor a single PMT channel."""

    def build(self):
        # Obtain the detection interface
        self.detection = self.registry.find_interface(DetectionInterface)
        # Obtain the PMT array
        self.pmt_array = self.detection.get_pmt_array()
        self.update_kernel_invariants('detection', 'pmt_array')
        self.logger.debug('Found PMT array with {:d} channel(s)'.format(len(self.pmt_array)))

        # Get max for PMT channel argument
        pmt_channel_max = len(self.pmt_array) - 1
        assert pmt_channel_max >= 0, 'PMT array can not be empty'

        # Arguments
        self.count_time = self.get_argument("PMT count window size", NumberValue(default=100 * ms, unit="ms", min=0.0))
        self.count_delay = self.get_argument("PMT count delay", NumberValue(default=10 * ms, unit="ms", min=0.0))
        self.pmt_channel = self.get_argument("PMT channel",
                                             NumberValue(default=0, step=1, min=0, max=pmt_channel_max, ndecimals=0))

        # Dataset related arguments
        self.reset_data = self.get_argument("Reset data", BooleanValue(default=True), group='Dataset')
        self.window_size = self.get_argument("Data window size",
                                             NumberValue(default=120 * s, unit='s', min=0, ndecimals=0, step=60),
                                             group='Dataset',
                                             tooltip='Data window size (use 0 for infinite window size)')
        self.dataset_key = self.get_argument("Dataset key", StringValue(default="tmp.pmt_monitor_count"),
                                             group='Dataset')

        # Applet specific arguments
        self.create_applet = self.get_argument("Create applet", BooleanValue(default=True), group='Applet')
        self.count_scale = self.get_argument("PMT count scale", NumberValue(default=1000, min=1, ndecimals=0, step=100),
                                             group='Applet', tooltip='Scale for the Y-axis')
        self.applet_update_delay = self.get_argument("Applet update delay",
                                                     NumberValue(default=0.5 * s, unit='s', min=0.0),
                                                     group='Applet')

        # Update kernel invariants
        self.update_kernel_invariants("count_time", "count_delay", "pmt_channel")

    def prepare(self):
        if self.window_size > 0 and self.count_time > 0.0:
            # Convert window size to dataset size
            self.window_size = int(self.window_size / self.count_time)
            self.logger.debug('Window size set to {:d}'.format(self.window_size))

    def run(self):
        # NOTE: there is no dax_init() in this experiment!

        # Initial value is reset to an empty list or try to obtain the previous value defaulting to an empty list
        init_value = [] if self.reset_data else self.get_dataset(self.dataset_key, default=[], archive=False)
        self.logger.debug('Appending to previous data' if init_value else 'Starting with empty list')

        # Set the result datasets to the correct mode
        self.set_dataset(self.dataset_key, init_value, broadcast=True, archive=False)

        if self.create_applet:
            # Use the CCB to create an applet
            ccb = get_ccb_tool(self)
            title = 'x{:d} count(s) per second'.format(self.count_scale)
            ccb.plot_xy('PMT monitor', self.dataset_key, sliding_window=self.window_size, title=title,
                        update_delay=self.applet_update_delay)

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

    @kernel
    def monitor(self):
        # Reset the core
        self.core.reset()

        while True:
            # Check for pause condition and return if true
            if self.scheduler.check_pause():
                return

            # Guarantee slack
            self.core.break_realtime()

            # Insert delay
            delay(self.count_delay)

            # Perform detection and get count
            self.pmt_array[self.pmt_channel].gate_rising(self.count_time)
            count = self.pmt_array[self.pmt_channel].fetch_count()

            # Store obtained count
            self.store_count(count)

    @rpc(flags={'async'})
    def store_count(self, count):
        # Calculate value to store
        value = count / self.count_time / self.count_scale

        # Append data to datasets
        self.append_to_dataset(self.dataset_key, value)
