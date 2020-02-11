import numpy as np

import artiq.coredevice.ttl
import artiq.coredevice.edge_counter

from dax.base import *
from dax.modules.interfaces.detection_if import *


class DetectionModule(DaxModule, DetectionInterface):
    """Module for ion state detection."""

    DURATION_KEY = 'duration'
    THRESHOLD_KEY = 'threshold'

    ACTIVE_PMT_CHANNELS_KEY = 'active_pmt_channels'

    def build(self, detect_sw, pmt_array, pmt_array_size):
        assert isinstance(pmt_array_size, int)

        # Detection laser switch
        self.setattr_device(detect_sw, 'detect_sw', (artiq.coredevice.ttl.TTLOut, artiq.coredevice.ttl.TTLInOut))

        # Array of PMT channels
        self.pmt_array_size = pmt_array_size
        self.pmt_array = [self.get_device('{:s}_{:d}'.format(pmt_array, i)) for i in range(self.pmt_array_size)]
        self.update_kernel_invariants('pmt_array', 'pmt_array_size')

        # Check if all PMT array devices have the same type
        pmt_device_types = {type(d) for d in self.pmt_array}
        if len(pmt_device_types) != 1:
            msg = 'All devices in the PMT array should have the same type'
            self.logger.error(msg)
            raise TypeError(msg)

        # Set parameters based on PMT array device type
        pmt_device_type = pmt_device_types.pop()
        if pmt_device_type == DummyDevice:
            pass
        elif pmt_device_type == artiq.coredevice.ttl.TTLInOut:
            self.EDGE_COUNTER = False
        elif pmt_device_type == artiq.coredevice.edge_counter.EdgeCounter:
            self.EDGE_COUNTER = True
        else:
            # Unsupported device type
            msg = 'PMT array has an unsupported device type'
            self.logger.error(msg)
            raise TypeError(msg)

        # Make flags kernel invariant
        self.update_kernel_invariants('EDGE_COUNTER')

    def load(self):
        # Duration of detection
        self.setattr_dataset_sys(self.DURATION_KEY, 150 * us)
        # Threshold for state discrimination
        self.setattr_dataset_sys(self.THRESHOLD_KEY, 1)

        # List of active PMT channels, also the mapping of ion to PMT channel (ions are ordered)
        self.setattr_dataset_sys(self.ACTIVE_PMT_CHANNELS_KEY, list(np.int32(i) for i in range(self.pmt_array_size)))

    @kernel
    def init(self):
        # Break realtime to get some slack
        self.core.break_realtime()

        # Initialize detection laser switch
        self.detect_sw.output()

        if not self.EDGE_COUNTER:
            # Configure the TTLInOut devices
            for p in self.pmt_array:
                # Set as input
                p.input()

    @kernel
    def config(self):
        # Break realtime to get some slack
        self.core.break_realtime()

        # Set detection laser switch to off
        self.set_detection_laser_o(False)

    @kernel
    def set_detection_laser_o(self, state):
        self.detect_sw.set_o(state)

    @kernel
    def detection_laser_on(self):
        self.set_detection_laser_o(True)

    @kernel
    def detection_laser_off(self):
        self.set_detection_laser_o(False)

    @kernel
    def detect_all(self):
        # Enable detection laser
        self.set_detection_laser_o(True)

        # Detect events on all channels
        with parallel:
            for i in range(self.pmt_array_size):
                self.pmt_array[i].gate_rising(self.duration)

        # Disable detection laser
        self.set_detection_laser_o(False)

        # Return the cursor at the end of the detection window
        return now_mu()

    @kernel
    def detect_active(self):
        # Enable detection laser
        self.set_detection_laser_o(True)

        # Detect events on active channels
        with parallel:
            for i in self.active_pmt_channels:
                self.pmt_array[i].gate_rising(self.duration)

        # Disable detection laser
        self.set_detection_laser_o(False)

        # Return the cursor at the end of the detection window
        return now_mu()

    @kernel
    def _pmt_array_count(self, index, detection_window_mu):
        # Get the count of a single PMT
        if self.EDGE_COUNTER:
            # EdgeCounter
            return self.pmt_array[index].fetch_count(detection_window_mu)
        else:
            # TTLInOut
            return self.pmt_array[index].count(detection_window_mu)

    @kernel
    def count_all(self, detection_window_mu):
        # Return a list of counts for all channels
        return [self._pmt_array_count(i, detection_window_mu) for i in range(self.pmt_array_size)]

    @kernel
    def count_active(self, detection_window_mu):
        # Return a list of counts for active channels
        return [self._pmt_array_count(i, detection_window_mu) for i in self.active_pmt_channels]

    @kernel
    def measure_all(self, detection_window_mu):
        # Get the counts of all channels
        counts = self.count_all(detection_window_mu)
        # Discriminate the counts based on the threshold and return the binary results
        return [c > self.threshold for c in counts]

    @kernel
    def measure_active(self, detection_window_mu):
        # Get the counts of the active channels
        counts = self.count_active(detection_window_mu)
        # Discriminate the counts based on the threshold and return the binary results
        return [c > self.threshold for c in counts]

    @kernel
    def detect(self):
        """Convenient alias of detect_active() to use with measure()."""
        return self.detect_active()

    @kernel
    def measure(self, detection_window_mu):
        """Convenience alias of measure_active() to use with detect()."""
        return self.measure_active(detection_window_mu)

    def set_active_channels(self, active_pmt_channels):
        # Set a new list of active channels
        self.set_dataset_sys(self.ACTIVE_PMT_CHANNELS_KEY, active_pmt_channels)
