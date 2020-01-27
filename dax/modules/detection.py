import numpy as np

from dax.base import *


class DetectionModule(DaxModule):
    """Module for ion state detection."""

    DURATION_KEY = 'duration'
    THRESHOLD_KEY = 'threshold'

    ACTIVE_PMT_CHANNELS_KEY = 'active_pmt_channels'

    DETECT_AOM_FREQ_KEY = 'detect_aom_freq'
    DETECT_AOM_PHASE_KEY = 'detect_aom_phase'
    DETECT_AOM_ATT_KEY = 'detect_aom_att'

    def build(self, aom, pmt_array, pmt_array_size):
        # Detection AOM
        self.setattr_device(aom, 'detect_aom')

        # Array of PMT channels
        self.pmt_array_size = pmt_array_size
        self.pmt_array = [self.get_device('{:s}_{:d}'.format(pmt_array, i)) for i in range(self.pmt_array_size)]
        self.update_kernel_invariants('pmt_array', 'pmt_array_size')

    def load_module(self):
        # Duration of detection
        self.setattr_dataset_sys(self.DURATION_KEY, 150 * us)
        # Threshold for state discrimination
        self.setattr_dataset_sys(self.THRESHOLD_KEY, 1)

        # List of active PMT channels, also the mapping of ion to PMT channel (ions are ordered)
        self.setattr_dataset_sys(self.ACTIVE_PMT_CHANNELS_KEY, np.arange(self.pmt_array_size, dtype=np.int32))

        # Detection AOM frequency, phase, and attenuation
        self.setattr_dataset_sys(self.DETECT_AOM_FREQ_KEY, 100 * MHz)
        self.setattr_dataset_sys(self.DETECT_AOM_PHASE_KEY, 0.0)
        self.setattr_dataset_sys(self.DETECT_AOM_ATT_KEY, 0.0 * dB)

    @kernel
    def init_module(self):
        # Break realtime to get some slack
        self.core.break_realtime()

        # Initialize and set detection AOM
        self.detect_aom.init()
        self.detect_aom.cfg_sw(0)
        self.detect_aom.set(self.detect_aom_freq, phase=self.detect_aom_phase)
        self.detect_aom.set_att(self.detect_aom_att)

        # Configure the PMT channels
        self._config_pmt_channels()

    def post_init_module(self):
        pass

    @kernel
    def _config_pmt_channels(self):
        # Configure the TTLInOut devices
        for p in self.pmt_array:
            # Set as input
            p.input()

    @kernel
    def _pmt_array_count(self, index, detection_window_mu):
        # Get the count of a single PMT
        return self.pmt_array[index].count(detection_window_mu)

    @kernel
    def detect_all(self):
        # Enable detection AOM
        self.detect_aom.cfg_sw(1)

        # Detect events on all channels
        with parallel:
            for i in range(self.pmt_array_size):
                self.pmt_array[i].gate_rising(self.duration)

        # Disable detection AOM
        self.detect_aom.cfg_sw(0)

        # Return the cursor at the end of the detection window
        return now_mu()

    @kernel
    def detect_active(self):
        # Enable detection AOM
        self.detect_aom.cfg_sw(1)

        # Detect events on active channels
        with parallel:
            for i in self.active_pmt_channels:
                self.pmt_array[i].gate_rising(self.duration)

        # Disable detection AOM
        self.detect_aom.cfg_sw(0)

        # Return the cursor at the end of the detection window
        return now_mu()

    @kernel
    def count_all(self, detection_window_mu):
        # Return a list of counts for all channels
        return [self._pmt_array_count(i, detection_window_mu) for i in range(self.pmt_array_size)]

    @kernel
    def count_active(self, detection_window_mu):
        # Return a list of counts for active channels
        return [self._pmt_array_count(i, detection_window_mu) for i in self.active_pmt_channels]

    @kernel
    def detect(self):
        """Convenient alias of detect_active() to use with measure()."""
        return self.detect_active()

    @kernel
    def measure(self, detection_window_mu):
        """Get measurement results after a call to detect()."""

        # Get the counts of the active channels
        counts = self.count_active(detection_window_mu)
        # Discriminate the counts based on the threshold and return the binary results
        return [c > self.threshold for c in counts]

    def set_active_channels(self, active_pmt_channels):
        # Set a new list of active channels
        self.set_dataset_sys(self.ACTIVE_PMT_CHANNELS_KEY, active_pmt_channels)


class DetectionModuleEc(DetectionModule):
    """Module for ion state detection. (For EdgeCounter devices)"""

    @kernel
    def _config_pmt_channels(self):
        # EdgeCounter devices do not require any configuration
        pass

    @kernel
    def _pmt_array_count(self, index, detection_window_mu):
        # Get the count of a single PMT
        return self.pmt_array[index].fetch_count(detection_window_mu)
