import numpy as np
import scipy.signal

from dax.base import *
from dax.modules.interfaces.global_beam_if import *
from dax.modules.interfaces.detection_if import *
from dax.modules.interfaces.trap_if import *


class LoadIonsService(DaxService):
    SERVICE_NAME = 'load_ions'

    def build(self):
        # Obtain required modules
        self.gbeam = self.registry.search_module(GlobalBeamInterface)
        self.detect = self.registry.search_module_dict(DetectionInterface)
        self.trap = self.registry.search_module_dict(TrapInterface)

        # Other variables
        self.num_channels = self.detect.num_channels()
        self.num_channels_shifter = np.int32(round(np.log2(self.num_channels)))
        self.update_kernel_invariants('num_channels', 'num_channels_shifter')

    def load(self):
        self.setattr_dataset_sys('sample_period', 150 * us)

    def init(self):
        pass

    def config(self):
        pass

    def load_ions(self, num_ions):
        # Convert data types
        num_ions = np.int32(num_ions)

        # Check input
        if not num_ions > 0:
            raise ValueError('Number of ions to load must be larger than 0')

        # Call load ions kernel
        last_pmt_counts = self._load_ions(num_ions)

        # Check the data on the host
        simple_count = self._get_ion_count(last_pmt_counts)
        active_channels = self._get_active_channels(last_pmt_counts)
        if simple_count != len(active_channels):
            msg = 'Analysis on number of loaded ions returned conflicting data: ' \
                  'simple_count={:d}, find_active_channels={}'.format(simple_count, active_channels)
            self.logger.error(msg)
            raise RuntimeError(msg)

        # Store active channels
        self.detect.set_active_channels(active_channels)

        # TODO, check with camera feed?
        # TODO, what to do when loading failed or we have conflicting data?
        # TODO, should we support unloading?

    @kernel
    def _load_ions(self, num_ions):
        # Reset the core
        self.core.reset()

        # TODO, set trap potential and understand how many ions we can hold

        # Configure global beam
        self.gbeam.sw_brc()
        delay(1 * us)

        # Enable devices
        self.trap.oven_on()
        self.trap.cool_on()
        self.trap.ion_on()
        self.gbeam.on()
        self.detect.detection_laser_on()

        # Start first detection window
        ion_count = np.int32(0)
        t_old = self.detect.detect_all(self.sample_period)

        # Keep sampling the detectors until we count enough ions
        while ion_count < num_ions:
            # Start new detection window
            t_new = self.detect.detect_all(self.sample_period, pulse_laser=False)
            # Store PMT counts of old detection window
            old_pmt_counts = self.detect.count_all(t_old)
            # Move timestamps
            t_old = t_new
            # Process old PMT counts
            ion_count = self._get_ion_count(old_pmt_counts)

        # Disable devices
        self.trap.oven_off()
        self.trap.cool_off()
        self.trap.ion_off()
        self.gbeam.off()
        self.detect.detection_laser_off()

        # There was one new detection window we did not checked yet
        last_pmt_counts = self.detect.count_all(t_old)
        # Return last counts for final check on host
        return last_pmt_counts

    @portable
    def _get_ion_count(self, counts):
        """Simple portable function to quickly count number of ions."""
        return self._count_ions_edge_detection(counts)

    def _get_active_channels(self, counts):
        """More sophisticated host function to find the active channels."""
        return self._find_active_channels_scipy_find_peaks(counts)

    @portable
    def _count_ions_edge_detection(self, counts):
        # Sum of counts
        counts_sum = np.int32(0)
        for c in counts:
            counts_sum += c

        # Calculate mean
        mean = counts_sum >> self.num_channels_shifter
        # Set threshold
        threshold = mean << 1

        # Count peaks / ions with edge-detection
        num_ions = np.int32(0)
        # See if first count is an "edge"
        if counts[0] > threshold:
            num_ions += 1
        # Count other edges
        for i in self.num_channels:
            if counts[i] <= threshold < counts[i + 1]:
                # Increment counter when detecting an edge
                num_ions += 1

        return num_ions

    @staticmethod
    def _find_active_channels_scipy_find_peaks(counts):
        peaks, _ = scipy.signal.find_peaks(counts, threshold=100, distance=1)
        return np.array(peaks, dtype=np.int32)
