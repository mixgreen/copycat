import numpy as np
import scipy.signal

from dax.experiment import *
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
        self.update_kernel_invariants('num_channels')

    def init(self):
        self.setattr_dataset_sys('sample_period', 150 * us)
        self.setattr_dataset_sys('per_ion_cutoff_time', 1 * s)

    def post_init(self):
        pass

    def load_ions(self, num_ions, fallback=True):
        # Default is fast-detect loading
        self.load_ions_fast_detect(num_ions, fallback)

    def load_ions_fast_detect(self, num_ions, fallback=True):
        # Convert data types
        num_ions = np.int32(num_ions)

        # Check input
        if not num_ions > 0:
            raise ValueError('Number of ions to load must be larger than 0')

        # Call load ions fast kernel
        last_pmt_counts = self._load_ions_fast_detect(num_ions)
        # TODO, should we support unloading?

        # Process the last data on the host
        simple_count = self._count_ions_edge_detection(last_pmt_counts)
        active_channels = self._find_active_channels_scipy_find_peaks(last_pmt_counts)

        # Check if results are consistent
        # TODO, also check with camera feed?
        if simple_count != len(active_channels):
            msg = 'Fast-detect ion loading procedure encountered inconsistent results'
            if fallback:
                self.logger.warning('{:s}, falling back to slow-detect ion loading'.format(msg))
                self.load_ions_slow_detect(num_ions, fallback)
            else:
                self.logger.error(msg)
                raise RuntimeError(msg)

        # Check if we reached desired result
        if simple_count < num_ions:
            msg = 'Fast-detect ion loading procedure was not able to load {:d} ions'.format(num_ions)
            self.logger.error(msg)
            raise RuntimeError(msg)

        # Store active channels
        self.detect.set_active_channels(active_channels)
        # Info message
        self.logger.info('Loaded {:d} ions with fast-detect'.format(num_ions))

    @kernel
    def _load_ions_fast_detect(self, num_ions):
        # Reset the core
        self.core.reset()

        # TODO, set trap potential and understand how many ions we can hold
        # Pre-loading
        self._pre_loading_sequence()

        # Start first detection window
        ion_count = np.int32(0)
        t_old = self.detect.detect_all(self.sample_period)
        t_cutoff = t_old + (self.core.seconds_to_mu(self.per_ion_cutoff_time) * num_ions)  # Calculated cutoff time

        # Keep sampling the detectors until we count enough ions or pass the cutoff time
        while ion_count < num_ions and t_old < t_cutoff:
            # Start new detection window
            t_new = self.detect.detect_all(self.sample_period, pulse_laser=False)
            # Store PMT counts of old detection window
            old_pmt_counts = self.detect.count_all(t_old)
            # Move timestamps
            t_old = t_new
            # Process old PMT counts
            ion_count = self._count_ions_edge_detection(old_pmt_counts)

        # Post-loading
        self._post_loading_sequence()

        # There was one new detection window we did not checked yet
        last_pmt_counts = self.detect.count_all(t_old)
        # Return last counts for final check on host
        return last_pmt_counts

    @kernel
    def _pre_loading_sequence(self):
        # Configure global beam
        self.gbeam.sw_brc()
        delay(1 * us)

        # Enable devices
        self.trap.oven_on()
        self.trap.cool_on()
        self.trap.ion_on()
        self.gbeam.on()
        self.detect.detection_laser_on()

    @kernel
    def _post_loading_sequence(self):
        # Disable devices
        self.trap.oven_off()
        self.trap.cool_off()
        self.trap.ion_off()
        self.gbeam.off()
        self.detect.detection_laser_off()

    def load_ions_slow_detect(self, num_ions, fallback=True):
        # Convert data types
        num_ions = np.int32(num_ions)

        # Check input
        if not num_ions > 0:
            raise ValueError('Number of ions to load must be larger than 0')

        # Call load ions fast kernel
        last_pmt_counts = self._load_ions_slow_detect(num_ions)
        # TODO, should we support unloading?

        # Process the last data on the host
        active_channels = self._find_active_channels_scipy_find_peaks(last_pmt_counts)

        # Check if results are consistent
        # TODO, check with camera feed? potentially fallback on camera detect loading?

        # Check if we reached desired result
        if simple_count < num_ions:
            msg = 'Slow-detect ion loading procedure was not able to load {:d} ions'.format(num_ions)
            self.logger.error(msg)
            raise RuntimeError(msg)

        # Store active channels
        self.detect.set_active_channels(active_channels)
        # Info message
        self.logger.info('Loaded {:d} ions with fast-detect'.format(num_ions))

    @kernel
    def _load_ions_slow_detect(self, num_ions):
        # Reset the core
        self.core.reset()

        # TODO, set trap potential and understand how many ions we can hold
        # Pre-loading
        self._pre_loading_sequence()

        # Start first detection window
        ion_count = np.int32(0)
        t_cutoff = now_mu() + (self.core.seconds_to_mu(self.per_ion_cutoff_time) * num_ions)  # Calculated cutoff time

        # Keep sampling the detectors until we count enough ions or pass the cutoff time
        while ion_count < num_ions and now_mu() < t_cutoff:
            # Start detection window
            t = self.detect.detect_all(self.sample_period, pulse_laser=False)
            # Store PMT counts (results in negative slack)
            pmt_counts = self.detect.count_all(t)
            # Process PMT counts (RPC call)
            ion_count = self._count_ions_scipy_find_peaks(pmt_counts)
            # Regain slack
            self.core.break_realtime()

        # Take one last sample just before the post-loading sequence for a final check on the host
        t = self.detect.detect_all(self.sample_period, pulse_laser=False)
        # Post-loading
        self._post_loading_sequence()
        # Get last PMT counts
        pmt_counts = self.detect.count_all(t)

        # Return last counts for final check on host
        return pmt_counts

    """Algorithms for ion counting and finding active channels"""

    @portable
    def _count_ions_threshold(self, counts):
        # TODO, count channels passing a threshold (threshold might be based on detection time)
        pass

    @portable
    def _count_ions_edge_detection(self, counts):
        # Find max
        counts_max = np.int32(0)
        for c in counts:
            if c > counts_max:
                counts_max = c

        # Set threshold
        threshold = counts_max >> 1

        # TODO, handle case where threshold is low and therefore no ions were found

        # Count peaks / ions with edge-detection
        num_ions = np.int32(0)
        # See if first count is an "edge"
        if counts[0] > threshold:
            num_ions += 1
        # Count other edges
        for i in range(self.num_channels - 1):
            if counts[i] <= threshold < counts[i + 1]:
                # Increment counter when detecting an edge
                num_ions += 1

        return num_ions

    @staticmethod
    def _find_active_channels_scipy_find_peaks(counts) -> TList(TInt32):
        threshold = np.max(counts) / 2
        # TODO, handle case where threshold is low and therefore no ions were found
        peaks, _ = scipy.signal.find_peaks(counts, height=threshold, distance=1)
        return peaks.astype(np.int32)

    def _count_ions_scipy_find_peaks(self, counts) -> TInt32:
        return np.int32(self._find_active_channels_scipy_find_peaks(counts).size)
