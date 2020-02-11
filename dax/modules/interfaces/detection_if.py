import abc

from dax.base import DaxModuleInterface


class DetectionInterface(DaxModuleInterface):
    """Interface for a detection module."""

    """Abstract module functionality"""

    @abc.abstractmethod
    def set_detection_laser_o(self, state):
        """Set the output state of the detection laser."""
        pass

    @abc.abstractmethod
    def detection_laser_on(self):
        """Switch detection laser on."""
        pass

    @abc.abstractmethod
    def detection_laser_off(self):
        """Switch detection laser off."""
        pass

    @abc.abstractmethod
    def detect_all(self):
        """Detect events on all channels."""
        pass

    @abc.abstractmethod
    def detect_active(self):
        """Detect events on active channels."""
        pass

    @abc.abstractmethod
    def count_all(self, detection_window_mu):
        """Return a list of counts for all channels."""
        pass

    @abc.abstractmethod
    def count_active(self, detection_window_mu):
        """Return a list of counts for active channels."""
        pass

    @abc.abstractmethod
    def measure_all(self, detection_window_mu):
        """Return a list of measurements for all channels."""
        pass

    @abc.abstractmethod
    def measure_active(self, detection_window_mu):
        """Return a list of measurements for active channels."""
        pass

    @abc.abstractmethod
    def detect(self):
        """Convenient alias of detect_active() to use with measure()."""
        pass

    @abc.abstractmethod
    def measure(self, detection_window_mu):
        """Convenience alias of measure_active() to use with detect()."""
        pass

    """Abstract module configuration"""

    @abc.abstractmethod
    def set_active_channels(self, active_pmt_channels):
        """Set a new list of active channels."""
        pass
