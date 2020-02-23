import abc

from dax.base import DaxModuleInterface


class DetectionInterface(DaxModuleInterface):
    """Interface for a detection module."""

    """Abstract module functionality"""

    # TODO, could add convenience functions for single target?
    # TODO, automatically switch detection laser for some functions?

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
    def detection_laser_pulse(self, duration):
        """Pulse detection laser."""
        pass

    @abc.abstractmethod
    def detection_laser_pulse_mu(self, duration):
        """Pulse detection laser."""
        pass

    @abc.abstractmethod
    def detect_all(self, duration, pulse_laser):
        """Detect events on all channels."""
        pass

    @abc.abstractmethod
    def detect_all_mu(self, duration, pulse_laser):
        """Detect events on all channels."""
        pass

    @abc.abstractmethod
    def detect_active(self, duration):
        """Detect events on active channels."""
        pass

    @abc.abstractmethod
    def detect_active_mu(self, duration):
        """Detect events on active channels."""
        pass

    @abc.abstractmethod
    def detect_targets(self, duration, targets):
        """Detect events on specific active channels."""
        pass

    @abc.abstractmethod
    def detect_targets_mu(self, duration, targets):
        """Detect events on specific active channels."""
        pass

    @abc.abstractmethod
    def count_all(self, up_to_timestamp_mu):
        """Return a list of counts for all channels."""
        pass

    @abc.abstractmethod
    def count_active(self, up_to_timestamp_mu):
        """Return a list of counts for active channels."""
        pass

    @abc.abstractmethod
    def count_targets(self, up_to_timestamp_mu, targets):
        """Return a list of counts for specific active channels."""
        pass

    @abc.abstractmethod
    def measure_all(self, up_to_timestamp_mu):
        """Return a list of measurements for all channels."""
        pass

    @abc.abstractmethod
    def measure_active(self, up_to_timestamp_mu):
        """Return a list of measurements for active channels."""
        pass

    @abc.abstractmethod
    def measure_targets(self, up_to_timestamp_mu, targets):
        """Return a list of measurements for specific active channels."""
        pass

    @abc.abstractmethod
    def detect(self):
        """Convenient alias of detect_active() with default time and laser pulse, to use with measure()."""
        pass

    @abc.abstractmethod
    def measure(self, up_to_timestamp_mu):
        """Convenience alias of measure_active() with to use with detect()."""
        pass

    """Abstract module configuration"""

    @abc.abstractmethod
    def num_channels(self):
        """Get the number of channels."""
        pass

    @abc.abstractmethod
    def num_active_channels(self):
        """Get the number of active channels."""
        pass

    @abc.abstractmethod
    def get_active_channels(self):
        """Get a list of active channels."""
        pass

    @abc.abstractmethod
    def set_active_channels(self, active_pmt_channels):
        """Set a new list of active channels."""
        pass
