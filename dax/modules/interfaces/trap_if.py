import abc

from dax.base import DaxModuleInterface


class TrapInterface(DaxModuleInterface):
    """Interface for a trap module."""

    """Abstract module functionality"""

    @abc.abstractmethod
    def set_oven_o(self, state):
        pass

    @abc.abstractmethod
    def oven_on(self):
        pass

    @abc.abstractmethod
    def oven_off(self):
        pass

    @abc.abstractmethod
    def set_cool_o(self, state):
        pass

    @abc.abstractmethod
    def cool_on(self):
        pass

    @abc.abstractmethod
    def cool_off(self):
        pass

    @abc.abstractmethod
    def cool(self, duration):
        """Cool for a specified duration."""
        pass

    @abc.abstractmethod
    def cool_mu(self, duration):
        """Cool for a specified duration in machine units."""
        pass

    @abc.abstractmethod
    def set_pump_o(self, state):
        pass

    @abc.abstractmethod
    def pump_on(self):
        pass

    @abc.abstractmethod
    def pump_off(self):
        pass

    @abc.abstractmethod
    def set_ion_o(self, state):
        pass

    @abc.abstractmethod
    def ion_on(self):
        pass

    @abc.abstractmethod
    def ion_off(self):
        pass

    @abc.abstractmethod
    def set_repump_o(self, state):
        pass

    @abc.abstractmethod
    def repump_on(self):
        pass

    @abc.abstractmethod
    def repump_off(self):
        pass

    """Abstract module configuration"""