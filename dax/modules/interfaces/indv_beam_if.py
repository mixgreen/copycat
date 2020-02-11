import abc

from dax.base import DaxModuleInterface


class IndvBeamInterface(DaxModuleInterface):
    """Interface for an individual beam module."""

    """Abstract module functionality"""

    @abc.abstractmethod
    def set_o(self, state):
        pass

    @abc.abstractmethod
    def on(self):
        pass

    @abc.abstractmethod
    def off(self):
        pass

    @abc.abstractmethod
    def pulse(self, duration):
        pass

    @abc.abstractmethod
    def pulse_mu(self, duration):
        pass

    """Abstract module configuration"""
