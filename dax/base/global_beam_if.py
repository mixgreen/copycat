import abc

from .base import DaxModuleInterface


class GlobalBeamInterface(DaxModuleInterface):
    """Interface for a global beam module."""

    """Abstract module functionality"""

    @abc.abstractmethod
    def set_sw(self, state):
        pass

    @abc.abstractmethod
    def sw_brc(self):
        pass

    @abc.abstractmethod
    def sw_z(self):
        pass

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
