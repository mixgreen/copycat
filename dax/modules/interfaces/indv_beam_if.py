import abc

from dax.base import DaxModuleInterface


class IndvBeamInterface(DaxModuleInterface):
    """Interface for an individual beam module."""

    """Abstract module functionality"""

    # TODO, functions for _all and _active could be added later (unsupported for MEMS, supported for multi-channel AOM)
    # TODO, could add convenience functions for single target?

    @abc.abstractmethod
    def set_targets_o(self, state, targets):
        pass

    @abc.abstractmethod
    def on_targets(self, targets):
        pass

    @abc.abstractmethod
    def off_targets(self, targets):
        pass

    @abc.abstractmethod
    def pulse_targets(self, duration, targets):
        pass

    @abc.abstractmethod
    def pulse_targets_mu(self, duration, targets):
        pass

    """Abstract module configuration"""
