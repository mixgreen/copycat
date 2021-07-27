import numpy as np
import typing

from dax.experiment import *

__all__ = ['BeamAssignmentError', 'BeamManager']


class BeamManager(DaxModule):
    """Module for managing a set of beams shared over a number of targets.

    The beam manager contains a kernel-friendly reservation algorithm that is able to manage
    the state of a set of beams shared over a number of targets. The manager object only
    tracks an internal state, and the user is responsible for acting appropriately on the
    returned results.

    An example use case of the beam manager is a situation where a limited number of beams
    are shared over an ion chain using mirrors for targeting.
    """

    NO_BEAM: typing.ClassVar[np.int32] = np.int32(-1)
    """Constant value for no beam."""

    class _BeamConfig:
        """Class that holds the configuration/state of a beam."""

        target: np.int32
        state: bool

        def __init__(self):
            # Initially we assume the beam is not targeted
            self.target = BeamManager.NO_BEAM
            # We assume beams are initially off
            self.state = False

    _num_beams: np.int32
    _beam_config: typing.List[_BeamConfig]

    def build(self, *, num_beams: typing.Union[int, np.integer]):  # type: ignore[override]
        """Build the beam manager object.

        :param num_beams: The number of beams
        """
        assert isinstance(num_beams, (int, np.integer)) and num_beams > 0, 'Number of beams must be of type int and > 0'

        # Store attributes
        self._num_beams = np.int32(num_beams)
        self.logger.debug(f'Number of beams: {self._num_beams}')
        self.update_kernel_invariants('_num_beams')

        # Mark class constant as kernel invariant
        self.update_kernel_invariants('NO_BEAM')

        # Current beam configurations
        self._beam_config = [self._BeamConfig() for _ in range(num_beams)]
        self.update_kernel_invariants('_beam_config')

    def init(self):
        pass

    def post_init(self):
        pass

    """Module functionality"""

    @portable
    def get_beam(self, target: TInt32, state: TBool) -> TInt32:
        """Provide a target and future state, beam index will be returned.

        The beam manager will update its internal state, but the user is responsible
        for acting on the results returned by the reservation system.

        When switching to the on state:

        - Returns :attr:`NO_BEAM` if a beam was already on and targeted (no action required)
        - Returns a beam index when a beam is available but not targeted
        - Raises exception when no beam was targeted and no beams are available

        When switching to the off state:

        - Returns :attr:`NO_BEAM` if no beam is on and targeted (no action required)
        - Returns a beam index when a beam is on and targeted

        :param target: The desired beam target
        :param state: The desired state
        :return: A beam index (read above description for detailed information)
        :raises BeamAssignmentError: Raised if a new beam assignment could not be completed
        """

        if state:  # We switch a beam on, which means we need to find a beam
            # Keep track of available beams in case we do not find a beam matching the target
            available_beam = self.NO_BEAM

            for b in range(self._num_beams):
                # Search for a beam matching the target
                c = self._beam_config[b]
                if c.target == target:
                    # We found a beam matching the target
                    if c.state == state:
                        # Beam already targeted and in the correct state, nothing needs to be done
                        return self.NO_BEAM
                    else:
                        # We found a beam matching the target with a different state, update state and return beam index
                        c.state = state
                        return b
                elif not c.state:
                    # We found an available beam, store it
                    available_beam = b

            # We exhausted the search which means that no beam is targeted at the given target
            if available_beam != self.NO_BEAM:
                # Save configuration
                self._beam_config[available_beam].target = np.int32(target)  # Extra conversion for type safety
                self._beam_config[available_beam].state = state
                # Return the index of the available beam
                return available_beam
            else:
                # No beam was available
                raise BeamAssignmentError('Requested more targets than available beams')

        else:  # We switch a beam off, which means we do not have to find a beam
            for b in range(self._num_beams):
                # Search for a beam matching the target
                c = self._beam_config[b]
                if c.target == target:
                    # We found a beam matching the target
                    if c.state == state:
                        # Beam already targeted and in the correct state, nothing needs to be done
                        return self.NO_BEAM
                    else:
                        # We found a beam matching the target with a different state, update state and return beam index
                        c.state = state
                        return b

            # We exhausted the search which means that no beam is targeted at the given target
            return self.NO_BEAM

    @portable
    def set_beam(self, beam: TInt32, target: TInt32, state: TBool):
        """Update the beam information with the given target and state.

        This function can be used when manually override the reservation system
        but keeping the beam information updated.

        :param beam: The beam index to update
        :param target: The new target of the beam
        :param state: The new state of the beam
        """
        self._beam_config[beam].target = np.int32(target)  # Extra conversion for type safety
        self._beam_config[beam].state = state


class BeamAssignmentError(RuntimeError):
    """Exception type in case of errors during beam assignment."""
    pass
