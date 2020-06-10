import abc
import typing

import artiq.coredevice.edge_counter

from dax.base.interface import DaxInterface

__all__ = ['DetectionInterface']


class DetectionInterface(DaxInterface, abc.ABC):

    @abc.abstractmethod
    def get_pmt_array(self) -> typing.List[artiq.coredevice.edge_counter.EdgeCounter]:
        """Get the array of PMT channels.

        :return: A list with EdgeCounter objects
        """
        pass

    @abc.abstractmethod
    def get_state_detection_threshold(self) -> int:
        """Get the state detection threshold.

        :return: State detection threshold
        """
        pass
