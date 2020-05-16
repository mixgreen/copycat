import abc
import typing

import artiq.coredevice.edge_counter

from dax.experiment import DaxInterface

__all__ = ['DetectionInterface']


class DetectionInterface(DaxInterface, abc.ABC):

    @abc.abstractmethod
    def get_pmt_array(self) -> typing.List[artiq.coredevice.edge_counter.EdgeCounter]:
        """Get the array of PMT channels.

        :return: A list with EdgeCounter objects.
        """
        pass
