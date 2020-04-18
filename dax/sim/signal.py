import abc
import typing
import vcd.writer  # type: ignore
import numpy as np

import artiq.language.core
from artiq.language.units import *

import dax.util.units


class DaxSignalManager(abc.ABC):
    """Abstract class for classes that manage simulated signals."""

    # The abstract signal type
    __S_T = typing.TypeVar('__S_T')

    @abc.abstractmethod
    def register(self, scope: str, name: str, type_: type, size: typing.Optional[int] = None) -> __S_T:
        pass

    @abc.abstractmethod
    def event(self, signal: __S_T, value: typing.Any, time: typing.Optional[np.int64] = None) -> None:
        pass

    @abc.abstractmethod
    def flush(self) -> None:
        pass

    @abc.abstractmethod
    def close(self) -> None:
        pass


class NullSignalManager(DaxSignalManager):
    """A signal manager that does nothing."""

    def register(self, scope: str, name: str, type_: type, size: typing.Optional[int] = None) -> typing.Any:
        return None

    def event(self, signal: typing.Any, value: typing.Any, time: typing.Optional[np.int64] = None) -> None:
        pass

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


class VcdSignalManager(DaxSignalManager):
    """VCD signal manager."""

    # The signal type
    __S_T = typing.Union[vcd.writer.RealVariable,
                         vcd.writer.ScalarVariable,
                         vcd.writer.StringVariable,
                         vcd.writer.VectorVariable]

    # Dict to convert Python types to VCD types
    _CONVERT_TYPE: typing.Dict[type, str] = {
        bool: 'reg',
        int: 'integer',
        np.int32: 'integer',
        np.int64: 'integer',
        float: 'real',
        str: 'string',
    }

    def __init__(self, output_file: str, timescale: float):
        assert isinstance(output_file, str), 'Output file must be of type str'
        assert isinstance(timescale, float), 'Timescale must be of type float'

        # Convert timescale
        timescale_str: str = dax.util.units.time_to_str(timescale, precision=0)

        # Open file
        self._output_file = open(output_file, mode='w')

        # Create VCD writer
        self._vcd = vcd.writer.VCDWriter(self._output_file, timescale=timescale_str)

    def register(self, scope: str, name: str, type_: type, size: typing.Optional[int] = None) -> __S_T:
        if type_ not in self._CONVERT_TYPE:
            raise TypeError('VCD signal manager can not handle type {}'.format(type_))

        # Get the var type
        var_type = self._CONVERT_TYPE[type_]

        # Register the signal with the VCD writer
        return self._vcd.register_var(scope, name, var_type=var_type, size=size)

    def event(self, signal: __S_T, value: typing.Any, time: typing.Optional[np.int64] = None) -> None:
        self._vcd.change(signal, artiq.language.core.now_mu() if time is None else time, value)

    def flush(self) -> None:
        # Flush the VCD file
        self._vcd.flush()

    def close(self) -> None:
        # Close the VCD writer
        self._vcd.close()
        # Close the file
        self._output_file.close()


# Singleton instance of the signal manager
_signal_manager: DaxSignalManager = NullSignalManager()


def get_signal_manager() -> DaxSignalManager:
    """Get the signal manager instance.

    The signal manager is used by simulated devices to register and change signals during simulation.

    :returns: The signal manager object
    """
    return _signal_manager


def set_signal_manager(signal_manager: DaxSignalManager) -> None:
    """Set a new signal manager.

    The old signal manager will be closed.

    :param signal_manager: The new signal manager object to use
    """

    # Close the current signal manager
    global _signal_manager
    _signal_manager.close()

    # Set the new signal manager
    _signal_manager = signal_manager
