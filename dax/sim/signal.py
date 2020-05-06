import abc
import typing
import vcd.writer  # type: ignore
import operator
import numpy as np

import artiq.language.core
from artiq.language.units import *

import dax.util.units

__all__ = ['DaxSignalManager', 'NullSignalManager', 'VcdSignalManager',
           'get_signal_manager', 'set_signal_manager']


class DaxSignalManager(abc.ABC):
    """Abstract class for classes that manage simulated signals."""

    __S_T = typing.TypeVar('__S_T')  # The abstract signal type

    @abc.abstractmethod
    def register(self, scope: str, name: str, type_: type,
                 size: typing.Optional[int] = None, init: typing.Any = None) -> __S_T:
        """Method used by devices to register a signal."""
        pass

    @abc.abstractmethod
    def event(self, signal: __S_T, value: typing.Any,
              time: typing.Optional[np.int64] = None, offset: typing.Optional[np.int64] = None) -> None:
        """Method used by devices to register events."""
        pass

    @abc.abstractmethod
    def flush(self, ref_period: float) -> None:
        """Flush the output of the signal manager."""
        pass

    @abc.abstractmethod
    def close(self) -> None:
        """Close the signal manager."""
        pass


class NullSignalManager(DaxSignalManager):
    """A signal manager that does nothing."""

    def register(self, scope: str, name: str, type_: type,
                 size: typing.Optional[int] = None, init: typing.Any = None) -> typing.Any:
        pass

    def event(self, signal: typing.Any, value: typing.Any,
              time: typing.Optional[np.int64] = None, offset: typing.Optional[np.int64] = None) -> None:
        pass

    def flush(self, ref_period: float) -> None:
        pass

    def close(self) -> None:
        pass


class VcdSignalManager(DaxSignalManager):
    """VCD signal manager."""

    __S_T = typing.Union[vcd.writer.RealVariable,
                         vcd.writer.ScalarVariable,
                         vcd.writer.StringVariable,
                         vcd.writer.VectorVariable]  # The signal type
    __T_T = typing.Type[typing.Union[bool, int, np.int32, np.int64, float, str, object]]  # The signal-type type
    __V_T = typing.Union[bool, int, np.int32, np.int64, float, str]  # The value types

    _CONVERT_TYPE = {
        bool: 'reg',
        int: 'integer',
        np.int32: 'integer',
        np.int64: 'integer',
        float: 'real',
        str: 'string',
        object: 'event',
    }
    """Dict to convert Python types to VCD types."""

    def __init__(self, output_file: str, timescale: float = ns):
        assert isinstance(output_file, str), 'Output file must be of type str'
        assert isinstance(timescale, float), 'Timescale must be of type float'
        assert timescale > 0.0, 'Timescale must be > 0.0'

        # Store timescale
        self._timescale = timescale

        # Open file
        self._output_file = open(output_file, mode='w')

        # Create VCD writer
        timescale_str = dax.util.units.time_to_str(timescale, precision=0)
        self._vcd = vcd.writer.VCDWriter(self._output_file, timescale=timescale_str, comment=output_file)
        # Create event buffer to support reverting time
        self._event_buffer = []  # type: typing.List[typing.Tuple[int, VcdSignalManager.__S_T, typing.Any]]

    def register(self, scope: str, name: str, type_: __T_T,
                 size: typing.Optional[int] = None, init: typing.Any = None) -> __S_T:
        """ Register a signal.

        Signals have to be registered before any events are committed.

        Possible types and expected arguments:
        - `bool` (a register with bit values `0`, `1`, `X`, `Z`), provide a size of the register
        - `int`, `np.int32`, `np.int64`
        - `float`
        - `str`
        - `object` (an event type with no value)

        :param scope: The scope of the signal, normally the device or module name
        :param name: The name of the signal
        :param type_: The type of the signal
        :param size: The size of the data (only for type bool)
        :param init: Initial value (defaults to `X`)
        :return: The signal object to use when committing events
        """
        if type_ not in self._CONVERT_TYPE:
            raise TypeError('VCD signal manager can not handle type {}'.format(type_))

        # Get the var type
        var_type = self._CONVERT_TYPE[type_]

        # Workaround for str init values
        if type_ is str and init is None:
            init = ''  # Shows up as `Z` instead of string value 'x'

        # Register the signal with the VCD writer
        return self._vcd.register_var(scope, name, var_type=var_type, size=size, init=init)

    def event(self, signal: __S_T, value: __V_T,
              time: typing.Optional[np.int64] = None, offset: typing.Optional[np.int64] = None) -> None:
        """Commit an event.

        Note that in a parallel context, :func:`delay` and :func:`delay_mu` do not directly
        influence the time returned by :func:`now_mu`.
        It is better to use the `time` or `offset` arguments to set events at different times.

        Bool type signals can have values `0`, `1`, `X`, `Z`.

        Event (`object`) type signals represent timestamps and do not have a value.
        We recommend to always use value `True` for event type signals.

        String type signals can use value `None` which is equivalent to `Z`

        :param signal: The signal that changed
        :param value: The new value of the signal
        :param time: Optional time in machine units when the event happened (:func:`now_mu` if no time was provided)
        :param offset: Optional offset from :func:`now_mu` in machine units when the event happened (default is `0`)
        """
        # Calculate time
        time = artiq.language.core.now_mu() if time is None else time
        time = time if offset is None else time + offset

        # Add event to buffer
        self._event_buffer.append((time, signal, value))

    def flush(self, ref_period: float) -> None:
        """Commit all buffered events.

        :param ref_period: The reference period (i.e. the time of one machine unit)
        """

        # Sort the list of events (VCD writer can only handle a linear timeline)
        self._event_buffer.sort(key=operator.itemgetter(0))

        if ref_period != self._timescale:
            # Scale the timestamps if the reference period does not match the timescale
            scalar = ref_period / self._timescale
            self._event_buffer = [(int(time * scalar), signal, value) for time, signal, value in self._event_buffer]

        try:
            # Submit sorted events to the VCD writer
            for time, signal, value in self._event_buffer:
                self._vcd.change(signal, time, value)
        except vcd.writer.VCDPhaseError:
            # Occurs when we try to submit a timestamp which is earlier than the last submitted timestamp
            raise RuntimeError('Attempt to go back in time too much') from None

        # Clear the event buffer
        self._event_buffer.clear()

    def close(self) -> None:
        """Close the VCD file."""
        # Close the VCD writer
        self._vcd.close()
        # Close the file
        self._output_file.close()


_signal_manager = NullSignalManager()  # type: DaxSignalManager
"""Singleton instance of the signal manager."""


def get_signal_manager() -> DaxSignalManager:
    """Get the signal manager instance.

    The signal manager is used by simulated devices to register and change signals during simulation.

    :return: The signal manager object
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
