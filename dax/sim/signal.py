import abc
import typing
import vcd.writer
import operator
import numpy as np
import datetime

import artiq.language.core
from artiq.language.units import *

from dax.sim.device import DaxSimDevice
from dax import __version__ as _dax_version
import dax.util.units

__all__ = ['DaxSignalManager', 'NullSignalManager', 'VcdSignalManager', 'PeekSignalManager', 'SignalNotSet',
           'get_signal_manager', 'set_signal_manager']

_S_T = typing.TypeVar('_S_T')  # The abstract signal type


class DaxSignalManager(abc.ABC, typing.Generic[_S_T]):
    """Abstract class for classes that manage simulated signals."""

    @abc.abstractmethod
    def register(self, scope: DaxSimDevice, name: str, type_: type, *,
                 size: typing.Optional[int] = None, init: typing.Any = None) -> _S_T:
        """Register a signal.

        Signals have to be registered before any events are committed.

        Possible types and expected arguments:

        - ``bool`` (a register with bit values ``0``, ``1``, ``'X'``, ``'Z'``), provide a size of the register
        - ``int``, ``np.int32``, ``np.int64``
        - ``float``
        - ``str``
        - ``object`` (an event type with no value)

        :param scope: The scope of the signal, which is the device object
        :param name: The name of the signal
        :param type_: The type of the signal
        :param size: The size of the data (only for type bool)
        :param init: Initial value (defaults to ``'X'``)
        :return: The signal object to use when committing events
        """
        pass

    @abc.abstractmethod
    def event(self, signal: _S_T, value: typing.Any, *,
              time: typing.Optional[np.int64] = None, offset: typing.Optional[np.int64] = None) -> None:
        """Commit an event.

        Note that in a parallel context, :func:`delay` and :func:`delay_mu` do not directly
        influence the time returned by :func:`now_mu`.
        It is better to use the time or offset parameters to set events at different times.

        Bool type signals can have values ``0``, ``1``, ``'X'``, ``'Z'``.
        A vector of a bool type signal has a value of type ``str`` (e.g. ``'1001XZ'``).

        Integer type variables can have any int value or any value legal for a bool type signal.

        Float type variables can only be assigned float values.

        Event (``object``) type signals represent timestamps and do not have a value.
        We recommend to always use value :const:`True` for event type signals.

        String type signals can use value :const:`None` which is equivalent to ``'Z'``.

        :param signal: The signal that changed
        :param value: The new value of the signal
        :param time: Optional time in machine units when the event happened (:func:`now_mu` if no time was provided)
        :param offset: Optional offset from the given time in machine units (default is :const:`0`)
        """
        pass

    @abc.abstractmethod
    def flush(self, ref_period: float) -> None:
        """Flush the output of the signal manager.

        :param ref_period: The reference period (i.e. the time of one machine unit)
        """
        pass

    @abc.abstractmethod
    def close(self) -> None:
        """Close the signal manager."""
        pass

    # noinspection PyMethodMayBeStatic
    def _get_timestamp(self, time: typing.Optional[np.int64] = None,
                       offset: typing.Optional[np.int64] = None) -> np.int64:
        """Return the timestamp of an event."""
        if time is None:
            time = artiq.language.core.now_mu()  # noqa: ATQ101
        if offset is not None:
            time += offset
        return time


class NullSignalManager(DaxSignalManager[None]):
    """A signal manager that does nothing."""

    def register(self, scope: DaxSimDevice, name: str, type_: type, *,
                 size: typing.Optional[int] = None, init: typing.Any = None) -> None:
        pass

    def event(self, signal: None, value: typing.Any, *,
              time: typing.Optional[np.int64] = None, offset: typing.Optional[np.int64] = None) -> None:
        pass

    def flush(self, ref_period: float) -> None:
        pass

    def close(self) -> None:
        pass


_VS_T = vcd.writer.Variable[vcd.writer.VarValue]  # The VCD signal type
_VT_T = typing.Type[typing.Union[bool, int, np.int32, np.int64, float, str, object]]  # The VCD signal-type type
_VV_T = typing.Union[bool, int, np.int32, np.int64, float, str, None]  # The VCD value types


class VcdSignalManager(DaxSignalManager[_VS_T]):
    """VCD signal manager."""

    # Dict of registered signals type
    __RS_T = typing.Dict[DaxSimDevice, typing.List[typing.Tuple[str, type, typing.Optional[int]]]]

    _CONVERT_TYPE: typing.Dict[type, str] = {
        bool: 'reg',
        int: 'integer',
        np.int32: 'integer',
        np.int64: 'integer',
        float: 'real',
        str: 'string',
        object: 'event',
    }
    """Dict to convert Python types to VCD types."""

    def __init__(self, file_name: str, timescale: float = ns):
        assert isinstance(file_name, str), 'Output file name must be of type str'
        assert isinstance(timescale, float), 'Timescale must be of type float'
        assert timescale > 0.0, 'Timescale must be > 0.0'

        # Store timescale
        self._timescale: float = timescale

        # Open file
        self._vcd_file = open(file_name, mode='w')

        # Create VCD writer
        timescale_str = dax.util.units.time_to_str(timescale, precision=0)
        self._vcd = vcd.writer.VCDWriter(self._vcd_file,
                                         timescale=timescale_str,
                                         date=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                         comment=file_name,
                                         version=_dax_version)

        # Create event buffer to support reverting time
        self._event_buffer: typing.List[typing.Tuple[typing.Union[int, np.int64], _VS_T, _VV_T]] = []
        # Create a registered signals data structure
        self._registered_signals: VcdSignalManager.__RS_T = {}

    def register(self, scope: DaxSimDevice, name: str, type_: _VT_T, *,
                 size: typing.Optional[int] = None, init: _VV_T = None) -> _VS_T:
        assert isinstance(scope, DaxSimDevice), 'The scope of the signal must be of type DaxSimDevice'
        assert isinstance(name, str), 'The name of the signal must be of type str'
        assert isinstance(size, int) or size is None, 'The size must be an int or None'

        if type_ not in self._CONVERT_TYPE:
            raise TypeError(f'VCD signal manager does not support signal type {type_}')

        # Get the var type
        var_type = self._CONVERT_TYPE[type_]

        # Workaround for str init values
        if type_ is str and init is None:
            init = ''  # Shows up as `Z` instead of string value 'x'

        # Register signal locally
        self._registered_signals.setdefault(scope, []).append((name, type_, size))
        # Register the signal with the VCD writer and return the signal object
        return self._vcd.register_var(scope.key, name, var_type=var_type, size=size, init=init)

    def event(self, signal: _VS_T, value: _VV_T, *,
              time: typing.Optional[np.int64] = None, offset: typing.Optional[np.int64] = None) -> None:
        # Add event to buffer
        self._event_buffer.append((self._get_timestamp(time, offset), signal, value))

    def flush(self, ref_period: float) -> None:
        # Sort the list of events (VCD writer can only handle a linear timeline)
        self._event_buffer.sort(key=operator.itemgetter(0))
        # Get a timestamp for now
        now: typing.Union[int, np.int64] = self._get_timestamp()

        if ref_period != self._timescale:
            # Scale the timestamps if the reference period does not match the timescale
            scalar = ref_period / self._timescale
            self._event_buffer = [(int(time * scalar), signal, value) for time, signal, value in self._event_buffer]
            # Scale the timestamp for now
            now = int(now * scalar)

        try:
            # Submit sorted events to the VCD writer
            for time, signal, value in self._event_buffer:
                self._vcd.change(signal, time, value)
        except vcd.writer.VCDPhaseError as e:
            # Occurs when we try to submit a timestamp which is earlier than the last submitted timestamp
            raise RuntimeError('Attempt to go back in time too much') from e
        else:
            # Flush the VCD writer
            self._vcd.flush(now)

        # Clear the event buffer
        self._event_buffer.clear()

    def close(self) -> None:
        # Close the VCD writer
        self._vcd.close()
        # Close the VCD file
        self._vcd_file.close()

    def get_registered_signals(self) -> __RS_T:
        """Return the registered signals.

        :return: A dictionary with devices and a list of signals
        """
        return self._registered_signals


_PS_T = typing.Tuple[DaxSimDevice, str]  # The peek signal manager signal type
_PT_T = _VT_T  # The peek signal manager signal-type type
_PV_T = _VV_T  # The peek signal manager signal type
_PD_T = typing.Dict[DaxSimDevice,  # The peek signal manager device list type
                    typing.Dict[str, typing.Tuple[_PT_T, typing.Optional[int], typing.Dict[np.int64, _PV_T]]]]


class _Meta(type):
    """Metaclass to have a pretty representation of a class."""

    def __repr__(cls) -> str:
        return cls.__name__


class SignalNotSet(metaclass=_Meta):
    """Class used to indicate that a signal was not set and no value could be returned."""
    pass


class PeekSignalManager(DaxSignalManager[_PS_T]):
    """Peek signal manager."""

    _CONVERT_TYPE: typing.Dict[type, _PT_T] = {
        bool: bool,
        int: int,
        np.int32: int,
        np.int64: int,
        float: float,
        str: str,
        object: object,
    }
    """Dict to convert Python types to peek signal manager internal types."""

    _CHECK_TYPE: typing.Dict[_PT_T, typing.Union[type, typing.Tuple[type, ...]]] = {
        bool: bool,
        int: (int, np.integer),
        float: float,
        str: str,
        object: bool,
    }
    """Dict to convert internal types to peek signal manager type-checking types."""

    _SPECIAL_VALUES: typing.Dict[_PT_T, typing.Set[typing.Any]] = {
        bool: {'x', 'X', 'z', 'Z', 0, 1},  # Also matches NumPy int and float
        int: {'x', 'X', 'z', 'Z'},
        float: set(),
        str: {None},
        object: set(),
    }
    """Dict with special allowed values for internal types."""

    def __init__(self) -> None:
        # Registered devices and buffer for signals/events
        self._event_buffer: _PD_T = {}

    def register(self, scope: DaxSimDevice, name: str, type_: _PT_T, *,
                 size: typing.Optional[int] = None, init: _PV_T = None) -> _PS_T:
        assert isinstance(scope, DaxSimDevice), 'The scope of the signal must be of type DaxSimDevice'
        assert isinstance(name, str), 'The name of the signal must be of type str'
        assert isinstance(size, int) or size is None, 'The size must be an int or None'

        # Check if type is supported and convert type if it is
        if type_ not in self._CONVERT_TYPE:
            raise ValueError(f'Peek signal manager does not support signal type {type_}')
        type_ = self._CONVERT_TYPE[type_]

        # Get signals of the given device
        signals = self._event_buffer.setdefault(scope, {})
        # Check if signal was already registered
        if name in signals:
            raise LookupError(f'Signal "{scope.key}.{name}" was already registered')

        # Check size
        if type_ is bool:
            if size is None or size < 1:
                raise TypeError('Provide a legal size for signal type bool')
        else:
            if size is not None:
                raise TypeError(f'Size not supported for signal type "{type_}"')

        if init is not None:
            # Check init value
            init = self._check_value(type_, size, init)

        # Register and initialize signal
        signals[name] = (type_, size, {} if init is None else {0: init})

        # Return the signal object
        return scope, name

    def event(self, signal: _PS_T, value: _PV_T, *,
              time: typing.Optional[np.int64] = None, offset: typing.Optional[np.int64] = None) -> None:
        assert isinstance(signal, tuple) and len(signal) == 2, 'Invalid signal object'
        assert time is None or isinstance(time, np.int64), 'Time must be of type np.int64 or None'
        assert offset is None or isinstance(offset, (int, np.integer)), 'Invalid type for offset'

        # Unpack device list
        device, name = signal
        type_, size, events = self._event_buffer[device][name]

        # Check value and add value to event buffer (overwrites old values)
        events[self._get_timestamp(time, offset)] = self._check_value(type_, size, value)

    def flush(self, ref_period: float) -> None:
        pass

    def close(self) -> None:
        pass

    def _check_value(self, type_: _PT_T, size: typing.Optional[int], value: _PV_T) -> _PV_T:
        """Check if value is valid, raise exception otherwise."""

        if size in {None, 1}:
            # noinspection PyTypeHints
            if isinstance(value, self._CHECK_TYPE[type_]):
                return value  # Value is legal (expected type)
            elif value in self._SPECIAL_VALUES[type_]:
                return value  # Value is legal (special value)
        elif type_ is bool and isinstance(value, str) and len(value) == size and all(
                v in {'x', 'X', 'z', 'Z', '0', '1'} for v in value):
            return value.lower()  # Value is legal (bool vector) (store lower case)

        # Value did not pass check
        raise ValueError(f'Invalid value {value} for signal type {type_}')

    """Peek functions"""

    def peek_and_type(self, scope: DaxSimDevice, signal: str, time: typing.Optional[np.int64] = None) -> \
            typing.Tuple[typing.Union[_PV_T, typing.Type[SignalNotSet]], type]:
        """Peek a value of a signal at a given time.

        :param scope: The scope of the signal
        :param signal: The signal name
        :param time: The time of interest (now if no time is given)
        :return: The type and value of the signal at the time of interest or :class:`SignalNotSet` if no value was found
        """

        assert isinstance(scope, DaxSimDevice), 'The given scope must be of type DaxSimDevice'
        assert isinstance(signal, str), 'The signal name must be of type str'
        assert isinstance(time, np.int64) or time is None

        try:
            # Get the device
            device = self._event_buffer[scope]
            # Get the signal
            type_, _, events = device[signal]
        except KeyError:
            raise KeyError(f'Signal "{scope.key}.{signal}" could not be found') from None

        if time is None:
            # Use the default time if none was provided
            time = artiq.language.core.now_mu()  # noqa: ATQ101

        # Return the last value before or at the given time stamp traversing the event list backwards
        value = next((v for t, v in sorted(events.items(), reverse=True) if t <= time), SignalNotSet)

        # Return the value and the type
        return value, type_

    def peek(self, scope: DaxSimDevice, signal: str,
             time: typing.Optional[np.int64] = None) -> typing.Union[_PV_T, typing.Type[SignalNotSet]]:
        """Peek a value of a signal at a given time.

        :param scope: The scope of the signal
        :param signal: The signal name
        :param time: The time of interest (now if no time is given)
        :return: The value of the signal at the time of interest or :class:`SignalNotSet` if no value was found
        """

        # Call the peek and type function
        value, _ = self.peek_and_type(scope, signal, time)
        # Return the value
        return value


_signal_manager: DaxSignalManager[typing.Any] = NullSignalManager()
"""Singleton instance of the signal manager."""


def get_signal_manager() -> DaxSignalManager[typing.Any]:
    """Get the signal manager instance.

    The signal manager is used by simulated devices to register and change signals during simulation.

    :return: The signal manager object
    """
    return _signal_manager


def set_signal_manager(signal_manager: DaxSignalManager[typing.Any]) -> None:
    """Set a new signal manager.

    The old signal manager will be closed.

    :param signal_manager: The new signal manager object to use
    """

    # Close the current signal manager
    global _signal_manager
    _signal_manager.close()

    # Set the new signal manager
    _signal_manager = signal_manager
