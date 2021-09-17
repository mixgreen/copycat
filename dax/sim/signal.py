import abc
import typing
import types
import operator
import datetime
import dataclasses
import numpy as np
import vcd.writer
import sortedcontainers

import artiq.language.core
from artiq.language.units import ns

from dax.sim.device import DaxSimDevice
from dax import __version__ as _dax_version
import dax.util.units

__all__ = ['DaxSignalManager', 'SignalNotSet', 'SignalNotSetError', 'SignalNotFoundError',
           'NullSignalManager', 'VcdSignalManager', 'PeekSignalManager',
           'get_signal_manager', 'set_signal_manager']


class _PrettyReprMeta(type):
    """Metaclass to have a pretty representation of a class."""

    def __repr__(cls) -> str:
        return cls.__name__


class SignalNotSet(metaclass=_PrettyReprMeta):
    """Class used to indicate that a signal was not set and no value could be returned."""
    pass


_T_T = np.int64  # Timestamp type
_O_T = typing.Union[int, np.int32, np.int64]  # Time offset type

_S_T = typing.TypeVar('_S_T')  # The abstract signal type variable
_ST_T = typing.Type[typing.Union[bool, int, np.int32, np.int64, float, str, object]]  # The signal-type type
_SV_T = typing.Union[bool, int, np.int32, np.int64, float, str, None]  # The signal-value type
_PV_T = typing.Union[_SV_T, typing.Type[SignalNotSet]]  # The peek-value type


class SignalNotSetError(RuntimeError):
    """This exception is raised when a signal value is requested but the signal is not set."""

    def __init__(self, scope: DaxSimDevice, name: str, time: _T_T, msg: str = ''):
        msg_ = f'Signal "{scope}.{name}" not set at time {time}{f": {msg}" if msg else ""}'
        super(SignalNotSetError, self).__init__(msg_)


class SignalNotFoundError(KeyError):
    """This exception is raised when a requested signal does not exist."""

    def __init__(self, scope: DaxSimDevice, name: str):
        super(SignalNotFoundError, self).__init__(f'Signal "{scope}.{name}" could not be found')


class DaxSignalManager(abc.ABC, typing.Generic[_S_T]):
    """Base class for classes that manage simulated signals."""

    __signals: typing.Dict[typing.Tuple[DaxSimDevice, str], _S_T]
    """Registered signals"""

    def __init__(self) -> None:
        self.__signals = {}

    def register(self, scope: DaxSimDevice, name: str, type_: _ST_T, *,
                 size: typing.Optional[int] = None, init: _SV_T = None) -> _S_T:
        """Register a signal.

        Signals have to be registered before any events are committed.
        Used by the device driver to register signals.

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
        :param init: Initial value (defaults to :const:`None` (``'X'``))
        :return: The signal object used to call other functions of this class
        :raises LookupError: Raised if the signal was already registered
        """
        assert isinstance(scope, DaxSimDevice), 'Signal scope must be of type DaxSimDevice'
        assert isinstance(name, str), 'Signal name must be of type str'
        assert isinstance(size, int) or size is None, 'The size must be an int or None'

        # Create the key
        key = (scope, name)

        if key in self.__signals:
            # A signal can not be registered more than once
            raise LookupError(f'Signal "{scope}.{name}" was already registered')

        # Register and return signal
        signal = self._register_signal(scope, name, type_, size=size, init=init)
        self.__signals[key] = signal
        return signal

    @abc.abstractmethod
    def _register_signal(self, scope: DaxSimDevice, name: str, type_: _ST_T, *,  # pragma: no cover
                         size: typing.Optional[int] = None, init: _SV_T = None) -> _S_T:
        """Internally register a signal.

        :param scope: The scope of the signal, which is the device object
        :param name: The name of the signal
        :param type_: The type of the signal
        :param size: The size of the data (only for type bool)
        :param init: Initial value (defaults to :const:`None` (``'X'``))
        :return: The signal object used to call other functions of this class
        :raises LookupError: Raised if the signal was already registered
        """
        pass

    def signal(self, scope: DaxSimDevice, name: str) -> _S_T:
        """Obtain an existing signal object.

        :param scope: The scope of the signal, which is the device object
        :param name: The name of the signal
        :return: The signal object used to call other functions of this class
        :raises SignalNotFoundError: Raised if the signal could not be found
        """
        # Create the key
        key = (scope, name)

        if key not in self.__signals:
            # Signal not found
            raise SignalNotFoundError(scope, name)

        # Return key
        return self.__signals[key]

    @abc.abstractmethod
    def push(self, signal: _S_T, value: _SV_T, *,  # pragma: no cover
             time: typing.Optional[_T_T] = None, offset: _O_T = 0) -> None:
        """Push an event (i.e. the change of a signal to a specified value at the given time).

        Note that in a parallel context, :func:`delay` and :func:`delay_mu` do not directly
        influence the time returned by :func:`now_mu`.
        It is recommended to use the time or offset parameters to set events at a different
        time without modifying the timeline.

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
    def pull(self, signal: _S_T, *,  # pragma: no cover
             time: typing.Optional[_T_T] = None, offset: _O_T = 0) -> _SV_T:
        """Pull the value of a signal at the given time.

        Note that in a parallel context, :func:`delay` and :func:`delay_mu` do not directly
        influence the time returned by :func:`now_mu`.
        It is recommended to use the time or offset parameters to get values at a different
        time without modifying the timeline.

        :param signal: The signal to obtain the value from
        :param time: Optional time in machine units to obtain the signal value (:func:`now_mu` if no time was provided)
        :param offset: Optional offset from the given time in machine units (default is :const:`0`)
        :return: The value of the given signal at the given time and offset
        :raises SignalNotSetError: Raised if the signal was not set at the given time
        """
        pass

    @abc.abstractmethod
    def flush(self, ref_period: float) -> None:  # pragma: no cover
        """Flush the output of the signal manager.

        :param ref_period: The reference period (i.e. the time of one machine unit)
        """
        pass

    @abc.abstractmethod
    def close(self) -> None:  # pragma: no cover
        """Close the signal manager.

        Note that this function must be reentrant!
        """
        pass


def _get_timestamp(time: typing.Optional[_T_T] = None, offset: _O_T = 0) -> _T_T:
    """Calculate the timestamp of an event."""
    if time is None:
        time = artiq.language.core.now_mu()  # noqa: ATQ101
    else:
        assert isinstance(time, np.int64), 'Time must be of type np.int64'
    return time + offset if offset else time


@dataclasses.dataclass(frozen=True)
class Signal:
    """Class to represent a signal."""

    scope: DaxSimDevice
    """Scope of the signal, which is the device object."""
    name: str
    """Name of the signal."""


@dataclasses.dataclass(frozen=True)
class ConstantSignal(Signal):
    """Class to represent a constant signal."""

    init: _SV_T
    """Init value."""


class NullSignalManager(DaxSignalManager[ConstantSignal]):
    """A signal manager with constant signals (i.e. all push events are dropped)."""

    def _register_signal(self, scope: DaxSimDevice, name: str, type_: _ST_T, *,
                         size: typing.Optional[int] = None, init: _SV_T = None) -> ConstantSignal:
        # Create and return the signal object
        return ConstantSignal(scope, name, init=init)

    def push(self, signal: ConstantSignal, value: _SV_T, *,
             time: typing.Optional[_T_T] = None, offset: _O_T = 0) -> None:
        pass  # Drop all events

    def pull(self, signal: ConstantSignal, *,
             time: typing.Optional[_T_T] = None, offset: _O_T = 0) -> _SV_T:
        if signal.init is None:
            # Signal was not set
            raise SignalNotSetError(signal.scope, signal.name, _get_timestamp(time, offset),
                                    msg='Signal not initialized')
        else:
            # Return the init value
            return signal.init

    def flush(self, ref_period: float) -> None:
        pass

    def close(self) -> None:
        pass


@dataclasses.dataclass(frozen=True)
class VcdSignal(ConstantSignal):
    """Class to represent a VCD signal."""

    vcd: vcd.writer.Variable[vcd.writer.VarValue]
    """The VCD variable."""


class VcdSignalManager(DaxSignalManager[VcdSignal]):
    """VCD signal manager."""

    __S_T = typing.Tuple[str, _ST_T, typing.Optional[int]]  # Signal-type type
    __RS_T = typing.Dict[DaxSimDevice, typing.List[__S_T]]  # Dict of registered signals type
    __RSM_T = typing.Mapping[DaxSimDevice, typing.List[__S_T]]  # Map of registered signals type
    _E_T = typing.Tuple[typing.Union[int, np.int64], VcdSignal, _SV_T]  # Event type

    _CONVERT_TYPE: typing.ClassVar[typing.Dict[_ST_T, str]] = {
        bool: 'reg',
        int: 'integer',
        np.int32: 'integer',
        np.int64: 'integer',
        float: 'real',
        str: 'string',
        object: 'event',
    }
    """Dict to convert Python types to VCD types."""

    _timescale: float
    _event_buffer: typing.List[_E_T]
    _registered_signals: __RS_T

    def __init__(self, file_name: str, timescale: float = 1 * ns):
        assert isinstance(file_name, str), 'Output file name must be of type str'
        assert isinstance(timescale, float), 'Timescale must be of type float'
        assert timescale > 0.0, 'Timescale must be > 0.0'

        # Call super
        super(VcdSignalManager, self).__init__()

        # Store timescale
        self._timescale = timescale

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
        self._event_buffer = []
        # Create a registered signals data structure
        self._registered_signals = {}

    def _register_signal(self, scope: DaxSimDevice, name: str, type_: _ST_T, *,
                         size: typing.Optional[int] = None, init: _SV_T = None) -> VcdSignal:
        # Get the var type
        if type_ not in self._CONVERT_TYPE:
            raise ValueError(f'VCD signal manager does not support signal type {type_}')
        var_type = self._CONVERT_TYPE[type_]

        # Workaround for str init values (shows up as `Z` instead of string value 'x')
        vcd_init = '' if type_ is str and init is None else init

        # Register signal
        self._registered_signals.setdefault(scope, []).append((name, type_, size))
        vcd_signal = self._vcd.register_var(scope.key, name, var_type=var_type, size=size, init=vcd_init)

        # Create and return the signal object
        return VcdSignal(scope, name, init=init, vcd=vcd_signal)

    def push(self, signal: VcdSignal, value: _SV_T, *,
             time: typing.Optional[_T_T] = None, offset: _O_T = 0) -> None:
        # Add event to buffer
        self._event_buffer.append((_get_timestamp(time, offset), signal, value))

    def pull(self, signal: VcdSignal, *,  # pragma: no cover
             time: typing.Optional[_T_T] = None, offset: _O_T = 0) -> _SV_T:
        if signal.init is None:
            # Signal was not initialized
            raise SignalNotSetError(signal.scope, signal.name, _get_timestamp(time, offset),
                                    msg='Signal not initialized')
        else:
            # Return the init value
            return signal.init

    def flush(self, ref_period: float) -> None:
        # Sort the list of events (VCD writer can only handle a linear timeline)
        self._event_buffer.sort(key=operator.itemgetter(0))
        # Get a timestamp for now
        now: typing.Union[int, _T_T] = _get_timestamp()

        if ref_period == self._timescale:
            # Just iterate over the event buffer
            event_buffer_iter: typing.Iterator[VcdSignalManager._E_T] = iter(self._event_buffer)
        else:
            # Scale the timestamps if the reference period does not match the timescale
            scalar = ref_period / self._timescale
            event_buffer_iter = ((int(time * scalar), signal, value) for time, signal, value in self._event_buffer)
            # Scale the timestamp for now
            now = int(now * scalar)

        try:
            # Submit sorted events to the VCD writer
            for time, signal, value in event_buffer_iter:
                self._vcd.change(signal.vcd, time, value)
        except vcd.writer.VCDPhaseError as e:
            # Occurs when we try to submit a timestamp which is earlier than the last submitted timestamp
            raise RuntimeError('Attempt to go back in time too much') from e
        else:
            # Flush the VCD writer
            self._vcd.flush(now)

        # Clear the event buffer
        self._event_buffer.clear()

    def close(self) -> None:
        # Close the VCD writer (reentrant)
        self._vcd.close()
        # Close the VCD file (reentrant)
        self._vcd_file.close()

    def get_registered_signals(self) -> __RSM_T:
        """Return the registered signals.

        :return: A dictionary with devices and a list of signals
        """
        return types.MappingProxyType(self._registered_signals)


# Workaround required for Python<3.9 and the usage of stubs for the sorted containers library
if typing.TYPE_CHECKING:
    _P_ES_T = sortedcontainers.SortedDict[_T_T, _SV_T]  # The peek signal manager event sequence type
    _P_TV_T = sortedcontainers.SortedKeysView[_T_T]  # The peek signal manager event sequence timestamp view type
else:
    _P_ES_T = sortedcontainers.SortedDict
    _P_TV_T = typing.KeysView[_T_T]  # Using generic KeysView, helps the PyCharm type checker


class PeekSignalManager(DaxSignalManager[Signal]):
    """Peek signal manager."""

    __EB_T = typing.Dict[DaxSimDevice,  # The peek signal manager device list and event buffer type
                         typing.Dict[str, typing.Tuple[_ST_T, typing.Optional[int], _P_ES_T, _P_TV_T]]]

    _CONVERT_TYPE: typing.ClassVar[typing.Dict[type, _ST_T]] = {
        bool: bool,
        int: int,
        np.int32: int,
        np.int64: int,
        float: float,
        str: str,
        object: object,
    }
    """Dict to convert Python types to peek signal manager internal types."""

    _CHECK_TYPE: typing.ClassVar[typing.Dict[_ST_T, typing.Union[type, typing.Tuple[type, ...]]]] = {
        bool: bool,
        int: (int, np.int32, np.int64),
        float: float,
        str: str,
        object: bool,
    }
    """Dict to convert internal types to peek signal manager type-checking types."""

    _SPECIAL_VALUES: typing.ClassVar[typing.Dict[_ST_T, typing.Set[typing.Any]]] = {
        bool: {'x', 'X', 'z', 'Z', 0, 1},  # Also matches NumPy int and float
        int: {'x', 'X', 'z', 'Z'},
        float: set(),
        str: {None},
        object: set(),
    }
    """Dict with special allowed values for internal types."""

    _event_buffer: __EB_T
    """Registered devices and buffer for signals/events."""

    def __init__(self) -> None:
        super(PeekSignalManager, self).__init__()
        self._event_buffer = {}

    def _register_signal(self, scope: DaxSimDevice, name: str, type_: _ST_T, *,
                         size: typing.Optional[int] = None, init: _SV_T = None) -> Signal:
        # Check if type is supported and convert type if it is
        if type_ not in self._CONVERT_TYPE:
            raise ValueError(f'Peek signal manager does not support signal type {type_}')
        type_ = self._CONVERT_TYPE[type_]

        # Get signals of the given device
        registered_signals = self._event_buffer.setdefault(scope, {})
        # Check if signal was already registered
        if name in registered_signals:
            raise LookupError(f'Signal "{scope}.{name}" was already registered')

        # Check size
        if type_ is bool:
            if size is None or size < 1:
                raise TypeError('Provide a legal size for signal type bool')
        else:
            if size is not None:
                raise TypeError(f'Size not supported for signal type "{type_}"')

        # Register signal
        events: _P_ES_T = sortedcontainers.SortedDict()
        if init is not None:
            # Check init value
            events[np.int64(0)] = self._check_value(type_, size, init)
        registered_signals[name] = (type_, size, events, events.keys())

        # Create and return the signal object
        return Signal(scope, name)

    def push(self, signal: Signal, value: _SV_T, *,
             time: typing.Optional[_T_T] = None, offset: _O_T = 0) -> None:
        # Unpack device list
        type_, size, events, _ = self._event_buffer[signal.scope][signal.name]

        # Check value and add value to event buffer
        # An existing value at the same timestamp will be overwritten, just as the ARTIQ RTIO system does
        events[_get_timestamp(time, offset)] = self._check_value(type_, size, value)

    def peek_and_type(self, scope: DaxSimDevice, name: str, *,
                      time: typing.Optional[_T_T] = None, offset: _O_T = 0) -> typing.Tuple[_PV_T, _ST_T]:
        """Peek the value of a signal at a given time.

        :param scope: The scope of the signal
        :param name: The name of the signal
        :param time: Optional time in machine units to obtain the signal value (:func:`now_mu` if no time was provided)
        :param offset: Optional offset from the given time in machine units (default is :const:`0`)
        :return: The type and value of the signal at the time of interest or :class:`SignalNotSet` if no value was found
        :raises KeyError: Raised if the signal could not be found
        """
        assert isinstance(scope, DaxSimDevice), 'The given scope must be of type DaxSimDevice'
        assert isinstance(name, str), 'The signal name must be of type str'

        try:
            # Get the device
            device = self._event_buffer[scope]
            # Get the signal
            type_, _, events, timestamps = device[name]
        except KeyError:
            raise KeyError(f'Signal "{scope}.{name}" could not be found') from None

        # Binary search for the insertion point (right) of the given timestamp
        index = events.bisect_right(_get_timestamp(time, offset))

        # Return the value and the type
        return events[timestamps[index - 1]] if index else SignalNotSet, type_

    def peek(self, scope: DaxSimDevice, name: str, *,
             time: typing.Optional[_T_T] = None, offset: _O_T = 0) -> _PV_T:
        """Peek the value of a signal at a given time.

        :param scope: The scope of the signal
        :param name: The name of the signal
        :param time: Optional time in machine units to obtain the signal value (:func:`now_mu` if no time was provided)
        :param offset: Optional offset from the given time in machine units (default is :const:`0`)
        :return: The type and value of the signal at the time of interest or :class:`SignalNotSet` if no value was found
        """
        # Call the peek and type function
        value, _ = self.peek_and_type(scope, name, time=time, offset=offset)
        # Return the value
        return value

    def pull(self, signal: Signal, *,
             time: typing.Optional[_T_T] = None, offset: _O_T = 0) -> _SV_T:
        # Peek the value
        value = self.peek(signal.scope, signal.name, time=time, offset=offset)

        if value is SignalNotSet:
            # Signal was not set, raise an exception
            raise SignalNotSetError(signal.scope, signal.name, _get_timestamp(time, offset))
        else:
            # Return the value
            return typing.cast(_SV_T, value)  # Cast required for mypy

    def flush(self, ref_period: float) -> None:
        pass

    def close(self) -> None:
        # Release resources
        self._event_buffer.clear()

    def _check_value(self, type_: _ST_T, size: typing.Optional[int], value: _PV_T) -> _SV_T:
        """Check if value is valid, raise exception otherwise."""

        if size in {None, 1}:
            # noinspection PyTypeHints
            if isinstance(value, self._CHECK_TYPE[type_]):
                return typing.cast(_SV_T, value)  # Value is legal (expected type), cast required for mypy
            elif value in self._SPECIAL_VALUES[type_]:
                return typing.cast(_SV_T, value)  # Value is legal (special value), cast required for mypy
        elif type_ is bool and isinstance(value, str) and len(value) == size and all(
                v in {'x', 'X', 'z', 'Z', '0', '1'} for v in value):
            return value.lower()  # Value is legal (bool vector) (store lower case)

        # Value did not pass check
        raise ValueError(f'Invalid value {value} for signal type {type_}')


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
