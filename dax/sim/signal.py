import abc
import typing
import operator
import datetime
import collections
import heapq
import numpy as np
import vcd.writer
import sortedcontainers

import artiq.language.core
from artiq.language.units import ns

from dax.sim.device import DaxSimDevice
from dax import __version__ as _dax_version
import dax.util.units

__all__ = [
    'Signal', 'SignalNotSetError', 'SignalNotFoundError', 'DaxSignalManager',
    'NullSignalManager', 'VcdSignalManager', 'PeekSignalManager',
    'get_signal_manager', 'set_signal_manager',
]

_T_T = np.int64  # Timestamp type
_O_T = typing.Union[int, np.int32, np.int64]  # Time offset type

_ST_T = typing.Union[typing.Type[bool], typing.Type[int], typing.Type[float],  # The signal type type
                     typing.Type[str], typing.Type[object]]
_SS_T = typing.Optional[int]  # The signal size type

_BOOL_T = typing.Union[bool, int, str]
_INT_T = typing.Union[int, np.int32, np.int64, str]
_FLOAT_T = float
_STR_T = str
_OBJ_T = bool
_SV_T = typing.Union[_BOOL_T, _INT_T, _FLOAT_T, _STR_T, _OBJ_T]  # The signal value type

_TIMESTAMP_MIN: _T_T = np.iinfo(np.int64).min  # type: ignore[attr-defined]
"""Minimum value for a timestamp."""
_INT_SPECIAL_VALUES: typing.FrozenSet[str] = frozenset('xXzZ')
"""Special values for int signals."""
_BOOL_VEC_VALUES: typing.FrozenSet[str] = frozenset('01xXzZ')
"""Legal characters for bool vector strings."""
_BOOL_VALUES: typing.FrozenSet[_BOOL_T] = _INT_SPECIAL_VALUES | {True, False, 0, 1}
"""Legal values for bool signals with size 1 (also matches float and NumPy int)."""


class _NormalizationError(ValueError):
    """Normalization error type."""

    def __init__(self, signal: 'Signal', value: typing.Any):  # Prevent using postponed evaluation of annotations
        super(_NormalizationError, self).__init__(
            f'Invalid value "{value}" for signal type {signal.type.__name__}'
            f'{"" if signal.size is None else f" with size {signal.size}"}'
        )


class Signal(abc.ABC):
    """Abstract class to represent a signal."""

    __scope: DaxSimDevice
    __name: str
    __type: _ST_T
    __size: _SS_T
    __normalize: typing.Callable[['Signal', typing.Any], _SV_T]  # Prevent using postponed evaluation of annotations

    _SIGNAL_TYPES: typing.ClassVar[typing.FrozenSet[_ST_T]] = frozenset([bool, int, float, str, object])
    """Valid signal types."""

    def __init__(self, scope: DaxSimDevice, name: str, type_: _ST_T, size: _SS_T = None):
        """Initialize a new signal object."""
        assert isinstance(scope, DaxSimDevice), 'Signal scope must be of type DaxSimDevice'
        assert isinstance(name, str), 'Signal name must be of type str'

        if not name.isidentifier():
            raise ValueError('Invalid signal name (must be a valid identifier)')
        if type_ not in self._SIGNAL_TYPES:
            raise ValueError('Invalid signal type')
        if type_ is bool:
            if not isinstance(size, int) or not size > 0:
                raise ValueError('Signal size must be an integer > 0 for signal type bool')
        else:
            if size is not None:
                raise ValueError(f'Size not supported for signal type "{type_}"')

        # Store attributes
        self.__scope = scope
        self.__name = name
        self.__type = type_
        self.__size = size

        # Select normalization function
        normalize_fn: typing.Dict[_ST_T, typing.Callable[[typing.Any], _SV_T]] = {
            bool: self._normalize_bool,
            int: self._normalize_int,
            float: self._normalize_float,
            str: self._normalize_str,
            object: self._normalize_object,
        }
        self.__normalize = normalize_fn[type_]  # type: ignore[assignment]

    @property
    def scope(self) -> DaxSimDevice:
        """Scope of the signal, which is the device object."""
        return self.__scope

    @property
    def name(self) -> str:
        """Name of the signal."""
        return self.__name

    @property
    def type(self) -> _ST_T:
        """Type of the signal."""
        return self.__type

    @property
    def size(self) -> _SS_T:
        """Size of the signal."""
        return self.__size

    def normalize(self, value: typing.Any) -> _SV_T:
        """Normalize a value for this signal.

        :param value: The value to normalize
        :return: The normalized value
        :raises ValueError: Raised if the value is invalid
        """
        return self.__normalize(value)

    def _normalize_bool(self, value: typing.Any) -> _BOOL_T:
        if self.size == 1 and value in _BOOL_VALUES:
            return value  # type: ignore[no-any-return]
        elif self.size is not None and isinstance(value, str) and 1 < self.size == len(value) and all(
                v in _BOOL_VEC_VALUES for v in value):
            return value.lower()  # Normalize to lower case
        else:
            raise _NormalizationError(self, value)

    def _normalize_int(self, value: typing.Any) -> _INT_T:
        if isinstance(value, (int, np.int32, np.int64)) or value in _INT_SPECIAL_VALUES:
            return value  # type: ignore[no-any-return]
        else:
            raise _NormalizationError(self, value)

    def _normalize_float(self, value: typing.Any) -> _FLOAT_T:
        if isinstance(value, float):
            return value
        else:
            raise _NormalizationError(self, value)

    def _normalize_str(self, value: typing.Any) -> _STR_T:
        if isinstance(value, str):
            return value
        else:
            raise _NormalizationError(self, value)

    def _normalize_object(self, value: typing.Any) -> _OBJ_T:
        if isinstance(value, bool):
            return value
        else:
            raise _NormalizationError(self, value)

    @abc.abstractmethod
    def push(self, value: typing.Any, *,  # pragma: no cover
             time: typing.Optional[_T_T] = None, offset: _O_T = 0) -> None:
        """Push an event to this signal (i.e. change the value of this signal at the given time).

        Values are automatically normalized before inserted into the signal manager (see :func:`normalize`).

        Note that in a parallel context, :func:`delay` and :func:`delay_mu` do not directly
        influence the time returned by :func:`now_mu`.
        It is recommended to use the time or offset parameters to set events at a different
        time without modifying the timeline.

        Bool type signals can have values ``0``, ``1``, ``'X'``, ``'Z'``.
        A vector of a bool type signal has a value of type ``str`` (e.g. ``'1001XZ'``).
        An integer can be converted to a bool vector with the following example code:
        ``f'{value & 0xFF:08b}'`` (size 8 bool vector).

        Integer type variables can have any int value or any value legal for a bool type signal.

        Float type variables can only be assigned float values.

        Event (``object``) type signals represent timestamps and do not have a value.
        We recommend to always use value :const:`True` for event type signals.

        String type signals can use value :const:`None` which is equivalent to ``'Z'``.

        :param value: The new value of this signal
        :param time: Optional time in machine units when the event happened (:func:`now_mu` if no time was provided)
        :param offset: Optional offset from the given time in machine units (default is :const:`0`)
        :raises ValueError: Raised if the value is invalid
        """
        pass

    @abc.abstractmethod
    def pull(self, *,  # pragma: no cover
             time: typing.Optional[_T_T] = None, offset: _O_T = 0) -> _SV_T:
        """Pull the value of this signal at the given time.

        Note that in a parallel context, :func:`delay` and :func:`delay_mu` do not directly
        influence the time returned by :func:`now_mu`.
        It is recommended to use the time or offset parameters to get values at a different
        time without modifying the timeline.

        :param time: Optional time in machine units to obtain the signal value (:func:`now_mu` if no time was provided)
        :param offset: Optional offset from the given time in machine units (default is :const:`0`)
        :return: The value of the given signal at the given time and offset
        :raises SignalNotSetError: Raised if the signal was not set at the given time
        """
        pass

    def __str__(self) -> str:
        """The key of the corresponding device followed by the name of this signal."""
        return f'{self.scope}.{self.name}'

    def __repr__(self) -> str:
        """See :func:`__str__`."""
        return str(self)


class SignalNotSetError(RuntimeError):
    """This exception is raised when a signal value is requested but the signal is not set."""

    def __init__(self, signal: Signal, time: _T_T, msg: str = ''):
        msg_ = f'Signal "{signal}" not set at time {time}{f": {msg}" if msg else ""}'
        super(SignalNotSetError, self).__init__(msg_)


class SignalNotFoundError(KeyError):
    """This exception is raised when a requested signal does not exist."""

    def __init__(self, scope: DaxSimDevice, name: str):
        super(SignalNotFoundError, self).__init__(f'Signal "{scope}.{name}" could not be found')


_S_T = typing.TypeVar('_S_T', bound=Signal)  # The abstract signal type variable


class DaxSignalManager(abc.ABC, typing.Generic[_S_T]):
    """Base class for classes that manage simulated signals."""

    __slots__ = ('__signals',)

    __signals: typing.Dict[typing.Tuple[DaxSimDevice, str], _S_T]
    """Registered signals"""

    def __init__(self) -> None:
        self.__signals = {}

    def register(self, scope: DaxSimDevice, name: str, type_: _ST_T, *,
                 size: _SS_T = None, init: typing.Optional[_SV_T] = None) -> _S_T:
        """Register a signal.

        Signals have to be registered before any events are committed.
        Used by the device driver to register signals.

        Possible types and expected arguments:

        - ``bool`` (a register with bit values ``0``, ``1``, ``'X'``, ``'Z'``), provide a size of the register
        - ``int``
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
        # Create the key
        key = (scope, name)

        if key in self.__signals:
            # A signal can not be registered more than once
            raise LookupError(f'Signal "{self.__signals[key]}" was already registered')

        # Create, register, and return signal
        signal = self._create_signal(scope, name, type_, size=size, init=init)
        self.__signals[key] = signal
        return signal

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

    def __iter__(self) -> typing.Iterator[_S_T]:
        """Obtain an iterator over the registered signals."""
        return iter(self.__signals.values())

    def __len__(self) -> int:
        """Get the number of registered signals."""
        return len(self.__signals)

    @abc.abstractmethod
    def _create_signal(self, scope: DaxSimDevice, name: str, type_: _ST_T, *,  # pragma: no cover
                       size: _SS_T = None, init: typing.Optional[_SV_T] = None) -> _S_T:
        """Create a new signal object.

        :param scope: The scope of the signal, which is the device object
        :param name: The name of the signal
        :param type_: The type of the signal
        :param size: The size of the data (only for type bool)
        :param init: Initial value (defaults to :const:`None` (``'X'``))
        :return: The signal object used to call other functions of this class
        :raises LookupError: Raised if the signal was already registered
        """
        pass

    @abc.abstractmethod
    def horizon(self) -> _T_T:
        """Return the time horizon.

        The time horizon is defined as the maximum of all event timestamps and the timeline cursor position.

        :return: The time horizon in machine units
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


class ConstantSignal(Signal):
    """Class to represent a constant signal."""

    _init: typing.Optional[_SV_T]

    def __init__(self, scope: DaxSimDevice, name: str, type_: _ST_T, size: _SS_T, *, init: typing.Optional[_SV_T]):
        super(ConstantSignal, self).__init__(scope, name, type_, size)
        self._init = None if init is None else self.normalize(init)

    def push(self, value: typing.Any, *,
             time: typing.Optional[_T_T] = None, offset: _O_T = 0) -> None:
        self.normalize(value)  # Do normalization (for exceptions) before dropping the event

    def pull(self, *,
             time: typing.Optional[_T_T] = None, offset: _O_T = 0) -> _SV_T:
        if self._init is None:
            # Signal was not set
            raise SignalNotSetError(self, _get_timestamp(time, offset), msg='Signal not initialized')
        else:
            # Return the init value
            return self._init


class NullSignal(ConstantSignal):
    """Class to represent a null signal."""

    _update_horizon: typing.Callable[[_T_T], None]

    def __init__(self, scope: DaxSimDevice, name: str, type_: _ST_T, size: _SS_T, *,
                 init: typing.Optional[_SV_T], update_horizon_fn: typing.Callable[[_T_T], None]):
        assert callable(update_horizon_fn)
        self._update_horizon = update_horizon_fn  # type: ignore[misc,assignment]
        super(NullSignal, self).__init__(scope, name, type_, size, init=init)
        if init is not None:
            self.push(init, time=np.int64(0))

    def push(self, value: typing.Any, *,
             time: typing.Optional[_T_T] = None, offset: _O_T = 0) -> None:
        self._update_horizon(_get_timestamp(time, offset))  # type: ignore[misc,call-arg]
        super(NullSignal, self).push(value, time=time, offset=offset)


class NullSignalManager(DaxSignalManager[NullSignal]):
    """A signal manager with constant signals (i.e. all push events to signals are dropped)."""

    __slots__ = ('_horizon',)

    _horizon: _T_T

    def __init__(self) -> None:
        super(NullSignalManager, self).__init__()
        self._horizon = _TIMESTAMP_MIN

    def _create_signal(self, scope: DaxSimDevice, name: str, type_: _ST_T, *,
                       size: _SS_T = None, init: typing.Optional[_SV_T] = None) -> NullSignal:
        return NullSignal(scope, name, type_, size, init=init, update_horizon_fn=self._update_horizon)

    def _update_horizon(self, t: _T_T) -> None:
        self._horizon = max(t, self._horizon)

    def horizon(self) -> _T_T:
        return max(self._horizon, _get_timestamp())

    def flush(self, ref_period: float) -> None:
        pass

    def close(self) -> None:
        pass


class VcdSignal(ConstantSignal):
    """Class to represent a VCD signal."""

    __VCD_T = vcd.writer.Variable[vcd.writer.VarValue]  # VCD variable type
    E_T = typing.Tuple[typing.Union[int, np.int64], 'VcdSignal', _SV_T]  # Event type (string literal forward reference)

    _events: typing.List[E_T]
    _vcd: __VCD_T

    _VCD_TYPE: typing.ClassVar[typing.Dict[_ST_T, str]] = {
        bool: 'reg',
        int: 'integer',
        float: 'real',
        str: 'string',
        object: 'event',
    }
    """Dict to convert Python types to VCD types."""

    def __init__(self, scope: DaxSimDevice, name: str, type_: _ST_T, size: _SS_T, *, init: typing.Optional[_SV_T],
                 vcd_: vcd.writer.VCDWriter, events: typing.List[E_T]):
        # Store reference to shared and mutable event buffer
        self._events = events
        # Call super
        super(VcdSignal, self).__init__(scope, name, type_, size, init=init)

        if type_ is str and init is None:
            # Workaround for str init values (shows up as `z` instead of string value 'x')
            init = ''

        # Register this variable with the VCD writer
        self._vcd = vcd_.register_var(scope.key, name,
                                      var_type=self._VCD_TYPE[type_], size=size, init=init)

        for alias in scope.aliases:
            # Register the alias to the variable
            vcd_.register_alias(alias, f"{alias}-{name}", self._vcd)

    def push(self, value: typing.Any, *,
             time: typing.Optional[_T_T] = None, offset: _O_T = 0) -> None:
        # Add event
        self._events.append((_get_timestamp(time, offset), self, self.normalize(value)))

    def _normalize_int(self, value: typing.Any) -> _INT_T:
        # Call super
        v = super(VcdSignal, self)._normalize_int(value)

        # Workaround for int values (NumPy int objects are not accepted)
        if isinstance(v, (np.int32, np.int64)):
            v = int(v)

        # Return value
        return v

    @property
    def vcd(self) -> __VCD_T:
        return self._vcd


class VcdSignalManager(DaxSignalManager[VcdSignal]):
    """VCD signal manager."""

    __slots__ = ('_timescale', '_file', '_vcd', '_events', '_flushed_horizon')

    _timescale: float
    _file: typing.IO[str]
    _vcd: vcd.writer.VCDWriter
    _events: typing.List[VcdSignal.E_T]
    _flushed_horizon: _T_T

    def __init__(self, file_name: str, *, timescale: float = 1 * ns):
        assert isinstance(file_name, str), 'Output file name must be of type str'
        assert isinstance(timescale, float), 'Timescale must be of type float'
        assert timescale > 0.0, 'Timescale must be > 0.0'

        # Call super
        super(VcdSignalManager, self).__init__()
        # Store timescale
        self._timescale = timescale

        # Open file
        self._file = open(file_name, mode='w')
        # Create VCD writer
        self._vcd = vcd.writer.VCDWriter(self._file,
                                         timescale=dax.util.units.time_to_str(timescale, precision=0),
                                         date=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                         comment=file_name,
                                         version=_dax_version)

        # Create the shared buffer for events
        self._events = []
        # Time horizon of flushed events
        self._flushed_horizon = 0  # VCD does not support negative timestamps, the initial horizon should be 0

    def _create_signal(self, scope: DaxSimDevice, name: str, type_: _ST_T, *,
                       size: _SS_T = None, init: typing.Optional[_SV_T] = None) -> VcdSignal:
        return VcdSignal(scope, name, type_, size, init=init, vcd_=self._vcd, events=self._events)

    def horizon(self) -> _T_T:
        # Sort existing events to easily get the maximum timestamp
        self._events.sort(key=operator.itemgetter(0))
        # Return the max of the latest event if available, the flushed horizon, and the current timestamp
        return max(self._events[-1][0] if self._events else 0, self._flushed_horizon, _get_timestamp())

    def flush(self, ref_period: float) -> None:
        # Get a timestamp for the new horizon
        horizon: _T_T = self.horizon()
        # Update the flushed horizon
        self._flushed_horizon = horizon

        # NOTE: self.horizon() sorts the events, which is required (VCD writer can only handle a linear timeline)

        if ref_period == self._timescale:
            # Just iterate over the events
            events_iter: typing.Iterator[VcdSignal.E_T] = iter(self._events)
        else:
            # Scale the timestamps if the reference period does not match the timescale
            scalar = ref_period / self._timescale
            events_iter = ((int(time * scalar), signal, value) for time, signal, value in self._events)
            # Scale the timestamp for the horizon
            horizon = np.int64(horizon * scalar)

        try:
            # Submit sorted events to the VCD writer
            for time, signal, value in events_iter:
                self._vcd.change(signal.vcd, time, value)
        except vcd.writer.VCDPhaseError as e:
            # Occurs when we try to submit a timestamp which is earlier than the last submitted timestamp
            raise RuntimeError('Attempt to go back in time too much') from e
        else:
            # Flush the VCD writer
            self._vcd.flush(int(horizon))

        # Clear the event buffer
        self._events.clear()

    def close(self) -> None:
        # Clear the event buffer
        self._events.clear()
        # Close the VCD writer (reentrant)
        self._vcd.close()
        # Close the VCD file (reentrant)
        self._file.close()


class PeekSignal(Signal):
    """Class to represent a peek signal."""

    # Workaround required for the local stubs of the sorted containers library
    if typing.TYPE_CHECKING:  # pragma: no cover
        _EB_T = sortedcontainers.SortedDict[_T_T, _SV_T]  # The peek signal event buffer type
        _TV_T = sortedcontainers.SortedKeysView[_T_T]  # The peek signal event buffer timestamp view type
    else:
        _EB_T = sortedcontainers.SortedDict
        _TV_T = typing.KeysView[_T_T]  # Using generic KeysView, helps the PyCharm type checker

    _buffer: typing.Deque[_SV_T]
    _events: _EB_T
    _timestamps: _TV_T

    def __init__(self, scope: DaxSimDevice, name: str, type_: _ST_T, size: _SS_T, *, init: typing.Optional[_SV_T]):
        # Call super
        super(PeekSignal, self).__init__(scope, name, type_, size)

        # Create buffer for push values
        self._buffer = collections.deque()
        # Create buffer for events
        self._events = sortedcontainers.SortedDict()
        # Create timestamp view
        self._timestamps = self._events.keys()

        if init is not None:
            self.push(init, time=np.int64(0))

    def push(self, value: typing.Any, *,
             time: typing.Optional[_T_T] = None, offset: _O_T = 0) -> None:
        # Normalize value and add value to the event buffer
        # An existing value at the same timestamp will be overwritten, just as the ARTIQ RTIO system does
        self._events[_get_timestamp(time, offset)] = self.normalize(value)

    def pull(self, *,
             time: typing.Optional[_T_T] = None, offset: _O_T = 0) -> _SV_T:
        if self._buffer:
            # Take an item from the buffer, push it, and return the value
            value = self._buffer.popleft()
            self.push(value, time=time, offset=offset)
            return value

        else:
            # Binary search for the insertion point (right) of the given timestamp
            index = self._events.bisect_right(_get_timestamp(time, offset))

            if index:
                # Return the value
                return self._events[self._timestamps[index - 1]]
            else:
                # Signal was not set, raise an exception
                raise SignalNotSetError(self, _get_timestamp(time, offset))

    def push_buffer(self, buffer: typing.Sequence[typing.Any]) -> None:
        """Push a buffer of values this signal.

        Values in the buffer will be pushed automatically at the next call to :func:`pull`. See also :func:`push`.

        :param buffer: The buffer of values to queue
        :raises ValueError: Raised if the value is invalid
        """
        # Add values to the push buffer
        self._buffer.extend(self.normalize(v) for v in buffer)

    def clear(self) -> None:
        """Clear buffers."""
        self._buffer.clear()
        self._events.clear()

    def horizon(self) -> _T_T:
        """Return the time horizon of this signal.

        See also :func:`DaxSignalManager.horizon`. For a :class:`PeekSignal`, the horizon is the timestamp of the
        latest event or a constant minimum timestamp value in case there are no events.

        :return: The time horizon in machine units
        """
        return self._events.keys()[-1] if self._events else _TIMESTAMP_MIN

    def __iter__(self) -> typing.Iterator[typing.Tuple[_T_T, _SV_T]]:
        """Return an iterator over the sorted events."""
        return iter(self._events.items())


class PeekSignalManager(DaxSignalManager[PeekSignal]):
    """Peek signal manager."""

    __slots__ = ()

    def _create_signal(self, scope: DaxSimDevice, name: str, type_: _ST_T, *,
                       size: _SS_T = None, init: typing.Optional[_SV_T] = None) -> PeekSignal:
        return PeekSignal(scope, name, type_, size, init=init)

    def horizon(self) -> _T_T:
        return max(max(signal.horizon() for signal in self), _get_timestamp())

    def flush(self, ref_period: float) -> None:
        pass

    def close(self) -> None:
        # Clear all signals
        for signal in self:
            signal.clear()

    def write_vcd(self, file_name: str, ref_period: float, **kwargs: typing.Any) -> None:
        """Write the contents of this signal manager into a VCD file.

        :param file_name: The file name of the VCD output file
        :param ref_period: The reference period (i.e. the time of one machine unit)
        :param kwargs: Keyword arguments passed to the VCD signal manager (see :class:`VcdSignalManager`)`
        """

        # Create a VCD signal manager
        vcd_ = VcdSignalManager(file_name, **kwargs)

        try:
            # Create signals
            signals = {s: vcd_.register(s.scope, s.name, s.type, size=s.size) for s in self}

            def repack(constant: typing.Any, iterator: typing.Iterator[typing.Tuple[typing.Any, ...]]) \
                    -> typing.Iterator[typing.Tuple[typing.Any, ...]]:
                for e in iterator:
                    # noinspection PyRedundantParentheses
                    yield (constant, *e)  # Parenthesis required for Python<3.8

            # Use a heap to merge the sorted events
            events = heapq.merge(*[repack(v, iter(s)) for s, v in signals.items()], key=operator.itemgetter(1))
            # Push all sorted events into the VCD signal manager
            for signal, time, value in events:
                signal.push(value, time=time)

        finally:
            # Flush and close the VCD signal manager
            vcd_.flush(ref_period)
            vcd_.close()


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
