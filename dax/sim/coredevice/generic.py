import typing

from dax.sim.device import DaxSimDevice
from dax.sim.signal import get_signal_manager, DaxSignalManager


class _GenericBase:

    def __init__(self, attr_name: typing.Optional[str],
                 signal_manager: DaxSignalManager[typing.Any],
                 signal_call: typing.Any, signal_function: typing.Any):
        assert isinstance(attr_name, str) or attr_name is None, 'Attribute name must be of type str or None'

        # Store attributes
        self._attr_name = attr_name
        self._signal_manager = signal_manager
        self._signal_call = signal_call
        self._signal_function = signal_function

    def __getattr__(self, item: str) -> typing.Any:
        # Non-existing attributes are added
        attr_name = item if self._attr_name is None else '.'.join([self._attr_name, item])
        obj = _GenericBase(attr_name, self._signal_manager, self._signal_call, self._signal_function)
        setattr(self, item, obj)
        return obj

    def __call__(self, *args: typing.Tuple[typing.Any, ...], **kwargs: typing.Dict[str, typing.Any]) -> None:
        # Make a string for the parameters
        parameters = '{:s}{:s}{:s}'.format(','.join(str(a) for a in args),
                                           ',' if args and kwargs else '',
                                           ','.join('{:s}={}'.format(k, v) for k, v in kwargs.items()))

        # Register the event
        self._signal_manager.event(self._signal_call, True)  # Register the timestamp of the call
        self._signal_manager.event(self._signal_function, '{:s}({})'.format(self._attr_name, parameters))


class Generic(_GenericBase, DaxSimDevice):

    def __init__(self, dmgr: typing.Any, **kwargs: typing.Any):
        # Call super for DaxSimDevice
        DaxSimDevice.__init__(self, dmgr, **kwargs)

        # Register signal
        signal_manager = get_signal_manager()
        signal_call = signal_manager.register(self, 'call', object)  # type: typing.Any
        signal_function = signal_manager.register(self, 'function', str)  # type: typing.Any

        # Call super for _GenericBase
        _GenericBase.__init__(self, None, signal_manager, signal_call, signal_function)

    def __call__(self, *args: typing.Tuple[typing.Any, ...], **kwargs: typing.Dict[str, typing.Any]) -> None:
        # The device can not be directly called, only its attributes
        raise TypeError('Generic device {:s} is not callable, only its attributes are'.format(self.key))
