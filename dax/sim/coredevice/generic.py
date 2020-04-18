import typing

from dax.sim.coredevice import *
from dax.sim.signal import DaxSignalManager


class _GenericBase:

    def __init__(self, attr_name: typing.Optional[str],
                 signal_manager: DaxSignalManager, signal_call, signal_function):
        assert isinstance(attr_name, str) or attr_name is None, 'Attribute name must be of type str or None'

        # Store attributes
        self._attr_name: typing.Optional[str] = attr_name
        self._signal_manager: DaxSignalManager = signal_manager
        self._signal_call = signal_call
        self._signal_function = signal_function

    def __getattr__(self, item: str) -> typing.Any:
        # Non-existing attributes are added
        attr_name: str = item if self._attr_name is None else '.'.join([self._attr_name, item])
        obj: typing.Any = _GenericBase(attr_name, self._signal_manager,
                                       self._signal_call, self._signal_function)
        setattr(self, item, obj)
        return obj

    def __call__(self, *args: typing.Tuple[typing.Any, ...], **kwargs: typing.Dict[str, typing.Any]):
        # Make a string for the parameters
        parameters: str = f'{",".join(str(a) for a in args)}' \
                          f'{"," if args and kwargs else ""}' \
                          f'{",".join(f"{k:s}={v}" for k, v in kwargs.items())}'

        # Register the event
        self._signal_manager.event(self._signal_call, None)  # Register the timestamp of the call
        self._signal_manager.event(self._signal_function, f'{self._attr_name:s}({parameters})')


class Generic(_GenericBase, DaxSimDevice):

    def __init__(self, dmgr, **kwargs):
        # Call super for DaxSimDevice
        DaxSimDevice.__init__(self, dmgr, **kwargs)

        # Register signal
        self._signal_manager = get_signal_manager()
        signal_call = self._signal_manager.register(self.key, 'call', object)
        signal_function = self._signal_manager.register(self.key, 'function', str)

        # Call super for _GenericBase
        _GenericBase.__init__(self, None, self._signal_manager, signal_call, signal_function)

    def __call__(self, *args: typing.Tuple[typing.Any, ...], **kwargs: typing.Dict[str, typing.Any]):
        # The device can not be directly called, only its attributes
        raise TypeError('Generic device {:s} is not callable, only its attributes are'.format(self.key))
