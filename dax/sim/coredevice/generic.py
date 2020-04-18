import typing

from dax.sim.coredevice import *
from dax.sim.signal import DaxSignalManager


class _GenericBase:

    def __init__(self, attr_name: typing.Optional[str],
                 signal_manager: DaxSignalManager, signal_call, signal_args, signal_kwargs):
        assert isinstance(attr_name, str) or attr_name is None, 'Attribute name must be of type str or None'

        # Store attributes
        self._attr_name: typing.Optional[str] = attr_name
        self._signal_manager: DaxSignalManager = signal_manager
        self._signal_call = signal_call
        self._signal_args = signal_args
        self._signal_kwargs = signal_kwargs

    def __getattr__(self, item: str) -> typing.Any:
        # Non-existing attributes are added
        attr_name: str = item if self._attr_name is None else '.'.join([self._attr_name, item])
        obj: typing.Any = _GenericBase(attr_name, self._signal_manager,
                                       self._signal_call, self._signal_args, self._signal_kwargs)
        setattr(self, item, obj)
        return obj

    def __call__(self, *args: typing.Tuple[typing.Any, ...], **kwargs: typing.Dict[str, typing.Any]):
        # Register an event
        self._signal_manager.event(self._signal_call, self._attr_name)
        self._signal_manager.event(self._signal_args, str(args).replace(' ', ''))
        self._signal_manager.event(self._signal_kwargs, str(kwargs).replace(' ', ''))


class Generic(_GenericBase, DaxSimDevice):

    def __init__(self, dmgr, **kwargs):
        # Call super for DaxSimDevice
        DaxSimDevice.__init__(self, dmgr, **kwargs)

        # Register signal
        self._signal_manager = get_signal_manager()
        signal_call = self._signal_manager.register(self.key, 'call', str)
        signal_args = self._signal_manager.register(self.key, 'args', str)
        signal_kwargs = self._signal_manager.register(self.key, 'kwargs', str)

        # Call super for _GenericBase
        _GenericBase.__init__(self, None, self._signal_manager, signal_call, signal_args, signal_kwargs)

    def __call__(self, *args: typing.Tuple[typing.Any, ...], **kwargs: typing.Dict[str, typing.Any]):
        # The device can not be directly called, only its attributes
        raise TypeError('Generic device {:s} is not callable, only its attributes are'.format(self.key))
