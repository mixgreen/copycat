import typing
import abc

import artiq.coredevice.core

__all__ = ['DaxSimDevice']


class DaxSimDevice(abc.ABC):
    """Abstract base class for simulated device drivers."""

    __key: str
    __core: artiq.coredevice.core.Core
    __kwargs: typing.Dict[str, typing.Any]

    def __init__(self, dmgr: typing.Any, *,
                 _key: str, _core: typing.Any = None, core_device: str = 'core', **kwargs: typing.Any):
        """Initialize a DAX simulation device.

        :param dmgr: The device manager, always first positional argument when ARTIQ constructs a device object
        :param core_device: ARTIQ default argument to change the device DB key of the core device
        :param _key: The key of this device, will be injected in the **kwargs arguments by DAX.sim
        :param _core: Used by DAX.sim to construct and pass the core object
        """

        assert isinstance(_key, str), 'Internal argument _key is expected to be type str'
        assert isinstance(core_device, str), 'Core device argument must be of type str'

        # Store device key
        self.__key = _key
        # Store core device (this will actually be a simulated core device, but we type it as an ARTIQ core device)
        self.__core = dmgr.get(core_device) if _core is None else _core

        # Store leftover kwargs, potentially useful for debugging
        self.__kwargs = kwargs

    @property
    def core(self) -> artiq.coredevice.core.Core:
        """Get the core object.

        :return: The core object
        """
        return self.__core

    @property
    def key(self) -> str:
        """Get the key of this device.

        :return: The unique device key as defined in the device DB
        """
        return self.__key

    def core_reset(self) -> None:
        """Called when ``core.reset()`` is used.

        By default this function does nothing.
        Mainly intended for clearing buffers of devices.
        """
        pass

    def close(self) -> None:
        """Called by ARTIQ to close the device.

        By default this function does nothing.
        """
        pass

    def __repr__(self) -> str:
        """Representation is the key of this device."""
        return self.key
