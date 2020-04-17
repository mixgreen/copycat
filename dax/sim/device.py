import typing

import artiq.coredevice.core  # type: ignore


class DaxSimDevice:
    def __init__(self, dmgr: typing.Any, _key: str,
                 _core: typing.Any = None, core_device: str = 'core',
                 **kwargs: typing.Dict[str, typing.Any]):
        """Initialize a DAX simulation device.

        :param dmgr: The device manager, always first positional argument when ARTIQ constructs a device object
        :param core_device: ARTIQ default argument to change the device DB key of the core device
        :param _key: The key of this device, will be injected in the **kwargs arguments by dax.sim
        :param _core: Used by dax.sim to construct and pass the core object
        """

        assert isinstance(_key, str), 'Internal argument _key is expected to be type str'
        assert isinstance(core_device, str), 'Core device argument must be of type str'

        # Store device key
        self.__key: str = _key
        # Store core device
        self.__core = dmgr.get(core_device) if _core is None else _core

        # Store leftover kwargs, potentially useful for debugging
        self.__kwargs = kwargs

    @property
    def core(self) -> artiq.coredevice.core.Core:
        """Get the core object.

        :returns: The core object
        """
        return self.__core

    @property
    def key(self) -> str:
        """Get the key of this device.

        :return: The unique device key as defined in the device DB
        """
        return self.__key

    def close(self) -> None:
        """Called by ARTIQ to close the device.

        By default this function does nothing.
        """
        pass
