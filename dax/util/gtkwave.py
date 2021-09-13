import typing
import datetime
import itertools
import operator
import logging
import numpy as np
import vcd.gtkw

import dax.base.system
from dax import __version__ as _dax_version
from dax.sim.signal import get_signal_manager, VcdSignalManager
from dax.util.output import get_file_name

__all__ = ['GTKWSaveGenerator']

_logger: logging.Logger = logging.getLogger(__name__)
"""Module logger object."""


class GTKWSaveGenerator:
    """Generator for GTKWave save file based on a given DAX system object.

    This class integrates tightly with the :class:`VcdSignalManager` to obtain
    signals while the DAX system object is used for the save file organization.
    Functionality to generate a save file was explicitly decoupled from the
    :class:`VcdSignalManager` class to maintain separation between the DAX
    base code and the simulation backend.
    """

    _CONVERT_TYPE: typing.ClassVar[typing.Dict[type, str]] = {
        bool: 'hex',
        int: 'dec',
        np.int32: 'dec',
        np.int64: 'dec',
        float: 'real',
        str: 'ascii',
        object: 'hex',
    }
    """Dict to convert Python types to GTKWave types."""

    def __init__(self, system: dax.base.system.DaxSystem):
        """Instantiate a new GTKWave save file generator.

        :param system: The system of interest
        """
        assert isinstance(system, dax.base.system.DaxSystem)

        # Verify that we are in simulation
        _logger.debug(f'DAX.sim enabled: {system.dax_sim_enabled}')
        if not system.dax_sim_enabled:
            raise RuntimeError('GTKWave safe file can only be generated when dax.sim is enabled')

        # Get the signal manager and verify that the VCD signal manager is used
        signal_manager: VcdSignalManager = typing.cast(VcdSignalManager, get_signal_manager())
        _logger.debug(f'Signal manager type: {type(signal_manager).__name__}')
        if not isinstance(signal_manager, VcdSignalManager):
            raise RuntimeError('GTKWave safe file can only be generated when using the VCD signal manager')
        # Get the registered signals
        registered_signals = {k.key: v for k, v in signal_manager.get_registered_signals().items()}
        _logger.debug(f'Found {len(registered_signals)} registered signal(s)')

        # Generate file name
        file_name: str = get_file_name(system.get_device('scheduler'), 'waves', 'gtkw')

        with open(file_name, mode='w') as f:
            # Create save file object and add generic metadata
            gtkw = vcd.gtkw.GTKWSave(f)
            gtkw.comment(f'System ID: {system.SYS_ID}',
                         f'System version: {system.SYS_VER}',
                         datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                         f'DAX version: {_dax_version}')
            gtkw.sst_expanded(False)

            # Iterator of registered devices grouped by parent
            parents = itertools.groupby(system.registry.get_device_parents().items(), operator.itemgetter(1))

            for (p, device_parent_iterator) in parents:
                # Unpack iterator
                devices: typing.List[str] = [d for d, _ in device_parent_iterator]
                _logger.debug(f'Found {len(devices)} device(s) for parent "{p.get_system_key()}"')

                if devices:
                    # Create a group for this parent (Parents are not nested)
                    gtkw.comment(f'Parent "{p}"')
                    with gtkw.group(p.get_system_key(), closed=True):
                        for d in devices:
                            # Add signals for each device in this parent
                            signals = registered_signals.pop(d, None)

                            if signals is not None:
                                # Add signals
                                self._add_signals(gtkw, d, signals)

            if registered_signals:
                # Handle leftover registered signals
                gtkw.comment('Leftover signals')
                _logger.debug(f'Adding signals for leftover device(s): {list(registered_signals)}')

                for d, signals in registered_signals.items():
                    # Add signals
                    self._add_signals(gtkw, d, signals)

    @classmethod
    def _add_signals(cls, gtkw: vcd.gtkw.GTKWSave, device: str,
                     signals: typing.List[typing.Tuple[str, type, typing.Optional[int]]]) -> None:
        """Add signals for a device.

        :param gtkw: The GTK Wave save file object
        :param device: The device key
        :param signals: Tuple with signal information
        """
        # Create a group for the signals of this device
        _logger.debug(f'Signals added for device "{device}": {[s for s, _, _ in signals]}')
        gtkw.comment(f'Signals for device "{device}"')
        with gtkw.group(device, closed=False):
            for s, t, size in signals:
                vector = '' if size is None or size == 1 else f'[{size - 1}:0]'
                # noinspection PyTypeChecker
                gtkw.trace(f'{device}.{s}{vector}', datafmt=cls._CONVERT_TYPE[t])
