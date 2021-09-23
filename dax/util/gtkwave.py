import typing
import datetime
import itertools
import operator
import logging
import vcd.gtkw

import dax.base.system
from dax import __version__ as _dax_version
from dax.sim.signal import get_signal_manager, Signal, DaxSignalManager
from dax.util.output import get_file_name

__all__ = ['GTKWSaveGenerator']

_logger: logging.Logger = logging.getLogger(__name__)
"""Module logger object."""


class GTKWSaveGenerator:
    """Generator for GTKWave save file based on a given DAX system object.

    This class integrates directly with DAX.sim to obtain
    signals while the DAX system object is used for the save file organization.
    """

    _GTKW_TYPE: typing.ClassVar[typing.Dict[type, str]] = {
        bool: 'hex',
        int: 'dec',
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

        # Get the signal manager
        signal_manager = get_signal_manager()
        _logger.debug(f'Signal manager type: {type(signal_manager).__name__}')
        if not isinstance(signal_manager, DaxSignalManager):
            raise RuntimeError('GTKWave safe file can only be generated when using DAX.sim')
        # Get the registered signals per device
        registered_devices = {scope.key: list(signals)
                              for scope, signals in itertools.groupby(signal_manager, operator.attrgetter('scope'))}
        _logger.debug(f'Found {len(registered_devices)} registered device(s)')

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

            for p, device_parent_iterator in parents:
                # Unpack iterator
                devices: typing.List[str] = [d for d, _ in device_parent_iterator]
                _logger.debug(f'Found {len(devices)} device(s) for parent "{p.get_system_key()}"')

                if devices:
                    # Create a group for this parent (Parents are not nested)
                    gtkw.comment(f'Parent "{p}"')
                    with gtkw.group(p.get_system_key(), closed=True):
                        for d in devices:
                            # Add signals for each device in this parent
                            signals = registered_devices.pop(d, None)

                            if signals is not None:
                                # Add signals
                                self._add_signals(gtkw, d, signals)

            if registered_devices:
                # Handle leftover registered signals
                gtkw.comment('Leftover signals')
                _logger.debug(f'Adding signals for leftover device(s): {list(registered_devices)}')

                for d, signals in registered_devices.items():
                    # Add signals
                    self._add_signals(gtkw, d, signals)

    @classmethod
    def _add_signals(cls, gtkw: vcd.gtkw.GTKWSave, device: str, signals: typing.Sequence[Signal]) -> None:
        """Add signals for a device.

        :param gtkw: The GTK Wave save file object
        :param device: The device key
        :param signals: Signals for this device
        """
        # Create a group for the signals of this device
        _logger.debug(f'Signals added for device "{device}": {", ".join([str(s) for s in signals])}')
        gtkw.comment(f'Signals for device "{device}"')
        with gtkw.group(device, closed=False):
            for s in signals:
                vector = '' if s.size is None or s.size == 1 else f'[{s.size - 1}:0]'
                # noinspection PyTypeChecker
                gtkw.trace(f'{s}{vector}', datafmt=cls._GTKW_TYPE[s.type])
