import typing
import vcd.gtkw
import datetime
import itertools
import operator

from dax import __version__ as _dax_version
from dax.base.dax import DaxSystem
from dax.sim.signal import get_signal_manager, VcdSignalManager
from dax.util.output import get_file_name

__all__ = ['GTKWSaveGenerator']


class GTKWSaveGenerator:
    """Generator for GTKWave save files based on a given DAX system object."""

    def __init__(self, system: DaxSystem):
        """Instantiate a new GTKWave save file generator.

        :param system: The system of interest
        """
        assert isinstance(system, DaxSystem)

        # Verify that we are in simulation
        if not system.dax_sim_enabled:
            raise RuntimeError('GTKWave safe file can only be generated when dax.sim is enabled')

        # Get the signal manager and verify that the VCD signal manager is used
        signal_manager: VcdSignalManager = typing.cast(VcdSignalManager, get_signal_manager())
        if not isinstance(signal_manager, VcdSignalManager):
            raise RuntimeError('GTKWave safe file can only be generated when using the VCD signal manager')
        # Get the registered signals
        registered_signals = {k.key: v for k, v in signal_manager.get_registered_signals().items()}

        # Generate file name
        file_name: str = get_file_name(system.get_device('scheduler'), 'waves', 'gtkw')

        with open(file_name, mode='w') as f:
            # Create save file object and add generic metadata
            gtkw = vcd.gtkw.GTKWSave(f)
            gtkw.comment(f'System ID: {system.SYS_ID}',
                         f'System version: {system.SYS_VER}',
                         datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                         f'DAX version: {_dax_version}')
            gtkw.savefile(file_name)

            # Iterator of registered devices grouped by parent
            parents = itertools.groupby(system.registry.get_device_parents().items(), operator.itemgetter(1))

            for (p, device_parent_iterator) in parents:
                # Unpack iterator
                devices: typing.List[str] = [d for d, _ in device_parent_iterator]

                if devices:
                    # Create a group for this parent (Parents are not nested)
                    with gtkw.group(p.get_system_key(), closed=True):
                        for d in devices:
                            # Add signals for each device in this parent
                            signals = registered_signals.get(d)

                            if signals is not None:
                                # Create a group for the signals of this device
                                with gtkw.group(d, closed=True):
                                    for s in signals:
                                        gtkw.trace(f'{d}.{s}')
