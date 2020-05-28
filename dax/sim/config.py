import logging
import typing

from dax import __version__ as _dax_version
from dax.sim.signal import set_signal_manager, get_signal_manager, VcdSignalManager, PeekSignalManager
from dax.util.output import get_file_name

__all__ = ['DaxSimConfig']

_logger = logging.getLogger(__package__)
"""The logger for this file and the root logger for dax.sim."""


class DaxSimConfig:
    """Virtual device class that configures the simulation through the device DB."""

    def __init__(self, dmgr: typing.Any,
                 logging_level: typing.Union[int, str], output: typing.Optional[str],
                 signal_mgr_kwargs: typing.Dict[str, typing.Any]):
        assert isinstance(logging_level, (int, str)), 'Logging level must be of type int or str'
        assert isinstance(output, str) or output is None, 'Invalid type for output parameter'
        assert isinstance(signal_mgr_kwargs, dict), 'Signal manager kwargs must be of type dict'

        # Set the dax.sim logging level and report that simulation is enabled
        _logger.setLevel(logging_level)
        _logger.info('DAX.sim enabled ({:s})'.format(_dax_version))

        if output is None:
            # Disable output
            self.__output_enabled = False

        elif output == 'vcd':
            # Enable output
            self.__output_enabled = True

            # Generate output file name for the signal manager
            scheduler = dmgr.get('scheduler')
            output_file_name = get_file_name(scheduler, 'trace', 'vcd')

            # Set the VCD signal manager
            _logger.debug('Initializing VCD signal manager...')
            set_signal_manager(VcdSignalManager(output_file_name, **signal_mgr_kwargs))
            _logger.debug('VCD signal manager initialized')

        elif output == 'peek':
            # Disable output
            self.__output_enabled = False

            # Set the peek signal manager
            set_signal_manager(PeekSignalManager())

        else:
            # Output type was not supported
            raise ValueError('Unsupported output type "{:s}"'.format(output))

    @property
    def output_enabled(self) -> bool:
        """Returns `True` if modules should generate output."""
        return self.__output_enabled

    @staticmethod
    def close() -> None:
        """Close the simulation.

        The signal manager will be closed.
        """
        _logger.debug('Closing signal manager')
        get_signal_manager().close()
