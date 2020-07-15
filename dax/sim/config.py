import logging
import typing

from dax import __version__ as _dax_version
from dax.sim.signal import set_signal_manager, get_signal_manager, VcdSignalManager, PeekSignalManager
from dax.util.output import get_file_name

__all__ = ['DaxSimConfig']

_logger: logging.Logger = logging.getLogger(__package__)
"""The logger for this file and the root logger for dax.sim."""


class DaxSimConfig:
    """Virtual device class that configures the simulation through the device DB."""

    def __init__(self, dmgr: typing.Any, *,
                 logging_level: typing.Union[int, str], output: str,
                 signal_mgr_kwargs: typing.Dict[str, typing.Any]):
        assert isinstance(logging_level, (int, str)), 'Logging level must be of type int or str'
        assert isinstance(output, str), 'Output must be of type str'
        assert isinstance(signal_mgr_kwargs, dict), 'Signal manager kwargs must be of type dict'

        # Set the dax.sim logging level and report that simulation is enabled
        _logger.setLevel(logging_level)
        _logger.info(f'DAX.sim enabled ({_dax_version:s})')

        if output == 'null':
            # Disable output
            self.__output_enabled = False

        elif output == 'vcd':
            # Enable output
            self.__output_enabled = True

            # Generate output file name for the signal manager
            output_file_name = get_file_name(dmgr.get('scheduler'), 'trace', 'vcd')

            # Set the VCD signal manager
            _logger.debug('Initializing VCD signal manager...')
            set_signal_manager(VcdSignalManager(output_file_name, **signal_mgr_kwargs))
            _logger.debug('VCD signal manager initialized')

        elif output == 'peek':
            # Disable output
            self.__output_enabled = False

            # Set the peek signal manager
            _logger.debug('Initializing peek signal manager...')
            if signal_mgr_kwargs:
                raise TypeError('PeekSignalManager() takes no arguments')
            set_signal_manager(PeekSignalManager())
            _logger.debug('Peek signal manager initialized')

        else:
            # Output type was not supported
            raise ValueError(f'Unsupported output type "{output:s}"')

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
