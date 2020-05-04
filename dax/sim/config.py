import logging
import typing

from dax.sim.time import DaxTimeManager
from dax.sim.signal import set_signal_manager, get_signal_manager, VcdSignalManager

__all__ = ['DaxSimConfig']

_logger = logging.getLogger(__name__)
"""The logger for this file."""


class DaxSimConfig:
    """Virtual device class that configures the simulation through the device DB."""

    def __init__(self, dmgr: typing.Any,
                 logging_level: typing.Union[int, str], timescale: float, output: bool):
        assert isinstance(logging_level, (int, str)), 'Logging level must be of type int or str'
        assert isinstance(timescale, float), 'Timescale must be of type float'
        assert isinstance(output, bool), 'Output flag must be of type bool'

        # Set the logging level and report that simulation is enabled
        _logger.setLevel(logging_level)
        _logger.info('DAX simulation enabled')

        # Store attributes
        self.__timescale = timescale
        self.__output_enabled = output

        # Make base name for output files
        scheduler = dmgr.get('scheduler')
        self.__base_name = '{:09d}-{}'.format(scheduler.rid, str(scheduler.expid.get("class_name")))

        if self.output_enabled:
            # Set the signal manager
            set_signal_manager(VcdSignalManager(self.get_output_file_name('vcd', postfix='trace'), timescale))
            _logger.debug('VCD signal manager initialized')

        # Set the time manager in ARTIQ
        _logger.debug('Initializing DAX time manager...')
        from artiq.language.core import set_time_manager
        set_time_manager(DaxTimeManager(timescale))
        _logger.debug('DAX time manager initialized')

    @property
    def timescale(self) -> float:
        """Return the timescale of the simulation."""
        return self.__timescale

    @property
    def output_enabled(self) -> bool:
        """Returns `True` if the user requested simulation output.

        :return: True if the `output_enabled` flag was set
        """
        return self.__output_enabled

    def get_output_file_name(self, ext: str, postfix: typing.Optional[str] = None) -> str:
        """ Convenience function to generate uniformly styled output file names.

        :param ext: The extension of the file
        :param postfix: A postfix for the base file name
        :return: A file name
        """
        assert isinstance(ext, str), 'File extension must be of type str'
        assert isinstance(postfix, str) or postfix is None, 'Postfix must be of type str or None'

        # Add postfix if provided
        output_file_name = self.__base_name if postfix is None else '{:s}-{:s}'.format(self.__base_name, postfix)
        # Return full file name
        return '{:s}.{:s}'.format(output_file_name, ext)

    @staticmethod
    def close() -> None:
        """Close the simulation.

        The signal manager will be closed.
        """
        _logger.debug('Closing signal manager')
        get_signal_manager().close()
