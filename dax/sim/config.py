import logging
import typing

from dax.sim.time import DaxTimeManager
from dax.sim.signal import set_signal_manager, get_signal_manager, VcdSignalManager

# The logger for this file
_logger: logging.Logger = logging.getLogger(__name__)


class DaxSimConfig:
    def __init__(self, dmgr: typing.Any,
                 logging_level: typing.Union[int, str], timescale: float):
        assert isinstance(logging_level, (int, str)), 'Logging level must be of type int or str'
        assert isinstance(timescale, float), 'Timescale must be of type float'

        # Store timescale
        self.__timescale: float = timescale

        # Set the logging level
        _logger.setLevel(logging_level)

        # Get The scheduler
        scheduler: typing.Any = dmgr.get('scheduler')

        # Set the signal manager
        _logger.debug('Initializing signal manager...')
        output_file: str = f'{scheduler.rid:09d}-{str(scheduler.expid.get("class_name"))}.vcd'
        set_signal_manager(VcdSignalManager(output_file, timescale))
        _logger.debug('Signal manager initialized')

        # Set the time manager in ARTIQ
        _logger.debug('Initializing time manager...')
        from artiq.language.core import set_time_manager
        set_time_manager(DaxTimeManager(timescale))
        _logger.debug('Time manager initialized')

    @property
    def timescale(self) -> float:
        """Return the timescale of the simulation."""
        return self.__timescale

    @staticmethod
    def close() -> None:
        # Close the signal manager
        _logger.debug('Closing signal manager')
        get_signal_manager().close()
