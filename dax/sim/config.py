import logging
import typing

from dax.sim.time import DaxTimeManager
from dax.sim.signal import set_signal_manager, get_signal_manager, VcdSignalManager

# The logger for this file
_logger: logging.Logger = logging.getLogger(__name__)


class DaxSimConfig:
    def __init__(self, dmgr: typing.Any,
                 logging_level: typing.Union[int, str], timescale: float, output: bool):
        assert isinstance(logging_level, (int, str)), 'Logging level must be of type int or str'
        assert isinstance(timescale, float), 'Timescale must be of type float'
        assert isinstance(output, bool), 'Output flag must be of type bool'

        # Set the logging level and report that simulation is enabled
        _logger.setLevel(logging_level)
        _logger.info('DAX simulation enabled')

        # Store attributes
        self.__timescale: float = timescale
        self.__output_enabled: bool = output

        # Make base name for output files
        scheduler: typing.Any = dmgr.get('scheduler')
        self.__base_name: str = f'{scheduler.rid:09d}-{str(scheduler.expid.get("class_name"))}'

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
        return self.__output_enabled

    def get_output_file_name(self, ext: str, postfix: typing.Optional[str] = None) -> str:
        assert isinstance(ext, str), 'File extension must be of type str'
        assert isinstance(postfix, str) or postfix is None, 'Postfix must be of type str or None'

        # Add postfix if provided
        output_file_name: str = self.__base_name if postfix is None else f'{self.__base_name:s}-{postfix:s}'
        # Return full file name
        return f'{output_file_name:s}.{ext:s}'

    @staticmethod
    def close() -> None:
        # Close the signal manager
        _logger.debug('Closing signal manager')
        get_signal_manager().close()
