from __future__ import annotations  # Postponed evaluation of annotations
from functools import lru_cache

import math
import typing
import pathlib
import numpy as np

from dax.experiment import *
from trap_dac_utils.reader import BaseReader, SpecialCharacter, SOLUTION_T

import artiq.coredevice.zotino  # type: ignore[import]
import artiq.coredevice.ad53xx  # type: ignore[import]

"""Zotino Path and Line types"""
_ZOTINO_KEY_T = typing.List[float]
_ZOTINO_KEY_T_MU = typing.List[int]
_ZOTINO_VALUE_T = typing.List[int]
_ZOTINO_LINE_T = typing.Tuple[_ZOTINO_KEY_T, _ZOTINO_VALUE_T]
_ZOTINO_SOLUTION_T = typing.List[_ZOTINO_LINE_T]
_ZOTINO_LINE_T_MU = typing.Tuple[_ZOTINO_KEY_T_MU, _ZOTINO_VALUE_T]
_ZOTINO_SOLUTION_T_MU = typing.List[_ZOTINO_LINE_T_MU]

__all__ = ['TrapDcModule', 'ZotinoReader']


class TrapDcModule(DaxModule):
    """A trap DC module using a Zotino device, inheriting from AD53XX.

    This module controls a Zotino used for trap DC. The device has 32 channels of DC voltage output that can be set
    using functions from this module.

    Solution files, which can be .csv or generated by a .py file, can be read into a python object and compressed
    for efficient zotino output. A map file is used to connect the output channels with the solution file pins.

    Using the prepared solution files, this module provides functions to shuttle these solutions at a predefined
    speed. There is the option to use DMA caching to lower the necessary amount of slack to prepend to your
    experiments.

    Notes when considering using this module:

    - Functions are provided to return the expected amount of slack needed to shuttle a solution at a given speed
      without underflow. However, this is meant to be an approximate calculation and can be configured as needed.
    - Everything in this module is Zotino specific. As other DC traps are needed they should be created separately.
    """

    _DMA_STARTUP_TIME: typing.ClassVar[float] = 1.728 * us
    """Startup time for DMA (s). Measured in the RTIO benchmarking tests during CI"""

    _zotino: artiq.coredevice.zotino.Zotino
    _solution_path: pathlib.Path
    _map_file: pathlib.Path
    _reader: ZotinoReader
    _min_line_delay_mu: np.int64
    _calculator: ZotinoCalculator

    def build(self,  # type: ignore[override]
              *,
              key: str,
              solution_path: str,
              map_file: str) -> None:
        """Build the trap DC module

        :param key: The key of the zotino device
        :param solution_path: The path name of the solution file directory
        :param map_file: The path name of a single map file
        """
        assert isinstance(key, str)
        assert isinstance(solution_path, str)
        assert isinstance(map_file, str)

        # Get devices
        self._zotino = self.get_device(key, artiq.coredevice.zotino.Zotino)
        self.update_kernel_invariants('_zotino')

        # Get the solution path
        self._solution_path = pathlib.Path(solution_path)

        # map file is the relative map file path
        self._map_file = pathlib.Path(map_file)

        # Initialize Zotino Reader
        self._reader = ZotinoReader(
            self._solution_path, self._map_file)

    @host_only
    def init(self) -> None:
        """Initialize this module."""
        # Get profile loader
        # Below calculated from set_dac_mu and load functions
        # https://m-labs.hk/artiq/manual/_modules/artiq/coredevice/ad53xx.html#AD53xx
        self._min_line_delay_mu = np.int64(self.core.seconds_to_mu(1500 * ns)
                                           + 2 * self._zotino.bus.ref_period_mu
                                           + len(self._reader._list_map_labels())
                                           * self._zotino.bus.xfer_duration_mu)
        self.update_kernel_invariants('_min_line_delay_mu')
        self._reader.init(self._zotino)
        self._calculator = ZotinoCalculator(np.int64(self.core.seconds_to_mu(self._DMA_STARTUP_TIME)))

    @host_only
    def post_init(self) -> None:
        pass

    @property
    def solution_path(self) -> str:
        """Get the solution path

        :return: The path to the solution file directory
        """
        return self._reader.solution_path

    @host_only
    def read_line_mu(self,
                     file_name: str,
                     index: int = 0,
                     multiplier: float = 1.0) -> _ZOTINO_LINE_T_MU:
        """Read in a single line of a solutions file and return the line in zotino form.
        Optionally apply multiplier to all voltages in path

        Note that the Zotino Path Voltages are given in **MU**.

        :param file_name: Solution file to parse the path from
        :param index: Line in path to get. A 0 indicates the first line
        :param multiplier: Optionally scale the voltages by a constant

        :return: Zotino module interpretable solution line with voltages in MU
        """
        path = self._read_line(file_name, index, multiplier)
        path_mu = (self._reader.convert_to_mu(path[0]), path[1])
        return path_mu

    @host_only
    def _read_line(self,
                   file_name: str,
                   index: int = 0,
                   multiplier: float = 1.0) -> _ZOTINO_LINE_T:
        """Read in a single line of a solutions file and return the line in zotino form.
        Optionally apply multiplier to all voltages in path

        Note that the Zotino Path Voltages are given in **V**.

        :param file_name: Solution file to parse the path from
        :param index: Line in path to get. A 0 indicates the first line
        :param multiplier: Optionally scale the voltages by a constant

        :return: Zotino module interpretable solution line with voltages in V
        """
        unprepared_line = self._reader.process_solution(self._reader.read_solution(file_name))[index]

        # multiply each solution list with multiplier
        line = (
            (np.asarray(unprepared_line[0]) * multiplier).tolist(),  # type: ignore[attr-defined]
            unprepared_line[1])

        return line

    @host_only
    def read_solution_mu(self,
                         file_name: str,
                         start: int = 0,
                         end: int = -1,
                         reverse: bool = False,
                         multiplier: float = 1.0) -> _ZOTINO_SOLUTION_T_MU:
        """Read in a segment of a solutions file and return the path in zotino form.
        Optionally reverse path and/or apply multiplier to all voltages in path

        Note that the Zotino Path Voltages are given in **MU**.

        :param file_name: Solution file to parse the path from
        :param start: Starting index of path (inclusive). Default 0 signals to start with first solution line
        :param end: End index of path (inclusive). Default -1 signals to end with last solution line
        :param reverse: Optionally return a reversed path. I.E. From end to start
        :param multiplier: Optionally scale the voltages by a constant

        :return: Zotino module interpretable solution path with voltages in MU
        """
        path = self._read_solution(file_name, start, end,
                                   reverse, multiplier)
        return self._reader.convert_solution_to_mu(path)

    @host_only
    def _read_solution(self,
                       file_name: str,
                       start: int = 0,
                       end: int = -1,
                       reverse: bool = False,
                       multiplier: float = 1.0) -> _ZOTINO_SOLUTION_T:
        """Read in a segment of a solutions file and return the path in zotino form.
        Optionally reverse path and/or apply multiplier to all voltages in path

        Note that the Zotino Path Voltages are given in **V**.

        :param file_name: Solution file to parse the path from
        :param start: Starting index of path (inclusive). Default 0 signals to start with first solution line
        :param end: End index of path (inclusive). Default -1 signals to end with last solution line
        :param reverse: Optionally return a reversed path. I.E. From end to start
        :param multiplier: Optionally scale the voltages by a constant

        :return: Zotino module interpretable solution path with voltages in V
        """

        solution = self._reader.process_solution(self._reader.read_solution(file_name))
        if end == -1:
            end = len(solution) - 1

        # multiply each solution list with multiplier
        for i, t in enumerate(solution):
            solution[i] = (
                (np.asarray(t[0]) * multiplier).tolist(), t[1])  # type: ignore[attr-defined]

        trimmed_solution = solution[start:end + 1]
        if reverse:
            trimmed_solution.reverse()

        path: _ZOTINO_SOLUTION_T = [trimmed_solution[0]]
        path.extend([self._reader.get_line_diff(t, trimmed_solution[i])
                     for i, t in enumerate(trimmed_solution[1:])])

        return path

    @host_only
    def list_solutions(self) -> typing.Sequence[str]:
        """Get a list of each solution file available in the solutions
        directory

        :return: The list of names of solution files available
        """

        return self._reader.list_solutions()

    @kernel
    def record_dma(self,
                   name: TStr,
                   solution: TList(TTuple([TList(TInt32), TList(TInt32)])),  # type: ignore[valid-type]
                   line_delay: TFloat) -> TStr:
        """Record the setting of sequential lines of voltages on the zotino device given a list
        of voltages (MU) and corresponding channels

        :param name: Name of DMA trace
        :param solution: A list of voltage lines to set and corresponding channels for each line
        :param line_delay: A delay (s) inserted after the line is set
            Must be greater than the SPI write time for the number of used channels

        :return: Unique key for DMA Trace
        """
        return self.record_dma_mu(name,
                                  solution,
                                  self.core.seconds_to_mu(line_delay))

    @kernel
    def record_dma_mu(self,
                      name: TStr,
                      solution: TList(TTuple([TList(TInt32), TList(TInt32)])),  # type: ignore[valid-type]
                      line_delay: TInt64) -> TStr:
        """Record the setting of sequential lines of voltages on the zotino device given a list
        of voltages (MU) and corresponding channels

        :param name: Name of DMA trace
        :param solution: A list of voltage lines to set and corresponding channels for each line
        :param line_delay: A delay (MU) inserted after the line is set
            Must be greater than the SPI write time for the number of used channels

        :return: Unique key for DMA Trace
        """
        if line_delay <= self._min_line_delay_mu:
            raise ValueError(f"Line Delay must be greater than {self._min_line_delay_mu}")
        dma_name = self.get_system_key(name)
        with self.core_dma.record(dma_name):
            for t in solution:
                self.set_line(t)
                delay_mu(line_delay)
        return dma_name

    @kernel
    def record_dma_rate(self,
                        name: TStr,
                        solution: TList(TTuple([TList(TInt32), TList(TInt32)])),  # type: ignore[valid-type]
                        line_rate: TFloat) -> TStr:
        """Record the setting of sequential lines of voltages on the zotino device given a list
        of voltages (MU) and corresponding channels

        :param name: Name of DMA trace
        :param solution: A list of voltage lines to set and corresponding channels for each line
        :param line_rate: A rate (Hz) to define speed to set each line
            Must be greater than the SPI write time for the number of used channels

        :return: Unique key for DMA Trace
        """
        return self.record_dma_mu(name,
                                  solution,
                                  self.core.seconds_to_mu(1.0 / line_rate))

    @kernel
    def get_dma_handle(self, key: TStr) -> TTuple([TInt32, TInt64, TInt32]):  # type: ignore[valid-type]
        """Get the DMA handle associated with the name of the recording

        :param key: Unique key of the recording

        :return: Handle used to playback the DMA Recording
        """
        return self.core_dma.get_handle(key)

    @kernel
    def shuttle_dma(self, key: TStr):
        """Play back a DMA recording specified by the key

        :param key: The key of the DMA recording to directly play back
        """
        self.core_dma.playback(key)

    @kernel
    def shuttle_dma_handle(self, handle: TTuple([TInt32, TInt64, TInt32])):  # type: ignore[valid-type]
        """Play back a DMA recording specified by the handle

        :param handle: The handle of the DMA recording to directly play back
        """
        self.core_dma.playback_handle(handle)

    @kernel
    def shuttle(self,
                solution: TList(TTuple([TList(TInt32), TList(TInt32)])),  # type: ignore[valid-type]
                line_delay: TFloat):
        """Set sequential lines of voltages on the zotino device given a list of voltages (MU) and
        corresponding channels

        :param solution: A list of voltage lines to set and corresponding channels for each line
        :param line_delay: A delay (s) inserted after the line is set
            Must be greater than the SPI write time for the number of used channels
        """
        self.shuttle_mu(solution, self.core.seconds_to_mu(line_delay))

    @kernel
    def shuttle_mu(self,
                   solution: TList(TTuple([TList(TInt32), TList(TInt32)])),  # type: ignore[valid-type]
                   line_delay: TInt64):
        """Set sequential lines of voltages on the zotino device given a list of voltages (MU) and
        corresponding channels

        :param solution: A list of voltage lines to set and corresponding channels for each line
        :param line_delay: A delay (MU) inserted after the line is set
            Must be greater than the SPI write time for the number of used channels
        """
        if line_delay <= self._min_line_delay_mu:
            raise ValueError(f"Line Delay must be greater than {self._min_line_delay_mu}")
        for t in solution:
            self.set_line(t)
            delay_mu(line_delay)

    @kernel
    def shuttle_rate(self,
                     solution: TList(TTuple([TList(TInt32), TList(TInt32)])),  # type: ignore[valid-type]
                     line_rate: TFloat):
        """Set sequential lines of voltages on the zotino device given a list of voltages (MU) and
        corresponding channels

        :param solution: A list of voltage lines to set and corresponding channels for each line
        :param line_rate: A rate (Hz) to define speed to set each line
            Must be greater than the SPI write time for the number of used channels
        """
        self.shuttle_mu(solution, self.core.seconds_to_mu(1 / line_rate))
        return

    @kernel
    def set_line(self,
                 line: TTuple([TList(TInt32), TList(TInt32)])):  # type: ignore[valid-type]
        """Set a line of voltages on the zotino device given a list of voltages (MU) and corresponding channels

        :param line: Up to 32 (# of Zotino channels) voltages and corresponding channel numbers
        """
        voltages, channels = line
        self._zotino.set_dac_mu(voltages, channels)

    @host_only
    def calculate_slack(self,
                        solution: _ZOTINO_SOLUTION_T_MU,
                        line_delay: float) -> float:
        """Calculate the slack required to shuttle solution with desired delay
        This method is used to prevent underflow when shuttling solutions
        If the desired line delay is >> than the communication delay, then the default amount
        of slack may be sufficient

        :param solution: The desired solution to shuttle
        :param line_delay: The desired line delay (s) to shuttle solution with

        :return: The necessary slack (s) to shuttle solution"""
        return self.core.mu_to_seconds(
            self.calculate_slack_mu(solution,
                                    self.core.seconds_to_mu(line_delay)))

    @host_only
    def calculate_slack_mu(self,
                           solution: _ZOTINO_SOLUTION_T_MU,
                           line_delay: np.int64) -> np.int64:
        """Calculate the slack required to shuttle solution with desired delay
        This method is used to prevent underflow when shuttling solutions
        If the desired line delay is >> than the communication delay, then the default amount
        of slack may be sufficient

        :param solution: The desired solution to shuttle
        :param line_delay: The desired line delay (MU) to shuttle solution with

        :return: The necessary slack (MU) to shuttle solution"""
        if line_delay < self._min_line_delay_mu:
            raise ValueError(f"Line Delay must be greater than {self._min_line_delay_mu}")
        return self._calculator.slack_mu(self._list_num_channels(solution),
                                         line_delay,
                                         self._min_line_delay_mu)

    @host_only
    def calculate_dma_slack(self,
                            solution: _ZOTINO_SOLUTION_T_MU,
                            line_delay: float) -> float:
        """Calculate the slack required to shuttle solution with dma and with desired delay
        This method is used to prevent underflow when shuttling solutions
        If the desired line delay is >> than the communication delay, then the default amount
        of slack may be sufficient

        :param solution: The desired solution to shuttle
        :param line_delay: The desired line delay (s) to shuttle solution with

        :return: The necessary slack (s) to shuttle solution"""
        return self.core.mu_to_seconds(
            self.calculate_dma_slack_mu(solution,
                                        self.core.seconds_to_mu(line_delay)))

    @host_only
    def calculate_dma_slack_mu(self,
                               solution: _ZOTINO_SOLUTION_T_MU,
                               line_delay: np.int64) -> np.int64:
        """Calculate the slack required to shuttle solution with dma and with desired delay
        This method is used to prevent underflow when shuttling solutions
        If the desired line delay is >> than the communication delay, then the default amount
        of slack may be sufficient

        :param solution: The desired solution to shuttle
        :param line_delay: The desired line delay (MU) to shuttle solution with

        :return: The necessary slack (MU) to shuttle solution"""
        if line_delay < self._min_line_delay_mu:
            raise ValueError(f"Line Delay must be greater than {self._min_line_delay_mu}")
        return self._calculator.slack_mu(self._list_num_channels(solution),
                                         line_delay,
                                         self._min_line_delay_mu,
                                         True)

    @host_only
    def _list_num_channels(self, solution: _ZOTINO_SOLUTION_T_MU) -> typing.Sequence[int]:
        """Given a zotino solution, list the length of each row in terms of number of channels

        :param solution: Any zotino solution

        :return: A list of number of channels that need to be set for each row"""
        return [len(t[0]) for t in solution]

    @host_only
    def configure_calculator(self,
                             *,
                             dma_startup_time: typing.Optional[float] = None,
                             comm_delay_intercept_mu: typing.Optional[np.int64] = None,
                             comm_delay_slope_mu: typing.Optional[np.int64] = None,
                             dma_comm_delay_intercept_mu: typing.Optional[np.int64] = None,
                             dma_comm_delay_slope_mu: typing.Optional[np.int64] = None) -> None:
        """Configure measured parameters that will affect slack calculations
        Each configuration is set if and only if the argument is passed in and is not None
        All original values were calculated from benchmarking

        :param dma_startup_time: The time it takes for DMA to start up in (s)
        :param comm_delay_intercept_mu: The intercept of the linear communication time between
            artiq and the kernel as a function of total channels
        :param comm_delay_slope_mu: The slope of the linear communication time between
            artiq and the kernel as a function of total channels
        :param dma_comm_delay_intercept_mu: The intercept of the linear communication time between
            artiq and the kernel for dma playback as a function of total channels
        :param dma_comm_delay_slope_mu: The slope of the linear communication time between
            artiq and the kernel for dma playback as a function of total channels
        """
        dma_startup_time_mu = None if dma_startup_time is None else np.int64(self.core.seconds_to_mu(dma_startup_time))
        self._calculator.configure(dma_startup_time_mu=dma_startup_time_mu,
                                   comm_delay_intercept_mu=comm_delay_intercept_mu,
                                   comm_delay_slope_mu=comm_delay_slope_mu,
                                   dma_comm_delay_intercept_mu=dma_comm_delay_intercept_mu,
                                   dma_comm_delay_slope_mu=dma_comm_delay_slope_mu)


class ZotinoCalculator:
    """This class is used to calculate the Zotino specific slack needed to shuttle solutions.

    The slack needed is calculated using core communications measurements as well as SPI
    communication measurements.

    The parameters used to calculate this slack can be overwritten through configuration
    if desired.

    The calculations done here are not a guarantee of solution shuttling success but can provide
    a helpful baseline.
    """

    _dma_startup_time_mu: np.int64
    _comm_delay_intercept_mu: np.int64
    _comm_delay_slope_mu: np.int64
    _dma_comm_delay_intercept_mu: np.int64
    _dma_comm_delay_slope_mu: np.int64

    def __init__(self, dma_startup_time_mu: np.int64):

        assert isinstance(dma_startup_time_mu, np.int64)
        assert dma_startup_time_mu > 0

        self._dma_startup_time_mu = dma_startup_time_mu
        self._comm_delay_intercept_mu = 33800
        self._comm_delay_slope_mu = 821
        self._dma_comm_delay_intercept_mu = 291
        self._dma_comm_delay_slope_mu = 131

    @host_only
    @lru_cache(maxsize=32)
    def _calculate_line_comm_delay_mu(self, num_channels: np.int64, dma: bool = False) -> np.int64:
        """Calculates the expected average communications delay when callng zotino.set_dac_mu
        Delay is a linear function of the number of channels being updated
        Linear line delay fit found from repeated Zotino benchmarking

        :param num_channels: Number of channels used to calculate expected avg delay
        :param dma: Should be true if calculating delay for DMA, otherwise false. Default is false

        :return: The expected average line delay for updating num_channels"""
        # linear line delay fit found from measurements on Zotino
        if dma:
            return self._dma_comm_delay_intercept_mu + self._dma_comm_delay_slope_mu * num_channels
        else:
            return self._comm_delay_intercept_mu + self._comm_delay_slope_mu * num_channels

    @host_only
    def slack_mu(self,
                 row_lens: typing.Sequence[np.int64],
                 line_delay_mu: np.int64,
                 offset_mu: np.int64,
                 dma: bool = False) -> np.int64:
        """This function calculates the required slack for a given solution and desired line delay
        All calculations are done in MU

        :param row_lens: The number of voltages to be sent for each row in the solution
        :param line_delay_mu: The desired line delay for shuttling in MU
        :param offset_mu: The slack offset which is a baseline for the wall clock time and cursor difference
        :param dma: Should be true if running experiments with DMA, otherwise false. Default is false

        :return: The amount of slack needed in MU to shuttle a solution of this form
        """
        # start with initial slack for the first line
        current_slack = 0
        added_slack = self._calculate_line_comm_delay_mu(row_lens[0], dma)
        # DMA startup time calculated from benchmark measurement
        if dma:
            added_slack += self._dma_startup_time_mu

        # Each line must delay long enough to account for the communication delay
        # If they do not, slack must be added at the beginning of experiment to account for this
        for row_len in row_lens[1:]:
            diff = line_delay_mu - self._calculate_line_comm_delay_mu(row_len, dma)
            current_slack += diff

            if current_slack < 0:
                added_slack -= current_slack
                current_slack = 0

        # reason for adding in offset at the end is to ensure that at no point
        # the current time is equal to the cursor time, but always ahead by at least the offset
        return added_slack + offset_mu

    @host_only
    def configure(self,
                  *,
                  dma_startup_time_mu: typing.Optional[np.int64] = None,
                  comm_delay_intercept_mu: typing.Optional[np.int64] = None,
                  comm_delay_slope_mu: typing.Optional[np.int64] = None,
                  dma_comm_delay_intercept_mu: typing.Optional[np.int64] = None,
                  dma_comm_delay_slope_mu: typing.Optional[np.int64] = None) -> None:
        """Configure measured parameters that will affect slack calculations
        Each configuration is set if and only if the argument is passed in and is not None
        All original values were calculated from benchmarking

        :param dma_startup_time: The time it takes for DMA to start up in (s)
        :param comm_delay_intercept_mu: The intercept of the linear communication time between
        artiq and the kernel as a function of total channels
        :param comm_delay_slope_mu: The slope of the linear communication time between
        artiq and the kernel as a function of total channels
        :param dma_comm_delay_intercept_mu: The intercept of the linear communication time between
        artiq and the kernel for dma playback as a function of total channels
        :param dma_comm_delay_slope_mu: The slope of the linear communication time between
        artiq and the kernel for dma playback as a function of total channels
        """
        if dma_startup_time_mu is not None:
            assert isinstance(dma_startup_time_mu, (int, np.int64))
            assert dma_startup_time_mu > 0
            self._dma_startup_time_mu = np.int64(dma_startup_time_mu)
        if comm_delay_intercept_mu is not None:
            assert isinstance(comm_delay_intercept_mu, (int, np.int64))
            assert comm_delay_intercept_mu > 0
            self._comm_delay_intercept_mu = np.int64(comm_delay_intercept_mu)
        if comm_delay_slope_mu is not None:
            assert isinstance(comm_delay_slope_mu, (int, np.int64))
            assert comm_delay_slope_mu > 0
            self._comm_delay_slope_mu = np.int64(comm_delay_slope_mu)
        if dma_comm_delay_intercept_mu is not None:
            assert isinstance(dma_comm_delay_intercept_mu, (int, np.int64))
            assert dma_comm_delay_intercept_mu > 0
            self._dma_comm_delay_intercept_mu = np.int64(dma_comm_delay_intercept_mu)
        if dma_comm_delay_slope_mu is not None:
            assert isinstance(dma_comm_delay_slope_mu, (int, np.int64))
            assert dma_comm_delay_slope_mu > 0
            self._dma_comm_delay_slope_mu = np.int64(dma_comm_delay_slope_mu)


class ZotinoReader(BaseReader[_ZOTINO_SOLUTION_T]):
    """A reader for the Zotino trap solution files.

    This reader extends the BaseReader functionality to support Zotino specific data structures.

    Additionally, a Zotino compression method, where channels do not need to be communicated if
    there is no change, is provided.
    """

    _CHANNEL: typing.ClassVar[str] = 'channel'
    """Column key for zotino channels."""

    _vref: float

    def __init__(self,
                 solution_path: pathlib.Path,
                 map_path: pathlib.Path,
                 allowed_specials: typing.FrozenSet[str]
                 = frozenset(SpecialCharacter)):
        """Constructor of a zotino reader class extending the base reader

        :param solution_path: Path to the directory containing solution files
        :param map_file: Path to the map file used to map pins to hardware output channels
        :param zotino: Zotino device driver
        :param allowed_specials: A set of string characters that are allowed in the solution files
        (not including numbers)
        """
        super(ZotinoReader, self).__init__(
            solution_path, map_path, allowed_specials)

    def init(self, zotino: artiq.coredevice.zotino.Zotino) -> None:
        self._vref = zotino.vref
        self._voltage_to_mu = zotino.voltage_to_mu

    def _check_init(self, func_name: str) -> None:
        if not hasattr(self, "_vref") or not hasattr(self, "_voltage_to_mu"):
            raise RuntimeError("Must initialize reader using init "
                               f"method to use function {func_name}")

    @property
    def voltage_low(self) -> float:
        self._check_init("voltage_low")
        return -self._vref * 2

    @property
    def voltage_high(self) -> float:
        self._check_init("voltage_high")
        return self._vref * 2

    @host_only
    def get_line_diff(self,
                      line: _ZOTINO_LINE_T,
                      previous: _ZOTINO_LINE_T) -> _ZOTINO_LINE_T:
        """Apply compression to a given path compared to the line before it

        For the zotino, if a channel output remains constant from line to line
        it does not need to be resent and therefore can be filtered out from the zotino path representation

        :param line: Line to filter unchanged voltages
        :param previous: Previous line to compare voltages

        :return: The voltage and channel lists for all changed values in current line
        """
        voltages = []
        channels = []
        for i, voltage in enumerate(line[0]):
            if previous[0][i] != voltage:
                voltages.append(voltage)
                channels.append(line[1][i])

        return voltages, channels

    @host_only
    def process_solution(self,
                         solution: SOLUTION_T) -> _ZOTINO_SOLUTION_T:
        """Implementation to take full solution file and convert it to zotino specific representation

        :param solution: Solutions file representation from :func:`read_solution_mu`

        :return: Solutions file representation for a zotino
        """
        channel_map_dict = self._simplify_map(self.map_file)

        parsed_solution = []
        for d in solution:
            voltages: typing.List[float] = []
            channels: typing.List[int] = []
            for key, val in d.items():
                if isinstance(val, SpecialCharacter):
                    voltage = self.process_specials(val)
                    if not math.isnan(voltage):
                        voltages.append(voltage)
                        channels.append(int(channel_map_dict[key]))
                else:
                    voltages.append(val)
                    channels.append(int(channel_map_dict[key]))

            parsed_solution.append((voltages, channels))

        return parsed_solution

    @host_only
    def process_specials(self, val: SpecialCharacter) -> float:
        """Implementation to handle a SpecialCharacter for the zotino

        :param val: SpecialCharacter from solution file

        :return: Handled value based on solution and zotino characteristics
        """
        self._check_init("process_specials")
        if val == SpecialCharacter.X:
            return math.nan
        elif val == SpecialCharacter.INF:
            return self.voltage_high
        elif val == SpecialCharacter.NEG_INF:
            return self.voltage_low
        else:
            # Special character not handled
            raise ValueError(f'Special character {val} is not yet handled')

    @host_only
    def _simplify_map(self,
                      channel_map: typing.Sequence[typing.Dict[str,
                                                               str]]) -> typing.Dict[str, str]:
        """Convert the map from a list of dictionaries to just a single dictionary where the key is the label

        This representation is more useful to parse the solution file with for a Zotino

        :param channel_map: Representation of csv file as list of dictionaries for each row

        :return: Representation of csv file as a single dictionary with the pin labels as the keys
        """

        return {d[self._LABEL]: d[self._CHANNEL] for d in channel_map}

    @host_only
    def convert_solution_to_mu(self,
                               solution: _ZOTINO_SOLUTION_T) -> _ZOTINO_SOLUTION_T_MU:
        """Convert all voltages in zotino path from volts to machine units

        :param solution: The full zotino path object with voltages in V

        :return: The full zotino path object with voltages in MU
        """
        return [(self.convert_to_mu(t[0]), t[1]) for t in solution]

    @host_only
    def convert_to_mu(self, voltages: _ZOTINO_KEY_T) -> _ZOTINO_KEY_T_MU:
        """Convert a list of voltages from volts to machine units

        :param voltages: A list of voltages in V

        :return: A list of voltages in MU
        """
        self._check_init("convert_to_mu")
        return [self._voltage_to_mu(v) for v in voltages]
