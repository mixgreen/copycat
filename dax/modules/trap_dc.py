from __future__ import annotations  # Postponed evaluation of annotations

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
    _zotino: artiq.coredevice.zotino.Zotino
    _solution_path: pathlib.Path
    _map_file: pathlib.Path
    _reader: ZotinoReader
    _min_line_delay_mu: np.int64

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

    @host_only
    def init(self) -> None:
        """Initialize this module."""
        # Get profile loader
        self._reader = ZotinoReader(
            self._solution_path, self._map_file, self._zotino)
        self._min_line_delay_mu = 1516 + len(self._reader._list_map_labels()) * 808

    @host_only
    def post_init(self) -> None:
        pass

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

    @kernel
    def record_dma(self,
                   name: TStr,
                   solution: TList(TTuple([TList(TInt32), TList(TInt32)])),  # type: ignore[valid-type]
                   line_delay: TFloat) -> TStr:
        """Record the setting of sequential lines of voltages on the zotino device given a list
        of voltages (MU) and corresponding channels

        :param name: Name of DMA trace
        :param solution: A list of voltage lines to set and corresponding channels for each line
        :param line_delay: A delay (s) inserted after the line is set with a minimum value of
        1517 + len(self._reader._list_map_labels()) * 808 MU

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
        :param line_delay: A delay (MU) inserted after the line is set with a minimum value of
        1517 + len(self._reader._list_map_labels()) * 808 MU

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
        :param line_rate: A rate (Hz) to define speed to set each line with a minimum value of
        1517 + len(self._reader._list_map_labels()) * 808 MU

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
        :param line_delay: A delay (s) inserted after the line is set with a minimum value of
        1517 + len(self._reader._list_map_labels()) * 808 MU
        """
        self.shuttle_mu(solution, self.core.seconds_to_mu(line_delay))

    @kernel
    def shuttle_mu(self,
                   solution: TList(TTuple([TList(TInt32), TList(TInt32)])),  # type: ignore[valid-type]
                   line_delay: TInt64):
        """Set sequential lines of voltages on the zotino device given a list of voltages (MU) and
        corresponding channels

        :param solution: A list of voltage lines to set and corresponding channels for each line
        :param line_delay: A delay (MU) inserted after the line is set with a minimum value of
        1517 + len(self._reader._list_map_labels()) * 808 MU
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
        :param line_rate: A rate (Hz) to define speed to set each line with a minimum value of
        1517 + len(self._reader._list_map_labels()) * 808 MU
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


class ZotinoReader(BaseReader[_ZOTINO_SOLUTION_T]):
    _CHANNEL: typing.ClassVar[str] = 'channel'
    """Column key for zotino channels."""
    _vref: float

    def __init__(self,
                 solution_path: pathlib.Path,
                 map_path: pathlib.Path,
                 zotino: artiq.coredevice.zotino.Zotino,
                 allowed_specials: typing.FrozenSet[str]
                 = frozenset(SpecialCharacter)):
        """Constructor of a zotino reader class extending the base reader

        :param solution_path: Path to the directory containing solution files
        :param map_file: Path to the map file used to map pins to hardware output channels
        :param zotino: Zotino device driver
        :param allowed_specials: A set of string characters that are allowed in the solution files
        (not including numbers)
        """
        self._vref = zotino.vref
        self._voltage_to_mu = zotino.voltage_to_mu
        super(ZotinoReader, self).__init__(
            solution_path, map_path, allowed_specials)

    @property
    def voltage_low(self) -> float:
        return -self._vref * 2

    @property
    def voltage_high(self) -> float:
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

    # TODO: add a method to convert payload to mu
    # also figure out if it needs to be done before creating payload
    def convert_to_mu(self, voltages: _ZOTINO_KEY_T) -> _ZOTINO_KEY_T_MU:
        """Convert a list of voltages from volts to machine units

        :param voltages: A list of voltages in V

        :return: A list of voltages in MU
        """
        return [self._voltage_to_mu(v) for v in voltages]
