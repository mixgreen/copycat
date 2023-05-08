from __future__ import annotations  # Postponed evaluation of annotations
from functools import lru_cache

import math
import typing
import pathlib
import numpy as np

from dax.experiment import *
from trap_dac_utils.reader import BaseReader
from trap_dac_utils.schemas import LINEAR_COMBO, SMART_DMA
from trap_dac_utils.types import LABEL_FIELD, LINE_T, SpecialCharacter, SOLUTION_T, MAP_T, CONFIG_T

import artiq.coredevice.zotino  # type: ignore[import]
import artiq.coredevice.ad53xx  # type: ignore[import]

"""Zotino Path and Line types"""
_ZOTINO_KEY_T = typing.List[float]
_ZOTINO_VALUE_T = typing.List[int]
_ZOTINO_LINE_T = typing.Tuple[_ZOTINO_KEY_T, _ZOTINO_VALUE_T]
_ZOTINO_SOLUTION_T = typing.List[_ZOTINO_LINE_T]
_ZOTINO_LINE_MU_T = typing.List[np.int32]
_ZOTINO_SOLUTION_MU_T = typing.List[_ZOTINO_LINE_MU_T]

__all__ = ['TrapDcModule', 'ZotinoReader', 'LinearCombo', 'SmartDma']


class _SolutionAttrs:
    """A module to represent a single shuttling solution and it's corresponding data fields
    """

    _attrs: typing.Dict[str, typing.Any]
    _trap_dc: TrapDcModule

    def __init__(self, cfg: CONFIG_T, trap_dc: TrapDcModule) -> None:
        self._attrs = dict(cfg)
        self._trap_dc = trap_dc

    def __getitem__(self, arg: str) -> typing.Any:
        return self._attrs[arg]

    def __setitem__(self, arg: str, newvalue: typing.Any) -> None:
        self._attrs[arg] = newvalue

    def get(self, arg: str, default: typing.Any) -> typing.Any:
        self._attrs.get(arg, default)

    def solution(self) -> SOLUTION_T:
        file = self._attrs.get('file', f'{self._attrs["name"]}.csv')
        start = self._attrs.get('start', 0)
        end = self._attrs.get('end', -1)
        return self._trap_dc.read_solution(file, start, end)

    def solution_mu(self) -> _ZOTINO_SOLUTION_MU_T:
        return self._trap_dc.solution_to_mu(self.solution(),
                                            reverse=self.reverse(),
                                            multiplier=self.multiplier())

    def line_delay(self) -> float:
        ld = self._attrs.get('line_delay')
        if ld is None or not isinstance(ld, float):
            raise ValueError("Line delay must be set to a float")
        return ld

    def reverse(self) -> bool:
        return self._attrs.get('reverse', False)

    def multiplier(self) -> float:
        return self._attrs.get('multiplier', 1.0)

    def name(self) -> str:
        name = self._attrs.get('name')
        if name is None or not isinstance(name, str):
            raise ValueError("Name must be set to a str")
        return str(name)


_SMART_DMA_CONFIG_T = typing.Dict[str, _SolutionAttrs]


class SmartDma(DaxModule):
    """A module to represent smart DMA configuration files

    This module can be used to get the shuttling solutions and intelligently
    store them in DMA for low latency use
    """
    _config: _SMART_DMA_CONFIG_T
    """The configuration dict that contains field names mapped to config data"""

    _trap_dc: TrapDcModule

    _incoming_dma_dict: typing.Dict[str, typing.Any]
    _keys: typing.List[str]
    _handles: typing.Sequence[typing.Tuple[np.int32, np.int64, np.int32]]
    _handle_map: typing.Dict[str, typing.Tuple[np.int32, np.int64, np.int32]]

    _names: typing.Sequence[str]
    _solution_mus: typing.Sequence[_ZOTINO_SOLUTION_MU_T]
    _line_delays: typing.Sequence[float]

    _erase_names: typing.Sequence[str]
    _record_names: typing.Sequence[str]

    def build(self,  # type: ignore[override]
              *, trap_dc: TrapDcModule, config: CONFIG_T) -> None:
        """Constructor of zotino linear combination module

        :param trap_dc: The trap DC module associated with this configuration
        :param config: The string name of the file where the configuration is
        """
        self._config = {d['name']: _SolutionAttrs(d, trap_dc) for d in config['params']}
        self._trap_dc = trap_dc

        self._names = self.names()
        self._solution_mus = self.solution_mus()
        self._line_delays = self.line_delays()

        self._incoming_dma_dict = self.solution_dict()
        self._keys = []
        self._handles = []

        self._erase_names = []
        self._record_names = []
        self._keys = [""] * len(self._names)

    @host_only
    def init(self) -> None:
        self._erase_names, self._record_names = self.compare_dma()
        self._recorded_names = [name for name in self._names if name not in self._record_names]
        self.init_kernel()

    @kernel
    def init_kernel(self) -> TNone:
        self.update_dma()

    @host_only
    def post_init(self) -> None:
        self._update_dma_dataset()
        self.post_init_kernel()

    @kernel
    def post_init_kernel(self) -> TNone:
        self._handles = [self.get_dma_handle(key) for key in self._keys]

    def solution_dict(self) -> typing.Dict[str, typing.Any]:
        """A method to return all the solutions in a dictionary for comparison

        :return: The incoming dma dictionary of solution details for comparison"""
        return {sol_attr.name(): (sol_attr.solution().hash,
                                  sol_attr.line_delay(),
                                  sol_attr.reverse(),
                                  sol_attr.multiplier())
                for sol_attr in self._config.values()}

    @host_only
    def compare_dma(self) -> typing.Tuple[typing.Sequence[str], typing.Sequence[str]]:
        """Compare the incoming dma dictionary populated by :func:`compare_dma` to the dictionary in
        the dataset
        Goal is to figure out the dma trace names that need to be erased (no longer needed or overwritten
        with a new hash or line delay) and the trace names that need to be recorded (weren't previously in
        trace or overwritted with a new hash or line_delay)

        :return: Two lists of dma trace names, the first for erasures, the second for recordings
        """
        dma_dict = self.get_dataset_sys("dma_dict", {}, archive=False)
        old_names = set(dma_dict.keys())
        incoming_names = set(self._incoming_dma_dict.keys())
        erase_names = old_names.difference(incoming_names)
        overwrite_names = old_names.intersection(incoming_names)
        record_names = incoming_names.difference(old_names)

        for name in overwrite_names:
            assert (name in dma_dict and name in self._incoming_dma_dict)
            if (dma_dict[name] != self._incoming_dma_dict[name]):
                erase_names.add(name)
                record_names.add(name)

        return list(erase_names), list(record_names)

    @kernel
    def update_dma(self, force_record: TBool = False) -> TNone:
        """Update the dma traces by erasing and recording the incoming data. The erase_names and new_record_names
        should come from the return of :func:`compare_dma`. The record_names, solutions, and line_delay fields should
        be the same data used as the argument to :func:`compare_dma` but transposed. The :func:`compare_dma` will
        transpose and return them.
        This function will also check for a powercycle of the core device
        This function should be called from the kernel in the init_kernel function of the module using it
        Whether there is a powercycle, force_record, or neither, the returned keys (i.e. the available dma traces)
        should always be the same

        :param record_names: List of all names that are needed in the dma trace
        :param solutions: List of solutions. One for each recording name. If force_record enable, all will be recorded
        :param line_delays: List of line delays. One for each shuttle solution
        :param erase_names: List of names to erase from the dma trace
        :param new_record_names: Names to record if force_record is not enabled
        :param force_record: False by default. If true all dma traces passed in will be recorded. Otherwise only the
            new dma traces will be recorded

        :return: List of dma trace keys that are available to the user. Keys can be used to get the corresponding
            handles
        """

        powercycle = len(self.core_cache.get(self.get_system_key("powercycle"))) == 0
        if force_record or powercycle:
            if not powercycle:
                for name in self._names:
                    self._trap_dc.erase_dma(name)

            self._keys = [self._trap_dc.record_dma(name, self._solution_mus[i], self._line_delays[i])
                          for i, name in enumerate(self._names)]

        else:
            for name in self._erase_names:
                self._trap_dc.erase_dma(name)

            i = 0
            for name in self._record_names:
                self.logger.debug("Recording trace %s", name)
                self._keys[i] = self._trap_dc.record_dma(name, self._solution_mus[i], self._line_delays[i])
                i = i + 1
            for name in self._recorded_names:
                self._keys[i] = self._trap_dc.get_dma_key(name)
                i = i + 1

    @rpc(flags={'async'})
    def _update_dma_dataset(self):
        """Updates the datset to the incoming dataset populated by :func:`compare_dma`
        Will reset the incoming dma dictionary variable once finished
        """
        self.set_dataset_sys("dma_dict", self._incoming_dma_dict, archive=False, data_store=False)
        self._incoming_dma_dict = {}

    @kernel
    def get_dma_handle(self,
                       key: TStr,
                       set_powercycle: TBool = True) -> TTuple([TInt32, TInt64, TInt32]):  # type: ignore[valid-type]
        """Get the DMA handle associated with the name of the recording

        :param key: Unique key of the recording
        :param set_powercycle: True by default. If set to True, will add an entry to the core cache which is used to
            determine powercycles

        :return: Handle used to playback the DMA Recording
        """
        if set_powercycle:
            self.core_cache.put(self.get_system_key("powercycle"), [1])

        return self._trap_dc.get_dma_handle(key)

    @host_only
    def get_dma_handle_host(self, key) -> typing.Tuple[np.int32, np.int64, np.int32]:
        """A method to return the handles retrieved during post init
        If no handle map exists yet, create it from the handles and keys

        :return: A list of all the handles"""
        if not self._handle_map:
            self._handle_map = {key: self._handles[i] for i, key in enumerate(self._keys)}

        return self._handle_map[key]

    def names(self) -> typing.Sequence[str]:
        """A method to get all of the string names

        :return: A list of string names
        """
        return [sol_attr.name() for sol_attr in self._config.values()]

    def solution_mus(self) -> typing.Sequence[_ZOTINO_SOLUTION_MU_T]:
        """A method to get all of the solutions in MU

        :return: A list of solutions in MU
        """
        return [sol_attr.solution_mu() for sol_attr in self._config.values()]

    def line_delays(self) -> typing.Sequence[float]:
        """A method to get all of the line delays as a list of floats

        :return: A list of floats of the line delays
        """
        return [sol_attr.line_delay() for sol_attr in self._config.values()]

    def fields(self) -> typing.Sequence[str]:
        """A method to get all of the config fields as a list of strings

        :return: A list of strings of the config fields
        """
        return list(self._config.keys())


class _LineAttrs:
    """A module to represent a single parameter and it's corresponding data fields
    """
    _attrs: typing.Dict[str, typing.Any]
    _trap_dc: TrapDcModule

    def __init__(self, cfg: CONFIG_T, trap_dc: TrapDcModule):
        self._attrs = dict(cfg)
        self._attrs.setdefault('value', 0.)
        self._trap_dc = trap_dc

    def __getitem__(self, arg: str) -> typing.Any:
        return self._attrs[arg]

    def __setitem__(self, arg: str, newvalue: typing.Any) -> None:
        self._attrs[arg] = newvalue

    def get(self, arg: str, default: typing.Any) -> typing.Any:
        self._attrs.get(arg, default)

    def line(self) -> LINE_T:
        file = self._attrs.get('file', f'{self._attrs["name"]}.csv')
        line = self._attrs.get('line', 0)
        return self._trap_dc.read_line(file, line)

    def in_range(self) -> bool:
        val = self._attrs['value']
        maxi = self._attrs.get('max', float('inf'))
        mini = self._attrs.get('min', float('-inf'))

        if val is None:
            return True
        assert isinstance(val, float) and isinstance(mini, float) and isinstance(maxi, float)
        assert mini < maxi
        return mini <= val <= maxi


_LINEAR_COMBO_CONFIG_T = typing.Dict[str, _LineAttrs]


class LinearCombo:
    """A module to represent linear combination configuration files

    This module can be used to get solution lines and combine them with others
    Then these new lines can be fed back into the zotino module to continue processing
    """
    _config: _LINEAR_COMBO_CONFIG_T
    """The configuration dict that contains field names mapped to config data"""

    def __init__(self, trap_dc: TrapDcModule, config: CONFIG_T):
        """Constructor of zotino linear combination module

        :param config_file: The string name of the file where the configuration is
        :param reader: A reader to use for retrieving solutions corresponding to parameter
        """
        self._config = {d['name']: _LineAttrs(d, trap_dc) for d in config['params']}

    def __getitem__(self, arg: str) -> _LineAttrs:
        """Method to make object indexible

        :param arg: The string argument to index on

        :return: The configuration attribute
        """
        return self._config[arg]

    def verify(self) -> None:
        """A method to check that the fields are all acceptible values as defined by
        the instance contraints
        """
        for f, attrs in self._config.items():
            if not attrs.in_range():
                raise ValueError(f'Field {f}={attrs["value"]} is out of range')

    def from_arguments(self, env: HasEnvironment, *,
                       prefix: str = '', group: typing.Optional[str] = None) -> None:
        """Updates the config with the values that the user input for each field

        :param env: The environment to get the arguments from
        :param prefix: The prefix to add to the field name when displaying
        :param group: The name of the group to put the argument under
        """
        assert isinstance(env, HasEnvironment)

        for f in self.fields():
            self._config[f]['value'] = self.get_argument(
                env, f, default=self._config[f]['value'], prefix=prefix, group=group)

    def get_argument(self, env: HasEnvironment, field: str, default: float, *,
                     prefix: str = '', group: typing.Optional[str], tooltip: typing.Optional[str] = None,
                     **kwargs: typing.Any):
        """A method to get the arguments from the environment, used in experiments to set the config fields

        :param env: The environment to get the arguments from
        :param field: The field to get the argument of
        :param default: The default value to set for the argument of the field
        :param prefix: The prefix to add to the field name when displaying
        :param group: The name of the group to put the argument under
        :param tooltip: A tooltip message to show for the argument

        :return: The user input for the field argument value
        """
        assert isinstance(env, HasEnvironment)
        assert isinstance(field, str)
        assert isinstance(default, float)
        assert isinstance(prefix, str)
        assert isinstance(group, str) or group is None
        assert isinstance(tooltip, str) or tooltip is None

        if group is None:
            group = 'Zotino overrides'

        number_kwargs = self._config[field].get('args', {})
        number_kwargs.update(kwargs)

        return env.get_argument(
            f'{prefix}{field.upper()}',
            NumberValue(default, **number_kwargs),
            group=group, tooltip=tooltip
        )

    def fields(self) -> typing.Sequence[str]:
        """A method to get all of the config fields as a list of strings

        :return: A list of strings of the config fields
        """
        return list(self._config.keys())


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
    _LC_T = typing.TypeVar('_LC_T', bound='LinearCombo')
    _SD_T = typing.TypeVar('_SD_T', bound='SmartDma')

    _DMA_STARTUP_TIME: typing.ClassVar[float] = 1.728 * us
    """Startup time for DMA (s). Measured in the RTIO benchmarking tests during CI"""

    _zotino: artiq.coredevice.zotino.Zotino
    _solution_path: pathlib.Path
    _map_file: pathlib.Path
    _reader: ZotinoReader
    _min_line_delay_mu: np.int64
    _calculator: ZotinoCalculator
    _incoming_dma_dict: typing.Dict[str, typing.Any]

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

        # Below calculated from set_dac_mu and load functions
        # https://m-labs.hk/artiq/manual/_modules/artiq/coredevice/ad53xx.html#AD53xx
        self._min_line_delay_mu = np.int64(self.core.seconds_to_mu(1500 * ns)
                                           + 2 * self._zotino.bus.ref_period_mu
                                           + self._reader.num_labels()
                                           * self._zotino.bus.xfer_duration_mu)
        self.update_kernel_invariants('_min_line_delay_mu')

    @host_only
    def init(self) -> None:
        """Initialize this module."""
        # Get profile loader
        # # Below calculated from set_dac_mu and load functions
        # # https://m-labs.hk/artiq/manual/_modules/artiq/coredevice/ad53xx.html#AD53xx
        # self._min_line_delay_mu = np.int64(self.core.seconds_to_mu(1500 * ns)
        #                                    + 2 * self._zotino.bus.ref_period_mu
        #                                    + self._reader.num_labels()
        #                                    * self._zotino.bus.xfer_duration_mu)
        # self.update_kernel_invariants('_min_line_delay_mu')
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
    def create_linear_combo(self,
                            config_file: str,
                            *args,
                            cls: typing.Type[_LC_T] = LinearCombo,  # type: ignore[assignment]
                            **kwargs) -> _LC_T:
        """Factory method to encapsulate creation of ZotinoLinearComboModule

        :param config_file: Name of the config file to read into the object

        :return: The ZotinoLinearComboModule object"""
        config = self._reader.read_config(config_file, schema=LINEAR_COMBO)
        return cls(self, config, *args, **kwargs)

    @host_only
    def create_smart_dma(self,
                         config_file: str,
                         *args,
                         cls: typing.Type[_SD_T] = SmartDma,  # type: ignore[assignment]
                         **kwargs) -> _SD_T:
        """Factory method to encapsulate creation of SmartDma

        :param config_file: Name of the config file to read into the object

        :return: The SmartDma object"""
        config = self._reader.read_config(config_file, schema=SMART_DMA)
        return cls(self, "smart_dma", trap_dc=self, config=config, *args, **kwargs)

    @host_only
    def read_line(self,
                  file_name: str,
                  index: int) -> LINE_T:
        """Read in a solutions file and return the specific line in base reader form

        Note that the Zotino Path Voltages are given in **V**.

        :param file_name: Solution file to parse the path from
        :param index: Line index in solution file

        :return: Base reader line form
        """
        solution = self._reader.read_solution(file_name)
        if index >= len(solution):
            raise ValueError("Index is out of the range of the solution file")
        return solution[index]

    @host_only
    def line_to_mu(self,
                   line: LINE_T,
                   *,
                   multiplier: float = 1.0) -> _ZOTINO_LINE_MU_T:
        """Read in a solutions file and return the line in zotino form.
        Optionally apply multiplier to all voltages in path

        Note that the Zotino Path Voltages are given in **MU**.

        :param reader_line: Line python object representation
        :param multiplier: Optionally scale the voltages by a constant

        :return: Zotino module interpretable solution line with voltages in V
        """
        unprepared_line = self._reader.process_solution(SOLUTION_T((line,)))[0]

        # multiply each solution list with multiplier
        prepared_line = (
            (np.asarray(unprepared_line[0]) * multiplier).tolist(),  # type: ignore[attr-defined]
            unprepared_line[1])

        return self._reader.line_to_mu(prepared_line)

    @host_only
    def read_line_mu(self,
                     file_name: str,
                     *,
                     index: int = 0,
                     multiplier: float = 1.0) -> _ZOTINO_LINE_MU_T:
        """Read in a single line of a solutions file and return the line in zotino form.
        Optionally apply multiplier to all voltages in path

        Note that the Zotino Path Voltages are given in **MU**.

        May provide either file_name and reader_solution. If both are provided the file name will be used

        :param file_name: Solution file to parse the path from
        :param index: Line in path to get. A 0 indicates the first line
        :param multiplier: Optionally scale the voltages by a constant

        :return: Zotino module interpretable solution line with packed voltages and channels
        """
        line = self.read_line(file_name, index)
        return self.line_to_mu(line=line, multiplier=multiplier)

    @host_only
    def read_solution(self,
                      file_name: str,
                      start: int = 0,
                      end: int = -1) -> SOLUTION_T:
        """Read in a solutions file and return the solution in base reader form

        Note that the Zotino Path Voltages are given in **V**.

        :param file_name: Solution file to parse the path from

        :return: Base reader solution form
        """
        return self._reader.read_solution(file_name, start=start, end=end)

    @host_only
    def solution_to_mu(self,
                       solution: SOLUTION_T,
                       *,
                       start: int = 0,
                       end: int = -1,
                       reverse: bool = False,
                       multiplier: float = 1.0) -> _ZOTINO_SOLUTION_MU_T:
        """Read in a segment of a solutions file and return the path in zotino form.
        Optionally reverse path and/or apply multiplier to all voltages in path

        Note that the Zotino Path Voltages are given in **V**.

        :param solution: Solution file to parse the path from
        :param start: Starting index of path (inclusive). Default 0 signals to start with first solution line
        :param end: End index of path (inclusive). Default -1 signals to end with last solution line
        :param reverse: Optionally return a reversed path. I.E. From end to start
        :param multiplier: Optionally scale the voltages by a constant

        :return: Zotino module interpretable solution path with voltages in V
        """

        if start >= len(solution):
            raise ValueError("Start index is out of the range of the solution file")
        if end >= len(solution):
            raise ValueError("End index is out of the range of the solution file")

        processed_solution = self._reader.process_solution(solution)
        # multiply each solution list with multiplier
        for i, t in enumerate(processed_solution):
            processed_solution[i] = (
                (np.asarray(t[0]) * multiplier).tolist(), t[1])  # type: ignore[attr-defined]

        # The modulus fixes the endpoint problem for -1
        trimmed_solution = processed_solution[start:(end % len(processed_solution)) + 1]
        if reverse:
            trimmed_solution.reverse()

        path: _ZOTINO_SOLUTION_T = [trimmed_solution[0]]
        path.extend([self._reader.get_line_diff(t, trimmed_solution[i])
                     for i, t in enumerate(trimmed_solution[1:])])

        return self._reader.solution_to_mu(path)

    @host_only
    def read_solution_mu(self,
                         file_name: str,
                         *,
                         start: int = 0,
                         end: int = -1,
                         reverse: bool = False,
                         multiplier: float = 1.0) -> _ZOTINO_SOLUTION_MU_T:
        """Read in a segment of a solutions file and return the path in zotino form.
        Optionally reverse path and/or apply multiplier to all voltages in path

        Note that the Zotino Path Voltages are given in **MU**.

        :param file_name: Solution file to parse the path from
        :param start: Starting index of path (inclusive). Default 0 signals to start with first solution line
        :param end: End index of path (inclusive). Default -1 signals to end with last solution line
        :param reverse: Optionally return a reversed path. I.E. From end to start
        :param multiplier: Optionally scale the voltages by a constant

        :return: Zotino module interpretable solution path with packed voltages and channels
        """
        solution = self.read_solution(file_name)
        return self.solution_to_mu(solution=solution, start=start, end=end, reverse=reverse, multiplier=multiplier)

    @host_only
    def list_solutions(self) -> typing.Sequence[str]:
        """Get a list of each solution file available in the solutions
        directory

        :return: The list of names of solution files available
        """

        return self._reader.list_solutions()

    @rpc
    def get_dma_key(self, name: str) -> TStr:
        """Return the dma trace key given a name
        :param name: Name for the dma trace

        :return: Key for the dma trace
        """
        return self.get_system_key(name)

    @kernel
    def record_dma(self,
                   name: TStr,
                   solution: TList(TList(TInt32)),  # type: ignore[valid-type]
                   line_delay: TFloat) -> TStr:
        """Record the setting of sequential lines of voltages on the zotino device given a list
        of voltages (MU) and corresponding channels

        :param name: Name of DMA trace
        :param solution: A list of packed voltage and channel lines to set
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
                      solution: TList(TList(TInt32)),  # type: ignore[valid-type]
                      line_delay: TInt64) -> TStr:
        """Record the setting of sequential lines of voltages on the zotino device given a list
        of voltages (MU) and corresponding channels

        :param name: Name of DMA trace
        :param solution: A list of packed voltage and channel lines to set
        :param line_delay: A delay (MU) inserted after the line is set
            Must be greater than the SPI write time for the number of used channels

        :return: Unique key for DMA Trace
        """
        if line_delay <= self._min_line_delay_mu:
            raise ValueError(f"Line Delay must be greater than {self._min_line_delay_mu}")
        dma_name = self.get_dma_key(name)
        with self.core_dma.record(dma_name):
            for t in solution:
                self.set_line(t)
                delay_mu(line_delay)
        return dma_name

    @kernel
    def record_dma_rate(self,
                        name: TStr,
                        solution: TList(TList(TInt32)),  # type: ignore[valid-type]
                        line_rate: TFloat) -> TStr:
        """Record the setting of sequential lines of voltages on the zotino device given a list
        of voltages (MU) and corresponding channels

        :param name: Name of DMA trace
        :param solution: A list of packed voltage and channel lines to set
        :param line_rate: A rate (Hz) to define speed to set each line
            Must be greater than the SPI write time for the number of used channels

        :return: Unique key for DMA Trace
        """
        return self.record_dma_mu(name,
                                  solution,
                                  self.core.seconds_to_mu(1.0 / line_rate))

    @kernel
    def erase_dma(self, name: TStr):
        """Erase a dma trace from the dma cache.

        :param name: Name of dma trace to erase
        """
        dma_name = self.get_dma_key(name)
        try:
            self.core_dma.erase(dma_name)
        except KeyError:
            self.logger.warn("Data not found for %s when erasing", dma_name)

    @kernel
    def get_dma_handle(self,
                       key: TStr) -> TTuple([TInt32, TInt64, TInt32]):  # type: ignore[valid-type]
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
                solution: TList(TList(TInt32)),  # type: ignore[valid-type]
                line_delay: TFloat):
        """Set sequential lines of voltages on the zotino device given a list of voltages (MU) and
        corresponding channels

        :param solution: A list of packed voltage and channel lines to set
        :param line_delay: A delay (s) inserted after the line is set
            Must be greater than the SPI write time for the number of used channels
        """
        self.shuttle_mu(solution, self.core.seconds_to_mu(line_delay))

    @kernel
    def shuttle_mu(self,
                   solution: TList(TList(TInt32)),  # type: ignore[valid-type]
                   line_delay: TInt64):
        """Set sequential lines of voltages on the zotino device given a list of voltages (MU) and
        corresponding channels

        :param solution: A list of packed voltage and channel lines to set
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
                     solution: TList(TList(TInt32)),  # type: ignore[valid-type]
                     line_rate: TFloat):
        """Set sequential lines of voltages on the zotino device given a list of voltages (MU) and
        corresponding channels

        :param solution: A list of packed voltage and channel lines to set
        :param line_rate: A rate (Hz) to define speed to set each line
            Must be greater than the SPI write time for the number of used channels
        """
        self.shuttle_mu(solution, self.core.seconds_to_mu(1 / line_rate))
        return

    @kernel
    def set_line(self,
                 line: TList(TInt32)):  # type: ignore[valid-type]
        """Set a line of voltages on the zotino device given a list of packed voltages and channels into one
        32-bit int

        :param line: Up to 32 (# of Zotino channels) 32-bit ints with voltage and channel packed in
        """
        for val in line:
            self._zotino.bus.write(val)
        self._zotino.load()

    @host_only
    def list_map_labels(self) -> typing.Sequence[str]:
        """Lists the labels that are in the map file

        :return: The labels from the map file
        """
        return self._reader.list_map_labels()

    @host_only
    def calculate_slack(self,
                        solution: _ZOTINO_SOLUTION_MU_T,
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
                           solution: _ZOTINO_SOLUTION_MU_T,
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
                            solution: _ZOTINO_SOLUTION_MU_T,
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
                               solution: _ZOTINO_SOLUTION_MU_T,
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
    def _list_num_channels(self, solution: _ZOTINO_SOLUTION_MU_T) -> typing.Sequence[int]:
        """Given a zotino solution, list the length of each row in terms of number of channels

        :param solution: Any zotino solution

        :return: A list of number of channels that need to be set for each row"""
        return [len(t) for t in solution]

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

        :param dma_startup_time_mu: The time it takes for DMA to start up in (MU)
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
        if not self.verify_solution(solution):
            raise ValueError("Solution is not a valid. (e.g. values are not in accepted range)")

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
        else:
            # Special character not handled
            raise ValueError(f'Special character {val} is not yet handled')

    @host_only
    def _simplify_map(self,
                      channel_map: MAP_T) -> typing.Mapping[str, str]:
        """Convert the map from a list of dictionaries to just a single dictionary where the key is the label

        This representation is more useful to parse the solution file with for a Zotino

        :param channel_map: Representation of csv file as list of dictionaries for each row

        :return: Representation of csv file as a single dictionary with the pin labels as the keys
        """

        return {d[LABEL_FIELD]: d[self._CHANNEL] for d in channel_map}

    @host_only
    def line_to_mu(self, line: _ZOTINO_LINE_T) -> _ZOTINO_LINE_MU_T:
        """Pack a line of values into a form directly writeable to the SPI bus

        :param line: The Zotino line to pack

        :return: The packed line which is a list of 32-bit integers
            where the most significant 24 bits are the packed value
        """
        self._check_init("line_to_mu")
        vs, chs = line
        return [np.int32(artiq.coredevice.ad53xx.ad53xx_cmd_write_ch(ch,
                                                                     self._voltage_to_mu(v),
                                                                     artiq.coredevice.ad53xx.AD53XX_CMD_DATA) << 8)
                for v, ch in zip(vs, chs)]

    @host_only
    def solution_to_mu(self, solution: _ZOTINO_SOLUTION_T) -> _ZOTINO_SOLUTION_MU_T:
        """Pack a solution of values into a form directly writeable to the SPI bus

        :param solution: The Zotino solution to pack

        :return: The packed solution where each solution row is a list of 32-bit integers
            where the most significant 24 bits are the packed value
        """
        return [self.line_to_mu(line) for line in solution]
