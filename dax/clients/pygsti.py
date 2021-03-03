from dax.experiment import *

from dax.interfaces.operation import OperationInterface
from dax.modules.hist_context import HistogramContext, HistogramAnalyzer

from dax.util.artiq import is_kernel
from dax.util.output import get_base_path

import numpy as np
import typing

import logging
_LOGGER = logging.getLogger(__name__)

# Could result in an ImportError, which will not load this experiment
try:
    import pygsti
except ImportError:
    _LOGGER.info(
        "PyGSTi is an optional dependency. However, the Randomized Benchmarking Clients require the package."
        "They will be unavailable unless it is included."
    )
    raise

__all__ = ['RandomizedBenchmarkingSQ']


@dax_client_factory
class RandomizedBenchmarkingSQ(DaxClient, Experiment):
    """Single-qubit randomized benchmarking using pyGSTi."""

    DEFAULT_QUBIT_NUMBER: int = 1
    """Load 1 ion at beginning of experiment"""

    DEFAULT_QUBIT_LABEL: str = 'Q0'
    """Default qubit label for pyGSTi analysis"""

    MAX_KASLI_MEMORY_LOAD: int = 20480
    """Rough approximation of maximum number of gates to load to Kasli at a time"""

    CIRCUIT_DEPTHS: list = [str(2 ** n) for n in range(int(np.log2(MAX_KASLI_MEMORY_LOAD)) + 1)]
    """Max depth to perform RB up to (automatically does every power of 2 below this number as well)"""

    STATE_DETECTION_THRESHOLD_KEY: str = 'state_detection_threshold'
    """Detection threshold to distinguish between 0 and 1 states"""

    def build(self):
        assert isinstance(self.DEFAULT_QUBIT_NUMBER, int), 'Default Qubit Number must be of type int'
        assert self.DEFAULT_QUBIT_NUMBER > 0, 'Default Qubit Number must be greater than zero'
        assert isinstance(self.DEFAULT_QUBIT_LABEL, str), 'Default Qubit Label must be of type str'
        assert isinstance(self.MAX_KASLI_MEMORY_LOAD, int), 'Max Kasli Memory Load must be of type int'
        assert self.MAX_KASLI_MEMORY_LOAD > 0, 'Max Kasli Memory Load must be greater than zero'
        assert isinstance(self.CIRCUIT_DEPTHS, list), 'Circuit Depths must be of type list'
        assert isinstance(self.STATE_DETECTION_THRESHOLD_KEY, str), 'State Detection Threshold Key must be of type str'
        assert is_kernel(self.device_setup), 'device_setup() must be a kernel function'
        assert is_kernel(self.device_cleanup), 'device_cleanup() must be a kernel function'
        assert not is_kernel(self.host_setup), 'host_setup() can not be a kernel function'
        assert not is_kernel(self.host_cleanup), 'host_cleanup() can not be a kernel function'

        # Add general arguments
        self._depth = self.get_argument("Max Depth", EnumerationValue(self.CIRCUIT_DEPTHS),
                                        tooltip=f"Max circuit depth, automatically generates every power of "
                                                f"2 below this value as well")
        self._num_circs = self.get_argument("Number of sample circuits",
                                            NumberValue(default=10, min=1, step=1, ndecimals=0))
        self._gate_delay = self.get_argument("Delay Between Gates", NumberValue(default=1 * us, ndecimals=1,
                                                                                step=1 * us, min=0 * us, unit='us'))
        self._loop_number = self.get_argument("Loop number", NumberValue(default=100, ndecimals=0,
                                                                         scale=1, step=1, min=0))

        # Flag for real time gates
        self._real_time = self.get_argument("Realtime gates",
                                            BooleanValue(True),
                                            tooltip="Only delay between gates will be the one you set plus dds latency")

        # Flags for saving files
        self._save_histograms = self.get_argument("Save Histogram PDFs", BooleanValue(False),
                                                  tooltip="Histograms will be saved as pdfs")

        # Update kernel invariants
        self.update_kernel_invariants("_depths", "_num_circs", "_gate_delay",
                                      "_loop_number", "_load_ion", "_real_time", "_save_histograms")

    def prepare(self):
        # Save pyGSTi version
        version = pygsti.__version__
        self.set_dataset('pygsti_version', version)
        self.logger.info(f'pyGSTi version: {version}')

        # Obtain operation interface
        self.operation = self.registry.find_interface(OperationInterface)
        self.update_kernel_invariants('operation')

        # OVERRIDE TO CHANGE AVAILABLE GATES
        self.AVAILABLE_GATES_DICT = {
            'Gxpi': self.operation.x,
            'Gxpi2': self.operation.sqrt_x,
            'Gxmpi2': self.operation.sqrt_x_dag,
            'Gypi': self.operation.y,
            'Gypi2': self.operation.sqrt_y,
            'Gympi2': self.operation.sqrt_y_dag,
            'Gh': self.operation.h
        }
        self.update_kernel_invariants('AVAILABLE_GATES_DICT')

        # Obtain Histogram Context
        self.histogram_context = self.registry.find_module(HistogramContext)
        self.update_kernel_invariants('histogram_context')

        # Get the scheduler
        self.scheduler = self.get_device('scheduler')
        self.update_kernel_invariants('scheduler')

        # Create depth list
        self._depth_list = [2 ** n for n in range(int(np.log2(int(self._depth))) + 1)]
        self.logger.debug(self._depth_list)

        # Create Processor Specifications and Experiment Design
        self._pspec = pygsti.obj.ProcessorSpec(self.DEFAULT_QUBIT_NUMBER, list(self.AVAILABLE_GATES_DICT.keys()),
                                               qubit_labels=[self.DEFAULT_QUBIT_LABEL], construct_models=('clifford',))
        self._exp_design = pygsti.protocols.CliffordRBDesign(self._pspec, self._depth_list, self._num_circs,
                                                             qubit_labels=[self.DEFAULT_QUBIT_LABEL])

        # Convert experiment design to circuit list
        try:
            self._total_circuit_list = [_get_gates(c.str, self.AVAILABLE_GATES_DICT, self.DEFAULT_QUBIT_LABEL) for
                                        c_list in self._exp_design.circuit_lists for c in c_list]
        except KeyError as e:
            self.logger.error(f'Gate "{e}" is not available according to the dictionary!')
            raise

        # Partition circuit list based on max size the Kasli can handle
        self._partition_list = _partition_gate_list(self._total_circuit_list, self.MAX_KASLI_MEMORY_LOAD)

    def host_setup(self) -> None:
        """Preparation on the host, called once at entry and after a pause."""
        pass

    @kernel
    def device_setup(self):  # type: () -> None
        """Preparation on the core device, called once at entry and after a pause.

        Should at least reset the core.
        """
        # Reset the core
        self.core.reset()

    def run(self):
        # Init Dax
        self.dax_init()

        # Set to realtime gates or not
        self.operation.set_realtime(self._real_time)

        # Plot Histograms
        self.histogram_context.plot_histogram()

        # Run kernel code
        self._terminated = False
        try:
            for circuit_list in self._partition_list:
                # Only send chunks to Kasli at a time
                self._circuit_list = circuit_list
                self.update_kernel_invariants("_circuit_list")
                self._run()
        except TerminationRequested:
            # Circuit was terminated
            self.logger.warning('Circuit was terminated by user request')
            self._terminated = True

    @kernel
    def _run(self):
        # Break realtime
        self.core.break_realtime()

        # Iterate through circuits
        current_depth = len(self._circuit_list[0])
        for circuit in self._circuit_list:
            with self.histogram_context:
                # Schedule two circuits to start to improve performance
                self._run_circuit(circuit)

                for _ in range(self._loop_number - 1):
                    self._run_circuit(circuit)
                    self.operation.store_measurements_all()

                # Perform final measurement
                self.operation.store_measurements_all()

            # Check if there is a pause condition when circuit depth changes
            if len(circuit) != current_depth:
                current_depth = len(circuit)
                if self.scheduler.check_pause():
                    # Interrupt current work, raise to break out of all loops
                    raise TerminationRequested

    @kernel
    def _run_circuit(self, circuit):
        # Guarantee slack
        self.core.break_realtime()

        # Initialize
        self.operation.prep_0_all()

        # Perform gates
        for gate in circuit:
            # Preconfigured Delay
            delay(self._gate_delay)

            # Run gate
            gate(0)

        # Measure state
        self.operation.m_z_all()

    @kernel
    def device_cleanup(self):  # type: () -> None
        """Cleanup on the core device, called before pausing and exiting."""
        pass

    def host_cleanup(self) -> None:
        """Cleanup on the host, called before pausing and exiting."""
        pass

    def analyze(self):
        # Do not run analysis if experiment was terminated (most likely won't work)
        if self._terminated:
            return

        # Create Histogram Analyzer
        h = HistogramAnalyzer(self)

        # Save histograms as pdfs
        if self._save_histograms:
            h.plot_all_histograms()

        # Create RB dataset
        ds = pygsti.objects.DataSet(outcomeLabels=['0', '1'])
        threshold = self.get_dataset_sys(self.STATE_DETECTION_THRESHOLD_KEY, 2)
        i = 0
        for c_list in self._exp_design.circuit_lists:
            for c in c_list:
                one = HistogramAnalyzer.histogram_to_one_count(h.histograms["histogram"][0][i], threshold)
                ds.add_count_dict(c, {'0': self._loop_number - one, '1': one})
                i += 1
        ds.done_adding_data()
        protocol_data = pygsti.protocols.ProtocolData(self._exp_design, ds)

        # Save data for later analysis
        dir_name = str(get_base_path(self.scheduler))
        self.logger.info(f"Saving pyGSTi data to {dir_name}")
        protocol_data.write(dir_name)

        # Run RB
        protocol = pygsti.protocols.RB()
        results = protocol.run(protocol_data)
        r = results.fits['full'].estimates['r']
        r_std = results.fits['full'].stds['r']
        r_a_fix = results.fits['A-fixed'].estimates['r']
        r_a_fix_std = results.fits['A-fixed'].stds['r']
        self.logger.info(f"r = {r:1.2e} +/- {2 * r_std:1.2e} (fit with a free asymptote)")
        self.logger.info(f"r = {r_a_fix:1.2e} +/- {2 * r_a_fix_std:1.2e} (fit with the asymptote fixed to 1/2^n)")


"""Circuit Util Functions"""


__G_T = typing.TypeVar('__G_T')  # Type variable for gates


def _get_gates(circuit: str, available_gates: typing.Dict[str, __G_T], separator: str) -> typing.List[__G_T]:
    # Create list of gates
    return [available_gates[g] for g in circuit.split(f':{separator}')
            if g != f'@({separator})' and g != f'{{}}@({separator})']


def _partition_gate_list(circuit_list: typing.Sequence[__G_T],
                         max_partition_size: int) -> typing.List[typing.List[__G_T]]:
    # Partition circuit list based on max size the hardware can handle
    partition_list = [[]]
    while len(circuit_list):
        if np.size(circuit_list[0]) > max_partition_size:
            raise ValueError(f"Single circuit of size {np.size(circuit_list)} too large")
        elif np.size(circuit_list[0]) + np.size(partition_list[-1]) <= max_partition_size:
            partition_list[-1].append(circuit_list.pop(0))
        else:
            partition_list.append([])

    return partition_list
