import numpy as np
import typing
import collections.abc
import itertools

try:
    # pyGSTi is an optional dependency
    import pygsti  # type: ignore
except ImportError:
    pass
except AttributeError:
    raise ImportError('Probably an import error due to module aliasing, '
                      'make sure none of your user files are named "pygsti"') from AttributeError

from dax.experiment import *
from dax.modules.hist_context import HistogramContext, HistogramAnalyzer
from dax.interfaces.operation import OperationInterface
from dax.interfaces.gate import GateInterface
from dax.interfaces.detection import DetectionInterface
from dax.util.artiq import is_kernel
from dax.util.output import get_base_path

__all__ = ['RandomizedBenchmarkingSQ']

__G_T = typing.TypeVar('__G_T')  # Type variable for gates


def _get_gates(circuit: str, available_gates: typing.Dict[str, __G_T], separator: str) -> typing.List[__G_T]:
    """Create list of gates from a circuit."""
    try:
        return [available_gates[g] for g in circuit.split(f':{separator}')
                if g != f'@({separator})' and g != f'{{}}@({separator})']
    except KeyError as e:
        raise KeyError(f'Gate "{e}" is not available according to the dictionary!') from None


def _partition_circuit_list(circuit_list: typing.Sequence[typing.Sequence[__G_T]],
                            max_partition_size: int) -> typing.Sequence[typing.Sequence[typing.Sequence[__G_T]]]:
    """Partition circuit list based on max size the hardware can handle."""
    partitions: typing.List[typing.List[typing.Sequence[__G_T]]] = [[]]

    for circuit in circuit_list:
        circuit_size = len(circuit)
        if circuit_size > max_partition_size:
            raise ValueError(f'Single circuit of size {circuit_size} too large')
        elif circuit_size + sum(len(p) for p in partitions[-1]) <= max_partition_size:
            partitions[-1].append(circuit)
        else:
            partitions.append([circuit])

    return partitions


@dax_client_factory
class RandomizedBenchmarkingSQ(DaxClient, Experiment):
    """Single-qubit randomized benchmarking using pyGSTi.

    Note that pyGSTi is an optional dependency of DAX and is lazily imported.

    To use this client, a system needs to have the following components available:

    - At least one :class:`OperationInterface`
    - A :class:`DetectionInterface`
    - A :class:`HistogramContext` that functions as a data context for the :class:`OperationInterface`

    This class can be customized by overriding the :func:`add_arguments`,
    :func:`device_setup`, :func:`device_cleanup`, :func:`host_setup`, and :func:`host_cleanup` functions.

    Keep in mind that files in your repository can **not** be named ``pygsti`` as this will
    confuse the Python import machinery due to module name aliasing.
    """

    QUBIT_LABELS: typing.Sequence[str] = ['Q0']
    """Qubit labels for pyGSTi analysis."""
    MAX_CIRCUIT_DEPTH: int = 2 ** 14
    """The maximum circuit depth available (preferably a power of 2), mostly a memory limitation."""

    def build(self) -> None:  # type: ignore
        assert isinstance(self.QUBIT_LABELS, collections.abc.Sequence), 'Qubit labels must be a sequence'
        assert all(isinstance(label, str) for label in self.QUBIT_LABELS), 'All qubit labels must be of type str'
        assert isinstance(self.MAX_CIRCUIT_DEPTH, int), 'Max circuit depth must be of type int'
        assert self.MAX_CIRCUIT_DEPTH > 0, 'Max circuit depth must be greater than zero'
        assert is_kernel(self.device_setup), 'device_setup() must be a kernel function'
        assert is_kernel(self.device_cleanup), 'device_cleanup() must be a kernel function'
        assert not is_kernel(self.host_setup), 'host_setup() can not be a kernel function'
        assert not is_kernel(self.host_cleanup), 'host_cleanup() can not be a kernel function'

        # Search for operation interfaces
        self._operation_interfaces = self.registry.search_interfaces(OperationInterface)  # type: ignore[misc]
        if not self._operation_interfaces:
            raise LookupError('No operation interfaces were found')

        # Calculate available circuit depths
        self._available_circuit_depths = {str(2 ** n): n for n in range(int(np.log2(self.MAX_CIRCUIT_DEPTH)) + 1)}

        # Add general arguments
        self._operation_interface: str = self.get_argument('Operation interface',
                                                           EnumerationValue(sorted(self._operation_interfaces)),
                                                           tooltip='The operation interface to use for benchmarking')
        self._max_depth: int = self.get_argument('Max depth',
                                                 EnumerationValue(sorted(self._available_circuit_depths)),
                                                 tooltip='Max circuit depth, automatically generates every power '
                                                         'of 2 below this value as well')
        self._num_circuits: int = self.get_argument('Number of circuits',
                                                    NumberValue(default=10, min=1, ndecimals=0, step=1),
                                                    tooltip='Number of circuits to sample from')
        self._num_samples: int = self.get_argument('Number of samples',
                                                   NumberValue(default=100, min=1, ndecimals=0, step=1),
                                                   tooltip='Number of samples per circuit')
        self.update_kernel_invariants('_num_samples')

        # Add user arguments
        self.add_arguments()

        # Advanced arguments
        self._gate_delay: float = self.get_argument('Gate delay',
                                                    NumberValue(default=1 * us, unit='us', min=0 * us, step=1 * us,
                                                                ndecimals=1),
                                                    group='Advanced',
                                                    tooltip='Delay time between gates')
        self._real_time: bool = self.get_argument('Realtime gates', BooleanValue(True),
                                                  group='Advanced',
                                                  tooltip='Limit delay between gates to configured latency and DDS '
                                                          'latency only')
        self._check_pause_timeout: float = self.get_argument('Check pause timeout',
                                                             NumberValue(default=3 * s, unit='s', min=0 * s,
                                                                         step=1 * s),
                                                             group='Advanced',
                                                             tooltip='Minimum time between consecutive pause checks')
        self.update_kernel_invariants('_gate_delay', '_check_pause_timeout')

        # pyGSTi arguments
        self._verbosity: int = self.get_argument('Verbosity',
                                                 NumberValue(default=0, min=0, ndecimals=0, step=1),
                                                 group='pyGSTi',
                                                 tooltip='pyGSTi verbosity level')

        # Arguments for plotting
        self._plot_histograms: bool = self.get_argument('Plot histograms', BooleanValue(False),
                                                        group='Plot',
                                                        tooltip='Plot histograms at runtime')
        self._save_histograms: bool = self.get_argument('Save Histogram PDFs', BooleanValue(False),
                                                        group='Plot',
                                                        tooltip='Histograms will be saved as PDF files')

    def prepare(self):
        try:
            # Try to use pyGSTi
            version = pygsti.__version__
        except NameError:
            raise ImportError('pyGSTi is not available on your system')
        else:
            # Save pyGSTi version
            self.set_dataset('pygsti_version', version)
            self.logger.info(f'pyGSTi version: {version}')

        # Obtain system components
        self._operation = self._operation_interfaces[self._operation_interface]
        self._histogram_context = self.registry.find_module(HistogramContext)
        self._detect = self.registry.find_interface(DetectionInterface)
        self.update_kernel_invariants('_operation', '_histogram_context')

        # Get the scheduler
        self._scheduler = self.get_device('scheduler')
        self.update_kernel_invariants('_scheduler')

        # Create circuit depths list
        circuit_depths = [2 ** n for n in range(self._available_circuit_depths[self._max_depth] + 1)]
        self.logger.debug(f'Circuit depths: {circuit_depths}')
        # Get the available gates
        available_gates = self.get_available_gates(self._operation)
        available_gates_list = sorted(available_gates)
        self.logger.debug(f'Available gates: {", ".join(available_gates_list)}')

        # Create Processor Specifications and Experiment Design
        self.logger.debug('Creating pyGSTi processor spec')
        pspec = pygsti.obj.ProcessorSpec(nQubits=1, gate_names=available_gates_list, qubit_labels=self.QUBIT_LABELS,
                                         construct_models=('clifford',), verbosity=self._verbosity)
        self.logger.debug('Creating pyGSTi protocol')
        self._exp_design = pygsti.protocols.CliffordRBDesign(pspec, circuit_depths, self._num_circuits,
                                                             qubit_labels=self.QUBIT_LABELS, verbosity=self._verbosity)

        # Convert experiment design to circuit list
        self.logger.debug('Converting circuits')
        all_circuits = [_get_gates(c.str, available_gates, self.QUBIT_LABELS[0])
                        for c_list in self._exp_design.circuit_lists for c in c_list]
        # Partition circuit list
        self.logger.debug('Partitioning circuits')
        self._partitions = _partition_circuit_list(all_circuits, self.MAX_CIRCUIT_DEPTH)
        self.logger.debug(f'Number of circuit partitions: {len(self._partitions)}')

    def run(self):
        """Entry point of the experiment."""

        # Set realtime
        self._operation.set_realtime(self._real_time)

        if self._plot_histograms:
            # Plot histograms
            self._histogram_context.plot_histogram()

        try:
            # Perform host setup
            self.host_setup()

            for circuit_list in self._partitions:
                if self._scheduler.check_pause():
                    # This experiment can not be paused and will be terminated instead
                    raise TerminationRequested

                # Send one partition to the core device
                self._run_circuit_list(circuit_list)

        except TerminationRequested:
            # Experiment was terminated
            self.logger.warning('Experiment interrupted (this experiment can not pause)')
            raise

        finally:
            # Perform host cleanup
            self.host_cleanup()

    @kernel  # noqa:ATQ306
    def _run_circuit_list(self, circuit_list):  # noqa: ATQ306
        try:
            # Device setup
            self.device_setup()

            # Keep track of the check pause timeout
            t_next_pause_check = np.int64(now_mu() + self.core.seconds_to_mu(self._check_pause_timeout))

            # Iterate through circuits
            for circuit in circuit_list:
                with self._histogram_context:
                    # Schedule two circuits to improve performance (pipelining)
                    self._run_circuit(circuit)

                    for _ in range(self._num_samples - 1):
                        self._run_circuit(circuit)
                        self._operation.store_measurements_all()

                    # Perform final measurement
                    self._operation.store_measurements_all()

                if t_next_pause_check > now_mu():
                    if self._scheduler.check_pause():
                        # Interrupt current work, raise to break out of all loops
                        raise TerminationRequested
                    else:
                        # Update check pause timeout
                        t_next_pause_check = np.int64(now_mu() + self.core.seconds_to_mu(self._check_pause_timeout))

        finally:
            # Device cleanup
            self.device_cleanup()

    @kernel  # noqa:ATQ306
    def _run_circuit(self, circuit):  # noqa: ATQ306
        # Guarantee slack
        self.core.break_realtime()

        # Initialize
        self._operation.prep_0_all()

        # Perform gates
        for gate in circuit:
            # Preconfigured Delay
            delay(self._gate_delay)

            # Run gate
            gate(0)

        # Measure state
        self._operation.m_z_all()

    def analyze(self):
        # Create Histogram Analyzer
        h = HistogramAnalyzer(self._histogram_context)

        # Create RB dataset
        ds = pygsti.objects.DataSet(outcomeLabels=['0', '1'])
        threshold = self._detect.get_state_detection_threshold()
        for i, circuit in enumerate(itertools.chain.from_iterable(self._exp_design.circuit_lists)):
            one = HistogramAnalyzer.histogram_to_one_count(h.histograms['histogram'][0][i], threshold)
            ds.add_count_dict(circuit, {'0': self._num_samples - one, '1': one})
        ds.done_adding_data()
        protocol_data = pygsti.protocols.ProtocolData(self._exp_design, ds)

        # Save data for later analysis
        dir_name = str(get_base_path(self._scheduler).joinpath('pygsti'))
        self.logger.info(f'Saving pyGSTi data to {dir_name}')
        protocol_data.write(dir_name)

        # Run RB
        protocol = pygsti.protocols.RandomizedBenchmarking()
        results = protocol.run(protocol_data)
        r = results.fits['full'].estimates['r']
        r_std = results.fits['full'].stds['r']
        r_a_fix = results.fits['A-fixed'].estimates['r']
        r_a_fix_std = results.fits['A-fixed'].stds['r']
        self.logger.info(f"r = {r:1.2e} +/- {2 * r_std:1.2e} (fit with a free asymptote)")
        self.logger.info(f"r = {r_a_fix:1.2e} +/- {2 * r_a_fix_std:1.2e} (fit with the asymptote fixed to 1/2^n)")

        if self._save_histograms:
            # Save histograms as PDFs
            h.plot_all_histograms()

    """Customization functions"""

    # noinspection PyMethodMayBeStatic
    def get_available_gates(self, gate: GateInterface) -> typing.Dict[str, typing.Callable[..., typing.Any]]:
        """Given a gate interface, define a string to gate mapping.

        Users can override this function to modify the mapping.

        :param gate: The current gate interface
        :return: A dict that maps gate names to functions in the gate interface
        """
        return {
            'Gxpi': gate.x,
            'Gxpi2': gate.sqrt_x,
            'Gxmpi2': gate.sqrt_x_dag,
            'Gypi': gate.y,
            'Gypi2': gate.sqrt_y,
            'Gympi2': gate.sqrt_y_dag,
            'Gh': gate.h,
        }

    def add_arguments(self) -> None:
        """Add custom arguments during the build phase."""
        pass

    def host_setup(self) -> None:
        """Setup on the host, called once at entry."""
        pass

    @kernel
    def device_setup(self):  # type: () -> None
        """Setup on the core device, called once at each entry of a kernel.

        Should at least reset the core.
        """
        # Reset the core
        self.core.reset()

    @kernel
    def device_cleanup(self):  # type: () -> None
        """Cleanup on the core device, called when leaving a kernel.

        Users should add a ``self.core.break_realtime()`` at the start of this function
        to make sure operations have enough slack to execute.
        """
        pass

    def host_cleanup(self) -> None:
        """Cleanup on the host, called once at exit."""
        pass
