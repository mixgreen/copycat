"""
Clients that use pyGSTi. See https://github.com/pyGSTio/pyGSTi.

Note that pyGSTi is an optional dependency of DAX.

Author: Jacob Whitlow
"""

import abc
import typing
import pathlib
import multiprocessing
import numpy as np
import natsort

from dax.experiment import *
from dax.modules.hist_context import HistogramContext, HistogramAnalyzer
from dax.interfaces.operation import OperationInterface
from dax.interfaces.gate import GateInterface
from dax.interfaces.data_context import DataContextInterface
from dax.util.artiq import is_kernel, default_enumeration_value
from dax.util.output import get_base_path
from dax.base.servo import DaxServo

__all__ = ['RandomizedBenchmarkingSQ', 'GateSetTomographySQ']


@dax_client_factory
class DaxVQEBase(DaxClient, DaxServo, Experiment):
    """VQE base client class for portable VQE experiments."""

    DEFAULT_OPERATION_KEY: typing.ClassVar[typing.Union[str, typing.Type[NoDefault]]] = NoDefault
    """Key of the default operation interface."""

    def build_servo(self) -> None:  # type: ignore[override]
        assert is_kernel(self.device_setup), 'device_setup() must be a kernel function'
        assert is_kernel(self.device_cleanup), 'device_cleanup() must be a kernel function'
        assert not is_kernel(self.host_setup), 'host_setup() can not be a kernel function'
        assert not is_kernel(self.host_cleanup), 'host_cleanup() can not be a kernel function'
        assert is_kernel(self.set_point), 'set_point() must be a kernel function'
        assert is_kernel(self.ansatz_circuit), 'ansatz_circuit() must be a kernel function'
        assert is_rpc(self.iterate), 'iterate() must be a rpc function'

        # Search for interfaces
        self._operation_interfaces = self.registry.search_interfaces(OperationInterface)  # type: ignore[misc]
        if not self._operation_interfaces:
            raise LookupError('No operation interfaces were found')
        self._data_contexts = self.registry.search_interfaces(DataContextInterface)  # type: ignore[misc]
        if not self._data_contexts:
            raise LookupError('No histogram/data contexts were found')

        # Add general arguments
        self._operation_interface_key: str = self.get_argument(
            'Operation interface',
            default_enumeration_value(sorted(self._operation_interfaces), default=self.DEFAULT_OPERATION_KEY),
            tooltip='The operation interface to use'
        )
        self._data_context_key: str = self.get_argument(
            'Histogram context',
            default_enumeration_value(sorted(self._data_contexts)),
            tooltip='The histogram/data context to use'
        )
        self._num_samples: int = self.get_argument(
            'Number of samples',
            NumberValue(100, min=1, ndecimals=0, step=1),
            tooltip='Number of samples per circuit'
        )
        self.update_kernel_invariants('_num_samples')

        # Add extra arguments
        self._add_arguments_internal()
        self.add_arguments()

        # Advanced arguments
        self._real_time: bool = self.get_argument(
            'Realtime gates',
            BooleanValue(True),
            group='Advanced',
            tooltip='Compensate device configuration latencies for gates'
        )

        # Arguments for plotting
        self._plot_histograms: bool = self.get_argument(
            'Plot histograms', BooleanValue(False),
            group='Plot',
            tooltip='Plot histograms at runtime'
        )
        self._plot_probability: bool = self.get_argument(
            'Plot probability', BooleanValue(False),
            group='Plot',
            tooltip='Plot state probability at runtime (shows up as the mean count plot to include error bars)'
        )
        self._save_probability: bool = self.get_argument(
            'Save probability plot', BooleanValue(False),
            group='Plot',
            tooltip='Probability plot will be saved as a PDF file (stored as mean count plots to include error bars)'
        )

    def _add_arguments_internal(self) -> None:
        """Add custom arguments.

        **For internal usage only**. See also :func:`add_arguments`.
        """
        pass

    def prepare(self) -> None:

        # Save configuration
        self.set_dataset('operation_interface_key', self._operation_interface_key)
        self.set_dataset('data_context_key', self._data_context_key)
        self.set_dataset('num_samples', self._num_samples)
        self.set_dataset('real_time', self._real_time)

        # Obtain system components
        self._operation = self._operation_interfaces[self._operation_interface_key]
        self._data_context = self._data_contexts[self._data_context_key]
        self.update_kernel_invariants('_operation', '_data_context')

        # Get the scheduler
        self._scheduler = self.get_device('scheduler')
        self.update_kernel_invariants('_scheduler')

    @abc.abstractmethod
    def iterate(self):
        data = self._data_context.get_raw()
        converge = self.convergence_condition(data)
        params = self.optimizer(data)
        return (converge, params)

    @abc.abstractmethod
    def optimizer(self, data):
        pass

    @abc.abstractmethod
    def ansatz_circuit(self, params):
        pass

    @abc.abstractmethod
    def convergence_condition(self, data):
        pass

    @kernel
    def run_point(self, point, index):  # type: (typing.Any, typing.Any) -> None  # pragma: no cover(self) -> None:
        """Entry point of the experiment."""

        try:
            self._run_circuit(point)

            self.update_new_result()

            converge, params = self.iterate()

            if converge:
                self.stop_servo()
            else:
                self.set_point(params, point)

        except TerminationRequested:
            # Experiment was terminated
            self.logger.warning('Experiment interrupted (this experiment can not pause)')
            raise

    @kernel  # noqa: ATQ306
    def _run_circuit(self, point):  # noqa: ATQ306

        for _ in range(self._num_samples):
            # Guarantee slack
            self.core.break_realtime()

            # Initialize
            self._operation.prep_0_all()

            # Perform gates
            with self._data_context:
                # Run gate
                self.ansatz_circuit(point)
                self._operation.store_measurements_all()

    @abc.abstractmethod
    def set_point(self, params, point):
        pass

    def analyze(self) -> None:
        self._analyze_internal()

    def _analyze_internal(self) -> None:  # pragma: no cover
        """Protocol-specific analysis.

        **For internal usage only**.
        """
        pass

    """Customization functions"""

    def add_arguments(self) -> None:
        """Add custom arguments during the build phase."""
        pass

    def host_setup(self) -> None:
        """Setup on the host, called once at entry."""
        # Save configuration
        self.set_dataset('num_qubits', self._operation.num_qubits)

        # Check if the target qubit is in range (must be done in run())
        if not 0 <= self._target_qubit < self._operation.num_qubits:
            raise ValueError(f'Target qubit is out of range (number of qubits: {self._operation.num_qubits})')

        # Set realtime
        self._operation.set_realtime(self._real_time)

        if isinstance(self._data_context, HistogramContext):
            # Enable plots
            if self._plot_histograms:
                self._data_context.plot_histogram(x_label='State')
            if self._plot_probability:
                # Mean counts will show the same as the probability plot (plus error bars) due to binary measurements
                self._data_context.plot_mean_count(x_label='Circuit', y_label='State')
        elif self._plot_histograms or self._plot_probability:
            self.logger.warning('Cannot enable real-time plots, requires a histogram context')

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
        self.logger.info(f"Converged parameters {self.get_servo_values()}")
        self.logger.info(f"Converged data {self._data_context.get_raw()}")
