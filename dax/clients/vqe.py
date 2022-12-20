"""
Clients for VQE experiments

Author: Aniket S. Dalvi
"""

import abc
import queue
import threading
import typing
import numpy as np
from qiskit.algorithms.optimizers import COBYLA  # type: ignore[import]

from dax.experiment import *
from dax.modules.hist_context import HistogramContext
from dax.interfaces.operation import OperationInterface
from dax.interfaces.data_context import DataContextInterface
from dax.util.artiq import is_kernel, default_enumeration_value, is_rpc
from dax.base.servo import DaxServo


class DaxVQEBase(DaxClient, DaxServo, Experiment):
    """VQE base client class for portable VQE experiments."""

    DEFAULT_OPERATION_KEY: typing.ClassVar[typing.Union[str, typing.Type[NoDefault]]] = NoDefault
    """Key of the default operation interface."""

    def build_servo(self) -> None:
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
        """Function to define each iteration of VQE"""
        pass

    @abc.abstractmethod
    def ansatz_circuit(self, params):
        """Function to define ansatz circuit representing the hamiltonian"""
        pass

    @abc.abstractmethod
    def get_result(self):
        """Functions to define metric of stored measurements that the VQE experiiment uses"""
        pass

    @kernel
    def run_point(self, point, index):  # noqa: ATQ306
        # type: (typing.Any, typing.Any) -> None  # pragma: no cover(self) -> None:
        """Entry point of the experiment."""

        try:

            converge, params = self.iterate()

            if converge:
                self.stop_servo()
            else:
                self.set_point(params, point)

            self._run_circuit(point)

        except TerminationRequested:
            # Experiment was terminated
            self.logger.warning('Experiment interrupted (this experiment can not pause)')
            raise

    @kernel
    def _run_circuit(self, point):  # noqa: ATQ306

        with self._data_context:
            for i in range(self._num_samples):
                # Guarantee slack
                self.core.break_realtime()

                # Initialize
                self._operation.prep_0_all()

                # Perform gates
                self.ansatz_circuit(point)
                self._operation.store_measurements_all()

    @abc.abstractmethod
    def set_point(self, params, point):
        """Function to set next point of VQE in the servo infrastructure"""
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


@dax_client_factory
class SingleQubitVQE(DaxVQEBase):

    def _add_arguments_internal(self) -> None:
        np.random.seed(99999)  # type: ignore[attr-defined]
        p0 = np.random.random()  # type: ignore[attr-defined]
        self.target_dist = {0: p0, 1: 1 - p0}

        self.params = np.random.rand(3)  # type: ignore[attr-defined]
        self.theta = self.add_servo_argument('theta', 'theta', NumberValue(self.params[0]))
        self.phi = self.add_servo_argument('phi', 'phi', NumberValue(self.params[1]))
        self.lamb = self.add_servo_argument('lamb', 'lamb', NumberValue(self.params[2]))

    def host_setup(self) -> None:
        super().host_setup()
        self.param_q: queue.Queue = queue.Queue(1)
        self.data_q: queue.Queue = queue.Queue(1)
        self.index = 0
        self.opt_thread = threading.Thread(target=self.optimizer,
                                           args=(3, self.objective_function),
                                           kwargs={'initial_point': self.params})
        self.opt_thread.start()

    @kernel
    def ansatz_circuit(self, point) -> TNone:  # noqa: ATQ306
        self._operation.rz(point.phi, 0)
        self._operation.rx(-self._operation.pi / 2, 0)
        self._operation.rz(point.theta, 0)
        self._operation.rx(self._operation.pi / 2, 0)
        self._operation.rz(point.lamb, 0)
        self._operation.m_z_all()

    def get_result(self):
        mean_data = self._data_context.get_mean_counts()
        last_run = mean_data[0][-1]
        return last_run

    def optimizer(self, num_vars, objective_function, initial_point=None):
        opt_method = COBYLA(maxiter=500, tol=0.0001)
        opt_method.optimize(num_vars, objective_function, initial_point=initial_point)
        self.param_q.put(None)

    def objective_function(self, params):
        self.param_q.put(params)
        mean_counts = self.data_q.get()
        output_dist = {0: 1 - mean_counts, 1: mean_counts}
        cost = sum(
            abs(self.target_dist.get(i, 0) - output_dist.get(i, 0))
            for i in range(2)
        )
        return cost

    @rpc
    def iterate(self) -> TTuple([TBool, TTuple([TFloat, TFloat, TFloat])]):  # type: ignore[valid-type]  # noqa: ATQ309
        if self.index > 0:
            self.data_q.put(self.get_result())

        new_params = self.param_q.get()
        if new_params is None:
            converge = True
            new_params = (0, 0, 0)
        else:
            converge = False

        self.index += 1

        return (converge, tuple(new_params))

    @kernel
    def set_point(self, params, point) -> TNone:  # noqa: ATQ306
        point.theta, point.phi, point.lamb = params

    def host_cleanup(self) -> None:
        super().host_cleanup()
