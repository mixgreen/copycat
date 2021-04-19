# mypy: disallow_untyped_defs = False
# mypy: disallow_incomplete_defs = False

import abc
import typing
import inspect

import numpy as np

from artiq.language import TInt32, TList, kernel

from dax.interfaces.gate import GateInterface
import dax.util.artiq

__all__ = ['OperationInterface', 'validate_interface']


class OperationInterface(GateInterface, abc.ABC):
    """The operation interface exposes a set of quantum operations and functions to handle measurement results.

    When this interface is implemented, :attr:`pi` and :attr:`num_qubits` need to be marked kernel invariant.

    All operation methods are expected to be kernel functions unless mentioned otherwise.
    Operations that are not implemented should raise a :class:`NotImplementedError`.

    Note that some functions of the operation interface must be used inside a
    :class:`dax.interfaces.data_context.DataContextInterface` context.
    """

    """Operations"""

    @abc.abstractmethod
    def prep_0_all(self):
        """Prepare all qubits to the ``|0>`` state."""
        pass

    @abc.abstractmethod
    def m_z_all(self):
        """Measure all qubits.

        This method only schedules measurement operations and does not return the result.
        The results, when it becomes available, will be kept in a buffer until the user retrieves it.
        All measurement results must be retrieved by the user using :func:`get_measurement`
        or :func:`store_measurements`.
        """
        pass

    """Measurement handling"""

    @abc.abstractmethod
    def get_measurement(self, qubit: TInt32) -> TInt32:
        """Get the binary result of a qubit measurement.

        This function can be used after one or more measurement operations were scheduled on the target qubit.
        The execution of this function will block until the measurement result is available.

        This function can be used in a list comprehension to obtain the measurement results of multiple qubits:

        ``measurements = [self.get_measurement(q) for q in range(self.num_qubits)]``

        :param qubit: The target qubit
        :return: The binary measurement result, 0 or 1
        """
        pass

    @abc.abstractmethod
    def store_measurements(self, qubits: TList(TInt32)):  # type: ignore[valid-type]
        """Store the binary measurement results of the given qubits in the archive.

        This function must be used inside a :class:`dax.interfaces.data_context.DataContextInterface` context.
        It can be used after one or more measurement operations were scheduled on each target qubit.
        The execution of this function will block until all measurement results are available.

        :param qubits: A list of target qubits
        """
        pass

    @kernel
    def store_measurements_all(self):
        """Store the binary measurement results of all qubits in the archive.

        This function must be used inside a :class:`dax.interfaces.data_context.DataContextInterface` context.
        It can be used after one or more measurement operations were scheduled on all qubits.
        The execution of this function will block until all measurement results are available.
        """
        self.store_measurements(list(range(self.num_qubits)))

    """Properties and configuration"""

    @property
    @abc.abstractmethod
    def num_qubits(self) -> np.int32:
        """The number of qubits in the system."""
        pass

    @abc.abstractmethod
    def set_realtime(self, realtime: bool) -> None:
        """Set the realtime flag.

        **This function can only be called from the host.**

        When enabled, all operations are performed realtime.
        This means that the operations are started exactly at the timestamp on which the function was called.
        If realtime is disabled, slack is added before every operation to relax timing constraints.

        Note that the realtime flag is not persistent over multiple experiments.
        It will be reset to its default value for every experiment.

        :param realtime: True to enable realtime for this experiment
        """
        pass


def validate_interface(interface: OperationInterface, *, num_qubits: typing.Optional[int] = None) -> bool:
    """Validate an operation interface object.

    :param interface: The operation interface object
    :param num_qubits: The exact number of qubits in the system (optional)
    :return: :const:`True`, to allow usage of this function in an ``assert`` statement
    :raise TypeError: Raised if validation failed
    """
    if not isinstance(interface, OperationInterface):
        raise TypeError('The provided interface is not of type OperationInterface')
    if not isinstance(num_qubits, (int, type(None))):
        raise TypeError('Num qubits must be of type int or None')

    # Validate properties
    properties: typing.Dict[str, typing.Callable[[typing.Any], bool]] = {
        'num_qubits': lambda p: isinstance(p, np.int32) and (p == num_qubits or num_qubits is None),
    }
    if not all(fn(getattr(interface, p, None)) for p, fn in properties.items()):
        raise TypeError('Not all properties return the correct types')

    # Validate kernel invariants
    kernel_invariants: typing.Set[str] = {'pi'} | properties.keys()
    if not all(i in getattr(interface, 'kernel_invariants', {}) for i in kernel_invariants):
        raise TypeError('Not all kernel invariants are correctly added')

    # Validate host only functions
    host_only_fn: typing.Set[str] = {'set_realtime'}
    if not all(dax.util.artiq.is_host_only(getattr(interface, fn, None)) for fn in host_only_fn):
        raise TypeError('Not all host only functions are decorated correctly')

    # Validate kernel functions
    kernel_fn: typing.Set[str] = {n for n, _ in inspect.getmembers(OperationInterface, inspect.isfunction)
                                  if not n.startswith('_') and n not in host_only_fn}
    if not all(dax.util.artiq.is_kernel(getattr(interface, fn, None)) for fn in kernel_fn):
        raise TypeError('Not all kernel functions are decorated correctly')

    # Return True
    return True
