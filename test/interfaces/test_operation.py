import unittest
import inspect
import numpy as np

from artiq.language import TFloat, TInt32, TList, kernel, host_only

import dax.interfaces.operation
from dax.util.artiq import is_kernel, is_host_only


class _OperationInstance(dax.interfaces.operation.OperationInterface):
    """This should be a correct implementation of the operation interface."""

    NUM_QUBITS: int = 8

    def __init__(self):
        self.kernel_invariants = {'pi', 'num_qubits'}

    @kernel
    def prep_0_all(self):
        pass

    @kernel
    def m_z_all(self):
        pass

    @kernel
    def get_measurement(self, qubit: TInt32) -> TInt32:
        return np.int32(0)

    @kernel
    def store_measurements(self, qubits: TList(TInt32)):  # type: ignore
        pass

    @property
    def num_qubits(self) -> np.int32:
        return np.int32(self.NUM_QUBITS)

    @host_only
    def set_realtime(self, realtime: bool) -> None:
        pass

    @kernel
    def i(self, qubit: TInt32):
        pass

    @kernel
    def x(self, qubit: TInt32):
        pass

    @kernel
    def y(self, qubit: TInt32):
        pass

    @kernel
    def z(self, qubit: TInt32):
        pass

    @kernel
    def h(self, qubit: TInt32):
        pass

    @kernel
    def sqrt_x(self, qubit: TInt32):
        pass

    @kernel
    def sqrt_x_dag(self, qubit: TInt32):
        pass

    @kernel
    def sqrt_y(self, qubit: TInt32):
        pass

    @kernel
    def sqrt_y_dag(self, qubit: TInt32):
        pass

    @kernel
    def sqrt_z(self, qubit: TInt32):
        pass

    @kernel
    def sqrt_z_dag(self, qubit: TInt32):
        pass

    @kernel
    def rx(self, theta: TFloat, qubit: TInt32):
        pass

    @kernel
    def ry(self, theta: TFloat, qubit: TInt32):
        pass

    @kernel
    def rz(self, theta: TFloat, qubit: TInt32):
        pass


class OperationInterfaceTestCase(unittest.TestCase):
    def test_correct_interface(self):
        self.assertIsNone(dax.interfaces.operation.validate_operation_interface(_OperationInstance(),
                                                                                _OperationInstance.NUM_QUBITS))

    def _validate_functions(self, fn_names):
        for fn in fn_names:
            with self.subTest(fn=fn):
                # Create a new interface object and replace the function to test with the dummy function
                interface = _OperationInstance()
                setattr(interface, fn, self._dummy_fn)

                with self.assertRaises(AssertionError, msg='Validate did not raise'):
                    dax.interfaces.operation.validate_operation_interface(interface, interface.NUM_QUBITS)

    def test_validate_kernel_fn(self):
        kernel_fn = [n for n, fn in inspect.getmembers(_OperationInstance, inspect.isfunction) if is_kernel(fn)]
        self.assertGreater(len(kernel_fn), 0, 'No kernel functions were found')
        self._validate_functions(kernel_fn)

    def test_validate_host_only_fn(self):
        host_only_fn = [n for n, fn in inspect.getmembers(_OperationInstance, inspect.isfunction) if is_host_only(fn)]
        self.assertGreater(len(host_only_fn), 0, 'No host only functions were found')
        self._validate_functions(host_only_fn)

    def test_validate_invariants(self):
        interface = _OperationInstance()

        for invariant in interface.kernel_invariants:
            with self.subTest(invariant=invariant):
                # Remove the invariant and test
                interface.kernel_invariants.remove(invariant)
                with self.assertRaises(AssertionError, msg='Validate did not raise'):
                    dax.interfaces.operation.validate_operation_interface(interface, interface.NUM_QUBITS)

                # Restore the invariant
                interface.kernel_invariants.add(invariant)

    def test_validate_num_qubits(self):
        class _Instance(_OperationInstance):
            @property
            def num_qubits(self):
                # Cast to int, which is the wrong type
                return int(super(_Instance, self).num_qubits)

        interface = _Instance()
        with self.assertRaises(AssertionError, msg='Validate did not raise'):
            dax.interfaces.operation.validate_operation_interface(interface, interface.NUM_QUBITS)

    def _dummy_fn(self):
        """Dummy function used for testing."""
        pass


if __name__ == '__main__':
    unittest.main()
