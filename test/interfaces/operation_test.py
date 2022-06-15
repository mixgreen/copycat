import unittest.mock
import inspect
import numpy as np

from artiq.language import TInt32, TList, kernel, host_only

from dax.base.interface import get_optionals
import dax.interfaces.operation
from dax.util.artiq import is_kernel, is_host_only

import test.interfaces.gate_test


# noinspection PyAbstractClass
class _MinimalOperationImplementation(dax.interfaces.operation.OperationInterface):
    """A minimal correct implementation of the operation interface, only implements mandatory functions."""

    NUM_QUBITS: int = 8
    kernel_invariants = {'pi', 'num_qubits'}

    @kernel
    def get_measurement(self, qubit: TInt32) -> TInt32:
        return np.int32(0)

    @kernel
    def store_measurements(self, qubits: TList(TInt32)):  # type: ignore[valid-type]
        pass

    @property
    def num_qubits(self) -> np.int32:
        return np.int32(self.NUM_QUBITS)

    @host_only
    def set_realtime(self, realtime: bool) -> None:
        pass


class OperationImplementation(_MinimalOperationImplementation, test.interfaces.gate_test.GateImplementation):
    """This should be a complete implementation of the operation interface."""

    @kernel
    def prep_0(self, qubit: TInt32):
        pass

    @kernel
    def prep_0_all(self):
        pass

    @kernel
    def m_z(self, qubit: TInt32):
        pass

    @kernel
    def m_z_all(self):
        pass


class OperationInterfaceTestCase(test.interfaces.gate_test.GateInterfaceTestCase):
    INTERFACE = dax.interfaces.operation.OperationInterface
    MINIMAL_IMPLEMENTATION = _MinimalOperationImplementation
    FULL_IMPLEMENTATION = OperationImplementation

    def test_get_operations(self):
        data = [
            (self.MINIMAL_IMPLEMENTATION, set()),
            (self.FULL_IMPLEMENTATION, get_optionals(dax.interfaces.operation.OperationInterface)),
        ]
        for cls, ref in data:
            for obj in [cls, cls()]:
                self.assertSetEqual(obj.get_operations(), ref)

    def test_validate_interface(self):
        for interface in [self.MINIMAL_IMPLEMENTATION(), self.FULL_IMPLEMENTATION()]:
            self.assertTrue(dax.interfaces.operation.validate_interface(interface))
            self.assertTrue(dax.interfaces.operation.validate_interface(
                interface, num_qubits=self.FULL_IMPLEMENTATION.NUM_QUBITS))

    def _validate_functions(self, fn_names):
        interface = self.FULL_IMPLEMENTATION()

        for fn in fn_names:
            with self.subTest(fn=fn):
                # Make sure the interface is valid
                dax.interfaces.operation.validate_interface(interface, num_qubits=interface.NUM_QUBITS)

                with unittest.mock.patch.object(interface, fn, self._dummy_fn):
                    # Patch interface and verify the validation fails
                    with self.assertRaises(TypeError, msg='Validate did not raise'):
                        dax.interfaces.operation.validate_interface(interface, num_qubits=interface.NUM_QUBITS)

    def test_validate_optional_fn(self):
        optionals = get_optionals(dax.interfaces.operation.OperationInterface)
        self.assertGreater(len(optionals), 0, 'No optional functions were found')
        self._validate_functions(optionals)

    def test_validate_kernel_fn(self):
        kernel_fn = [n for n, fn in inspect.getmembers(self.FULL_IMPLEMENTATION, inspect.isfunction) if is_kernel(fn)]
        self.assertGreater(len(kernel_fn), 0, 'No kernel functions were found')
        self._validate_functions(kernel_fn)

    def test_validate_host_only_fn(self):
        host_only_fn = [n for n, fn in inspect.getmembers(self.FULL_IMPLEMENTATION, inspect.isfunction) if
                        is_host_only(fn)]
        self.assertGreater(len(host_only_fn), 0, 'No host only functions were found')
        self._validate_functions(host_only_fn)

    def test_validate_invariants(self):
        interface = self.FULL_IMPLEMENTATION()

        for invariant in interface.kernel_invariants:
            with self.subTest(invariant=invariant):
                # Remove the invariant and test
                interface.kernel_invariants.remove(invariant)
                with self.assertRaises(TypeError, msg='Validate did not raise'):
                    dax.interfaces.operation.validate_interface(interface, num_qubits=interface.NUM_QUBITS)

                # Restore the invariant
                interface.kernel_invariants.add(invariant)

    def test_validate_num_qubits(self):
        class _Instance(self.FULL_IMPLEMENTATION):
            @property
            def num_qubits(self):
                # Cast to int, which is the wrong type
                return int(super(_Instance, self).num_qubits)

        interface = _Instance()
        with self.assertRaises(TypeError, msg='Validate did not raise'):
            dax.interfaces.operation.validate_interface(interface, num_qubits=interface.NUM_QUBITS)

    def _dummy_fn(self):
        """Dummy function used for testing."""
        pass
