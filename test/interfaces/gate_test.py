import typing
import unittest

from artiq.language import TFloat, TInt32, kernel

from dax.base.interface import get_optionals
import dax.interfaces.gate


# noinspection PyAbstractClass
class _MinimalGateImplementation(dax.interfaces.gate.GateInterface):
    """A minimal correct implementation of the gate interface, only implements mandatory functions."""

    kernel_invariants = {'pi'}


class GateImplementation(_MinimalGateImplementation):
    """This should be a complete implementation of the gate interface."""

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

    @kernel
    def rphi(self, theta: TFloat, phi: TFloat, qubit: TInt32):
        pass

    @kernel
    def xx(self, control: TInt32, target: TInt32):
        pass

    @kernel
    def xx_dag(self, control: TInt32, target: TInt32):
        pass

    @kernel
    def xx_pi_2(self, control: TInt32, target: TInt32):
        pass

    @kernel
    def xx_pi_2_dag(self, control: TInt32, target: TInt32):
        pass

    @kernel
    def rxx(self, theta: TFloat, control: TInt32, target: TInt32):
        pass

    @kernel
    def cz(self, control: TInt32, target: TInt32):
        pass

    @kernel
    def cnot(self, control: TInt32, target: TInt32):
        pass


class GateInterfaceTestCase(unittest.TestCase):
    INTERFACE: typing.ClassVar[typing.Type] = dax.interfaces.gate.GateInterface
    MINIMAL_IMPLEMENTATION: typing.ClassVar[typing.Type] = _MinimalGateImplementation
    FULL_IMPLEMENTATION: typing.ClassVar[typing.Type] = GateImplementation

    def test_optionals(self):
        optionals = get_optionals(self.INTERFACE)
        instance = self.MINIMAL_IMPLEMENTATION()
        signatures = [(0,), (), (0.0, 0), (0, 1), (0.0, 0, 1)]  # Standard test signatures for optional functions

        for fn_name in optionals:
            fn = getattr(instance, fn_name)
            for args in signatures:
                try:
                    fn(*args)  # Test the implementation of the optional method (without overriding it)
                except NotImplementedError:
                    break  # This is the correct optional implementation
                except TypeError:
                    continue  # The signature is incorrect, continue to the next
                else:
                    self.fail(f'Optional function {fn_name} does not raise a NotImplementedError by default')
            else:
                self.fail(f'Optional function {fn_name} can not be matched to a test signature')

    def test_implemented_optionals(self):
        self.assertSetEqual(get_optionals(self.MINIMAL_IMPLEMENTATION),
                            get_optionals(self.INTERFACE),
                            'An optional method was implemented in the minimal implementation test class')
        self.assertSetEqual(get_optionals(self.FULL_IMPLEMENTATION), set(),
                            'Not all optional methods are implemented in the full implementation test class')

    def test_get_gates(self):
        data = [
            (self.MINIMAL_IMPLEMENTATION, set()),
            (self.FULL_IMPLEMENTATION, get_optionals(dax.interfaces.gate.GateInterface)),
        ]
        for cls, ref in data:
            for obj in [cls, cls()]:
                self.assertSetEqual(obj.get_gates(), ref)
