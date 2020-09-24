# mypy: disallow_untyped_defs = False
# mypy: disallow_incomplete_defs = False

import abc
import math

from artiq.language.types import TInt32, TFloat

from dax.base.interface import DaxInterface

__all__ = ['GateInterface']


class GateInterface(DaxInterface, abc.ABC):
    """The gate interface exposes a set of quantum gates.

    When this interface is implemented :attr:`pi` needs to be marked kernel invariant.

    Gates that are not implemented should raise a :attr:`NotImplementedError`.
    """

    # TODO: future candidate functions: cnot, cz, xx
    # TODO: candidate operations: prep_0, m_z (probably for a different interface)
    # TODO: add sqrt and inverse rotations?

    @property
    def pi(self) -> float:
        """Return the constant pi.

        :return: Float value of pi.
        """
        return float(math.pi)

    """Pauli gates"""

    @abc.abstractmethod
    def i(self, qubit: TInt32):
        """Pauli I.

        :param qubit: Target qubit
        """
        pass

    @abc.abstractmethod
    def x(self, qubit: TInt32):
        """Pauli X.

        :param qubit: Target qubit
        """
        pass

    @abc.abstractmethod
    def y(self, qubit: TInt32):
        """Pauli Y.

        :param qubit: Target qubit
        """
        pass

    @abc.abstractmethod
    def z(self, qubit: TInt32):
        """Pauli Z.

        :param qubit: Target qubit
        """
        pass

    """Clifford gates"""

    @abc.abstractmethod
    def h(self, qubit: TInt32):
        """Hadamard.

        :param qubit: Target qubit
        """
        pass

    """Arbitrary rotations"""

    @abc.abstractmethod
    def rx(self, theta: TFloat, qubit: TInt32):
        """Arbitrary X rotation.

        :param theta: Rotation angle in radians (:attr:`pi` available for usage)
        :param qubit: Target qubit
        """
        pass

    @abc.abstractmethod
    def ry(self, theta: TFloat, qubit: TInt32):
        """Arbitrary Y rotation.

        :param theta: Rotation angle in radians (:attr:`pi` available for usage)
        :param qubit: Target qubit
        """
        pass

    @abc.abstractmethod
    def rz(self, theta: TFloat, qubit: TInt32):
        """Arbitrary Z rotation.

        :param theta: Rotation angle in radians (:attr:`pi` available for usage)
        :param qubit: Target qubit
        """
        pass
