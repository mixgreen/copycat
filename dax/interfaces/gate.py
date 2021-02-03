# mypy: disallow_untyped_defs = False
# mypy: disallow_incomplete_defs = False

import abc
import math

from artiq.language.types import TInt32, TFloat

from dax.base.interface import DaxInterface

__all__ = ['GateInterface']


class GateInterface(DaxInterface, abc.ABC):
    """The gate interface exposes a set of quantum gates.

    When this interface is implemented, :attr:`pi` needs to be marked kernel invariant.

    Normally, all gate methods are expected to be kernels.
    Gates that are not implemented should raise a :attr:`NotImplementedError`.
    """

    # TODO: candidate gates: cnot, cz, xx
    # TODO: add machine unit variations for gates? (including pi_mu and conversion functions)

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

    @abc.abstractmethod
    def sqrt_x(self, qubit: TInt32):
        """sqrt(X).

        :param qubit: Target qubit
        """
        pass

    @abc.abstractmethod
    def sqrt_x_dag(self, qubit: TInt32):
        """sqrt(X) dagger.

        :param qubit: Target qubit
        """
        pass

    @abc.abstractmethod
    def sqrt_y(self, qubit: TInt32):
        """sqrt(Y).

        :param qubit: Target qubit
        """
        pass

    @abc.abstractmethod
    def sqrt_y_dag(self, qubit: TInt32):
        """sqrt(Y) dagger.

        :param qubit: Target qubit
        """
        pass

    @abc.abstractmethod
    def sqrt_z(self, qubit: TInt32):
        """sqrt(Z).

        :param qubit: Target qubit
        """
        pass

    @abc.abstractmethod
    def sqrt_z_dag(self, qubit: TInt32):
        """sqrt(Z) dagger.

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
