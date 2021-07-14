# mypy: disallow_untyped_defs = False
# mypy: disallow_incomplete_defs = False

import abc
import math
import typing

from artiq.language.types import TInt32, TFloat
from artiq.language.core import host_only

from dax.base.interface import DaxInterface, optional, get_optionals

__all__ = ['GateInterface']


class GateInterface(DaxInterface, abc.ABC):  # pragma: no cover
    """The gate interface exposes a set of quantum gates.

    When this interface is implemented, :attr:`pi` needs to be marked kernel invariant.

    Normally, all gate methods are expected to be kernels.
    Gates that are not implemented raise a :attr:`NotImplementedError`.
    """

    # TODO: candidate gates: cnot, cz, xx
    # TODO: add machine unit variations for gates? (including pi_mu and conversion functions)

    @property
    def pi(self) -> float:
        """Return the constant pi.

        :return: Float value of pi.
        """
        return math.pi

    """Pauli gates"""

    @optional
    def i(self, qubit: TInt32):
        """Pauli I.

        :param qubit: Target qubit
        """
        raise NotImplementedError

    @optional
    def x(self, qubit: TInt32):
        """Pauli X.

        :param qubit: Target qubit
        """
        raise NotImplementedError

    @optional
    def y(self, qubit: TInt32):
        """Pauli Y.

        :param qubit: Target qubit
        """
        raise NotImplementedError

    @optional
    def z(self, qubit: TInt32):
        """Pauli Z.

        :param qubit: Target qubit
        """
        raise NotImplementedError

    """Clifford gates"""

    @optional
    def h(self, qubit: TInt32):
        """Hadamard.

        :param qubit: Target qubit
        """
        raise NotImplementedError

    @optional
    def sqrt_x(self, qubit: TInt32):
        """sqrt(X).

        :param qubit: Target qubit
        """
        raise NotImplementedError

    @optional
    def sqrt_x_dag(self, qubit: TInt32):
        """sqrt(X) dagger.

        :param qubit: Target qubit
        """
        raise NotImplementedError

    @optional
    def sqrt_y(self, qubit: TInt32):
        """sqrt(Y).

        :param qubit: Target qubit
        """
        raise NotImplementedError

    @optional
    def sqrt_y_dag(self, qubit: TInt32):
        """sqrt(Y) dagger.

        :param qubit: Target qubit
        """
        raise NotImplementedError

    @optional
    def sqrt_z(self, qubit: TInt32):
        """sqrt(Z).

        :param qubit: Target qubit
        """
        raise NotImplementedError

    @optional
    def sqrt_z_dag(self, qubit: TInt32):
        """sqrt(Z) dagger.

        :param qubit: Target qubit
        """
        raise NotImplementedError

    """Arbitrary rotations"""

    @optional
    def rx(self, theta: TFloat, qubit: TInt32):
        """Arbitrary X rotation.

        :param theta: Rotation angle in radians (:attr:`pi` available for usage)
        :param qubit: Target qubit
        """
        raise NotImplementedError

    @optional
    def ry(self, theta: TFloat, qubit: TInt32):
        """Arbitrary Y rotation.

        :param theta: Rotation angle in radians (:attr:`pi` available for usage)
        :param qubit: Target qubit
        """
        raise NotImplementedError

    @optional
    def rz(self, theta: TFloat, qubit: TInt32):
        """Arbitrary Z rotation.

        :param theta: Rotation angle in radians (:attr:`pi` available for usage)
        :param qubit: Target qubit
        """
        raise NotImplementedError

    """Properties and configuration"""

    @classmethod
    @host_only
    def get_gates(cls) -> typing.Set[str]:
        """Get a set of available gates.

        :return: The implemented gate functions as a set of strings
        """
        return get_optionals(GateInterface) - get_optionals(cls)
