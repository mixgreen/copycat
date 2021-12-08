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

    Candidate gates for the gate interface are low- to intermediate-level theorist gates
    and low-level target-specific gates. Intermediate-level theorist gates are included for
    convenience and simulation. We consider the CNOT gate the ceiling for intermediate-level
    gates. The gate interface will include arbitrary-angle and fixed-angle gates where the
    latter can be pre-calculated and should therefore have shorter execution times.
    """

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

    """Clifford gates (single-qubit)"""

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

    """Arbitrary rotations (single-qubit)"""

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

    @optional
    def rphi(self, theta: TFloat, phi: TFloat, qubit: TInt32):
        """Rotation theta around the cos(phi)X + sin(phi)Y axis.

        :param theta: Rotation angle in radians (:attr:`pi` available for usage)
        :param phi: Angle of the rotation axis in the XY plane (:attr:`pi` available for usage)
        :param qubit: Target qubit
        """
        raise NotImplementedError

    """Two-qubit gates"""

    @optional
    def cz(self, control: TInt32, target: TInt32):
        """Controlled Z gate.

        :param control: Control qubit
        :param target: Target qubit
        """
        raise NotImplementedError

    @optional
    def cnot(self, control: TInt32, target: TInt32):
        """Controlled X gate.

        :param control: Control qubit
        :param target: Target qubit
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
