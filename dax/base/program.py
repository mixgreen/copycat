import typing
import abc

import artiq.experiment
import artiq.coredevice.core

import dax.base.system
import dax.interfaces.operation

__all__ = ['DaxProgram']


class DaxProgram(dax.base.system.DaxBase, abc.ABC):
    """Base class for DAX programs.

    DAX programs are generic experiments that can be dynamically linked to a DAX system.
    Normally, a DAX program would also inherit from the ARTIQ :class:`Experiment` or :class:`EnvExperiment`
    class and implement the :func:`prepare`, :func:`run`, and :func:`analyze` functions to
    define an execution flow. Additionally, a :func:`build` function can be implemented.

    A DAX program is designed like a regular ARTIQ experiment and has access to the following additional attributes:

    - :attr:`core`, is already populated with the core device driver
    - :attr:`q`, gate-level access to the quantum domain (see :class:`dax.interfaces.operation.OperationInterface`)
      (note: this attribute should only be used in the :func:`run` and :func:`analyze` functions)
    - :attr:`logger`, program logger (see also :class:`dax.base.system.DaxBase`)
      (note: should not be used in kernels)

    The ARTIQ environment of a DAX program is partially decoupled from the environment that hosts the DAX system.
    The device DB is empty, arguments are not passed to/from the DAX system, and datasets are isolated.
    """

    def __init__(self, managers_or_parent: typing.Any,
                 *args: typing.Any,
                 core: artiq.coredevice.core.Core,
                 interface: dax.interfaces.operation.OperationInterface,
                 **kwargs: typing.Any):
        """Construct a DAX program object.

        :param managers_or_parent: Manager or parent of this program
        :param core: The core object
        :param interface: The operation interface object
        :param args: Positional arguments forwarded to the :func:`build` function
        :param kwargs: Keyword arguments forwarded to the :func:`build` function
        """

        # Call super
        super(DaxProgram, self).__init__(managers_or_parent, *args, **kwargs)

        # Store attributes after ``build()``
        self.__core: artiq.coredevice.core.Core = core
        self.__q: dax.interfaces.operation.OperationInterface = interface
        # Update kernel invariants
        self.update_kernel_invariants('core', 'q')

    @property
    def core(self) -> artiq.coredevice.core.Core:
        """The core device driver.

        This attribute should only be used in the :func:`run` and :func:`analyze` functions.
        """
        return self.__core

    @property
    def q(self) -> dax.interfaces.operation.OperationInterface:
        """Property that provides gate-level access to the quantum domain.

        This attribute should only be used in the :func:`run` and :func:`analyze` functions.
        """
        return self.__q

    @artiq.experiment.host_only
    def get_identifier(self) -> str:
        """Return the class name."""
        return f'({self.__class__.__name__})'
