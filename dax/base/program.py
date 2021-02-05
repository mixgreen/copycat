import typing
import abc

import dax.base.system
import dax.interfaces.operation

__all__ = ['DaxProgram']


class DaxProgram(dax.base.system.DaxClient, abc.ABC):
    """Base class for DAX programs.

    DAX programs are generic experiments that can be dynamically linked to a DAX system.
    Normally, a program would inherit from the ARTIQ :class:`Experiment` or :class:`EnvExperiment`
    class and implement the :func:`prepare`, :func:`run`, and :func:`analyze` functions to
    define an execution flow. Additionally, a :func:`build` function can be implemented to
    parametrize the program using arguments.

    Note that the :func:`build` function does not need to call `super()`.
    The decorator will make sure all classes are build in the correct order.

    A DAX program is designed like a regular ARTIQ experiments.
    Additionally, users have access to regular DAX constructs and a special attribute :attr:`q`.

    - :attr:`core`, :attr:`core_dma` and :attr:`core_cache` are already populated
    - :attr:`q`, gate-level access to the quantum domain (see :class:`dax.interface.operation.OperationInterface`)
    - :attr:`logger`, logger (should not be used in kernels)
    - :func:`init`, DAX initialization function
    - :func:`post_init`, DAX post-initialization function
    - :attr:`DAX_INIT`, flag to enable/disable DAX initialization (enabled by default)

    See also :class:`dax.base.system.DaxClient`, :class:`dax.base.system.DaxHasSystem`,
    and :class:`dax.base.system.DaxHasSystem`.
    """

    def __init__(self, managers_or_parent: dax.base.system.DaxSystem,
                 *args: typing.Any, **kwargs: typing.Any):
        """Construct the DAX client object.

        :param managers_or_parent: Manager or parent of this program
        :param args: Positional arguments forwarded to the :func:`build` function
        :param kwargs: Keyword arguments forwarded to the :func:`build` function
        """
        assert isinstance(managers_or_parent, dax.base.system.DaxSystem), 'The parent must be a DaxSystem'

        # Obtain operation interface from parent before the users build() function is called
        self.__q = managers_or_parent.registry.find_interface(
            dax.interfaces.operation.OperationInterface)  # type: ignore[misc]
        # Validate the operation interface
        dax.interfaces.operation.validate_operation_interface(self.q)
        # Add kernel invariants
        self.update_kernel_invariants('q')
        # Call super
        super(DaxProgram, self).__init__(managers_or_parent, *args, **kwargs)

    @property
    def q(self) -> dax.interfaces.operation.OperationInterface:
        """Property that provides gate-level access to the quantum domain."""
        return self.__q
