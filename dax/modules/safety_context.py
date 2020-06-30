import typing
import numpy as np

from dax.experiment import *

__all__ = ['SafetyContext', 'SafetyContextError']


class SafetyContextError(RuntimeError):
    """Class for safety context errors.

    The constructor of this class should not be modified to maintain compatibility
    with the ARTIQ compiler.
    """
    pass


class SafetyContext(DaxModule):
    """Context class for safety controls when entering and exiting a context.

    Callback functions for enter and exit should be provided by the
    `enter_cb` and `exit_cb` kwargs respectively.

    This context is not reentrant and has all portable functions.
    Other objects can check if the context is entered by using
    the :func:`in_context` function.

    This class can optionally be subclassed and the class attribute
    :attr:`EXCEPTION_TYPE` can be overridden if desired.
    """

    EXCEPTION_TYPE: type = SafetyContextError
    """The exception type raised (must be a subclass of :class:`SafetyContextError`)."""

    def build(self, enter_cb, exit_cb) -> None:  # type: ignore
        """Build the safety context module.

        :param enter_cb: The callback function for entering the context.
        :param exit_cb: The callback function for exiting the context.
        """
        assert callable(enter_cb), 'Provided enter callback is not a callable'
        assert callable(exit_cb), 'Provided exit callback is not callable'

        # Add class constants as kernel invariants
        assert issubclass(self.EXCEPTION_TYPE, SafetyContextError), \
            'The provided exception type is not a subclass of SafetyContextError'
        self.update_kernel_invariants('EXCEPTION_TYPE', 'ENTER_ERR_MSG', 'EXIT_ERR_MSG')

        # Store references to the callback functions
        self._enter_cb: typing.Callable[[], None] = enter_cb
        self._exit_cb: typing.Callable[[], None] = exit_cb
        self.update_kernel_invariants('_enter_cb', '_exit_cb')

        # Store custom error messages
        self._enter_err_msg: str = f'Safety context "{self.get_name()}" is non-reentrant'
        self._exit_err_msg: str = f'Safety context "{self.get_name()}" has been exited more times than entered'
        self.update_kernel_invariants('_enter_err_msg', '_exit_err_msg')

        # By default we are not in context
        self._in_context: np.int32 = np.int32(0)  # This variable is NOT kernel invariant

    def init(self) -> None:
        pass

    def post_init(self) -> None:
        pass

    @portable
    def in_context(self) -> TBool:
        """True if we are in context."""
        return bool(self._in_context)

    @portable
    def __enter__(self):  # type: () -> None
        """Enter the safety context.

        Normally this function should not be called directly but by the `with` statement instead.
        """
        if self._in_context != 0:
            # Prevent nested context
            raise self.EXCEPTION_TYPE(self._enter_err_msg)

        # Call enter callback function
        self._enter_cb()
        # Increment in context counter after successful enter callback
        self._in_context += 1

    @portable
    def __exit__(self, exc_type, exc_val, exc_tb):  # type: (typing.Any, typing.Any, typing.Any) -> None
        """Exit the safety context.

        Normally this function should not be called directly but by the `with` statement instead.

        It is not possible to assign default values to the argument as the ARTIQ compiler only
        accepts `__exit__` functions with exactly four positional arguments.
        """
        if self._in_context <= 0:
            # Enter and exit calls were out of sync
            raise self.EXCEPTION_TYPE(self._exit_err_msg)

        # Call exit callback function
        self._exit_cb()
        # Decrement context counter after successful exit callback
        self._in_context -= 1