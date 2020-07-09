import typing
import numpy as np

from dax.experiment import *

__all__ = ['ReentrantSafetyContext', 'SafetyContext', 'SafetyContextError']


class SafetyContextError(RuntimeError):
    """Class for safety context errors.

    The constructor of this class should not be modified to maintain compatibility
    with the ARTIQ compiler.
    """
    pass


class ReentrantSafetyContext(DaxModule):
    """Context class for safety controls when entering and exiting a context.

    Callback functions for enter and exit should be provided by the
    `enter_cb` and `exit_cb` kwargs respectively.

    **This context is reentrant** and has all portable functions.
    The callback functions will only be called in the outermost context
    and not when entering or exiting a nested context.
    Other objects can check if the context is entered by using
    the :func:`in_context` function.

    This class can optionally be subclassed and the class attribute
    :attr:`EXCEPTION_TYPE` can be overridden if desired.
    """

    __CB_T = typing.Callable[[], None]  # Callback function type

    EXCEPTION_TYPE: type = SafetyContextError
    """The exception type raised (must be a subclass of :class:`SafetyContextError`)."""

    def build(self, *, enter_cb: __CB_T, exit_cb: __CB_T) -> None:  # type: ignore
        """Build the safety context module.

        :param enter_cb: The callback function for entering the context
        :param exit_cb: The callback function for exiting the context
        """
        assert callable(enter_cb), 'Provided enter callback is not a callable'
        assert callable(exit_cb), 'Provided exit callback is not callable'

        # Add class constants as kernel invariants
        assert issubclass(self.EXCEPTION_TYPE, SafetyContextError), \
            'The provided exception type is not a subclass of SafetyContextError'
        self.update_kernel_invariants('EXCEPTION_TYPE')

        # Store references to the callback functions
        self._enter_cb: ReentrantSafetyContext.__CB_T = enter_cb
        self._exit_cb: ReentrantSafetyContext.__CB_T = exit_cb
        self.update_kernel_invariants('_enter_cb', '_exit_cb')

        # Store custom error messages
        self._exit_err_msg: str = f'Safety context "{self.get_name()}" has been exited more times than entered'
        self.update_kernel_invariants('_exit_err_msg')

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
        if self._in_context == 0:
            # Call enter callback function
            self._enter_cb()

        # Increment in context counter
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

        if self._in_context == 1:
            # Call exit callback function
            self._exit_cb()

        # Decrement context counter
        self._in_context -= 1


class SafetyContext(ReentrantSafetyContext):
    """Context class for safety controls when entering and exiting a context.

    Callback functions for enter and exit should be provided by the
    `enter_cb` and `exit_cb` kwargs respectively.

    **This context is not reentrant** and has all portable functions.
    Other objects can check if the context is entered by using
    the :func:`in_context` function.

    This class can optionally be subclassed and the class attribute
    :attr:`EXCEPTION_TYPE` can be overridden if desired.
    """

    def build(self, **kwargs: typing.Any) -> None:  # type: ignore
        """Build the safety context module."""

        # Call super
        super(SafetyContext, self).build(**kwargs)

        # Store custom error message
        self._enter_err_msg: str = f'Safety context "{self.get_name()}" is non-reentrant'
        self.update_kernel_invariants('_enter_err_msg')

    @portable
    def __enter__(self):  # type: () -> None
        """Enter the safety context.

        Normally this function should not be called directly but by the `with` statement instead.
        """
        if self._in_context != 0:
            # Prevent nested context
            raise self.EXCEPTION_TYPE(self._enter_err_msg)

        # Note: not using super() because the ARTIQ compiler does not support it
        # Note: not using ReentrantSafetyContext.__enter__() because the ARTIQ compiler can not unify it
        if self._in_context == 0:
            # Call enter callback function
            self._enter_cb()

        # Increment in context counter
        self._in_context += 1
