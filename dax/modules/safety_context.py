import typing
import numpy as np

from dax.experiment import *
from dax.util.artiq import is_kernel, is_host_only

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
    ``enter_cb`` and ``exit_cb`` kwargs respectively.

    For the safety context to work correct, **host and kernel synchronization must be taken into account**.
    See :func:`in_context` for more information.

    **This context is reentrant** and has all portable functions.
    The callback functions will only be called in the outermost context
    and not when entering or exiting a nested context.
    Other objects can check if the context is entered by using
    the :func:`in_context` function.

    This class can optionally be subclassed and the class attribute
    :attr:`EXCEPTION_TYPE` can be overridden if desired.
    """

    __CB_T = typing.Callable[[], None]  # Callback function type

    EXCEPTION_TYPE: typing.ClassVar[type] = SafetyContextError
    """The exception type raised (must be a subclass of :class:`SafetyContextError`)."""

    _safety_context_enter_cb: __CB_T
    _safety_context_exit_cb: __CB_T
    _safety_context_exit_error: bool
    _safety_context_exit_error_msg: str
    _safety_context_rpc: bool
    _safety_context_entries: np.int32

    def build(self, *, enter_cb: __CB_T, exit_cb: __CB_T,  # type: ignore[override]
              exit_error: bool = False, rpc_: bool = False) -> None:
        """Build the safety context module.

        :param enter_cb: The callback function for entering the context
        :param exit_cb: The callback function for exiting the context
        :param exit_error: Raise an error if exit is called more times than enter
        :param rpc_: Enter and exit events are handled in **async RPC's**
        """
        assert callable(enter_cb), 'Provided enter callback is not a callable'
        assert callable(exit_cb), 'Provided exit callback is not callable'
        assert isinstance(exit_error, bool), 'Exit error flag must be of type bool'
        assert isinstance(rpc_, bool), 'RPC flag must be of type bool'

        # Add class constants as kernel invariants
        assert issubclass(self.EXCEPTION_TYPE, SafetyContextError), \
            'The provided exception type is not a subclass of SafetyContextError'
        self.update_kernel_invariants('EXCEPTION_TYPE')

        # Store references to the callback functions
        self._safety_context_enter_cb = enter_cb  # type: ignore[misc,assignment]
        self._safety_context_exit_cb = exit_cb  # type: ignore[misc,assignment]
        self.update_kernel_invariants('_safety_context_enter_cb', '_safety_context_exit_cb')

        # Store exit error flag and custom error message
        self._safety_context_exit_error = exit_error
        self._safety_context_exit_error_msg = f'Safety context "{self.get_name()}" has been exited too many times'
        self.update_kernel_invariants('_safety_context_exit_error', '_safety_context_exit_error_msg')

        # Store RPC flag
        self._safety_context_rpc = rpc_
        self.update_kernel_invariants('_safety_context_rpc')
        if self._safety_context_rpc:
            assert not is_kernel(enter_cb), 'Enter callback can not be a kernel when RPC is enabled'
            assert not is_kernel(exit_cb), 'Exit callback can not be a kernel when RPC is enabled'
        else:
            assert not is_host_only(enter_cb), 'Enter callback can not be host only when RPC is disabled'
            assert not is_host_only(exit_cb), 'Exit callback can not be host only when RPC is disabled'

        # By default we are not in context
        self._safety_context_entries = np.int32(0)  # This variable is NOT kernel invariant

    def init(self) -> None:
        pass

    def post_init(self) -> None:
        pass

    @portable
    def in_context(self) -> TBool:
        """:const:`True` if we are in context.

        When using this function, **host and kernel synchronization must be taken into account**.
        Due to the nature of the context, enter and exit will both be called on the host or in a kernel.

        **RPC is disabled:** If the context is entered in a kernel, the host will not be aware that the
        context was entered when handling RPC calls due to delayed synchronization of variables.
        Hence, this function should not be used in RPC calls.

        **RPC is enabled:** If the context is entered in a kernel, the kernel will not be aware that the
        context was entered because context entering was handled on the host using RPC.
        Hence, this function should only be used in RPC calls or host-only functions.
        """
        return bool(self._safety_context_entries)

    @portable
    def _safety_context_enter(self):  # type: () -> None
        """Handle context enter (portable)."""
        if self._safety_context_entries == 0:
            # Call enter callback function
            self._safety_context_enter_cb()  # type: ignore[misc]

        # Increment in context counter after enter callback was successfully executed
        self._safety_context_entries += 1

    @rpc(flags={'async'})
    def _safety_context_enter_rpc(self):  # type: () -> None
        """Handle context enter (RPC)."""
        self._safety_context_enter()

    @portable
    def _enter(self):  # type: () -> None
        """Enter the safety context.

        This function can be used by subclasses to call the safety context enter procedure.
        """
        if self._safety_context_rpc:
            self._safety_context_enter_rpc()
        else:
            self._safety_context_enter()

    @portable
    def __enter__(self):  # type: () -> None
        """Enter the safety context.

        Normally this function should not be called directly but by the ``with`` statement instead.
        """
        self._enter()

    @portable
    def _safety_context_exit(self):  # type: () -> None
        """Handle context exit (portable)."""
        if self._safety_context_exit_error and self._safety_context_entries <= 0:
            # Enter and exit calls were out of sync
            raise self.EXCEPTION_TYPE(self._safety_context_exit_error_msg)

        if self._safety_context_entries > 0:
            # Decrement context counter before exit callback is executed
            self._safety_context_entries -= 1

            if self._safety_context_entries == 0:
                # Call exit callback function
                self._safety_context_exit_cb()  # type: ignore[misc]

    @rpc(flags={'async'})
    def _safety_context_exit_rpc(self):  # type: () -> None
        """Handle context exit (RPC)."""
        self._safety_context_exit()

    @portable
    def _exit(self):  # type: () -> None
        """Exit the safety context.

        This function can be used by subclasses to call the safety context exit procedure.
        """
        if self._safety_context_rpc:
            self._safety_context_exit_rpc()
        else:
            self._safety_context_exit()

    @portable  # noqa:ATQ306
    def __exit__(self, exc_type, exc_val, exc_tb):  # type: (typing.Any, typing.Any, typing.Any) -> None # noqa: ATQ306
        """Exit the safety context.

        Normally this function should not be called directly but by the ``with`` statement instead.

        It is not possible to assign default values to the argument as the ARTIQ compiler only
        accepts :func:`__exit__` functions with exactly four positional arguments.
        """
        self._exit()


class SafetyContext(ReentrantSafetyContext):
    """Context class for safety controls when entering and exiting a context.

    Callback functions for enter and exit should be provided by the
    ``enter_cb`` and ``exit_cb`` kwargs respectively.

    For the safety context to work correct, **host and kernel synchronization must be taken into account**.
    See :func:`in_context` for more information.

    **This context is not reentrant** and has all portable functions.
    Other objects can check if the context is entered by using
    the :func:`in_context` function.

    This class can optionally be subclassed and the class attribute
    :attr:`EXCEPTION_TYPE` can be overridden if desired.
    """

    _safety_context_enter_error_msg: str

    def build(self, **kwargs: typing.Any) -> None:  # type: ignore[override]
        """Build the safety context module."""

        # Call super
        super(SafetyContext, self).build(**kwargs)

        # Store custom error message
        self._safety_context_enter_error_msg = f'Safety context "{self.get_name()}" is non-reentrant'
        self.update_kernel_invariants('_safety_context_enter_error_msg')

    @portable
    def __enter__(self):  # type: () -> None
        """Enter the safety context.

        Normally this function should not be called directly but by the ``with`` statement instead.
        """
        if self._safety_context_entries != 0:
            # Prevent nested context
            raise self.EXCEPTION_TYPE(self._safety_context_enter_error_msg)

        # Note: not using super() because the ARTIQ compiler does not support it
        # Note: not using ReentrantSafetyContext.__enter__() because the ARTIQ compiler can not unify it
        if self._safety_context_entries == 0:
            # Call enter callback function
            self._safety_context_enter_cb()  # type: ignore[misc]

        # Increment in context counter after enter callback was successfully executed
        self._safety_context_entries += 1
