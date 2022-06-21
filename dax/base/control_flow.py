import abc
import typing

from artiq.language.core import host_only, kernel, portable, TerminationRequested
from artiq.language.types import TBool

import dax.base.system
import dax.util.artiq

__all__ = ['DaxControlFlow']


class DaxControlFlow(dax.base.system.DaxBase, abc.ABC):
    """Control flow class for standardized experiment sequence.

    The control flow class contains an experiment control flow for setup and cleanup with host and kernel functions
    before entering a main run function that runs until completion. Other control-flow related classes can inherit from
    this base class to build experiment templates that make use of such control flows.

    The control flow is implemented in the following order by the corresponding functions:

    1. :func:`host_enter` (end-user function)
    2. :func:`host_setup` (end-user function)
    3. :func:`_dax_control_flow_setup`
    4. :func:`device_setup` (end-user function)
    5. :func:`_dax_control_flow_run`
    6. :func:`device_cleanup` (end-user function)
    7. :func:`_dax_control_flow_cleanup`
    8. :func:`host_cleanup` (end-user function)
    9. :func:`host_exit` (end-user function)

    Inheriting classes that implement an experiment template can or must implement the following functions:

    1. :func:`_dax_control_flow_while`
    2. :func:`_dax_control_flow_is_kernel`
    3. :func:`_dax_control_flow_run`
    4. :func:`_dax_control_flow_setup` (optional)
    5. :func:`_dax_control_flow_cleanup` (optional)
    """

    DAX_CONTROL_FLOW_CORE_ATTR: typing.ClassVar[str] = 'core'
    """Attribute name of the core device."""

    __terminated: bool
    __dax_control_flow_scheduler: typing.Any

    def build(self, *args: typing.Any, **kwargs: typing.Any) -> None:
        """Build the control flow object.

        :param args: Positional arguments forwarded to the superclass
        :param kwargs: Keyword arguments forwarded to the superclass
        """

        assert isinstance(self.DAX_CONTROL_FLOW_CORE_ATTR, str), 'Core attribute name must be of type str'
        assert dax.util.artiq.is_portable(self._dax_control_flow_while), 'While condition must be portable'
        assert not dax.util.artiq.is_kernel(self._dax_control_flow_is_kernel), 'Is kernel cannot be a kernel'
        assert dax.util.artiq.is_portable(self._dax_control_flow_run), 'Run must be portable'
        assert not dax.util.artiq.is_kernel(self._dax_control_flow_setup), 'Setup cannot be a kernel'
        assert not dax.util.artiq.is_kernel(self._dax_control_flow_cleanup), 'Cleanup cannot be a kernel'

        # Check if host_*() functions are all non-kernel functions
        if any(dax.util.artiq.is_kernel(fn) for fn in [self.host_enter, self.host_setup,
                                                       self.host_cleanup, self.host_exit]):
            raise TypeError('host_*() functions cannot be kernels')

        # Check if device_*() functions are kernels or portable if the main control flow is a kernel
        if self._dax_control_flow_is_kernel() and not all(dax.util.artiq.is_kernel(fn) or dax.util.artiq.is_portable(fn)
                                                          for fn in [self.device_setup, self.device_cleanup]):
            raise TypeError('device_*() functions must be kernels or portable when the main control flow is a kernel')

        # Initialize variables
        self.__terminated = False

        # Call super and forward arguments, for compatibility with other libraries
        # noinspection PyArgumentList
        super(DaxControlFlow, self).build(*args, **kwargs)

        # The scheduler object
        self.__dax_control_flow_scheduler = self.get_device('scheduler')
        self.update_kernel_invariants('_dax_control_flow_scheduler')  # Mark the property as kernel invariant

    @property
    def _dax_control_flow_is_terminated(self) -> bool:
        """:const:`True` if the control flow was terminated by the user."""
        return self.__terminated

    @property
    def _dax_control_flow_scheduler(self) -> typing.Any:
        """The scheduler object."""
        return self.__dax_control_flow_scheduler

    @host_only
    def run(self) -> None:
        """Entry point of the control flow."""

        if not hasattr(self, self.DAX_CONTROL_FLOW_CORE_ATTR):
            raise AttributeError(f'Core attribute "{self.DAX_CONTROL_FLOW_CORE_ATTR}" could not be found')

        try:
            # Call the host enter code
            self.logger.debug('Performing host enter')
            self.host_enter()

            while self._dax_control_flow_while():

                while self._dax_control_flow_scheduler.check_pause():
                    # Pause the run
                    self.logger.debug('Pausing run')
                    getattr(self, self.DAX_CONTROL_FLOW_CORE_ATTR).comm.close()  # Close communications before pausing
                    self._dax_control_flow_scheduler.pause()  # Can raise a TerminationRequested exception
                    self.logger.debug('Resuming run')

                try:
                    # Coming from a host context, perform host setup
                    self.logger.debug('Performing host setup')
                    self.host_setup()
                    self.logger.debug('Performing DAX control flow setup')
                    self._dax_control_flow_setup()

                    # Execute the run_point control flow
                    if self._dax_control_flow_is_kernel():
                        self.logger.debug('Running experiment on core device')
                        self._dax_control_flow_run_kernel()
                    else:
                        self.logger.debug('Running experiment on host')
                        self._dax_control_flow_run_portable()

                finally:
                    # One time host cleanup
                    self.logger.debug('Performing DAX control flow cleanup')
                    self._dax_control_flow_cleanup()
                    self.logger.debug('Performing host cleanup')
                    self.host_cleanup()

        except TerminationRequested:
            # Run was terminated
            self.__terminated = True
            self.logger.warning('Run was terminated by user request')

        else:
            # Call the host exit code
            self.logger.debug('Performing host exit')
            self.host_exit()

    @kernel
    def _dax_control_flow_run_kernel(self) -> None:
        """Run the control flow in a kernel."""
        self._dax_control_flow_run_portable()

    @portable
    def _dax_control_flow_run_portable(self) -> None:
        """Run the control flow (portable)."""
        try:
            # Perform device setup
            self.device_setup()

            while self._dax_control_flow_while():
                # Check for pause condition
                if self._dax_control_flow_scheduler.check_pause():
                    break  # Break to exit kernel

                # Run the main control flow
                self._dax_control_flow_run()

        finally:
            # Perform device cleanup
            self.device_cleanup()

    """Control-flow functions"""

    @abc.abstractmethod
    def _dax_control_flow_while(self) -> TBool:  # pragma: no cover
        """Internal function that returns :const:`True` if the main control loop should continue.

        This is different to a check pause, as this should check for experiment specific conditions.
        Additionally, this function must be portable.
        """
        pass

    @abc.abstractmethod
    def _dax_control_flow_is_kernel(self) -> bool:  # pragma: no cover
        """Internal function that returns :const:`True` if the main run function should be executed as a kernel."""
        pass

    @abc.abstractmethod
    def _dax_control_flow_run(self) -> None:  # pragma: no cover
        """Internal function that defines the main run function inside the control flow.

        This function must be portable.
        """
        pass

    def _dax_control_flow_setup(self) -> None:  # pragma: no cover
        """Internal function for host setup procedures called right after :func:`host_setup`."""
        pass

    def _dax_control_flow_cleanup(self) -> None:  # pragma: no cover
        """Internal function for host cleanup procedures called right before :func:`host_cleanup`."""
        pass

    """End-user functions"""

    @host_only
    def host_enter(self) -> None:  # pragma: no cover
        """1. Entry code on the host, called once."""
        pass

    @host_only
    def host_setup(self) -> None:  # pragma: no cover
        """2 Setup on the host, called once at entry and after a pause."""
        pass

    @portable
    def device_setup(self) -> None:  # pragma: no cover
        """3. Setup on the core device, called once at entry and after a pause.

        Can for example be used to reset the core.
        """
        pass

    @portable
    def device_cleanup(self) -> None:  # pragma: no cover
        """5. Cleanup on the core device, called once after scanning and before a pause.

        In case the device cleanup function is a kernel, it is good to add a ``self.core.break_realtime()``
        at the start of this function to make sure operations can execute in case of an underflow exception.
        Device cleanup often also contains a call like ``self.core.wait_until_mu(now_mu())``.
        """
        pass

    def host_cleanup(self) -> None:  # pragma: no cover
        """6. Cleanup on the host, called once after scanning and before a pause."""
        pass

    def host_exit(self) -> None:  # pragma: no cover
        """7. Exit code on the host, called if the scan finished without exceptions.

        This function is often used to finalize calibration procedures and store the latest calibration values.
        """
        pass
