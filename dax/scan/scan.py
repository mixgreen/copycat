import abc
import typing
import collections

import dax.base.dax
from dax.experiment import *

__all__ = ['DaxScan']


class DaxScan(dax.base.dax.DaxBase, abc.ABC):

    def build(self, *args: typing.Any, **kwargs: typing.Any) -> None:
        """Build the scan object using the :func:`build_scan` function.

        Normally users would build their scan object by overriding the :func:`build_scan` function.
        In specific cases where this function might be overridden, do not forget to call super.build().

        :param args: Positional arguments forwarded to the superclass
        :param kwargs: Keyword arguments forwarded to the superclass
        """

        # Call super and forward arguments, for compatibility with other libraries
        # noinspection PyArgumentList
        super(DaxScan, self).build(*args, **kwargs)

        # Collection of scannables
        self._scannables = collections.OrderedDict()  # type: typing.Dict[str, typing.Any]

        # The scheduler object
        self._scan_scheduler = self.get_device('scheduler')

        # Build this scan (no args or kwargs available)
        self.logger.debug('Building scan')
        self.__in_build = True
        self.build_scan()
        # TODO: add standard options (e.g. continuous looping?)
        self.__in_build = False

    @abc.abstractmethod
    def build_scan(self) -> None:
        """Users should override this method to build their scan.

        To build the scan, use the :func:`add_scan` function.
        Additionally, users can also add normal arguments using the standard ARTIQ functions.
        """
        pass

    def add_scan(self, key: str, scannable: Scannable,
                 group: typing.Optional[str] = None, tooltip: typing.Optional[str] = None) -> None:
        """Register a scannable.

        Scannables are normal ARTIQ `Scannable` objects and will appear in the user interface.

        :param key: Key of the scan
        :param scannable: An ARTIQ `Scannable` object
        :param group: The argument group name
        :param tooltip: The shown tooltip
        """

        # Verify this function was called in the build_scan() function
        if not self.__in_build:
            raise TypeError('add_scan() can only be called in the build_scan() method')

        # Verify type of the given scannable
        if not isinstance(scannable, Scannable):
            raise TypeError('The given processor must be a scannable')

        # Add argument to the list of scannables
        self._scannables[key] = self.get_argument(key, scannable, group=group, tooltip=tooltip)

    def add_result(self):
        pass  # TODO

    """Run functions"""

    @host_only
    def run(self) -> None:
        """Entry point of the experiment implemented by the scan class.

        Normally users do not have to override this method.
        Users implement the :func:`run_point` function instead.
        """

        # Check if build() was called
        assert hasattr(self, '_scannables'), 'DaxScan.build() was not called'

        # Make the multi-scan manager and construct the list with points
        self._scan_points = list(MultiScanManager(*self._scannables.items())) if self._scannables else []
        self.update_kernel_invariants('_scan_points')
        self.logger.debug('Prepared {:d} scan point(s) with {:d} scan parameter(s)'.
                          format(len(self._scan_points), len(self._scannables)))

        if not self._scan_points:
            # There are no scan points
            self.logger.warning('No scan points found, aborting experiment')
            return

        try:
            # One time host setup
            self.logger.debug('Performing host setup')
            self.host_setup()

            # Run the scan
            if _is_kernel(self.run_point):
                self.logger.debug('Running scan on core device')
                self._run_scan_in_kernel()
            else:
                self.logger.debug('Running scan on host')
                self._run_scan()

        except TerminationRequested:
            # Scan was terminated
            self.logger.info('Scan was terminated by user request')

        finally:
            # One time host cleanup
            self.logger.debug('Performing host cleanup')
            self.host_cleanup()

    @kernel
    def _run_scan_in_kernel(self):
        """Run scan on the core device."""
        self._run_scan()

    @portable
    def _run_scan(self):
        """Portable run scan function."""
        try:
            for p in self._scan_points:
                # Check for pause condition
                if self._scan_scheduler.check_pause():
                    self._pause_scan()

                # Run for one point
                self.run_point(p)

        finally:
            # Perform device cleanup
            self.device_cleanup()

    def _pause_scan(self):
        """Pause the scan."""

        # Disconnect from the core device
        self.core.comm.close()
        # Pause
        self.scheduler.pause()  # Can raise TerminationRequested exception
        # Setup host again before resuming scan
        self.host_setup()

    def host_setup(self) -> None:
        """1. Preparation on the host, called once at entry and after a pause."""
        pass

    @abc.abstractmethod
    def run_point(self, point):
        """2. Code to run for a single point, called as many times as there are points.

        :param point: Point object containing the current scan parameters
        """
        pass

    @portable
    def device_cleanup(self):
        """3. Cleanup on the core device, called once."""
        pass

    def host_cleanup(self) -> None:
        """4. Cleanup on the host, called once."""
        pass


def _is_kernel(func: typing.Any) -> bool:
    """Helper function to detect if a function is a kernel or not.

    :param func: The function of interest
    :return: True if the given function is a kernel
    """
    meta = getattr(func, 'artiq_embedded', None)
    return False if meta is None else (meta.core_name is not None and not meta.portable)
