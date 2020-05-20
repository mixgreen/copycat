import abc
import typing
import collections
import re
import numpy as np

import dax.base.dax
from dax.experiment import *

__all__ = ['DaxScan']

_KEY_RE = re.compile(r'\w+')
"""Regex for matching valid keys."""


def _is_valid_key(key: str) -> bool:
    """Return true if the given key is valid."""
    assert isinstance(key, str), 'The given key should be a string'
    return bool(_KEY_RE.fullmatch(key))


def _is_kernel(func: typing.Any) -> bool:
    """Helper function to detect if a function is a kernel or not.

    :param func: The function of interest
    :return: True if the given function is a kernel
    """
    meta = getattr(func, 'artiq_embedded', None)
    return False if meta is None else (meta.core_name is not None and not meta.portable)


class DaxScan(dax.base.dax.DaxBase, abc.ABC):
    """Scanning class for standardized scanning functionality.

    Users can inherit from this class to implement their scanning experiments.
    """

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
        self.__in_build = False

    @abc.abstractmethod
    def build_scan(self) -> None:
        """Users should override this method to build their scan.

        To build the scan, use the :func:`add_scan` function.
        Additionally, users can also add normal arguments using the standard ARTIQ functions.
        """
        pass

    def add_scan(self, key: str, name: str, scannable: Scannable,
                 group: typing.Optional[str] = None, tooltip: typing.Optional[str] = None) -> None:
        """Register a scannable.

        Scannables are normal ARTIQ `Scannable` objects and will appear in the user interface.

        :param key: Unique key of the scan, used to obtain the value later
        :param name: The name of the argument
        :param scannable: An ARTIQ `Scannable` object
        :param group: The argument group name
        :param tooltip: The shown tooltip
        """

        # Verify this function was called in the build_scan() function
        if not self.__in_build:
            raise TypeError('add_scan() can only be called in the build_scan() method')

        # Verify type of the given scannable
        if not isinstance(scannable, Scannable):
            raise TypeError('The given processor is not a scannable')

        # Verify the key is valid and not in use
        if not _is_valid_key(key):
            raise ValueError('Provided key "{:s}" is not valid'.format(key))
        if key in self._scannables:
            raise LookupError('Provided key "{:s}" was already in use'.format(key))

        # Add argument to the list of scannables
        self._scannables[key] = self.get_argument(name, scannable, group=group, tooltip=tooltip)

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

        # Index of current scan point
        self._scan_index = np.int32(0)

        try:
            while self._scan_index < len(self._scan_points):
                while self._scan_scheduler.check_pause():
                    # Pause the scan
                    self._scan_scheduler.pause()

                try:
                    # Coming from a host context, perform host setup
                    self.logger.debug('Performing host setup')
                    self.host_setup()

                    # Run the scan
                    if _is_kernel(self.run_point):
                        self.logger.debug('Running scan on core device')
                        self._run_scan_in_kernel()
                    else:
                        self.logger.debug('Running scan on host')
                        self._run_scan()

                finally:
                    # One time host cleanup
                    self.logger.debug('Performing host cleanup')
                    self.host_cleanup()

        except TerminationRequested:
            # Scan was terminated
            self.logger.info('Scan was terminated by user request')

        else:
            # Call the host exit code
            self.logger.debug('Performing host exit procedure')
            self.host_exit()

    @kernel
    def _run_scan_in_kernel(self):
        """Run scan on the core device."""
        self._run_scan()

    @portable
    def _run_scan(self):
        """Portable run scan function."""
        try:
            while self._scan_index < len(self._scan_points):
                # Check for pause condition
                if self._scan_scheduler.check_pause():
                    break  # Break to exit the run scan function

                # Run for one point
                self.run_point(self._scan_points[self._scan_index])
                # Increment index
                self._scan_index += 1

        finally:
            # Perform device cleanup
            self.device_cleanup()

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
        """3. Cleanup on the core device, called once after scanning and before a pause."""
        pass

    def host_cleanup(self) -> None:
        """4. Cleanup on the host, called once after scanning and before a pause."""
        pass

    def host_exit(self) -> None:
        """5. Exit code on the host if the scan finished successfully."""
        pass
