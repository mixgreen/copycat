import abc
import typing
import collections
import re
import numpy as np

from artiq.experiment import *

import dax.base.dax

__all__ = ['DaxScan']

_KEY_RE = re.compile(r'[a-zA-Z_]\w*')
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
    The first step is to build the scan by overriding the :func:`build_scan` function.
    Use the :func:`add_scan` function to add normal ARTIQ scannables to this scan object.
    Other ARTIQ functions are available to obtain other arguments.

    The scan class implements a :func:`run` function that controls the overall scanning flow.
    The :func:`prepare` and :func:`analyze` functions are not implemented by the scan class, but users
    are free to provide their own implementations.

    The following functions can be overridden to define scanning behavior:

    1. :func:`host_setup`
    2. :func:`device_setup`
    3. :func:`run_point` (must be implemented)
    4. :func:`device_cleanup`
    5. :func:`host_cleanup`

    Finally, the :func:`host_exit` function can be overridden to implement any functionality
    executed just before leaving the :func:`run` function.

    To exit a scan early, call the :func:`stop_scan` function.

    By default, an infinite scan option is added to the arguments which allows infinite looping
    over the available points.
    An infinite scan can be stopped by using the :func:`stop_scan` function or by using
    the "Terminate experiment" button in the dashboard.
    It is possible to disable the infinite scan argument for an experiment by setting the
    :attr:`INFINITE_SCAN_ARGUMENT` class attribute to `False`.
    The default setting of the infinite scan argument can be modified by setting the
    :attr:`INFINITE_SCAN_DEFAULT` class attribute.

    In case scanning is performed in a kernel, users are responsible for setting
    up the right devices to actually run a kernel.
    """

    INFINITE_SCAN_ARGUMENT = True  # type: bool
    """Flag to enable the infinite scan argument."""
    INFINITE_SCAN_DEFAULT = False  # type: bool
    """Default setting of the infinite scan argument (if enabled)."""

    SCAN_ARCHIVE_KEY_FORMAT = 'scan.{key:s}'  # type: str
    """Dataset key format for archiving independent scans."""

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
        self._scan_scannables = collections.OrderedDict()  # type: typing.Dict[str, typing.Iterable[typing.Any]]

        # The scheduler object
        self._scan_scheduler = self.get_device('scheduler')

        # Build this scan (no args or kwargs available)
        self.logger.debug('Building scan')
        self.__in_build = True
        self.build_scan()
        self.__in_build = False

        if self.INFINITE_SCAN_ARGUMENT:
            # Add an argument for infinite scan
            self._scan_infinite = self.get_argument('Infinite scan', BooleanValue(self.INFINITE_SCAN_DEFAULT),
                                                    group='DAX.scan',
                                                    tooltip='Loop infinitely over the scan points')  # type: bool
        else:
            # If infinite scan is disabled, the value is always False
            self._scan_infinite = False

        # Update kernel invariants
        self.update_kernel_invariants('_scan_scheduler', '_scan_infinite')

    @abc.abstractmethod
    def build_scan(self) -> None:
        """Users should override this method to build their scan.

        To build the scan, use the :func:`add_scan` function.
        Additionally, users can also add normal arguments using the standard ARTIQ functions.
        """
        pass

    @property
    def is_infinite_scan(self) -> bool:
        """True if the scan was set to be an infinite scan."""
        if hasattr(self, '_scan_infinite'):
            return self._scan_infinite
        else:
            raise AttributeError('is_scan_infinite can only be obtained after build() was called')

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
        if key in self._scan_scannables:
            raise LookupError('Provided key "{:s}" was already in use'.format(key))

        # Add argument to the list of scannables
        self._scan_scannables[key] = self.get_argument(name, scannable, group=group, tooltip=tooltip)

    @host_only
    def get_scan_points(self) -> typing.Dict[str, typing.List[typing.Any]]:
        """Get the scan points (including product) for analysis.

        A list of values is returned on a per-key basis.
        The values are returned in the same sequence as was provided to the actual run,
        including the product of all scannables.

        To get the values without applying the product, see :func:`get_scannables`.

        :return: A dict containing all the scan points on a per-key basis
        """
        if hasattr(self, '_scan_points'):
            return {key: [getattr(point, key) for point in self._scan_points] for key in self._scan_scannables}
        else:
            raise AttributeError('Scan points can only be obtained after run() was called')

    @host_only
    def get_scannables(self) -> typing.Dict[str, typing.List[typing.Any]]:
        """Get the scan points without product.

        For every key, a list of scan values is returned.
        These values are the individual values for each key, without applying the product.

        To get the values including the product, see :func:`get_scan_points`.

        :return: A dict containing the individual scan values on a per-key basis
        """
        return {key: list(scannable) for key, scannable in self._scan_scannables.items()}

    @host_only
    def _make_scan_points(self) -> typing.List[typing.Any]:
        """Make and return the scan points

        :return: A list of scan points
        """
        if self._scan_scannables:
            # Make scan points using the ARTIQ multi scan manager
            scan_points = list(MultiScanManager(*self._scan_scannables.items()))  # type: ignore

            # For every scan point, mark the parameters as kernel invariant
            keys = set(self._scan_scannables)
            for point in scan_points:
                setattr(point, 'kernel_invariants', keys)

            # Return the scan points
            return scan_points

        else:
            # Return an empty list
            return []

    """Run functions"""

    @host_only
    def run(self) -> None:
        """Entry point of the experiment implemented by the scan class.

        Normally users do not have to override this method.
        Users implement the :func:`run_point` function instead.
        """

        # Check if build() was called
        assert hasattr(self, '_scan_scannables'), 'DaxScan.build() was not called'

        # Make the scan points
        self._scan_points = self._make_scan_points()
        self.update_kernel_invariants('_scan_points')
        self.logger.debug('Prepared {:d} scan point(s) with {:d} scan parameter(s)'.
                          format(len(self._scan_points), len(self._scan_scannables)))

        if not self._scan_points:
            # There are no scan points
            self.logger.warning('No scan points found, aborting experiment')
            return

        for key in self._scan_scannables:
            # Archive product of all scan points (separate dataset for every key)
            self.set_dataset(key, [getattr(point, key) for point in self._scan_points], archive=True)
        if len(self._scan_scannables) > 1:
            # Archive values of each independent scan if we have more than 1 scan
            for key, scannable in self._scan_scannables.items():
                self.set_dataset(self.SCAN_ARCHIVE_KEY_FORMAT.format(key=key), [e for e in scannable], archive=True)

        # Index of current scan point
        self._scan_index = np.int32(0)

        try:
            while self._scan_index < len(self._scan_points):
                while self._scan_scheduler.check_pause():
                    # Pause the scan
                    self.logger.debug('Pausing scan')
                    self._scan_scheduler.pause()
                    self.logger.debug('Resuming scan')

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
    def _run_scan_in_kernel(self):  # type: () -> None
        """Run scan on the core device."""
        self._run_scan()

    @portable
    def _run_scan(self):  # type: () -> None
        """Portable run scan function."""
        try:
            # Perform device setup
            self.device_setup()

            while self._scan_index < len(self._scan_points):
                # Check for pause condition
                if self._scan_scheduler.check_pause():
                    break  # Break to exit the run scan function

                # Run for one point
                self.run_point(self._scan_points[self._scan_index])

                # Increment index
                self._scan_index += np.int32(1)

                # Handle infinite scan
                if self._scan_infinite and self._scan_index == len(self._scan_points):
                    self._scan_index = np.int32(0)

        finally:
            # Perform device cleanup
            self.device_cleanup()

    @portable
    def stop_scan(self):  # type: () -> None
        """Stop the scan after the current point.

        This function can only be called from the :func:`run_point` function.
        """

        # Stop the scan by moving the index to the end (+1 to differentiate with infinite scan)
        self._scan_index = len(self._scan_points) + np.int32(1)

    """Functions to be implemented by the user"""

    def host_setup(self) -> None:
        """1. Preparation on the host, called once at entry and after a pause."""
        pass

    @portable
    def device_setup(self):  # type: () -> None
        """2. Preparation on the core device, called once at entry and after a pause.

        Can for example be used to reset the core.
        """
        pass

    @abc.abstractmethod
    def run_point(self, point):  # type: (typing.Any) -> None
        """3. Code to run for a single point, called as many times as there are points.

        :param point: Point object containing the current scan parameters
        """
        pass

    @portable
    def device_cleanup(self):  # type: () -> None
        """4. Cleanup on the core device, called once after scanning and before a pause.

        In case the device cleanup function is a kernel, it is good to add a `self.core.break_realtime()`
        at the start of this function to make sure operations can execute in case of an
        underflow exception.
        """
        pass

    def host_cleanup(self) -> None:
        """5. Cleanup on the host, called once after scanning and before a pause."""
        pass

    def host_exit(self) -> None:
        """6. Exit code on the host if the scan finished successfully."""
        pass
