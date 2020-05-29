import abc
import typing
import itertools
import re
import numpy as np
from collections import OrderedDict

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


class ScanProductGenerator:
    """Generator class for a cartesian product of scans.

    This class is inspired by the ARTIQ MultiScanManager class.
    """

    # Iterator type
    __I_T = typing.Iterator[typing.Tuple['ScanProductGenerator.ScanPoint', 'ScanProductGenerator.ScanIndex']]

    class ScanItem:
        def __init__(self, **kwargs: typing.Any):
            # Mark all attributes as kernel invariant
            self.kernel_invariants = set(kwargs)

        def __repr__(self) -> str:
            """Return a string representation of this object."""
            attributes = ', '.join('{:s}={}'.format(k, getattr(self, k)) for k in self.kernel_invariants)
            return self.__class__.__name__.format(attributes)

    class ScanPoint(ScanItem):
        def __init__(self, **kwargs: typing.Any):
            # Call super
            super(ScanProductGenerator.ScanPoint, self).__init__(**kwargs)
            # Set the attributes of this object
            for k, v in kwargs.items():
                setattr(self, k, v)

    class ScanIndex(ScanItem):
        def __init__(self, **kwargs: int):
            # Call super
            super(ScanProductGenerator.ScanIndex, self).__init__(**kwargs)
            # Set the attributes of this object
            for k, v in kwargs.items():
                setattr(self, k, np.int32(v))

    def __init__(self, *scans: typing.Tuple[str, typing.Iterable[typing.Any]],
                 enable_index: bool = True):
        """Create a new scan product generator.

        The `enable_index` parameter dan be used to disable index objects,
        potentially reducing the memory footprint of the scan.

        :param scans: A list of tuples with the key and values of the scan
        :param enable_index: If false, empty index objects are returned
        """
        assert isinstance(enable_index, bool), 'The enable index flag must be of type bool'

        # Unpack scan tuples
        self._keys, self._scans = tuple(zip(*scans))

        # Store enable index flag
        self._enable_index = enable_index

    def _point_generator(self) -> typing.Iterator['ScanProductGenerator.ScanPoint']:
        """Returns a generator for scan points."""
        for values in itertools.product(*self._scans):
            yield self.ScanPoint(**{k: v for k, v in zip(self._keys, values)})

    def _index_generator(self) -> typing.Iterator['ScanProductGenerator.ScanIndex']:
        """Returns a generator for scan indices."""
        if self._enable_index:
            for indices in itertools.product(*(range(len(s)) for s in self._scans)):
                # Yield a scan index object for every set of indices
                yield self.ScanIndex(**{k: v for k, v in zip(self._keys, indices)})
        else:
            # Create one empty scan index
            si = self.ScanIndex()
            for _ in range(np.prod([len(s) for s in self._scans])):  # type: ignore
                # Yield the empty scan index object for all
                yield si

    def __iter__(self) -> __I_T:
        """Returns a generator that returns tuples of a scan point and a scan index."""
        # Storing a list of tuples instead of two list hopefully results in better data locality
        return zip(self._point_generator(), self._index_generator())


class DaxScan(dax.base.dax.DaxBase, abc.ABC):
    """Scanning class for standardized scanning functionality.

    Users can inherit from this class to implement their scanning experiments.
    The first step is to build the scan by overriding the :func:`build_scan` function.
    Use the :func:`add_scan` function to add normal ARTIQ scannables to this scan object.
    Other ARTIQ functions are available to obtain other arguments.

    Adding multiple scans results automatically in a multi-dimensional scan by scanning over
    all value combinations in the cartesian product of the scans.
    The first scan added represents the dimension which is only scanned once.
    All next scans are repeatedly performed to form the cartesian product.

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

    The :func:`run_point` function has access to a point and an index argument.
    Users can disable the index argument to reduce the memory footprint of the experiment.
    The :attr:`ENABLE_INDEX` attribute can be used to configure this behavior (default: `True`).
    When the index is disabled, the passed `index` argument will be empty.

    In case scanning is performed in a kernel, users are responsible for setting
    up the right devices to actually run a kernel.
    """

    INFINITE_SCAN_ARGUMENT = True  # type: bool
    """Flag to enable the infinite scan argument."""
    INFINITE_SCAN_DEFAULT = False  # type: bool
    """Default setting of the infinite scan argument (if enabled)."""

    ENABLE_INDEX = True  # type: bool
    """Flag to enable the index argument in the run_point() function."""

    SCAN_ARCHIVE_KEY_FORMAT = '_scan.{key:s}'  # type: str
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
        self._scan_scannables = OrderedDict()  # type: OrderedDict[str, typing.Iterable[typing.Any]]

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

        The first scan added represents the dimension which is only scanned once.
        All next scans are repeatedly performed to form the cartesian product.
        Hence, the last added scan will be repeated the most times.

        Note that scans can be reordered using the :func:`set_scan_order` function.
        The order in the ARTIQ dashboard will not change, but the product will be generated differently.

        Scannables are normal ARTIQ `Scannable` objects and will appear in the user interface.

        This function can only be called in the :func:`build_scan` function.

        :param key: Unique key of the scan, used to obtain the value later
        :param name: The name of the argument
        :param scannable: An ARTIQ `Scannable` object
        :param group: The argument group name
        :param tooltip: The shown tooltip
        """
        assert isinstance(key, str), 'Key must be of type str'
        assert isinstance(name, str), 'Name must be of type str'
        assert isinstance(group, str) or group is None, 'Group must be of type str or None'
        assert isinstance(tooltip, str) or tooltip is None, 'Tooltip must be of type str or None'

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
    def set_scan_order(self, *keys: str) -> None:
        """Set the scan order given a sequence of keys.

        The given sequence of scan keys will determine the new order of the scans.
        In the list of scans, the given keys will be moved to the end of the list in the given order.
        Keys of scans that are not in the new scan order will remain in their relative position.
        For more information about the order of scans and the cartesian product, see :func:`add_scan`.

        This function can only be called in the :func:`build_scan` function.

        :param keys: The scan keys in the desired order
        """

        # Verify this function was called in the build_scan() function
        if not self.__in_build:
            raise TypeError('set_scan_order() can only be called in the build_scan() method')

        for k in keys:
            # Move key to the end of the list
            self._scan_scannables.move_to_end(k)

    @host_only
    def get_scan_points(self) -> typing.Dict[str, typing.List[typing.Any]]:
        """Get the cartesian product of scan points for analysis.

        A list of values is returned on a per-key basis.
        The values are returned in the same sequence as was provided to the actual run,
        as the cartesian product of all scannables.

        To get the values without applying the product, see :func:`get_scannables`.

        This function can only be used after the :func:`run` function was called.

        :return: A dict containing all the scan points on a per-key basis
        """
        if hasattr(self, '_scan_elements'):
            return {key: [getattr(point, key) for point, _ in self._scan_elements] for key in self._scan_scannables}
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

    """Run functions"""

    @host_only
    def run(self) -> None:
        """Entry point of the experiment implemented by the scan class.

        Normally users do not have to override this method.
        Users implement the :func:`run_point` function instead.
        """

        # Check if build() was called
        assert hasattr(self, '_scan_scannables'), 'DaxScan.build() was not called'

        # Make the scan elements
        if self._scan_scannables:
            self._scan_elements = list(ScanProductGenerator(*self._scan_scannables.items(),  # type: ignore
                                                            enable_index=self.ENABLE_INDEX))
        else:
            self._scan_elements = []
        self.update_kernel_invariants('_scan_elements')
        self.logger.debug('Prepared {:d} scan point(s) with {:d} scan parameter(s)'.
                          format(len(self._scan_elements), len(self._scan_scannables)))

        if not self._scan_elements:
            # There are no scan points
            self.logger.warning('No scan points found, aborting experiment')
            return

        for key in self._scan_scannables:
            # Archive cartesian product of all scan points (separate dataset for every key)
            self.set_dataset(key, [getattr(point, key) for point, _ in self._scan_elements], archive=True)
        # Archive values of each independent scan
        for key, scannable in self._scan_scannables.items():
            self.set_dataset(self.SCAN_ARCHIVE_KEY_FORMAT.format(key=key), [e for e in scannable], archive=True)

        # Index of current scan element
        self._scan_index = np.int32(0)

        try:
            while self._scan_index < len(self._scan_elements):
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

            while self._scan_index < len(self._scan_elements):
                # Check for pause condition
                if self._scan_scheduler.check_pause():
                    break  # Break to exit the run scan function

                # Run for one point
                point, index = self._scan_elements[self._scan_index]
                self.run_point(point, index)

                # Increment index
                self._scan_index += np.int32(1)

                # Handle infinite scan
                if self._scan_infinite and self._scan_index == len(self._scan_elements):
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
        self._scan_index = len(self._scan_elements) + np.int32(1)

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
    def run_point(self, point, index):  # type: (typing.Any, typing.Any) -> None
        """3. Code to run for a single point, called as many times as there are points.

        :param point: Point object containing the current scan parameter values
        :param index: Index object containing the current scan parameter indices
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
