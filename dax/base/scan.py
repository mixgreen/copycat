import abc
import typing
import itertools
import re
import numpy as np
import collections
import os
import h5py  # type: ignore

from artiq.experiment import *
import artiq.coredevice.core  # type: ignore

import dax.base.system
import dax.util.artiq

__all__ = ['DaxScan', 'DaxScanReader']

_KEY_RE: typing.Pattern[str] = re.compile(r'[a-zA-Z_]\w*')
"""Regex for matching valid keys."""


def _is_valid_key(key: str) -> bool:
    """Return true if the given key is valid."""
    assert isinstance(key, str), 'The given key should be a string'
    return bool(_KEY_RE.fullmatch(key))


class _ScanProductGenerator:
    """Generator class for a cartesian product of scans.

    This class is inspired by the ARTIQ MultiScanManager class.
    """

    class _ScanItem:
        def __init__(self, **kwargs: typing.Any):
            # Mark all attributes as kernel invariant
            self.kernel_invariants: typing.Set[str] = set(kwargs)

        def __repr__(self) -> str:
            """Return a string representation of this object."""
            attributes: str = ', '.join(f'{k}={getattr(self, k)}' for k in self.kernel_invariants)
            return f'{self.__class__.__name__}: {attributes}'

    class ScanPoint(_ScanItem):
        def __init__(self, **kwargs: typing.Any):
            # Call super
            super(_ScanProductGenerator.ScanPoint, self).__init__(**kwargs)
            # Set the attributes of this object
            for k, v in kwargs.items():
                setattr(self, k, v)

    class ScanIndex(_ScanItem):
        def __init__(self, **kwargs: int):
            # Call super
            super(_ScanProductGenerator.ScanIndex, self).__init__(**kwargs)
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
        self._enable_index: bool = enable_index

    def _point_generator(self) -> typing.Iterator[ScanPoint]:
        """Returns a generator for scan points."""
        for values in itertools.product(*self._scans):
            yield self.ScanPoint(**{k: v for k, v in zip(self._keys, values)})

    def _index_generator(self) -> typing.Iterator[ScanIndex]:
        """Returns a generator for scan indices."""
        if self._enable_index:
            for indices in itertools.product(*(range(len(s)) for s in self._scans)):
                # Yield a scan index object for every set of indices
                yield self.ScanIndex(**{k: v for k, v in zip(self._keys, indices)})
        else:
            # Create one empty scan index
            si = self.ScanIndex()
            for _ in range(np.prod([len(s) for s in self._scans])):  # type: ignore[attr-defined]
                # Yield the empty scan index object for all
                yield si

    def __iter__(self) -> typing.Iterator[typing.Tuple[ScanPoint, ScanIndex]]:
        """Returns a generator that returns tuples of a scan point and a scan index."""
        # Storing a list of tuples instead of two list hopefully results in better data locality
        return zip(self._point_generator(), self._index_generator())


class DaxScan(dax.base.system.DaxBase, abc.ABC):
    """Scanning class for standardized scanning functionality.

    Users can inherit from this class to implement their scanning experiments.
    The first step is to build the scan by overriding the :func:`build_scan` function.
    Use the :func:`add_scan` function to add normal ARTIQ scannables to this scan object.
    Static scans can be added using the :func:`add_static_scan` function.
    Regular ARTIQ functions are available to obtain other arguments.

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

    Finally, the :func:`host_enter` and :func:`host_exit` functions can be overridden to implement any
    functionality executed once at the start of or just before leaving the :func:`run` function.

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
    The :attr:`ENABLE_SCAN_INDEX` attribute can be used to configure this behavior (default: `True`).
    When the index is disabled, the passed `index` argument will be empty.

    In case scanning is performed in a kernel, users are responsible for setting
    up the right devices to actually run a kernel.

    Arguments passed to the :func:`build` function are passed to the super class for
    compatibility with other libraries and abstraction layers.
    It is still possible to pass arguments to the :func:`build_scan` function by using special
    keyword arguments of the :func:`build` function of which the keywords are defined
    in the :attr:`SCAN_ARGS_KEY` and :attr:`SCAN_KWARGS_KEY` attributes.
    """

    INFINITE_SCAN_ARGUMENT: bool = True
    """Flag to enable the infinite scan argument."""
    INFINITE_SCAN_DEFAULT: bool = False
    """Default setting of the infinite scan."""

    ENABLE_SCAN_INDEX: bool = True
    """Flag to enable the index argument in the run_point() function."""

    SCAN_GROUP: str = 'scan'
    """The group name for archiving data."""
    SCAN_KEY_FORMAT: str = SCAN_GROUP + '/{key}'
    """Dataset key format for archiving independent scans."""
    SCAN_PRODUCT_GROUP: str = 'product'
    """The sub-group name for archiving product data."""
    SCAN_PRODUCT_KEY_FORMAT: str = f'{SCAN_GROUP}/{SCAN_PRODUCT_GROUP}/{{key}}'
    """Dataset key format for archiving scan products."""

    SCAN_ARGS_KEY: str = 'scan_args'
    """:func:`build` keyword argument for positional arguments passed to :func:`build_scan`."""
    SCAN_KWARGS_KEY: str = 'scan_kwargs'
    """:func:`build` keyword argument for keyword arguments passed to :func:`build_scan`."""

    def build(self, *args: typing.Any, **kwargs: typing.Any) -> None:
        """Build the scan object using the :func:`build_scan` function.

        Normally users would build their scan object by overriding the :func:`build_scan` function.
        In specific cases where this function might be overridden, do not forget to call `super.build()`.

        :param args: Positional arguments forwarded to the superclass
        :param kwargs: Keyword arguments forwarded to the superclass (includes args and kwargs for :func:`build_scan`)
        """

        assert isinstance(self.INFINITE_SCAN_ARGUMENT, bool), 'Infinite scan argument flag must be of type bool'
        assert isinstance(self.INFINITE_SCAN_DEFAULT, bool), 'Infinite scan default flag must be of type bool'
        assert isinstance(self.ENABLE_SCAN_INDEX, bool), 'Enable scan index flag must be of type bool'
        assert isinstance(self.SCAN_ARGS_KEY, str), 'Scan args keyword must be of type str'
        assert isinstance(self.SCAN_KWARGS_KEY, str), 'Scan kwargs keyword must be of type str'

        # Obtain the scan args and kwargs
        scan_args: typing.Sequence[typing.Any] = kwargs.pop(self.SCAN_ARGS_KEY, ())
        scan_kwargs: typing.Dict[str, typing.Any] = kwargs.pop(self.SCAN_KWARGS_KEY, {})
        assert isinstance(scan_args, collections.abc.Sequence), 'Scan args must be a sequence'
        assert isinstance(scan_kwargs, dict), 'Scan kwargs must be a dict'
        assert all(isinstance(k, str) for k in scan_kwargs), 'All scan kwarg keys must be of type str'

        # Check if host_*() functions are all non-kernel functions
        if any(dax.util.artiq.is_kernel(f) for f in [self.host_enter, self.host_setup,
                                                     self.host_cleanup, self.host_exit]):
            raise TypeError('host_*() functions can not be kernels')

        # Call super and forward arguments, for compatibility with other libraries
        # noinspection PyArgumentList
        super(DaxScan, self).build(*args, **kwargs)

        # Collection of scannables
        self._dax_scan_scannables: collections.OrderedDict[str, typing.Iterable[typing.Any]] = collections.OrderedDict()

        # The scheduler object
        self._dax_scan_scheduler: typing.Any = self.get_device('scheduler')

        # Build this scan (no args or kwargs available)
        self.logger.debug('Building scan')
        self.__in_build: bool = True
        # noinspection PyArgumentList
        self.build_scan(*scan_args, **scan_kwargs)  # type: ignore[call-arg]
        self.__in_build = False

        # Confirm we have a core attribute
        if not hasattr(self, 'core'):
            raise AttributeError('DaxScan could not find a "core" attribute')
        self.core: artiq.coredevice.core.Core  # Type annotation for core attribute

        if self.INFINITE_SCAN_ARGUMENT:
            # Add an argument for infinite scan
            self._dax_scan_infinite: bool = self.get_argument('Infinite scan', BooleanValue(self.INFINITE_SCAN_DEFAULT),
                                                              group='DAX.scan',
                                                              tooltip='Loop infinitely over the scan points')
        else:
            # If infinite scan argument is disabled, the value is always the default one
            self._dax_scan_infinite = self.INFINITE_SCAN_DEFAULT

        # Update kernel invariants
        self.update_kernel_invariants('_dax_scan_scheduler', '_dax_scan_infinite')

    @abc.abstractmethod
    def build_scan(self) -> None:
        """Users should override this method to build their scan.

        To build the scan, use the :func:`add_scan` and :func:`add_static_scan` functions.
        Additionally, users can also add normal arguments using the standard ARTIQ functions.

        It is possible to pass arguments from the constructor to this function using the
        `scan_args` and `scan_kwargs` keyword arguments.
        """
        pass

    @property
    def is_infinite_scan(self) -> bool:
        """True if the scan was set to be an infinite scan."""
        if hasattr(self, '_dax_scan_infinite'):
            return self._dax_scan_infinite
        else:
            raise AttributeError('is_scan_infinite can only be obtained after build() was called')

    def add_scan(self, key: str, name: str, scannable: Scannable, *,
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
            raise ValueError(f'Provided key "{key}" is not valid')
        if key in self._dax_scan_scannables:
            raise LookupError(f'Provided key "{key}" is already in use')

        # Add argument to the list of scannables
        self._dax_scan_scannables[key] = self.get_argument(name, scannable, group=group, tooltip=tooltip)

    def add_static_scan(self, key: str, points: typing.Sequence[typing.Any]) -> None:
        """Register a static scan.

        A static scan is handled the same as a regular scan (see :func:`add_scan`).
        The difference is that a static scan has predefined points and
        therefore does not appear in the user interface.

        :param key: Unique key of the scan, used to obtain the value later
        :param points: A sequence with points
        """
        assert isinstance(key, str), 'Key must be of type str'

        # Verify this function was called in the build_scan() function
        if not self.__in_build:
            raise TypeError('add_scan() can only be called in the build_scan() method')

        # Verify type of the points
        if not isinstance(points, collections.abc.Sequence):
            raise TypeError('Points must be a sequence')
        if isinstance(points, np.ndarray):
            if not any(np.issubdtype(points.dtype, t) for t in [np.integer, np.floating, np.bool_, np.character]):
                raise TypeError('The NumPy point type is not supported')
            if points.ndim != 1:
                raise TypeError('Only NumPy arrays with one dimension are supported')
        else:
            if not all(isinstance(e, type(points[0])) for e in points):
                raise TypeError('The point types must be homogeneous')
            if len(points) > 0 and not isinstance(points[0], (int, float, bool, str)):
                raise TypeError('The point type is not supported')

        # Verify the key is valid and not in use
        if not _is_valid_key(key):
            raise ValueError(f'Provided key "{key}" is not valid')
        if key in self._dax_scan_scannables:
            raise LookupError(f'Provided key "{key}" is already in use')

        # Add points to the list of scannables
        self._dax_scan_scannables[key] = points

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
            self._dax_scan_scannables.move_to_end(k)

    @host_only
    def get_scan_points(self) -> typing.Dict[str, typing.List[typing.Any]]:
        """Get the cartesian product of scan points for analysis.

        A list of values is returned on a per-key basis.
        The values are returned in the same sequence as was provided to the actual run,
        as the cartesian product of all scannables.

        To get the values without applying the product, see :func:`get_scannables`.

        This function can only be used after the :func:`run` function was called
        which normally means it is not available during the build and prepare phase.

        :return: A dict containing all the scan points on a per-key basis
        """
        if hasattr(self, '_dax_scan_elements'):
            return {key: [getattr(point, key) for point, _ in self._dax_scan_elements]
                    for key in self._dax_scan_scannables}
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
        return {key: list(scannable) for key, scannable in self._dax_scan_scannables.items()}

    """Run functions"""

    @host_only
    def run(self) -> None:
        """Entry point of the experiment implemented by the scan class.

        Normally users do not have to override this method.
        Once-executed entry code can use the :func:`host_enter` function instead.
        """

        # Check if build() was called
        assert hasattr(self, '_dax_scan_scannables'), 'DaxScan.build() was not called'

        # Make the scan elements
        if self._dax_scan_scannables:
            self._dax_scan_elements: typing.List[typing.Any] = list(
                _ScanProductGenerator(*self._dax_scan_scannables.items(),  # type: ignore[arg-type]
                                      enable_index=self.ENABLE_SCAN_INDEX))
        else:
            self._dax_scan_elements = []
        self.update_kernel_invariants('_dax_scan_elements')
        self.logger.debug(f'Prepared {len(self._dax_scan_elements)} scan point(s) '
                          f'with {len(self._dax_scan_scannables)} scan parameter(s)')

        if not self._dax_scan_elements:
            # There are no scan points
            self.logger.warning('No scan points found, aborting experiment')
            return

        # Reporting infinite scan flag
        self.logger.debug(f'Infinite scan: {self.is_infinite_scan}')

        for key, scannable in self._dax_scan_scannables.items():
            # Archive values of independent scan
            self.set_dataset(self.SCAN_KEY_FORMAT.format(key=key), [e for e in scannable], archive=True)
            # Archive cartesian product of scan point (separate dataset for every key)
            self.set_dataset(self.SCAN_PRODUCT_KEY_FORMAT.format(key=key),
                             [getattr(point, key) for point, _ in self._dax_scan_elements], archive=True)

        # Index of current scan element
        self._dax_scan_index: np.int32 = np.int32(0)

        try:
            # Call the host enter code
            self.logger.debug('Performing host enter procedure')
            self.host_enter()

            while self._dax_scan_index < len(self._dax_scan_elements):
                while self._dax_scan_scheduler.check_pause():
                    # Pause the scan
                    self.logger.debug('Pausing scan')
                    self.core.comm.close()  # Close communications before pausing
                    self._dax_scan_scheduler.pause()  # Can raise a TerminationRequested exception
                    self.logger.debug('Resuming scan')

                try:
                    # Coming from a host context, perform host setup
                    self.logger.debug('Performing host setup')
                    self.host_setup()

                    # Run the scan
                    if dax.util.artiq.is_kernel(self.run_point):
                        self.logger.debug('Running scan on core device')
                        self._run_dax_scan_in_kernel()
                    else:
                        self.logger.debug('Running scan on host')
                        self._run_dax_scan()

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
    def _run_dax_scan_in_kernel(self):  # type: () -> None
        """Run scan on the core device."""
        self._run_dax_scan()

    @portable
    def _run_dax_scan(self):  # type: () -> None
        """Portable run scan function."""
        try:
            # Perform device setup
            self.device_setup()

            while self._dax_scan_index < len(self._dax_scan_elements):
                # Check for pause condition
                if self._dax_scan_scheduler.check_pause():
                    break  # Break to exit the run scan function

                # Run for one point
                point, index = self._dax_scan_elements[self._dax_scan_index]
                self.run_point(point, index)

                # Increment index
                self._dax_scan_index += np.int32(1)

                # Handle infinite scan
                if self._dax_scan_infinite and self._dax_scan_index == len(self._dax_scan_elements):
                    self._dax_scan_index = np.int32(0)

        finally:
            # Perform device cleanup
            self.device_cleanup()

    @portable
    def stop_scan(self):  # type: () -> None
        """Stop the scan after the current point.

        This function can only be called from the :func:`run_point` function.
        """

        # Stop the scan by moving the index to the end (+1 to differentiate with infinite scan)
        self._dax_scan_index = np.int32(len(self._dax_scan_elements) + 1)

    """Functions to be implemented by the user"""

    def host_enter(self) -> None:
        """0. Entry code on the host, called once."""
        pass

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
        """6. Exit code on the host, called if the scan finished without exceptions."""
        pass


class DaxScanReader:
    """Reader class to retrieve scan data from and HDF5 archive or a live :class:`DaxScan` object.

    This class will read an HDF5 file, extract scan data, and expose it through attributes.
    It is also possible to provide a :class:`DaxScan` object as source and the same data
    will be exposed, though :class:`DaxScan` also provides methods to obtain data directly.

    :attr:`keys` is a list of keys for which scan data is available.

    :attr:`scannables` is a dict which for each key contains the list of values.
    These values are the individual values for each key, without applying the product.

    :attr:`scan_points` is a dict with for each key contains the list of scan points.
    The values are returned in the same sequence as was provided to the actual run,
    as the cartesian product of all scannables.
    """

    def __init__(self, source: typing.Union[DaxScan, str, h5py.File]):
        """Create a new DAX scan reader object.

        :param source: The source of the scan data
        """

        # Input conversion
        if isinstance(source, str):
            # Open HDF5 file
            source = h5py.File(os.path.expanduser(source), mode='r')

        if isinstance(source, DaxScan):
            # Get data from scan object
            self.scannables: typing.Dict[str, typing.List[typing.Any]] = source.get_scannables()
            self.scan_points: typing.Dict[str, typing.List[typing.Any]] = source.get_scan_points()
            self.keys: typing.List[str] = list(self.scannables.keys())

        elif isinstance(source, h5py.File):
            # Verify format of HDF5 file
            group_name = 'datasets/' + DaxScan.SCAN_GROUP
            if group_name not in source:
                raise KeyError('The HDF5 file does not contain scanning data')

            # Get the group which contains all data
            group = source[group_name]

            # Read and convert data from HDF5 file
            self.keys = [k for k in group.keys() if k != DaxScan.SCAN_PRODUCT_GROUP]
            self.scannables = {k: group[k][()] for k in self.keys}
            group = group[DaxScan.SCAN_PRODUCT_GROUP]  # Switch to group that contains scan points
            self.scan_points = {k: group[k][()] for k in self.keys}

        else:
            raise TypeError('Unsupported source type')
