import collections
import collections.abc
import typing
import abc
import numpy as np

import artiq.coredevice.edge_counter

from dax.experiment import *
from dax.interfaces.detection import DetectionInterface
from dax.util.ccb import get_ccb_tool, CcbTool, CcbToolBase
from dax.util.artiq import is_kernel
from dax.util.units import time_to_str

__all__ = ['PmtMonitor', 'MultiPmtMonitor']

# Workaround required for Python<3.9
if typing.TYPE_CHECKING:  # pragma: no cover
    _C_T = collections.OrderedDict[str, float]  # Count scales type
else:
    _C_T = collections.OrderedDict


class _PmtMonitorBase(DaxClient, Experiment, abc.ABC):
    """Base PMT monitor class."""

    APPLET_GROUP: typing.ClassVar[str]
    """Group of the applet."""
    DEFAULT_DATASET: typing.ClassVar[str]
    """Default dataset for output."""
    DEFAULT_BUFFER_SIZE: typing.ClassVar[int] = 1
    """Default buffer size in samples."""
    MAX_BUFFER_SIZE: typing.ClassVar[int] = 32
    """Maximum buffer size in samples."""
    DEFAULT_SLIDING_WINDOW_SIZE: typing.ClassVar[int] = 120
    """Default sliding window size in seconds (zero for no sliding window)."""
    DEFAULT_SUBSAMPLE_FACTOR: typing.ClassVar[int] = 0
    """Default subsampling factor (<=1 for no subsampling)."""
    DEFAULT_APPLET_UPDATE_DELAY: typing.ClassVar[float] = 0.1
    """Default applet update delay in seconds."""
    DATA_CLEANUP_THRESHOLD_MULTIPLIER: typing.ClassVar[int] = 4
    """Data cleanup will occur when the number of samples exceeds the window size multiplied by this value."""
    DATA_CLEANUP_PRESERVE_MULTIPLIER: typing.ClassVar[int] = 2
    """Data cleanup will preserve the number of samples in a window multiplied by this value."""

    CCB_TOOL_CLASS: typing.Optional[typing.Type[CcbToolBase]] = CcbTool
    """The CCB tool class to use, use :const:`None` to fallback on the default CCB tool."""

    _RAW_COUNT: typing.ClassVar[str] = '<Raw counts>'
    """Count scale key for raw count (i.e. no scaling)."""
    _COUNT_SCALES: typing.ClassVar[_C_T] = collections.OrderedDict(
        [(_RAW_COUNT, -1.0)], GHz=GHz, MHz=MHz, kHz=kHz, Hz=Hz, mHz=mHz)
    """Scales that can be used for the Y-axis."""

    DAX_INIT = False
    """Disable DAX init."""

    def build(self) -> None:  # type: ignore[override]
        assert isinstance(self.APPLET_GROUP, str), 'Applet group must be of type str'
        assert isinstance(self.DEFAULT_DATASET, str), 'Default dataset must be of type str'
        assert isinstance(self.DEFAULT_BUFFER_SIZE, int), 'Default buffer size must be of type int'
        assert self.DEFAULT_BUFFER_SIZE > 0, 'Default buffer size must be greater than zero'
        assert isinstance(self.MAX_BUFFER_SIZE, int), 'Maximum buffer size must be of type int'
        assert self.MAX_BUFFER_SIZE > 0, 'Maximum buffer size must be greater than zero'
        assert self.DEFAULT_BUFFER_SIZE <= self.MAX_BUFFER_SIZE, 'Default buffer size greater than the max buffer size'
        assert isinstance(self.DEFAULT_SLIDING_WINDOW_SIZE, int), 'Default sliding window size must be of type int'
        assert self.DEFAULT_SLIDING_WINDOW_SIZE >= 0, 'Default sliding window size must be zero or greater'
        assert isinstance(self.DEFAULT_SUBSAMPLE_FACTOR, int), 'Default subsample factor must be of type int'
        assert self.DEFAULT_SUBSAMPLE_FACTOR >= 0, 'Default subsample factor must be zero or greater'
        assert isinstance(self.DEFAULT_APPLET_UPDATE_DELAY, float), 'Default applet update delay must be of type float'
        assert self.DEFAULT_APPLET_UPDATE_DELAY > 0.0, 'Default applet update delay must be greater than zero'
        assert isinstance(self.DATA_CLEANUP_THRESHOLD_MULTIPLIER, int), \
            'Data cleanup threshold multiplier must be of type int'
        assert self.DATA_CLEANUP_THRESHOLD_MULTIPLIER > 0, 'Data cleanup threshold multiplier must be greater than zero'
        assert isinstance(self.DATA_CLEANUP_PRESERVE_MULTIPLIER, int), \
            'Data cleanup preserve multiplier must be of type int'
        assert self.DATA_CLEANUP_PRESERVE_MULTIPLIER > 0, 'Data cleanup preserve multiplier must be greater than zero'
        assert self.CCB_TOOL_CLASS is None or issubclass(self.CCB_TOOL_CLASS, CcbToolBase), \
            'CCB tool class must be a subclass of CcbToolBase or None'
        assert is_kernel(self.device_setup), 'device_setup() must be a kernel function'
        assert is_kernel(self.device_cleanup), 'device_cleanup() must be a kernel function'
        assert not is_kernel(self.host_setup), 'host_setup() can not be a kernel function'
        assert not is_kernel(self.host_cleanup), 'host_cleanup() can not be a kernel function'
        assert is_kernel(self._detect)
        assert is_kernel(self._count)

        # Obtain the PMT array
        self.pmt_array: typing.List[artiq.coredevice.edge_counter.EdgeCounter] = self.get_pmt_array()
        assert isinstance(self.pmt_array, list), 'PMT array is not a list'
        self.update_kernel_invariants('pmt_array')
        self.logger.debug(f'Found PMT array with {len(self.pmt_array)} channel(s)')
        assert self.pmt_array, 'The PMT array can not be empty'

        # Get the scheduler and CCB tool
        self.scheduler = self.get_device('scheduler')
        self.ccb = get_ccb_tool(self) if self.CCB_TOOL_CLASS is None else self.CCB_TOOL_CLASS(self)
        self.update_kernel_invariants('scheduler')

        # Standard arguments
        self.detection_window: float = self.get_argument(
            'PMT detection window size',
            NumberValue(100 * ms, unit='ms', min=0.0),
            tooltip='Detection window duration'
        )
        self.detection_delay: float = self.get_argument(
            'PMT detection delay',
            NumberValue(0 * ms, unit='ms', min=0.0),
            tooltip='Delay between detection windows (when set to zero, the delay will be 8 machine units)'
        )
        self.buffer_size: int = self.get_argument(
            'Buffer size',
            NumberValue(self.DEFAULT_BUFFER_SIZE, min=1, max=self.MAX_BUFFER_SIZE, ndecimals=0, step=1),
            tooltip='Buffer size in number of samples'
        )
        self.count_scale_label: str = self.get_argument(
            'PMT count scale',
            EnumerationValue(list(self._COUNT_SCALES), default='kHz'),
            tooltip='Scaling factor for the PMT counts graph'
        )
        self.sliding_window: int = self.get_argument(
            'Data window size',
            NumberValue(self.DEFAULT_SLIDING_WINDOW_SIZE, unit='s', min=0, ndecimals=0, step=10),
            tooltip='Data window size (use 0 for infinite window size)'
        )
        self.update_kernel_invariants('detection_window', 'buffer_size')

        # Add extra arguments
        self._add_arguments_internal()
        self.add_arguments()

        # Dataset related arguments
        self.reset_data: bool = self.get_argument(
            'Reset data',
            BooleanValue(True),
            group='Dataset',
            tooltip='Clear old data at start'
        )
        self.reset_data_resume: bool = self.get_argument(
            'Reset data when resuming',
            BooleanValue(False),
            group='Dataset',
            tooltip='Clear old data when resuming from a pause'
        )
        self.auto_data_cleanup: bool = self.get_argument(
            'Automatic data cleanup',
            BooleanValue(True),
            group='Dataset',
            tooltip='Clean data outside the window periodically (does not clean data if data window size is infinite)'
        )
        self.archive_data: bool = self.get_argument(
            'Archive data',
            BooleanValue(False),
            group='Dataset',
            tooltip='Archive the obtained data in the HDF5 output file'
        )
        self.dataset_key: str = self.get_argument(
            'Dataset key',
            StringValue(self.DEFAULT_DATASET),
            group='Dataset',
            tooltip='Dataset key to which plotting data will be written'
        )

        # Applet specific arguments
        self.create_applet: bool = self.get_argument(
            'Create applet',
            BooleanValue(True),
            group='Applet',
            tooltip='Call CCB create applet command at start'
        )
        self.subsample: int = self.get_argument(
            'Subsample',
            NumberValue(self.DEFAULT_SUBSAMPLE_FACTOR, min=0, ndecimals=0, step=1),
            group='Applet',
            tooltip='Subsample data by the given factor before plotting (<=1 for no subsampling)'
        )
        self.applet_update_delay: float = self.get_argument(
            'Applet update delay',
            NumberValue(self.DEFAULT_APPLET_UPDATE_DELAY, unit='s', min=0.1, step=0.1),
            group='Applet',
            tooltip='Delay between plot interface updates'
        )
        self.applet_auto_close: bool = self.get_argument(
            'Close applet automatically',
            BooleanValue(True),
            group='Applet',
            tooltip='Close applet when experiment is terminated'
        )
        self._add_arguments_applet()

    def _add_arguments_internal(self) -> None:
        """Add custom arguments.

        **For internal usage only**. See also :func:`add_arguments`.
        """
        pass

    def _add_arguments_applet(self) -> None:
        """Add custom arguments specifically in the applets group."""
        pass

    @abc.abstractmethod
    def _create_applet(self, *args: typing.Any, **kwargs: typing.Any) -> None:  # pragma: no cover
        """Create applet.

        :param args: Positional arguments for plotting **excluding the applet name**
        :param kwargs: Keyword arguments for plotting
        """
        pass

    def prepare(self) -> None:
        if self.detection_window <= 0.0:
            raise ValueError('Detection window must be greater than zero')

        # Prepare dataset kwargs
        self.dataset_kwargs = {'archive': self.archive_data, 'broadcast': True, 'persist': False}

        if self.archive_data and not self.auto_data_cleanup:
            self.logger.warning('Auto data cleanup disabled and data archiving enabled, '
                                'this can result in a large HDF5 output file')

        # Convert the detection delay to machine units
        # The lowest number of machine units required to separate two events is ``core.ref_multiplier``
        # Without event separation, the core device fuses consecutive detection windows to one
        self.detection_delay_mu: np.int64 = np.int64(max(self.core.seconds_to_mu(self.detection_delay),
                                                         self.core.ref_multiplier))
        self.update_kernel_invariants('detection_delay_mu')

        if self.sliding_window > 0:
            # Calculate window size in samples
            self.window_size_samples: int = round(self.sliding_window / (self.detection_window + self.detection_delay))
        else:
            # Disable sliding window
            self.window_size_samples = 0

        if self.auto_data_cleanup and self.window_size_samples > 0:
            # Calculate threshold for data cleanup
            self.data_cleanup_threshold: int = self.window_size_samples * self.DATA_CLEANUP_THRESHOLD_MULTIPLIER
        else:
            # Disable automatic data cleanup
            self.auto_data_cleanup = False
            self.data_cleanup_threshold = 0

        if self.count_scale_label != self._RAW_COUNT:
            # Pre-calculate Y-scalar
            self.y_scalar: float = 1.0 / self.detection_window / self._COUNT_SCALES[self.count_scale_label]
        else:
            # No scaling
            self.y_scalar = 1.0

    def analyze(self):
        if self.archive_data:
            # Archive relevant data
            self.set_dataset('detection_window', self.detection_window)
            self.set_dataset('detection_delay', self.detection_delay)
            self.set_dataset('y_scalar', self.y_scalar)
            self.logger.info(f'Number of data points prepared for archiving: {len(self._data)}')

    def run(self) -> None:
        # Initial value is reset to an empty list or try to obtain the previous value defaulting to an empty list
        if self.reset_data:
            self.logger.info('Starting with empty dataset')
            self._data: typing.List[typing.Any] = []
        else:
            previous_data = self.get_dataset(self.dataset_key, default=[], archive=False)
            if isinstance(previous_data, list):
                self.logger.info('Appending to previous dataset')
                self._data = previous_data
            else:
                self.logger.info('Previous dataset invalid, starting with empty dataset')
                self._data = []

        # Set the result datasets
        self.set_dataset(self.dataset_key, self._data.copy(), **self.dataset_kwargs)

        # Log messages from the prepare phase
        if self.window_size_samples > 0:
            self.logger.debug(f'Window size set to {self.window_size_samples} sample(s)')
        self.logger.debug(f'Automatic data cleanup: {self.auto_data_cleanup}')

        if self.create_applet:
            # Construct X-label
            if self.sliding_window > 0:
                x_label: str = f'Window size: {time_to_str(self.sliding_window, precision=0)}'
            else:
                x_label = 'Sample'

            # Create the applet Y-label
            if self.count_scale_label != self._RAW_COUNT:
                y_label: str = f'Counts per second ({self.count_scale_label})'
            else:
                y_label = 'Raw counts'

            # Create the applet
            self._create_applet(
                self.dataset_key, group=self.APPLET_GROUP, update_delay=self.applet_update_delay,
                sliding_window=self.window_size_samples, subsample=self.subsample, x_label=x_label, y_label=y_label
            )

        try:
            # Only stop when termination is requested
            while True:
                # Host setup
                self.host_setup()
                # Monitor
                self.logger.info('Start monitoring')
                self.monitor()
                # Host cleanup
                self.host_cleanup()

                # To pause, close communications and call the pause function
                self.logger.info('Pausing')
                self.core.comm.close()
                self.scheduler.pause()  # Can raise a TerminationRequested exception

                if self.reset_data_resume:
                    # Reset dataset when resuming
                    self._data = []
                    self.set_dataset(self.dataset_key, self._data.copy(), **self.dataset_kwargs)

        except TerminationRequested:
            # Experiment was terminated, gracefully end the experiment
            self.logger.info('Terminated gracefully')

        except RTIOUnderflow:
            # Underflow exception
            self.logger.exception('RTIO underflow exception; increase buffer size, detection window size, '
                                  'or detection delay')

        except RTIOOverflow:
            # Buffer overflow
            self.logger.exception('RTIO overflow exception; buffer size exceeds the size of the hardware buffers')

        finally:
            if self.applet_auto_close:
                # Disable the applet
                self.ccb.disable_applet_group(self.APPLET_GROUP)

    @kernel
    def monitor(self):  # type: () -> None
        # Device setup
        self.device_setup()

        for _ in range(self.buffer_size):
            # Build up a buffer
            delay_mu(self.detection_delay_mu)
            self._detect()

        try:
            while not self.scheduler.check_pause():
                # Detect
                delay_mu(self.detection_delay_mu)
                self._detect()
                # Count
                self._count()
        finally:
            # Drop buffered data by resetting core
            self.core.reset()
            # Device cleanup
            self.device_cleanup()

    def store(self, data: typing.Any) -> None:
        # Append data
        self._data.append(data)

        if self.auto_data_cleanup and len(self._data) > self.data_cleanup_threshold:
            # Cleanup data
            self._data = self._data[-self.window_size_samples * self.DATA_CLEANUP_PRESERVE_MULTIPLIER:]
            self.set_dataset(self.dataset_key, self._data.copy(), **self.dataset_kwargs)
        else:
            # Append to dataset
            self.append_to_dataset(self.dataset_key, data)

    @abc.abstractmethod
    def _detect(self) -> None:  # pragma: no cover
        """Perform detection."""
        pass

    @abc.abstractmethod
    def _count(self) -> None:  # pragma: no cover
        """Get counts, and store result."""
        pass

    """Customization functions"""

    def add_arguments(self) -> None:
        """Add custom arguments during the build phase."""
        pass

    def host_setup(self) -> None:
        """Setup on the host, called once at entry and after a pause."""
        pass

    @kernel
    def device_setup(self):  # type: () -> None
        """Setup on the core device, called once at entry and after a pause.

        Should at least reset the core.
        """
        # Reset the core
        self.core.reset()

    @kernel
    def device_cleanup(self):  # type: () -> None
        """Cleanup on the core device, called before pausing and exiting."""
        pass

    def host_cleanup(self) -> None:
        """Cleanup on the host, called before pausing and exiting."""
        pass

    def get_pmt_array(self) -> typing.List[artiq.coredevice.edge_counter.EdgeCounter]:
        """Get the PMT array from the system.

        By default, search for a detection interface and request the PMT array.

        :return: A list with :class:`EdgeCounter` objects
        """
        # Obtain the detection interface
        detection = self.registry.find_interface(DetectionInterface)  # type: ignore[misc]
        # Return its PMT array
        return detection.get_pmt_array()


@dax_client_factory
class PmtMonitor(_PmtMonitorBase):
    """PMT monitor utility to monitor a single PMT channel."""

    APPLET_GROUP = 'dax.pmt_monitor'
    DEFAULT_DATASET = 'plot.dax.pmt_monitor'

    DEFAULT_MOVING_AVERAGE_PLOT_XY: typing.ClassVar[int] = 0
    """Default uniform moving average window size in samples for the plot XY applet."""
    DEFAULT_NUM_DIGITS_BIG_NUMBER: typing.ClassVar[int] = 5
    """Default number of digits to display for the big number applet."""

    _PLOT_XY: typing.ClassVar[str] = 'Plot XY'
    """Key for plot XY applet type."""
    _BIG_NUMBER: typing.ClassVar[str] = 'Big number'
    """Key for big number applet type."""

    def _add_arguments_internal(self) -> None:
        assert isinstance(self.DEFAULT_MOVING_AVERAGE_PLOT_XY, int)
        assert self.DEFAULT_MOVING_AVERAGE_PLOT_XY >= 0, 'Default moving average window size must be zero or greater'
        assert isinstance(self.DEFAULT_NUM_DIGITS_BIG_NUMBER, int)
        assert self.DEFAULT_NUM_DIGITS_BIG_NUMBER >= 0, 'Default number of digits must be zero or greater'

        # Dict with available applet types
        self._applet_types: typing.Dict[str, typing.Callable[..., None]] = {
            self._PLOT_XY: self.ccb.plot_xy,
            self._BIG_NUMBER: self.ccb.big_number,
        }

        # Arguments
        self.pmt_channel: int = self.get_argument(
            'PMT channel',
            NumberValue(0, step=1, min=0, max=len(self.pmt_array) - 1, ndecimals=0),
            tooltip='PMT channel to monitor'
        )
        self.applet_type: str = self.get_argument(
            'Applet type',
            EnumerationValue(list(self._applet_types), default=self._PLOT_XY),
            tooltip='Choose an applet type'
        )
        self.update_kernel_invariants('pmt_channel')

    def _add_arguments_applet(self) -> None:
        # Plot XY applet args
        self.moving_average: int = self.get_argument(
            'Moving average',
            NumberValue(self.DEFAULT_MOVING_AVERAGE_PLOT_XY, step=1, min=0, ndecimals=0),
            group='Applet',
            tooltip='Number of samples used for uniform moving average window '
                    '(0 to disable moving average, for plot XY applet only)'
        )
        # Big number applet args
        self.num_digits: int = self.get_argument(
            'Num digits',
            NumberValue(self.DEFAULT_NUM_DIGITS_BIG_NUMBER, step=1, min=0, max=99, ndecimals=0),
            group='Applet',
            tooltip='Number of digits to show (for big number applet only)'
        )

    def _create_applet(self, *args: typing.Any, **kwargs: typing.Any) -> None:
        if self.applet_type == self._PLOT_XY:
            # Modify keyword arguments
            kwargs.setdefault('last', True)
            kwargs.setdefault('moving_average', self.moving_average)
        elif self.applet_type == self._BIG_NUMBER:
            # Modify keyword arguments
            kwargs = {k: v for k, v in kwargs.items() if k in {'group', 'update_delay'}}
            kwargs.setdefault('digit_count', self.num_digits)

        # Create applet based on chosen applet type
        self._applet_types[self.applet_type](self.applet_type, *args, **kwargs)

    @kernel
    def _detect(self):  # type: () -> None
        # Perform detection
        self.pmt_array[self.pmt_channel].gate_rising(self.detection_window)

    @kernel
    def _count(self):  # type: () -> None
        # Get count
        count = self.pmt_array[self.pmt_channel].fetch_count()
        # Store obtained count
        self._store(count)

    @rpc(flags={'async'})
    def _store(self, count):  # type: (int) -> None
        # Calculate data to store
        data = count * self.y_scalar
        # Store data
        self.store(data)


@dax_client_factory
class MultiPmtMonitor(_PmtMonitorBase):
    """PMT monitor utility to monitor multiple PMT channels simultaneously."""

    APPLET_GROUP = 'dax.multi_pmt_monitor'
    DEFAULT_DATASET = 'plot.dax.multi_pmt_monitor'

    TITLES: typing.ClassVar[typing.Sequence[typing.Optional[str]]] = []
    """A sequence of applet titles when using separate applets."""

    def _add_arguments_internal(self) -> None:
        assert isinstance(self.TITLES, collections.abc.Sequence), 'Separate titles must be a sequence'
        assert not self.TITLES or len(self.TITLES) == len(self.pmt_array), \
            'The sequence of applet titles must be empty or have the same length as the PMT array'
        assert all(t is None or isinstance(t, str) for t in self.TITLES), 'Titles must be of type str or None'

        # Arguments
        self.separate_applets: bool = self.get_argument(
            'Separate applets',
            BooleanValue(False),
            tooltip='Create a separate applet for each PMT'
        )

        # Channels
        self.enabled_channels: typing.List[bool] = [
            self.get_argument(
                f'Channel {i}',
                BooleanValue(True), group='Channels',
                tooltip=f'Enable monitoring for channel {i}'
            )
            for i in range(len(self.pmt_array))]
        self.update_kernel_invariants('enabled_channels')

    def _create_applet(self, *args: typing.Any, **kwargs: typing.Any) -> None:
        if all(not c for c in self.enabled_channels):
            self.logger.warning('No channels were enabled, no applet will be started')
            return

        # Modify keyword arguments
        kwargs.setdefault('plot_names', 'PMT')

        if self.separate_applets:
            # Assemble titles
            if self.TITLES:
                titles: typing.Sequence[typing.Optional[str]] = self.TITLES
            else:
                titles = [None] * len(self.pmt_array)

            # Create separate applets for each channel using multi-plot XY
            for i, t in zip(range(len(self.pmt_array)), titles):
                if self.enabled_channels[i]:
                    self.ccb.plot_xy_multi(f'pmt_{i}', *args, index=i, title=t, **kwargs)
        else:
            # Plot all data using multi-plot XY
            index = [i for i, c in enumerate(self.enabled_channels) if c]
            self.ccb.plot_xy_multi('plot_all', *args, index=index, **kwargs)

    @kernel
    def _detect(self):  # type: () -> None
        # Perform detection (using low-level control for maximum performance)
        for p in self.pmt_array:
            p.set_config(True, False, False, True)
        delay(self.detection_window)
        for p in self.pmt_array:
            p.set_config(False, False, True, False)

    @kernel
    def _count(self):  # type: () -> None
        # Get counts
        counts = [p.fetch_count() for p in self.pmt_array]
        # Store obtained counts
        self._store(counts)

    @rpc(flags={'async'})
    def _store(self, counts):  # type: (typing.List[int]) -> None
        # Calculate data to store
        data = [c * self.y_scalar for c in counts]
        # Store data
        self.store(data)
