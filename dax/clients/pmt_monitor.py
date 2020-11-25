import collections
import collections.abc
import typing
import abc
import numpy as np

import artiq.coredevice.edge_counter

from dax.experiment import *
from dax.interfaces.detection import DetectionInterface
from dax.util.ccb import get_ccb_tool
from dax.util.artiq import is_kernel
from dax.util.units import time_to_str

__all__ = ['PmtMonitor', 'MultiPmtMonitor']


class _PmtMonitorBase(DaxClient, EnvExperiment, abc.ABC):
    """Base PMT monitor class."""

    APPLET_GROUP: str
    """Group of the applet."""
    DEFAULT_DATASET: str
    """Default dataset for output."""
    DEFAULT_BUFFER_SIZE: int = 1
    """Default buffer size in samples."""
    MAX_BUFFER_SIZE: int = 32
    """Maximum buffer size in samples."""

    _RAW_COUNT: str = '<Raw counts>'
    """Count scale key for raw count (i.e. no scaling)."""
    _COUNT_SCALES = collections.OrderedDict([(_RAW_COUNT, -1.0)], GHz=GHz, MHz=MHz, kHz=kHz, Hz=Hz, mHz=mHz)
    """Scales that can be used for the Y-axis."""

    DAX_INIT: bool = False
    """Disable DAX init."""

    def build(self) -> None:  # type: ignore
        assert isinstance(self.APPLET_GROUP, str), 'Applet group must be of type str'
        assert isinstance(self.DEFAULT_DATASET, str), 'Default dataset must be of type str'
        assert isinstance(self.DEFAULT_BUFFER_SIZE, int), 'Default buffer size must be of type int'
        assert self.DEFAULT_BUFFER_SIZE > 0, 'Default buffer size must be greater than zero'
        assert isinstance(self.MAX_BUFFER_SIZE, int), 'Maximum buffer size must be of type int'
        assert self.MAX_BUFFER_SIZE > 0, 'Maximum buffer size must be greater than zero'
        assert self.DEFAULT_BUFFER_SIZE <= self.MAX_BUFFER_SIZE, 'Default buffer size greater than the max buffer size'
        assert is_kernel(self.device_setup), 'device_setup() must be a kernel function'
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
        self.ccb = get_ccb_tool(self)
        self.update_kernel_invariants('scheduler')

        # Standard arguments
        self.detection_window: float = self.get_argument('PMT detection window size',
                                                         NumberValue(default=100 * ms, unit='ms', min=0.0),
                                                         tooltip='Detection window duration')
        self.detection_delay: float = self.get_argument('PMT detection delay',
                                                        NumberValue(default=0 * ms, unit='ms', min=0.0),
                                                        tooltip='Delay between detection windows (when set to zero, '
                                                                'the delay will be 10 machine units)')
        self.buffer_size: int = self.get_argument('Buffer size',
                                                  NumberValue(default=self.DEFAULT_BUFFER_SIZE, min=1,
                                                              max=self.MAX_BUFFER_SIZE, ndecimals=0, step=1),
                                                  tooltip='Buffer size in number of samples')
        self.count_scale_label: str = self.get_argument('PMT count scale',
                                                        EnumerationValue(list(self._COUNT_SCALES), default='kHz'),
                                                        tooltip='Scaling factor for the PMT counts graph '
                                                                '(requires applet restart)')
        self.sliding_window: int = self.get_argument('Data window size',
                                                     NumberValue(default=120 * s, unit='s', min=0, ndecimals=0, step=1),
                                                     tooltip='Data window size (use 0 for infinite window size) '
                                                             '(requires applet restart)')
        self.update_kernel_invariants('detection_window', 'buffer_size')

        # Add custom arguments here
        self._add_custom_arguments()

        # Dataset related arguments
        self.reset_data: bool = self.get_argument('Reset data',
                                                  BooleanValue(default=True),
                                                  group='Dataset',
                                                  tooltip='Clear old data at start')
        self.reset_data_resume: bool = self.get_argument('Reset data when resuming',
                                                         BooleanValue(default=False),
                                                         group='Dataset',
                                                         tooltip='Clear old data when resuming from a pause')
        self.dataset_key: str = self.get_argument('Dataset key',
                                                  StringValue(default=self.DEFAULT_DATASET),
                                                  group='Dataset',
                                                  tooltip='Dataset key to which plotting data will be written '
                                                          '(requires applet restart)')

        # Applet specific arguments
        self.create_applet: bool = self.get_argument('Create applet',
                                                     BooleanValue(default=True),
                                                     group='Applet',
                                                     tooltip='Call CCB create applet command at start')
        self.applet_update_delay: float = self.get_argument('Applet update delay',
                                                            NumberValue(default=0.1 * s, unit='s', min=0.0),
                                                            group='Applet',
                                                            tooltip='Delay between plot interface updates')
        self.applet_auto_close: bool = self.get_argument('Close applet automatically',
                                                         BooleanValue(default=True),
                                                         group='Applet',
                                                         tooltip='Close applet when experiment is terminated')

    def _add_custom_arguments(self) -> None:
        """Add custom arguments."""
        pass

    @abc.abstractmethod
    def _create_applet(self, *args: typing.Any, **kwargs: typing.Any) -> None:
        """Create applet.

        :param args: Positional arguments for plotting **excluding the applet name**
        :param kwargs: Keyword arguments for plotting
        """
        pass

    def prepare(self) -> None:
        if self.detection_window <= 0.0:
            raise ValueError('Detection window must be greater than zero')

        # Convert the detection delay to machine units
        # The lowest number of machine units required to separate two events is `core.ref_multiplier`
        # Without event separation, the core device fuses consecutive detection windows to one
        self.detection_delay_mu: np.int64 = np.int64(max(self.core.seconds_to_mu(self.detection_delay),
                                                         self.core.ref_multiplier))
        self.update_kernel_invariants('detection_delay_mu')

        if self.count_scale_label != self._RAW_COUNT:
            # Pre-calculate Y-scalar
            self.y_scalar: float = 1.0 / self.detection_window / self._COUNT_SCALES[self.count_scale_label]
        else:
            # No scaling
            self.y_scalar = 1.0

    def run(self) -> None:
        # NOTE: there is no dax_init() in this experiment!

        # Initial value is reset to an empty list or try to obtain the previous value defaulting to an empty list
        init_value = [] if self.reset_data else self.get_dataset(self.dataset_key, default=[], archive=False)
        self.logger.debug('Appending to previous data' if init_value else 'Starting with empty list')

        # Set the result datasets to the correct mode
        dataset_kwargs: typing.Dict[str, bool] = {'broadcast': True, 'archive': False}
        self.set_dataset(self.dataset_key, init_value, **dataset_kwargs)

        if self.create_applet:
            if self.sliding_window > 0:
                # Calculate window size in samples
                window_size_samples: int = round(self.sliding_window / (self.detection_window + self.detection_delay))
                self.logger.debug(f'Window size set to {window_size_samples} sample(s)')
                # Construct X-label
                x_label: str = f'Window size: {time_to_str(self.sliding_window, precision=0)}'
            else:
                # No sliding window
                window_size_samples = 0
                # Construct X-label
                x_label = f'Sample'

            # Create the applet Y-label
            if self.count_scale_label != self._RAW_COUNT:
                y_label: str = f'Counts per second ({self.count_scale_label})'
            else:
                y_label = 'Raw counts'

            # Create the applet
            self._create_applet(self.dataset_key, group=self.APPLET_GROUP, update_delay=self.applet_update_delay,
                                sliding_window=window_size_samples, x_label=x_label, y_label=y_label)

        try:
            # Only stop when termination is requested
            while True:
                # Host setup
                self.host_setup()
                # Monitor
                self.logger.debug('Start monitoring')
                self.monitor()

                # To pause, close communications and call the pause function
                self.logger.debug('Pausing')
                self.core.comm.close()
                self.scheduler.pause()  # Can raise a TerminationRequested exception

                if self.reset_data_resume:
                    # Reset dataset when resuming
                    self.set_dataset(self.dataset_key, [], **dataset_kwargs)

        except TerminationRequested:
            # Experiment was terminated, gracefully end the experiment
            self.logger.debug('Terminated gracefully')

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

    @abc.abstractmethod
    def _detect(self) -> None:
        """Perform detection."""
        pass

    @abc.abstractmethod
    def _count(self) -> None:
        """Get counts, and store result."""
        pass

    """Customization functions"""

    def host_setup(self) -> None:
        """Preparation on the host, called once at entry and after a pause."""
        pass

    @kernel
    def device_setup(self):  # type: () -> None
        """Preparation on the core device, called once at entry and after a pause.

        Should at least reset the core.
        """
        # Reset the core
        self.core.reset()

    def get_pmt_array(self) -> typing.List[artiq.coredevice.edge_counter.EdgeCounter]:
        """Get the PMT array from the system.

        By default, search for a detection interface and request the PMT array.

        :return: A list with `EdgeCounter` objects
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

    NUM_DIGITS_BIG_NUMBER: int = 5
    """Number of digits to display for the big number applet."""

    _PLOT_XY: str = 'Plot XY'
    """Key for plot XY applet type."""
    _BIG_NUMBER: str = 'Big number'
    """Key for big number applet type."""

    def _add_custom_arguments(self) -> None:
        assert self.NUM_DIGITS_BIG_NUMBER >= 0, 'Number of digits must be zero or greater'

        # Dict with available applet types
        self._applet_types: typing.Dict[str, typing.Callable[..., None]] = {
            self._PLOT_XY: self.ccb.plot_xy,
            self._BIG_NUMBER: self.ccb.big_number,
        }

        # Arguments
        self.pmt_channel: int = self.get_argument('PMT channel',
                                                  NumberValue(default=0, step=1, min=0, max=len(self.pmt_array) - 1,
                                                              ndecimals=0),
                                                  tooltip='PMT channel to monitor')
        self.applet_type: str = self.get_argument('Applet type',
                                                  EnumerationValue(list(self._applet_types), self._PLOT_XY),
                                                  tooltip='Choose an applet type (requires applet restart)')
        self.update_kernel_invariants('pmt_channel')

    def _create_applet(self, *args: typing.Any, **kwargs: typing.Any) -> None:
        if self.applet_type == self._PLOT_XY:
            # Modify keyword arguments
            kwargs.setdefault('last', True)
        elif self.applet_type == self._BIG_NUMBER:
            # Modify keyword arguments
            kwargs = {k: v for k, v in kwargs.items() if k in {'group', 'update_delay'}}
            kwargs.setdefault('digit_count', self.NUM_DIGITS_BIG_NUMBER)

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
        # Calculate value to store
        value = count * self.y_scalar
        # Append data to datasets
        self.append_to_dataset(self.dataset_key, value)


@dax_client_factory
class MultiPmtMonitor(_PmtMonitorBase):
    """PMT monitor utility to monitor multiple PMT channels simultaneously."""

    APPLET_GROUP = 'dax.multi_pmt_monitor'
    DEFAULT_DATASET = 'plot.dax.multi_pmt_monitor'

    TITLES: typing.Sequence[typing.Optional[str]] = []
    """A sequence of applet titles when using separate applets."""

    def _add_custom_arguments(self) -> None:
        assert isinstance(self.TITLES, collections.abc.Sequence), 'Separate titles must be a sequence'
        assert not self.TITLES or len(self.TITLES) == len(self.pmt_array), \
            'The sequence of applet titles must be empty or have the same length as the PMT array'
        assert all(t is None or isinstance(t, str) for t in self.TITLES), 'Titles must be of type str or None'

        # Arguments
        self.separate_applets: bool = self.get_argument('Separate applets',
                                                        BooleanValue(False),
                                                        tooltip='Create a separate applet for each PMT')

    def _create_applet(self, *args: typing.Any, **kwargs: typing.Any) -> None:
        # Modify keyword arguments
        kwargs.setdefault('plot_names', 'PMT')
        kwargs.setdefault('markers_only', True)

        if self.separate_applets:
            # Assemble titles
            if self.TITLES:
                titles: typing.Sequence[typing.Optional[str]] = self.TITLES
            else:
                titles = [None] * len(self.pmt_array)

            # Create separate applets for each channel using multi-plot XY
            for i, t in zip(range(len(self.pmt_array)), titles):
                self.ccb.plot_xy_multi(f'pmt_{i}', *args, index=i, title=t, **kwargs)
        else:
            # Plot all data using multi-plot XY
            self.ccb.plot_xy_multi('plot_all', *args, **kwargs)

    @kernel
    def _detect(self):  # type: () -> None
        # Perform detection
        with parallel:
            for p in self.pmt_array:
                p.gate_rising(self.detection_window)

    @kernel
    def _count(self):  # type: () -> None
        # Get counts
        counts = [p.fetch_count() for p in self.pmt_array]
        # Store obtained counts
        self._store(counts)

    @rpc(flags={'async'})
    def _store(self, counts):  # type: (typing.List[int]) -> None
        # Calculate value to store
        value = [c * self.y_scalar for c in counts]
        # Append data to datasets
        self.append_to_dataset(self.dataset_key, value)
