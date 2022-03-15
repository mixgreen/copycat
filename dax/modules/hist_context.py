import typing
import collections
import collections.abc
import numpy as np
import h5py
import natsort
import os.path
import math

from dax.experiment import *
from dax.interfaces.data_context import DataContextInterface, DataContextError
from dax.interfaces.detection import DetectionInterface
from dax.util.ccb import get_ccb_tool
from dax.util.output import FileNameGenerator, BaseFileNameGenerator
from dax.util.units import UnitsFormatter

__all__ = ['HistogramContext', 'HistogramAnalyzer', 'HistogramContextError']


class HistogramContextError(DataContextError):
    """Class for histogram context errors."""
    pass


_DATA_T = typing.Union[bool, int]  # Type of raw data


class HistogramContext(DaxModule, DataContextInterface):
    """Context class for managing storage of PMT histogram data.

    This module can be used as a sub-module of a service providing state measurement abilities.
    The HistogramContext object can directly be passed to the user which can use it as a context
    or call its additional functions.

    Note that the histogram context requires a :class:`DetectionInterface` in your system.
    This class implements the :class:`DataContextInterface`.

    The histogram context objects manages all result values, but the user is responsible for tracking
    "input parameters".
    """

    HISTOGRAM_PLOT_KEY_FORMAT: typing.ClassVar[str] = 'plot.{base}.histogram_context.histogram'
    """Dataset name for plotting latest histogram."""
    HISTOGRAM_PLOT_NAME: typing.ClassVar[str] = 'histogram'
    """Name of the histogram plot applet."""

    PROBABILITY_PLOT_KEY_FORMAT: typing.ClassVar[str] = 'plot.{base}.histogram_context.probability'
    """Dataset name for plotting latest individual probability graph."""
    PROBABILITY_PLOT_NAME: typing.ClassVar[str] = 'probability'
    """Name of the individual probability plot applet."""

    MEAN_COUNT_PLOT_KEY_FORMAT: typing.ClassVar[str] = 'plot.{base}.histogram_context.mean_count'
    """Dataset name for plotting latest mean count graph."""
    STDEV_COUNT_PLOT_KEY_FORMAT: typing.ClassVar[str] = 'plot.{base}.histogram_context.stdev_count'
    """Dataset name for plotting standard deviation on latest mean count graph."""
    MEAN_COUNT_PLOT_NAME: typing.ClassVar[str] = 'mean count'
    """Name of the mean count plot applet."""

    STATE_PROBABILITY_PLOT_KEY_FORMAT: typing.ClassVar[str] = 'plot.{base}.histogram_context.state_probability'
    """Dataset name for plotting latest full state probability graph."""
    STATE_PROBABILITY_PLOT_NAME: typing.ClassVar[str] = 'state_probability'
    """Name of the full state probability plot applet."""

    PLOT_GROUP_FORMAT: typing.ClassVar[str] = '{base}.histogram_context'
    """Group to which the plot applets belong."""

    DATASET_GROUP: typing.ClassVar[str] = 'histogram_context'
    """The group name for archiving histogram data."""
    RAW_DATASET_GROUP: typing.ClassVar[str] = 'raw'
    """The group name for archiving raw data."""
    DATASET_KEY_FORMAT: typing.ClassVar[str] = f'{DATASET_GROUP}/{{dataset_key}}/{RAW_DATASET_GROUP}/{{index}}'
    """Format string for raw sub-dataset keys."""
    DEFAULT_DATASET_KEY: typing.ClassVar[str] = 'histogram'
    """The default dataset key of the output sub-datasets."""

    _default_dataset_key: str
    _units_fmt: UnitsFormatter
    _in_context: np.int32
    _buffer: typing.List[typing.Sequence[_DATA_T]]
    _first_close: bool
    _raw_cache: typing.Dict[str, typing.List[typing.Sequence[typing.Sequence[_DATA_T]]]]
    _histogram_cache: typing.Dict[str, typing.List[typing.Sequence[typing.Counter[_DATA_T]]]]
    _dataset_key: str
    _plot_base_key: str
    _plot_group_base_key: str
    _open_datasets: typing.Counter[str]
    _histogram_plot_key: str
    _probability_plot_key: str
    _mean_count_plot_key: str
    _stdev_count_plot_key: str
    _state_probability_plot_key: str
    _plot_group: str

    def build(self, *,  # type: ignore[override]
              default_dataset_key: typing.Optional[str] = None,
              plot_base_key: str = 'dax',
              plot_group_base_key: typing.Optional[str] = None) -> None:
        """Build the histogram context module.

        The plot base key can be used to group plot datasets and applets as desired.
        The base keys are formatted with the ARTIQ ``scheduler`` object which allows users to
        add experiment-specific information in the base keys.

        By default, plot datasets and applets both are reused. Examples of common settings include:

         - Reuse plot datasets and applets (default)
         - Create unique plot datasets and applets based on the experiment RID:
           ``plot_base_key="{scheduler.rid}"``
         - Create unique plot datasets based on the experiment RID but reuse applets:
           ``plot_base_key="{scheduler.rid}", plot_group_base_key=""``

        :param default_dataset_key: Default dataset name used for storing histogram data
        :param plot_base_key: Base key for plot dataset keys
        :param plot_group_base_key: Base key for the plot group, same as ``plot_base_key`` if :const:`None`
        """
        assert isinstance(default_dataset_key, str) or default_dataset_key is None, \
            'Provided default dataset key must be None or of type str'
        assert isinstance(plot_base_key, str), 'Plot base key must be of type str'
        assert isinstance(plot_group_base_key, str) or plot_group_base_key is None, \
            'Plot group base key must be None or of type str'

        # Store default dataset key
        self._default_dataset_key = self.DEFAULT_DATASET_KEY if default_dataset_key is None else default_dataset_key

        # Get CCB tool
        self._ccb = get_ccb_tool(self)
        # Get scheduler
        self._scheduler = self.get_device('scheduler')
        # Units formatter
        self._units_fmt = UnitsFormatter()

        # By default we are not in context
        self._in_context = np.int32(0)
        # The count buffer (buffer appending is a bit faster than dict operations)
        self._buffer = []
        # Flag for the first call to close()
        self._first_close = True

        # Cache for raw data
        self._raw_cache = {}
        # Cache for histogram data
        self._histogram_cache = {}

        # Target dataset key
        self._dataset_key = self._default_dataset_key
        # Store plot base key
        self._plot_base_key = plot_base_key
        # Store plot group base key
        self._plot_group_base_key = plot_base_key if plot_group_base_key is None else plot_group_base_key
        # Open datasets stored as counters, which represent the length of the data
        self._open_datasets = collections.Counter()

    def init(self) -> None:
        # Generate plot keys
        base: str = self._plot_base_key.format(scheduler=self._scheduler)
        self._histogram_plot_key = self.HISTOGRAM_PLOT_KEY_FORMAT.format(base=base)
        self._probability_plot_key = self.PROBABILITY_PLOT_KEY_FORMAT.format(base=base)
        self._mean_count_plot_key = self.MEAN_COUNT_PLOT_KEY_FORMAT.format(base=base)
        self._stdev_count_plot_key = self.STDEV_COUNT_PLOT_KEY_FORMAT.format(base=base)
        self._state_probability_plot_key = self.STATE_PROBABILITY_PLOT_KEY_FORMAT.format(base=base)
        # Generate applet plot group
        base = self._plot_group_base_key.format(scheduler=self._scheduler)
        self._plot_group = self.PLOT_GROUP_FORMAT.format(base=base)

    def post_init(self) -> None:
        # Obtain the state detection threshold
        detection = self.registry.find_interface(DetectionInterface)  # type: ignore[misc]
        self._state_detection_threshold = detection.get_state_detection_threshold()
        self.update_kernel_invariants('_state_detection_threshold')

    """Data handling functions"""

    @portable
    def in_context(self) -> TBool:
        """True if we are in context."""
        return bool(self._in_context)

    @rpc(flags={'async'})
    def append(self, data):  # type: (typing.Sequence[_DATA_T]) -> None
        """Append PMT data to the histogram (async RPC).

        This function is intended to be fast to allow high input data throughput.
        No type checking is performed on the data.

        :param data: A list of ints representing the PMT counts of different ions
        :raises HistogramContextError: Raised if called outside the histogram context
        """
        if not self._in_context:
            # Called out of context
            raise HistogramContextError('The histogram append function can only be called inside the histogram context')

        # Append the given element to the buffer
        self._buffer.append(data)

    @rpc(flags={'async'})
    def config_dataset(self, key=None, *args, **kwargs):  # type: (typing.Optional[str], typing.Any, typing.Any) -> None
        """Optional configuration of the histogram context output dataset (async RPC).

        Set the dataset base key used for the following histograms.
        Use :const:`None` to reset the dataset base key to its default value.

        Within ARTIQ kernels it is not possible to use string formatting functions.
        Instead, the key can be a string that includes formatting annotations while
        formatting parameters can be provided as positional and keyword arguments.
        The formatting function will be called on the host.

        The formatter uses an extended format and it is possible to convert float values
        to human-readable format using conversion flags such as ``'{!t}'`` and ``'{!f}'``.
        See :class:`dax.util.units.UnitsFormatter` for more information about the available conversion flags.
        Note that the formatter has the default precision of 6 digits which is not likely
        to generate unique keys. An other field can be added to make sure the keys are unique.

        This function can not be used when already in context.

        :param key: Key for the result dataset using standard Python formatting notation
        :param args: Python ``str.format()`` positional arguments
        :param kwargs: Python ``str.format()`` keyword arguments
        :raises HistogramContextError: Raised if called inside the histogram context
        """
        assert isinstance(key, str) or key is None, 'Provided dataset key must be of type str or None'

        if self._in_context:
            # Called in context
            raise HistogramContextError('Setting the target dataset can only be done when not in context')

        # Update the dataset key
        self._dataset_key = self._default_dataset_key if key is None else self._units_fmt.vformat(key, args, kwargs)

    @rpc(flags={'async'})
    def open(self):  # type: () -> None
        """Open the histogram context.

        Opening the histogram context will prepare the target dataset and clear the buffer.
        Optionally, this context can be configured using the :func:`config` function.

        This function can be used to manually enter the histogram context.
        We strongly recommend to use the ``with`` statement instead.

        :raises HistogramContextError: Raised if the histogram context was already entered (context is non-reentrant)
        """

        if self._in_context:
            # Prevent context reentry
            raise HistogramContextError('The histogram context is non-reentrant')

        # Create a new buffer (clearing it might result in data loss due to how the dataset manager works)
        self._buffer = []
        # Increment in context counter
        self._in_context += 1

    @rpc(flags={'async'})
    def close(self):  # type: () -> None
        """Close the histogram context.

        This function can be used to manually exit the histogram context.
        We strongly recommend to use the ``with`` statement instead.

        :raises HistogramContextError: Raised if the histogram context was not entered
        """

        if not self._in_context:
            # Called exit out of context
            raise HistogramContextError('The exit function can only be called from inside the histogram context')

        if self._first_close:
            # Prepare the probability and mean count plot datasets by clearing them
            self.clear_probability_plot()
            self.clear_mean_count_plot()
            # Clear flag
            self._first_close = False

        # Create a sub-dataset keys for archiving this result (HDF5 only supports static array dimensions)
        archive_dataset_key: str = self.DATASET_KEY_FORMAT.format(
            dataset_key=self._dataset_key, index=self._open_datasets[self._dataset_key])

        if len(self._buffer):
            # Check consistency of data in the buffer
            if any(len(b) != len(self._buffer[0]) for b in self._buffer):
                raise RuntimeError('Data in the buffer is ragged')
            if len(self._buffer[0]) == 0:
                raise RuntimeError('Data elements in the buffer are empty')

            # Store raw data in the cache
            self._raw_cache.setdefault(self._dataset_key, []).append(self._buffer)
            # Archive raw data
            self.set_dataset(archive_dataset_key, self._buffer, archive=True)

            # Transform buffer data to pack counts per ion and convert into histograms
            histograms: typing.List[typing.Counter[_DATA_T]] = HistogramAnalyzer.raw_to_histograms(self._buffer)
            # Store histograms in the cache
            self._histogram_cache.setdefault(self._dataset_key, []).append(histograms)

            # Flatten dict-like histograms to uniformly-sized list-style histograms
            max_count: int = max(max(h) for h in histograms)
            flat_histograms = [HistogramAnalyzer.counter_to_ndarray(h, max_count=max_count) for h in histograms]
            # Write result to histogram plotting dataset
            self.set_dataset(self._histogram_plot_key, flat_histograms, broadcast=True, archive=False)

            # Calculate individual state probabilities
            probabilities: typing.List[float] = [self._histogram_to_probability(h) for h in histograms]
            # Append result to probability plotting dataset
            self.append_to_dataset(self._probability_plot_key, probabilities)

            # Calculate count mean and standard deviation per histogram
            mean_stdev_counts = (HistogramAnalyzer.histogram_to_mean_stdev_count(h) for h in histograms)
            mean_counts, stdev_counts = list(zip(*mean_stdev_counts))  # Transpose data and unpack
            # Append results to count mean and standard deviation plotting datasets
            self.append_to_dataset(self._mean_count_plot_key, mean_counts)
            self.append_to_dataset(self._stdev_count_plot_key, stdev_counts)

        else:
            # Add empty element to the caches (keeps indexing consistent)
            self._raw_cache.setdefault(self._dataset_key, []).append([])
            self._histogram_cache.setdefault(self._dataset_key, []).append([])
            # Write empty element to sub-dataset for archiving (keeps indexing consistent)
            self.set_dataset(archive_dataset_key, [], archive=True)

        # Update counter for this dataset key
        self._open_datasets[self._dataset_key] += 1
        # Update context counter
        self._in_context -= 1

    def _histogram_to_probability(self, histogram: typing.Counter[_DATA_T],
                                  state_detection_threshold: typing.Optional[int] = None) -> float:
        """Convert a histogram to an individual state probability.

        Falls back on default state detection threshold if none is given.
        """
        if state_detection_threshold is None:
            # Use default state_detection_threshold if not set
            state_detection_threshold = self._state_detection_threshold

        return HistogramAnalyzer.histogram_to_probability(
            histogram, state_detection_threshold=state_detection_threshold)

    """Applet plotting functions"""

    @rpc(flags={'async'})
    def plot_histogram(self, **kwargs):  # type: (typing.Any) -> None
        """Open the applet that shows a plot of the latest histogram.

        This function can only be called after the module is initialized.

        :param kwargs: Extra keyword arguments for the plot
        """

        # Set default arguments
        kwargs.setdefault('x_label', 'Number of counts')
        kwargs.setdefault('y_label', 'Frequency')
        kwargs.setdefault('title', f'RID {self._scheduler.rid}')
        # Plot
        self._ccb.plot_hist_multi(self.HISTOGRAM_PLOT_NAME, self._histogram_plot_key, group=self._plot_group, **kwargs)

    @rpc(flags={'async'})
    def plot_probability(self, **kwargs):  # type: (typing.Any) -> None
        """Open the applet that shows a plot of individual state probabilities (one for each histogram).

        Note that if the data points are randomized the user should provide X values
        to sort the points and plot the graph correctly (``x`` kwarg).

        This function can only be called after the module is initialized.

        :param kwargs: Extra keyword arguments for the plot
        """

        # Set defaults
        kwargs.setdefault('y_label', 'State probability')
        kwargs.setdefault('title', f'RID {self._scheduler.rid}')
        # Plot
        self._ccb.plot_xy_multi(self.PROBABILITY_PLOT_NAME, self._probability_plot_key,
                                group=self._plot_group, **kwargs)

    @rpc(flags={'async'})
    def plot_mean_count(self, **kwargs):  # type: (typing.Any) -> None
        """Open the applet that shows a plot of mean count per histogram.

        This function can only be called after the module is initialized.

        :param kwargs: Extra keyword arguments for the plot
        """

        # Set defaults
        kwargs.setdefault('y_label', 'Mean count')
        kwargs.setdefault('title', f'RID {self._scheduler.rid}')
        # Plot
        self._ccb.plot_xy_multi(self.MEAN_COUNT_PLOT_NAME, self._mean_count_plot_key,
                                error=self._stdev_count_plot_key, group=self._plot_group, **kwargs)

    @rpc(flags={'async'})
    def plot_state_probability(self, dataset_key=None,
                               **kwargs):  # type: (typing.Optional[str], typing.Any) -> None
        """Open the applet that shows the full state probability graph.

        **This function can only be called after all data was obtained.**
        Plotting state probabilities does not scale well and therefore we do not support real-time plotting.
        Instead, the contents of the graph are computed once when calling this function.
        If this function is called before data is available, an error will be logged but no exception is raised.

        Note that this function computes and redraws the plot for each call.
        Therefore, this function should preferably be called only once after the experiment finished.

        :param dataset_key: Key of the dataset to plot
        :param kwargs: Extra keyword arguments for the plot
        """

        try:
            # Obtain raw data
            raw = self.get_raw(dataset_key)
        except KeyError:
            # No data available (yet)
            self.logger.error('No data available, state probability can only be plotted after all data is obtained')
        else:
            # Transform and broadcast data
            state_probabilities = HistogramAnalyzer.raw_to_flat_state_probabilities(
                raw, state_detection_threshold=self._state_detection_threshold)
            self.set_dataset(self._state_probability_plot_key, state_probabilities,
                             broadcast=True, archive=False)

            # Set defaults
            kwargs.setdefault('y_label', '|State> probability')
            kwargs.setdefault('title', f'RID {self._scheduler.rid}')
            kwargs.setdefault('plot_names', '|{index}>')
            # Plot
            self._ccb.plot_xy_multi(self.STATE_PROBABILITY_PLOT_NAME, self._state_probability_plot_key,
                                    group=self._plot_group, **kwargs)

    @rpc(flags={'async'})
    def clear_probability_plot(self):  # type: () -> None
        """Clear the probability plot.

        This function can only be called after the module is initialized.
        """
        # Set the probability dataset to an empty list
        self.set_dataset(self._probability_plot_key, [], broadcast=True, archive=False)

    @rpc(flags={'async'})
    def clear_mean_count_plot(self):  # type: () -> None
        """Clear the mean count plot.

        This function can only be called after the module is initialized.
        """
        # Set the mean/stdev count datasets to empty lists
        self.set_dataset(self._mean_count_plot_key, [], broadcast=True, archive=False)
        self.set_dataset(self._stdev_count_plot_key, [], broadcast=True, archive=False)

    @rpc(flags={'async'})
    def disable_histogram_plot(self):  # type: () -> None
        """Close the histogram plot.

        This function can only be called after the module is initialized.
        """
        self._ccb.disable_applet(self.HISTOGRAM_PLOT_NAME, self._plot_group)

    @rpc(flags={'async'})
    def disable_probability_plot(self):  # type: () -> None
        """Close the probability plot.

        This function can only be called after the module is initialized.
        """
        self._ccb.disable_applet(self.PROBABILITY_PLOT_NAME, self._plot_group)

    @rpc(flags={'async'})
    def disable_mean_count_plot(self):  # type: () -> None
        """Close the probability plot.

        This function can only be called after the module is initialized.
        """
        self._ccb.disable_applet(self.MEAN_COUNT_PLOT_NAME, self._plot_group)

    @rpc(flags={'async'})
    def disable_state_probability_plot(self):  # type: () -> None
        """Close the full state probability plot.

        This function can only be called after the module is initialized.
        """
        self._ccb.disable_applet(self.STATE_PROBABILITY_PLOT_NAME, self._plot_group)

    @rpc(flags={'async'})
    def disable_all_plots(self):  # type: () -> None
        """Close all histogram context plots.

        This function can only be called after the module is initialized.
        """
        self._ccb.disable_applet_group(self._plot_group)

    """Data access functions"""

    @host_only
    def get_keys(self) -> typing.List[str]:
        """Get the keys for which histogram data was recorded.

        The returned keys can be used for the :func:`get_raw`, :func:`get_histograms`,
        and :func:`get_probabilities` functions.

        :return: A list with keys
        """
        return natsort.natsorted(self._raw_cache)

    @host_only
    def get_raw(self, dataset_key: typing.Optional[str] = None) \
            -> typing.List[typing.Sequence[typing.Sequence[_DATA_T]]]:
        """Obtain the raw data captured by the histogram context for a specific key.

        Data is formatted as a 3-dimensional list.
        To access the data of histogram N of data point P of channel C: ``get_raw()[N][P][C]``.

        In case no dataset key is provided, the default dataset key is used.

        :param dataset_key: Key of the dataset to obtain the raw data of
        :return: All raw data for the specified key
        :raises KeyError: Raised if no data is available for the given dataset key
        """
        return self._raw_cache[self._default_dataset_key if dataset_key is None else dataset_key][:]

    @host_only
    def get_histograms(self, dataset_key: typing.Optional[str] = None) \
            -> typing.List[typing.Sequence[collections.Counter]]:
        """Obtain all histogram objects recorded by this histogram context for a specific key.

        The data is formatted as a list of histograms per channel.
        So to access histogram N of channel C: ``get_histograms()[C][N]``.

        In case no dataset key is provided, the default dataset key is used.

        :param dataset_key: Key of the dataset to obtain the histograms of
        :return: All histogram data for the specified key
        :raises KeyError: Raised if no data is available for the given dataset key
        """
        return list(zip(*self._histogram_cache[self._default_dataset_key if dataset_key is None else dataset_key]))

    @host_only
    def get_probabilities(self, dataset_key: typing.Optional[str] = None,
                          state_detection_threshold: typing.Optional[int] = None) -> typing.List[typing.List[float]]:
        """Obtain all individual state probabilities recorded by this histogram context for a specific key.

        The data is formatted as a list of probabilities per channel.
        So to access probability N of channel C: ``get_probabilities()[C][N]``.

        If measurements were performed using counts, the state detection threshold will be used
        to decide the probability of a state.
        For binary measurements, the state detection threshold is ignored.

        :param dataset_key: Key of the dataset to obtain the probabilities of
        :param state_detection_threshold: State detection threshold used to calculate the probabilities
        :return: All probability data for the specified key
        :raises KeyError: Raised if no data is available for the given dataset key
        """
        return [[self._histogram_to_probability(h, state_detection_threshold) for h in histograms]
                for histograms in self.get_histograms(dataset_key)]

    @host_only
    def get_mean_counts(self, dataset_key: typing.Optional[str] = None) -> typing.List[typing.List[float]]:
        """Obtain all mean counts recorded by this histogram context for a specific key.

        The data is formatted as a list of counts per channel.
        So to access mean count N of channel C: ``get_mean_counts()[C][N]``.

        For binary measurements, the mean count returns a value in the range ``[0..1]``.

        :param dataset_key: Key of the dataset to obtain the mean counts of
        :return: All mean count data for the specified key
        :raises KeyError: Raised if no data is available for the given dataset key
        """
        return [[HistogramAnalyzer.histogram_to_mean_count(h) for h in histograms]
                for histograms in self.get_histograms(dataset_key)]

    @host_only
    def get_stdev_counts(self, dataset_key: typing.Optional[str] = None) -> typing.List[typing.List[float]]:
        """Obtain all standard deviations of counts recorded by this histogram context for a specific key.

        The data is formatted as a list of standard deviations per channel.
        So to access standard deviation N of channel C: ``get_stdev_counts()[C][N]``.

        :param dataset_key: Key of the dataset to obtain the mean counts of
        :return: All mean count data for the specified key
        :raises KeyError: Raised if no data is available for the given dataset key
        """
        return [[HistogramAnalyzer.histogram_to_stdev_count(h) for h in histograms]
                for histograms in self.get_histograms(dataset_key)]


class HistogramAnalyzer:
    """Basic automated analysis and offline plotting of data obtained by the histogram context.

    Various data sources can be provided and presented data should have a uniform format.
    Simple automated plotting functions are provided, but users can also access data directly
    for manual processing and analysis.

    :attr:`keys` is a list of keys for which data is available.

    :attr:`histograms` is a dict which for each key contains a list of histograms per channel.
    The first dimension is the channel and the second dimension are the histograms.
    Note that histograms are stored as ``Counter`` objects, which behave like dicts.

    :attr:`probabilities` is a dict which for each key contains a list of individual state probabilities.
    This attribute is only available if a state detection threshold is available.
    The probabilities are a mapped version of the :attr:`histograms` data.

    :attr:`mean_counts` is a dict which for each key contains a list of mean counts.

    :attr:`stdev_counts` is a dict which for each key contains a list of count standard deviations.

    :attr:`raw` is a 3-dimensional array with raw PMT data.
    *This attribute is only available if raw data was stored (DAX>0.4)*.
    The first dimension is the histogram index, the second dimension the data point,
    and the third dimension the channel number.

    Various helper functions for data processing are also available.
    :func:`histogram_to_probability` converts a single histogram, formatted as a
    ``Counter`` object, to a state probability based on a given state detection threshold.
    :func:`histograms_to_probabilities` maps a list of histograms per channel (2D array of ``Counter`` objects)
    to a list of probabilities per channel based on a given state detection threshold.
    :func:`histogram_to_mean_count` converts a single histogram, formatted as a ``Counter`` object, to a mean count.
    :func:`histograms_to_mean_counts` maps a list of histograms per channel (2D array of ``Counter`` objects)
    to a list of mean counts per channel.
    :func:`histogram_to_stdev_count` converts a single histogram, formatted as a ``Counter`` object,
    to a count standard deviation.
    :func:`histograms_to_stdev_counts` maps a list of histograms per channel (2D array of ``Counter`` objects)
    to a list of count standard deviations per channel.
    :func:`counter_to_ndarray` and :func:`ndarray_to_counter` convert a single histogram
    stored as a ``Counter`` object to an array representation and vice versa.
    :func:`raw_to_states` converts raw data to sequences of integer states based on a given detection threshold.
    :func:`raw_to_state_probabilities` converts raw data to full state probabilities
    based on a given detection threshold.
    """

    HISTOGRAM_PLOT_FILE_FORMAT: typing.ClassVar[str] = '{key}_{index}'
    """File name format for histogram plot files."""
    PROBABILITY_PLOT_FILE_FORMAT: typing.ClassVar[str] = '{key}_probability'
    """File name format for individual state probability plot files."""
    MEAN_COUNT_PLOT_FILE_FORMAT: typing.ClassVar[str] = '{key}_mean_count'
    """File name format for mean count plot files."""
    STATE_PROBABILITY_PLOT_FILE_FORMAT: typing.ClassVar[str] = '{key}_state_probability'
    """File name format for full state probability plot files."""

    def __init__(self, source: typing.Union[DaxSystem, HistogramContext, str, h5py.File],
                 state_detection_threshold: typing.Optional[int] = None, *,
                 hdf5_group: typing.Optional[str] = None):
        """Create a new histogram analyzer object.

        :param source: The source of the histogram data
        :param state_detection_threshold: The state detection threshold used to calculate state probabilities
        :param hdf5_group: HDF5 group containing the data, defaults to root of the HDF5 file
        """
        assert isinstance(state_detection_threshold, int) or state_detection_threshold is None
        assert isinstance(hdf5_group, str) or hdf5_group is None

        # Input conversion
        if isinstance(source, DaxSystem):
            # Obtain histogram context module
            source = source.registry.find_module(HistogramContext)
        elif isinstance(source, str):
            # Open HDF5 file
            source = h5py.File(os.path.expanduser(source), mode='r')

        if isinstance(source, HistogramContext):
            if state_detection_threshold is None:
                # Obtain the state detection threshold
                detection = source.registry.find_interface(DetectionInterface)  # type: ignore[misc]
                self.state_detection_threshold: int = detection.get_state_detection_threshold()
            else:
                # Store provided state detection threshold
                self.state_detection_threshold = state_detection_threshold

            # Get data from histogram context module
            self.keys: typing.List[str] = source.get_keys()
            self.histograms: typing.Dict[str, typing.List[typing.Sequence[typing.Counter[_DATA_T]]]] = \
                {k: source.get_histograms(k) for k in self.keys}
            self.probabilities: typing.Dict[str, np.ndarray] = \
                {k: np.asarray(source.get_probabilities(k, state_detection_threshold)) for k in self.keys}
            self.mean_counts: typing.Dict[str, np.ndarray] = \
                {k: np.asarray(source.get_mean_counts(k)) for k in self.keys}
            self.stdev_counts: typing.Dict[str, np.ndarray] = \
                {k: np.asarray(source.get_stdev_counts(k)) for k in self.keys}
            self.raw: typing.Dict[str, typing.Sequence[np.ndarray]] = \
                {k: [np.asarray(r) for r in source.get_raw(k)] for k in self.keys}

            # Obtain the file name generator
            self._file_name_generator: BaseFileNameGenerator = FileNameGenerator(source.get_device('scheduler'))

        elif isinstance(source, h5py.File):
            # Construct HDF5 group name
            path = [] if hdf5_group is None else [hdf5_group]
            group_name = '/'.join(path + ['datasets', HistogramContext.DATASET_GROUP])
            # Verify format of HDF5 file
            if group_name not in source:
                raise KeyError('The HDF5 file does not contain histogram data')

            # Get the group which contains all data
            group = source[group_name]

            # Read keys
            self.keys = natsort.natsorted(group)

            # Read data from HDF5 file
            if self.keys and HistogramContext.RAW_DATASET_GROUP in group[self.keys[0]]:
                # Raw data available
                self.raw = {k: [np.asarray(group[k][HistogramContext.RAW_DATASET_GROUP][index])
                                for index in natsort.natsorted(group[k][HistogramContext.RAW_DATASET_GROUP])]
                            for k in self.keys}
                # Reconstruct histograms from raw data
                self.histograms = {k: list(zip(*(self.raw_to_histograms(r) for r in raw)))
                                   for k, raw in self.raw.items()}
            else:
                # No raw data available (DAX<0.4), using legacy histogram storage
                histograms = ((k, (group[k][index] for index in natsort.natsorted(group[k]))) for k in self.keys)
                self.histograms = {k: [[self.ndarray_to_counter(values) for values in channel]
                                       for channel in zip(*datasets)] for k, datasets in histograms}

            if state_detection_threshold is not None:
                self.state_detection_threshold = state_detection_threshold  # Store state detection threshold
            try:
                state_detection_threshold = -1 if state_detection_threshold is None else state_detection_threshold
                self.probabilities = {k: self.histograms_to_probabilities(h, state_detection_threshold)
                                      for k, h in self.histograms.items()}
            except TypeError:
                pass  # Could not obtain probabilities without a provided state detection threshold
            self.mean_counts = {k: self.histograms_to_mean_counts(h) for k, h in self.histograms.items()}
            self.stdev_counts = {k: self.histograms_to_stdev_counts(h) for k, h in self.histograms.items()}

            # Get a file name generator
            self._file_name_generator = BaseFileNameGenerator()

        else:
            raise TypeError('Unsupported source type')

    """Helper functions"""

    @classmethod
    def histogram_to_one_count(cls, counter: typing.Counter[_DATA_T], state_detection_threshold: int = -1) -> int:
        """Helper function to count the number of one measurements in a histogram.

        This function works correct for both binary measurements and detection counts.
        For detection counts, counts *greater than* the state detection threshold are considered to be in state one.

        :param counter: The ``Counter`` object representing the histogram
        :param state_detection_threshold: The state detection threshold to use (optional)
        :return: The number of one measurements
        """
        assert isinstance(state_detection_threshold, int), 'State detection threshold must be of type int'

        if state_detection_threshold < 0:
            if not all(isinstance(c, (bool, np.bool_)) for c in counter):
                raise TypeError('All measurements must be binary when no state detection threshold is given')

            # Count the number of one measurements, works only for binary measurements
            return sum(f for c, f in counter.items() if c is True)
        else:
            # Count the number of one measurements, works both for binary measurements and detection counts
            return sum(f for c, f in counter.items() if c is True or c > state_detection_threshold)

    @classmethod
    def histogram_to_probability(cls, counter: typing.Counter[_DATA_T], state_detection_threshold: int = -1) -> float:
        """Helper function to convert a histogram to an individual state probability.

        Counts *greater than* the state detection threshold are considered to be in state one.

        :param counter: The ``Counter`` object representing the histogram
        :param state_detection_threshold: The state detection threshold to use (optional)
        :return: The state probability as a float
        """
        # Obtain number of one measurements
        one = cls.histogram_to_one_count(counter, state_detection_threshold=state_detection_threshold)
        # Total number of measurements
        total = sum(counter.values())
        # Return probability
        return one / total

    @classmethod
    def histograms_to_probabilities(cls, histograms: typing.Sequence[typing.Sequence[collections.Counter]],
                                    state_detection_threshold: int = -1) -> np.ndarray:
        """Convert histograms to individual state probabilities based on a state detection threshold.

        Histograms are provided as a 2D array of ``Counter`` objects.
        The first dimension is the channel, the second dimension is the sequence of counters.

        :param histograms: The input histograms
        :param state_detection_threshold: The state detection threshold to use (optional)
        :return: Array of probabilities with the same shape as the input histograms
        """
        # Calculate the probabilities
        probabilities = [
            [cls.histogram_to_probability(h, state_detection_threshold=state_detection_threshold) for h in channel]
            for channel in histograms
        ]
        # Return the probabilities as an ndarray
        return np.asarray(probabilities)

    @classmethod
    def _histogram_to_mean_count(cls, counter: typing.Counter[_DATA_T]) -> typing.Tuple[float, int]:
        """Helper function to calculate the mean count of a histogram.

        :param counter: The ``Counter`` object representing the histogram
        :return: A tuple with the mean count as a float and the total number of samples as an int
        """
        num_samples = sum(counter.values())
        mean = sum(c * v for c, v in counter.items()) / num_samples
        return mean, num_samples

    @classmethod
    def histogram_to_mean_count(cls, counter: typing.Counter[_DATA_T]) -> float:
        """Helper function to calculate the mean count of a histogram.

        :param counter: The ``Counter`` object representing the histogram
        :return: The mean count as a float
        """
        mean, _ = cls._histogram_to_mean_count(counter)
        return mean

    @classmethod
    def histograms_to_mean_counts(cls, histograms: typing.Sequence[typing.Sequence[collections.Counter]]) -> np.ndarray:
        """Convert histograms to mean counts.

        Histograms are provided as a 2D array of ``Counter`` objects.
        The first dimension is the channel, the second dimension is the sequence of counters.

        :param histograms: The input histograms
        :return: Array of counts with the same shape as the input histograms
        """
        counts = [[cls.histogram_to_mean_count(h) for h in channel] for channel in histograms]
        return np.asarray(counts)

    @classmethod
    def histogram_to_mean_stdev_count(cls, counter: typing.Counter[_DATA_T]) -> typing.Tuple[float, float]:
        """Helper function to calculate the count mean and standard deviation of a histogram.

        This helper function is more efficient than calculating mean and standard deviation separately.
        It is mainly intended to be used for real-time data processing in :class:`HistogramContext`.

        :param counter: The ``Counter`` object representing the histogram
        :return: The mean and standard deviation of the histogram as a tuple of floats
        """
        mean, num_samples = cls._histogram_to_mean_count(counter)
        squared_mean = sum(c ** 2 * v for c, v in counter.items()) / num_samples
        return mean, math.sqrt(squared_mean - mean ** 2)

    @classmethod
    def histogram_to_stdev_count(cls, counter: typing.Counter[_DATA_T]) -> float:
        """Helper function to calculate the count standard deviation of a histogram.

        :param counter: The ``Counter`` object representing the histogram
        :return: The standard deviation of the histogram as a float
        """
        _, stdev = cls.histogram_to_mean_stdev_count(counter)
        return stdev

    @classmethod
    def histograms_to_stdev_counts(cls, histograms: typing.Sequence[typing.Sequence[collections.Counter]]) \
            -> np.ndarray:
        """Convert histograms to count standard deviations.

        Histograms are provided as a 2D array of ``Counter`` objects.
        The first dimension is the channel, the second dimension is the sequence of counters.

        :param histograms: The input histograms
        :return: Array of standard deviations with the same shape as the input histograms
        """
        counts = [[cls.histogram_to_stdev_count(h) for h in channel] for channel in histograms]
        return np.asarray(counts)

    @classmethod
    def counter_to_ndarray(cls, histogram: typing.Counter[_DATA_T], *,
                           max_count: typing.Optional[int] = None) -> np.ndarray:
        """Convert a histogram stored as a ``Counter`` object to an ndarray.

        Note that histograms with binary measurement results will be converted to an array with length 2.
        Hence, the information that the histogram only contains binary measurements is lost.

        :param histogram: The histogram in ``Counter`` format
        :param max_count: The maximum count the array should be able to store (optional)
        :return: ndarray that represents the same histogram
        """
        assert isinstance(max_count, int) or max_count is None, 'Max must be of type int or None'

        if max_count is None:
            # Choose length such that the highest count in the histogram fits
            max_count = max(histogram)

        return np.asarray([histogram[i] for i in range(max_count + 1)])

    @classmethod
    def ndarray_to_counter(cls, histogram: typing.Sequence[int]) -> collections.Counter:
        """Convert a histogram stored as an ndarray to a ``Counter`` object.

        Note that it is not possible to determine if arrays only contain binary measurement results,
        this information is lost when converting the counter to an array.
        See also :func:`counter_to_ndarray`.

        :param histogram: The histogram in ndarray format
        :return: ``Counter`` object that represents the same histogram
        """
        return collections.Counter({i: v for i, v in enumerate(histogram) if v > 0})

    @classmethod
    def raw_to_histograms(cls, raw: typing.Sequence[typing.Sequence[_DATA_T]]) -> typing.List[typing.Counter[_DATA_T]]:
        """Convert raw data to a histogram per channel.

        :param raw: The raw data to process, one buffer of data
        :return: A list of ``Counter`` objects that represent the histogram for each channel
        """
        return [collections.Counter(c) for c in zip(*raw)]

    @classmethod
    def _vector_to_int(cls, vector: typing.Sequence[_DATA_T], state_detection_threshold: int) -> int:
        """Convert a vector of raw counts to an integer state."""

        # Accumulated result
        acc: int = 0

        for count in reversed(vector):
            # Shift accumulator
            acc <<= 1
            # Add bit
            acc |= count is True or count > state_detection_threshold

        # Return the accumulated result
        return acc

    @classmethod
    def raw_to_states(cls, raw: typing.Sequence[typing.Sequence[typing.Sequence[_DATA_T]]],
                      state_detection_threshold: int) -> typing.List[typing.List[int]]:
        """Convert raw data to integer states.

        :param raw: The raw data to process
        :param state_detection_threshold: The state detection threshold to use
        :return: A 2-dimensional list with integer states (number of histograms * number of points)
        """
        assert isinstance(state_detection_threshold, int), 'State detection threshold must be of type int'

        # Return the converted result
        return [[cls._vector_to_int(point, state_detection_threshold) for point in histogram] for histogram in raw]

    @classmethod
    def _states_to_probabilities(cls, states: typing.Sequence[int]) -> typing.Dict[int, float]:
        """Convert a sequence of integer states to a dictionary with state probabilities."""

        # Reduce using a counter
        counter = collections.Counter(states)
        # The total number of measured states
        total = len(states)
        # Convert counts to state probabilities
        return {k: v / total for k, v in counter.items()}

    @classmethod
    def raw_to_state_probabilities(cls, raw: typing.Sequence[typing.Sequence[typing.Sequence[_DATA_T]]],
                                   state_detection_threshold: int) -> typing.List[typing.Dict[int, float]]:
        """Convert raw data into full state probabilities.

        :param raw: The raw data to process
        :param state_detection_threshold: The state detection threshold to use
        :return: A list of sparse dictionaries where each dictionary contains integer states and their probability
        """

        # Return the converted result
        return [cls._states_to_probabilities(states) for states in cls.raw_to_states(raw, state_detection_threshold)]

    @classmethod
    def raw_to_flat_state_probabilities(cls, raw: typing.Sequence[typing.Sequence[typing.Sequence[_DATA_T]]],
                                        state_detection_threshold: int) -> typing.List[typing.List[float]]:
        """Convert raw data into flattened full state probabilities.

        :param raw: The raw data to process
        :param state_detection_threshold: The state detection threshold to use
        :return: A 2-dimensional list containing full state probability data (iteration, integer state)
        """

        # Get the state probabilities as dicts
        state_probabilities = cls.raw_to_state_probabilities(raw, state_detection_threshold=state_detection_threshold)

        try:
            # Obtain the number of bits and states (assumes there is at least one measurement)
            num_bits = len(raw[0][0])
        except IndexError:
            # No data, return empty list
            return []
        else:
            # Calculate the number of states
            num_states = 2 ** num_bits
            # Flatten data and return result
            return [[p.get(i, 0.0) for i in range(num_states)] for p in state_probabilities]

    """Plotting functions"""

    def plot_histogram(self, key: str, *,
                       x_label: typing.Optional[str] = 'Count',
                       y_label: typing.Optional[str] = 'Frequency',
                       labels: typing.Optional[typing.Sequence[str]] = None,
                       width: float = 1.0,
                       legend_loc: typing.Optional[typing.Union[str, typing.Tuple[float, float]]] = None,
                       fig_size: typing.Optional[typing.Tuple[float, float]] = None,
                       ext: str = 'pdf',
                       **kwargs: typing.Any) -> None:
        """Plot the histograms for a given key.

        :param key: The key of the data to plot
        :param x_label: X-axis label
        :param y_label: Y-axis label
        :param labels: List of plot labels
        :param width: Total width of all bars
        :param legend_loc: Location of the legend
        :param fig_size: The figure size
        :param ext: Output file extension
        :param kwargs: Keyword arguments for the plot function
        """
        assert isinstance(key, str)
        assert isinstance(x_label, str) or x_label is None
        assert isinstance(y_label, str) or y_label is None
        assert isinstance(labels, collections.abc.Sequence) or labels is None
        assert isinstance(width, float)
        assert isinstance(ext, str)

        # Lazy import
        import matplotlib.pyplot as plt
        import matplotlib.ticker

        if fig_size is None:
            # Make the default width of the figure wider
            fig_w, fig_h = plt.rcParams.get('figure.figsize')
            fig_size = (fig_w * 2, fig_h)

        # Get the histograms associated with the given key
        histograms = self.histograms[key]

        # Create figure
        fig, ax = plt.subplots(figsize=fig_size)

        for index, h in enumerate(zip(*histograms)):
            # Obtain X and Y values (for all channels)
            x_values = np.arange(max(max(c) for c in h) + 1)
            y_values = [[c[x] for x in x_values] for c in h]

            # Current labels
            current_labels = [f'Plot {i}' for i in range(len(y_values))] if labels is None else labels
            if len(current_labels) < len(y_values):
                # Not enough labels
                raise IndexError('Number of labels is less than the number of plots')

            # Plot
            ax.cla()  # Clear axes
            bar_width = width / len(h)
            for i, (y, label) in enumerate(zip(y_values, current_labels)):
                ax.bar(x_values + (bar_width * i) - (width / 2), y,
                       width=bar_width, align='edge', label=label, **kwargs)

            # Formatting
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))  # Only integer ticks
            ax.legend(loc=legend_loc)

            # Save figure
            file_name = self._file_name_generator(self.HISTOGRAM_PLOT_FILE_FORMAT.format(key=key, index=index), ext)
            fig.savefig(file_name, bbox_inches='tight')

        # Close the figure
        plt.close(fig)

    def plot_all_histograms(self, **kwargs: typing.Any) -> None:
        """Plot histograms for all keys available in the data.

        :param kwargs: Keyword arguments passed to :func:`plot_histogram`
        """
        for key in self.keys:
            self.plot_histogram(key, **kwargs)

    def plot_probability(self, key: str, *,
                         x_values: typing.Optional[typing.Sequence[typing.Union[float, int]]] = None,
                         x_label: typing.Optional[str] = None,
                         y_label: typing.Optional[str] = 'State probability',
                         labels: typing.Optional[typing.Sequence[str]] = None,
                         legend_loc: typing.Optional[typing.Union[str, typing.Tuple[float, float]]] = None,
                         fig_size: typing.Optional[typing.Tuple[float, float]] = None,
                         ext: str = 'pdf',
                         **kwargs: typing.Any) -> None:
        """Plot the individual state probability graph for a given key.

        In the individual state probability graph, states are plotted independently for each qubit.
        For a full state probability graph, see :func:`plot_state_probability`.

        Note that if the data points are randomized the user should provide X values
        to sort the points and plot the graph correctly.

        :param key: The key of the data to plot
        :param x_values: The sequence with X values for the graph
        :param x_label: X-axis label
        :param y_label: Y-axis label
        :param labels: List of plot labels
        :param legend_loc: Location of the legend
        :param fig_size: The figure size
        :param ext: Output file extension
        :param kwargs: Keyword arguments for the plot function
        """
        assert isinstance(key, str)
        assert isinstance(x_values, collections.abc.Sequence) or x_values is None
        assert isinstance(x_label, str) or x_label is None
        assert isinstance(y_label, str) or y_label is None
        assert isinstance(labels, collections.abc.Sequence) or labels is None
        assert isinstance(ext, str)
        assert hasattr(self, 'probabilities'), \
            'Probability data not available, probably because no state detection threshold was provided'

        # Get the probabilities associated with the provided key
        probabilities = [np.asarray(p) for p in self.probabilities[key]]

        if not len(probabilities):
            # No data to plot
            return

        if x_values is None:
            # Generate generic X values
            x_values = np.arange(len(probabilities[0]))
        else:
            # Sort data based on the given x values
            x_values = np.asarray(x_values)
            ind = x_values.argsort()
            x_values = x_values[ind]
            probabilities = [p[ind] for p in probabilities]

        # Current labels
        current_labels = [f'Plot {i}' for i in range(len(probabilities))] if labels is None else labels
        if len(current_labels) < len(probabilities):
            # Not enough labels
            raise IndexError('Number of labels is less than the number of plots')

        # Plotting defaults
        kwargs.setdefault('marker', 'o')
        kwargs.setdefault('linestyle', '')

        # Lazy import
        import matplotlib.pyplot as plt

        # Plot
        fig, ax = plt.subplots(figsize=fig_size)
        for y, label in zip(probabilities, current_labels):
            ax.plot(x_values, y, label=label, **kwargs)

        # Plot formatting
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.ticklabel_format(axis='x', scilimits=(0, 1))
        ax.legend(loc=legend_loc)

        # Save and close figure
        file_name = self._file_name_generator(self.PROBABILITY_PLOT_FILE_FORMAT.format(key=key), ext)
        fig.savefig(file_name, bbox_inches='tight')
        plt.close(fig)

    def plot_all_probabilities(self, **kwargs: typing.Any) -> None:
        """Plot individual state probability graphs for all keys available in the data.

        In individual state probability graphs, states are plotted independently for each qubit.
        For full state probability graphs, see :func:`plot_all_state_probabilities`.

        Note that if the data points are randomized the user should provide X values
        to sort the points and plot the graph correctly (``x_values`` kwarg).

        :param kwargs: Keyword arguments passed to :func:`plot_probability`
        """
        for key in self.keys:
            self.plot_probability(key, **kwargs)

    def plot_mean_count(self, key: str, *,
                        x_values: typing.Optional[typing.Sequence[typing.Union[float, int]]] = None,
                        x_label: typing.Optional[str] = None,
                        y_label: typing.Optional[str] = 'Mean count',
                        labels: typing.Optional[typing.Sequence[str]] = None,
                        legend_loc: typing.Optional[typing.Union[str, typing.Tuple[float, float]]] = None,
                        fig_size: typing.Optional[typing.Tuple[float, float]] = None,
                        ext: str = 'pdf',
                        **kwargs: typing.Any) -> None:
        """Plot the mean count graph for a given key.

        Note that if the data points are randomized the user should provide X values
        to sort the points and plot the graph correctly.

        :param key: The key of the data to plot
        :param x_values: The sequence with X values for the graph
        :param x_label: X-axis label
        :param y_label: Y-axis label
        :param labels: List of plot labels
        :param legend_loc: Location of the legend
        :param fig_size: The figure size
        :param ext: Output file extension
        :param kwargs: Keyword arguments for the plot function
        """
        assert isinstance(key, str)
        assert isinstance(x_values, collections.abc.Sequence) or x_values is None
        assert isinstance(x_label, str) or x_label is None
        assert isinstance(y_label, str) or y_label is None
        assert isinstance(labels, collections.abc.Sequence) or labels is None
        assert isinstance(ext, str)

        # Get the data associated with the provided key
        mean_counts = [np.asarray(p) for p in self.mean_counts[key]]
        stdev_counts = [np.asarray(p) for p in self.stdev_counts[key]]

        if not len(mean_counts):
            # No data to plot
            return

        if x_values is None:
            # Generate generic X values
            x_values = np.arange(len(mean_counts[0]))
        else:
            # Sort data based on the given x values
            x_values = np.asarray(x_values)
            ind = x_values.argsort()
            x_values = x_values[ind]
            mean_counts = [p[ind] for p in mean_counts]
            stdev_counts = [p[ind] for p in stdev_counts]

        # Current labels
        current_labels = [f'Plot {i}' for i in range(len(mean_counts))] if labels is None else labels
        if len(current_labels) < len(mean_counts):
            # Not enough labels
            raise IndexError('Number of labels is less than the number of plots')

        # Plotting defaults
        kwargs.setdefault('marker', 'o')
        kwargs.setdefault('linestyle', '')

        # Lazy import
        import matplotlib.pyplot as plt

        # Plot
        fig, ax = plt.subplots(figsize=fig_size)
        for y, stdev, label in zip(mean_counts, stdev_counts, current_labels):
            ax.errorbar(x_values, y, yerr=stdev, label=label, **kwargs)

        # Plot formatting
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.ticklabel_format(axis='x', scilimits=(0, 1))
        ax.legend(loc=legend_loc)

        # Save and close figure
        file_name = self._file_name_generator(self.MEAN_COUNT_PLOT_FILE_FORMAT.format(key=key), ext)
        fig.savefig(file_name, bbox_inches='tight')
        plt.close(fig)

    def plot_all_mean_counts(self, **kwargs: typing.Any) -> None:
        """Plot mean count graphs for all keys available in the data.

        Note that if the data points are randomized the user should provide X values
        to sort the points and plot the graph correctly (``x_values`` kwarg).

        :param kwargs: Keyword arguments passed to :func:`plot_mean_count`
        """
        for key in self.keys:
            self.plot_mean_count(key, **kwargs)

    def plot_state_probability(self, key: str, *,
                               x_values: typing.Optional[typing.Sequence[typing.Union[float, int]]] = None,
                               x_label: typing.Optional[str] = None,
                               y_label: typing.Optional[str] = '|State> probability',
                               labels: typing.Optional[typing.Sequence[str]] = None,
                               legend_loc: typing.Optional[typing.Union[str, typing.Tuple[float, float]]] = None,
                               fig_size: typing.Optional[typing.Tuple[float, float]] = None,
                               ext: str = 'pdf',
                               **kwargs: typing.Any) -> None:
        """Plot the full state probability graph for a given key.

        In the full state probability graph, states are plotted as full system qubit states.
        For an individual state probability graph, see :func:`plot_probability`.

        Note that if the data points are randomized the user should provide X values
        to sort the points and plot the graph correctly.

        :param key: The key of the data to plot
        :param x_values: The sequence with X values for the graph
        :param x_label: X-axis label
        :param y_label: Y-axis label
        :param labels: List of plot labels
        :param legend_loc: Location of the legend
        :param fig_size: The figure size
        :param ext: Output file extension
        :param kwargs: Keyword arguments for the plot function
        """
        assert isinstance(key, str)
        assert isinstance(x_values, collections.abc.Sequence) or x_values is None
        assert isinstance(x_label, str) or x_label is None
        assert isinstance(y_label, str) or y_label is None
        assert isinstance(labels, collections.abc.Sequence) or labels is None
        assert isinstance(ext, str)
        assert hasattr(self, 'raw'), 'Provided data source does not contain required raw data (DAX<0.4)'

        # Get the state probabilities associated with the provided key
        state_probabilities = np.asarray(self.raw_to_flat_state_probabilities(
            self.raw[key], state_detection_threshold=self.state_detection_threshold))

        if not state_probabilities.size:
            # No data to plot
            return

        # Obtain the number of bits and states (assumes there is at least one measurement)
        num_bits = len(self.raw[key][0][0])
        num_states = state_probabilities.shape[1]
        assert 2 ** num_bits == num_states, 'Data transformation error, array sizes do not align'

        if x_values is None:
            # Generate generic X values
            x_values = np.arange(len(state_probabilities))
        else:
            # Sort data based on the given x values
            x_values = np.asarray(x_values)
            ind = x_values.argsort()
            x_values = x_values[ind]
            state_probabilities = state_probabilities[ind]

        # Transpose data for plotting
        y_data = state_probabilities.transpose()

        # Current labels
        current_labels = [f'|{i:0{num_bits}b}>' for i in range(num_states)] if labels is None else labels
        if len(current_labels) < num_states:
            # Not enough labels
            raise IndexError('Number of labels is less than the number of plots')

        # Plotting defaults
        kwargs.setdefault('marker', 'o')
        kwargs.setdefault('linestyle', '')

        # Lazy import
        import matplotlib.pyplot as plt

        # Plot
        fig, ax = plt.subplots(figsize=fig_size)
        for y, label in zip(y_data, current_labels):
            ax.plot(x_values, y, label=label, **kwargs)

        # Plot formatting
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.ticklabel_format(axis='x', scilimits=(0, 1))
        ax.legend(loc=legend_loc)

        # Save and close figure
        file_name = self._file_name_generator(self.STATE_PROBABILITY_PLOT_FILE_FORMAT.format(key=key), ext)
        fig.savefig(file_name, bbox_inches='tight')
        plt.close(fig)

    def plot_all_state_probabilities(self, **kwargs: typing.Any) -> None:
        """Plot full state probability graphs for all keys available in the data.

        Note that if the data points are randomized the user should provide X values
        to sort the points and plot the graph correctly (``x_values`` kwarg).

        :param kwargs: Keyword arguments passed to :func:`plot_state_probability`
        """
        for key in self.keys:
            self.plot_state_probability(key, **kwargs)
