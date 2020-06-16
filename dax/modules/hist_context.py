import typing
import collections
import numpy as np
import h5py  # type: ignore
import natsort
import os

import matplotlib.pyplot as plt  # type: ignore
import matplotlib.ticker  # type: ignore

from dax.experiment import *
from dax.interfaces.detection import DetectionInterface
from dax.util.ccb import get_ccb_tool
from dax.util.output import get_file_name_generator, dummy_file_name_generator
from dax.util.units import UnitsFormatter

__all__ = ['HistogramContext', 'HistogramAnalyzer', 'HistogramContextError']


class HistogramContext(DaxModule):
    """Context class for managing storage of PMT histogram data.

    This module can be used as a sub-module of a service providing state measurement abilities.
    The HistogramContext object can directly be passed to the user which can use it as a context
    or call its additional functions.

    Note that the histogram context requires a :class:`DetectionInterface` in your system.

    The histogram context objects manages all result values, but the user is responsible for tracking
    "input parameters".
    """

    HISTOGRAM_PLOT_KEY = 'plot.dax.histogram_context.histogram'
    """Dataset name for plotting latest histogram."""
    HISTOGRAM_PLOT_NAME = 'histogram'
    """Name of the histogram plot applet."""

    PROBABILITY_PLOT_KEY = 'plot.dax.histogram_context.probability'
    """Dataset name for plotting latest probability graph."""
    PROBABILITY_PLOT_NAME = 'probability'
    """Name of the probability plot applet."""

    PLOT_GROUP = 'dax.histogram_context'
    """Group to which the plot applets belong."""

    DATASET_GROUP = 'histogram_context'
    """The group name for archiving data."""
    DATASET_KEY_FORMAT = DATASET_GROUP + '/{dataset_key:s}/{index:d}'
    """Format string for sub-dataset keys."""
    DEFAULT_DATASET_KEY = 'histogram'
    """The default name of the output sub-dataset."""

    def build(self, default_dataset_key: typing.Optional[str] = None) -> None:  # type: ignore
        """Build the histogram context module.

        :param default_dataset_key: Default dataset name used for storing histogram data
        """
        assert isinstance(default_dataset_key, str) or default_dataset_key is None, \
            'Provided default dataset key must be None or of type str'

        # Store default dataset key
        self._default_dataset_key = self.DEFAULT_DATASET_KEY if default_dataset_key is None else default_dataset_key

        # Get CCB tool
        self._ccb = get_ccb_tool(self)
        # Units formatter
        self._units_fmt = UnitsFormatter()

        # By default we are not in context
        self._in_context = np.int32(0)
        # The count buffer (buffer appending is a bit faster than dict operations)
        self._buffer = []  # type: typing.List[typing.Sequence[int]]

        # Archive to analyze high level data at the end of the experiment
        self._histogram_archive = {}  # type: typing.Dict[str, typing.List[typing.Sequence[collections.Counter]]]

        # Target dataset key
        self._dataset_key = self._default_dataset_key
        # Datasets that are initialized with a counter, which represents the length of the data
        self._open_datasets = collections.Counter()  # type: typing.Dict[str, int]

    def init(self) -> None:
        # Prepare the probability plot dataset by clearing it
        self.clear_probability_plot()

    def post_init(self) -> None:
        # Obtain the state detection threshold
        detection = self.registry.find_interface(DetectionInterface)  # type: ignore[misc]
        self._state_detection_threshold = detection.get_state_detection_threshold()
        self.update_kernel_invariants('_state_detection_threshold')

    """Data handling functions"""

    @portable
    def in_context(self) -> bool:
        """True if we are in context."""
        return bool(self._in_context)

    @rpc(flags={'async'})
    def append(self, data):  # type: (typing.Sequence[int]) -> None
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
        Use `None` to reset the dataset base key to its default value.

        Within ARTIQ kernels it is not possible to use string formatting functions.
        Instead, the key can be a string that includes formatting annotations while
        formatting parameters can be provided as positional and keyword arguments.
        The formatting function will be called on the host.

        The formatter uses an extended format and it is possible to convert float values
        to human-readable format using conversion flags `{!t}` and `{!f}`.
        Note that the formatter has the default precision of 6 digits which is not likely
        to generate unique keys. An other field can be added to make sure the keys are unique.

        This function can not be used when already in context.

        :param key: Key for the result dataset using standard Python formatting notation
        :param args: Python `str.format()` positional arguments
        :param kwargs: Python `str.format()` keyword arguments
        :raises HistogramContextError: Raised if called inside the histogram context
        """
        assert isinstance(key, str) or key is None, 'Provided dataset key must be of type str or None'

        if self._in_context:
            # Called in context
            raise HistogramContextError('Setting the target dataset can only be done when not in context')

        # Update the dataset key
        self._dataset_key = self._default_dataset_key if key is None else self._units_fmt.vformat(key, args, kwargs)

    @portable
    def __enter__(self):  # type: () -> None
        """Enter the histogram context.

        Entering the histogram context will prepare the target dataset and clear the buffer.
        Optionally, this context can be configured using the :func:`config` function before entering the context.
        """
        self.open()

    @portable
    def __exit__(self, exc_type, exc_val, exc_tb):  # type: (typing.Any, typing.Any, typing.Any) -> None
        """Exit the histogram context."""
        self.close()

    @rpc(flags={'async'})
    def open(self):  # type: () -> None
        """Enter the histogram context manually.

        Optionally, this context can be configured using the :func:`config` function.

        This function can be used to manually enter the histogram context.
        We strongly recommend to use the `with` statement instead.

        :raises HistogramContextError: Raised if already in histogram context (context is non-reentrant)
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
        """Exit the histogram context manually.

        This function can be used to manually exit the histogram context.
        We strongly recommend to use the `with` statement instead.

        :raises HistogramContextError: Raised if called outside the histogram context
        """

        if not self._in_context:
            # Called exit out of context
            raise HistogramContextError('The exit function can only be called from inside the histogram context')

        # Create a sub-dataset key for this result (HDF5 only supports fixed size elements in a list)
        sub_dataset_key = self.DATASET_KEY_FORMAT.format(dataset_key=self._dataset_key,
                                                         index=self._open_datasets[self._dataset_key])

        if len(self._buffer):
            # Check consistency of data in the buffer
            if any(len(b) != len(self._buffer[0]) for b in self._buffer):
                raise RuntimeError('Data in the buffer is not consistent, data probably corrupt')

            # Transform buffer data to pack counts per ion and convert into histograms
            histograms = [collections.Counter(c) for c in zip(*self._buffer)]
            # Store histograms in the archive
            self._histogram_archive.setdefault(self._dataset_key, []).append(histograms)

            # Obtain maximum count over all histograms (HDF5 only supports fixed size arrays)
            max_count = max(max(h) for h in histograms)
            # Flatten dict-like histograms to same-size list-style histograms (HDF5 does not support mapping types)
            flat_histograms = [[h[i] for i in range(max_count + 1)] for h in histograms]

            # Write result to sub-dataset for archiving
            self.set_dataset(sub_dataset_key, flat_histograms, archive=True)
            # Write result to histogram plotting dataset
            self.set_dataset(self.HISTOGRAM_PLOT_KEY, flat_histograms, broadcast=True, archive=False)

            # Calculate state probabilities
            probabilities = [self._histogram_to_probability(h) for h in histograms]
            # Append result to probability plotting dataset
            self.append_to_dataset(self.PROBABILITY_PLOT_KEY, probabilities)

        else:
            # Add empty element to the archive (keeps indexing consistent)
            self._histogram_archive.setdefault(self._dataset_key, []).append([])
            # Write empty element to sub-dataset for archiving (keeps indexing consistent)
            self.set_dataset(sub_dataset_key, [], archive=True)

        # Update counter for this dataset key
        self._open_datasets[self._dataset_key] += 1
        # Update context counter
        self._in_context -= 1

    def _histogram_to_probability(self, counter: collections.Counter,
                                  state_detection_threshold: typing.Optional[int] = None) -> float:
        """Convert a histogram to a state probability.

        Falls back on default state detection threshold if none is given.
        """

        if state_detection_threshold is None:
            # Use default state_detection_threshold if not set
            state_detection_threshold = self._state_detection_threshold

        return self.histogram_to_probability(counter, state_detection_threshold)

    @staticmethod
    def histogram_to_probability(counter: collections.Counter, state_detection_threshold: int) -> float:
        """Helper function to convert a histogram to a state probability."""

        # One measurements (recognizes binary measurements and counts)
        one = sum(f for c, f in counter.items() if c is True or c > state_detection_threshold)
        # Total measurements
        total = sum(counter.values())
        # Return probability
        return one / total

    """Applet plotting functions"""

    @rpc(flags={'async'})
    def plot_histogram(self, **kwargs):  # type: (typing.Any) -> None
        """Open the applet that shows a plot of the latest histogram.

        :param kwargs: Extra keyword arguments for the plot
        """

        # Set default arguments
        kwargs.setdefault('x_label', 'Number of counts')
        kwargs.setdefault('y_label', 'Frequency')
        # Plot
        self._ccb.plot_hist(self.HISTOGRAM_PLOT_NAME, self.HISTOGRAM_PLOT_KEY, group=self.PLOT_GROUP, **kwargs)

    @rpc(flags={'async'})
    def plot_probability(self, **kwargs):  # type: (typing.Any) -> None
        """Open the applet that shows a plot of state probabilities over multiple histograms.

        :param kwargs: Extra keyword arguments for the plot
        """

        # Set default label
        kwargs.setdefault('y_label', 'State probability')
        # Plot
        self._ccb.plot_xy_multi(self.PROBABILITY_PLOT_NAME, self.PROBABILITY_PLOT_KEY, group=self.PLOT_GROUP, **kwargs)

    @rpc(flags={'async'})
    def clear_probability_plot(self):  # type: () -> None
        """Clear the probability plot."""
        # Set the probability dataset to an empty list
        self.set_dataset(self.PROBABILITY_PLOT_KEY, [], broadcast=True, archive=False)

    @rpc(flags={'async'})
    def disable_histogram_plot(self):  # type: () -> None
        """Close the histogram plot."""
        self._ccb.disable_applet(self.HISTOGRAM_PLOT_NAME, self.PLOT_GROUP)

    @rpc(flags={'async'})
    def disable_probability_plot(self):  # type: () -> None
        """Close the probability plot."""
        self._ccb.disable_applet(self.PROBABILITY_PLOT_NAME, self.PLOT_GROUP)

    @rpc(flags={'async'})
    def disable_all_plots(self):  # type: () -> None
        """Close all histogram context plots."""
        self._ccb.disable_applet_group(self.PLOT_GROUP)

    """Data access functions"""

    @host_only
    def get_keys(self) -> typing.List[str]:
        """Get the keys for which histogram data was recorded.

        The returned keys can be used for the :func:`get_histograms` and :func:`get_probabilities` functions.

        :return: A list with keys
        """
        return list(self._histogram_archive)

    @host_only
    def get_histograms(self, dataset_key: typing.Optional[str] = None) \
            -> typing.List[typing.Sequence[collections.Counter]]:
        """Obtain all histogram objects recorded by this histogram context for a specific key.

        The data is formatted as a list of histograms per channel.
        So to access histogram N of channel C: `get_histograms()[C][N]`.

        In case no dataset key is provided, the default dataset key is used.

        :param dataset_key: Key of the dataset to obtain the histograms of
        :return: All histogram data for the specified key
        """
        return list(zip(*self._histogram_archive[self._default_dataset_key if dataset_key is None else dataset_key]))

    @host_only
    def get_probabilities(self, dataset_key: typing.Optional[str] = None,
                          state_detection_threshold: typing.Optional[int] = None) -> typing.List[typing.List[float]]:
        """Obtain all state probabilities recorded by this histogram context.

        The data is formatted as a list of probabilities per channel.
        So to access probability N of channel C: `get_probabilities()[C][N]`.

        If measurements were performed using counts, the state detection threshold will be used
        to decide the probability of a state.
        For binary measurements, the state detection threshold is ignored.

        :param dataset_key: Key of the dataset to obtain the probabilities of
        :param state_detection_threshold: State detection threshold used to calculate the probabilities
        :return: All probability data for the specified key
        """
        return [[self._histogram_to_probability(h, state_detection_threshold) for h in histograms]
                for histograms in self.get_histograms(dataset_key)]


class HistogramAnalyzer:
    """Basic automated analysis and offline plotting of data obtained by the histogram context."""

    HISTOGRAM_PLOT_FILE_FORMAT = '{key:s}_{index:d}'
    """File name format for histogram plot files."""
    PROBABILITY_PLOT_FILE_FORMAT = '{key:s}_probability'
    """File name format for probability plot files."""

    def __init__(self, source: typing.Union[DaxSystem, HistogramContext, str, h5py.File],
                 state_detection_threshold: typing.Optional[int] = None):
        """Create a new histogram analyzer object.

        :param source: The source of the histogram data
        :param state_detection_threshold: The state detection threshold used to calculate state probabilities
        """
        assert isinstance(state_detection_threshold, int) or state_detection_threshold is None

        # Input conversion
        if isinstance(source, DaxSystem):
            # Obtain histogram context module
            source = source.registry.find_module(HistogramContext)
        elif isinstance(source, str):
            # Open HDF5 file
            source = h5py.File(os.path.expanduser(source), mode='r')

        if isinstance(source, HistogramContext):
            # Get data from histogram context module
            self.keys = source.get_keys()
            self.histograms = {k: source.get_histograms(k) for k in self.keys}
            self.probabilities = {k: np.asarray(source.get_probabilities(k, state_detection_threshold))
                                  for k in self.keys}

            # Obtain the file name generator
            self._file_name_generator = get_file_name_generator(source.get_device('scheduler'))

        elif isinstance(source, h5py.File):
            # Verify format of HDF5 file
            group_name = 'datasets/' + HistogramContext.DATASET_GROUP
            if group_name not in source:
                raise KeyError('The HDF5 file does not contain histogram data')

            # Get the group which contains all data
            group = source[group_name]

            # Read and convert data from HDF5 file
            self.keys = list(group)
            histograms = ((k, (group[k][index] for index in natsort.natsorted(group[k]))) for k in self.keys)
            self.histograms = {k: [[self.ndarray_to_counter(values) for values in channel]
                                   for channel in zip(*datasets)] for k, datasets in histograms}
            if state_detection_threshold is not None:
                self.probabilities = {k: np.asarray(self._get_probabilities(h, state_detection_threshold))
                                      for k, h in self.histograms.items()}

            # Get a file name generator
            self._file_name_generator = dummy_file_name_generator

        else:
            raise TypeError('Unsupported source type')

    """Helper functions"""

    @staticmethod
    def _get_probabilities(histograms: typing.Sequence[typing.Sequence[collections.Counter]],
                           state_detection_threshold: int) -> typing.List[typing.List[float]]:
        """Convert a sequence of histograms to a sequence of probabilities."""
        return [[HistogramContext.histogram_to_probability(h, state_detection_threshold) for h in channel]
                for channel in histograms]

    @staticmethod
    def counter_to_ndarray(histogram: collections.Counter) -> np.ndarray:
        """Convert a histogram stored as a Counter object to an ndarray.

        :param histogram: The histogram in Counter format
        :return: ndarray that represents the same histogram
        """
        return np.asarray([histogram[i] for i in range(max(histogram) + 1)])

    @staticmethod
    def ndarray_to_counter(histogram: typing.Sequence[int]) -> collections.Counter:
        """Convert a histogram stored as an ndarray to a Counter object.

        :param histogram: The histogram in ndarray format
        :return: Counter object that represents the same histogram
        """
        return collections.Counter({i: v for i, v in enumerate(histogram) if v > 0})

    """Plotting functions"""

    def plot_histogram(self, key: str,
                       x_label: typing.Optional[str] = 'Count', y_label: typing.Optional[str] = 'Frequency',
                       width: float = 0.8,
                       legend_loc: typing.Optional[typing.Union[str, typing.Tuple[float, float]]] = None,
                       ext: str = 'pdf',
                       **kwargs: typing.Any) -> None:
        """Plot the histograms for a given key.

        :param key: The key of the data to plot
        :param x_label: X-axis label
        :param y_label: Y-axis label
        :param width: Total width of a bar
        :param legend_loc: Location of the legend
        :param ext: Output file extension
        :param kwargs: Keyword arguments for the plot function
        """
        assert isinstance(key, str)
        assert isinstance(x_label, str) or x_label is None
        assert isinstance(y_label, str) or y_label is None
        assert isinstance(width, float)
        assert isinstance(ext, str)

        # Get the histograms associated with the given key
        histograms = self.histograms[key]

        # Create figure
        fig, ax = plt.subplots()

        for index, h in enumerate(zip(*histograms)):
            # Obtain X and Y values (for all channels)
            x_values = np.arange(max(max(c) for c in h) + 1)
            y_values = [[c[x] for x in x_values] for c in h]

            # Plot
            ax.cla()  # Clear axes
            bar_width = width / len(h)
            for i, y in enumerate(y_values):
                ax.bar(x_values + (bar_width * i) - (width / 2), y,
                       width=bar_width, align='edge', label='Channel {:d}'.format(i), **kwargs)

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
        for key in self.histograms:
            self.plot_histogram(key, **kwargs)

    def plot_probability(self, key: str,
                         x_values: typing.Optional[typing.Sequence[typing.Union[float, int]]] = None,
                         x_label: typing.Optional[str] = None, y_label: typing.Optional[str] = 'State probability',
                         legend_loc: typing.Optional[typing.Union[str, typing.Tuple[float, float]]] = None,
                         ext: str = 'pdf',
                         **kwargs: typing.Any) -> None:
        """Plot the probability graph for a given key.

        :param key: The key of the data to plot
        :param x_values: The sequence with X values for the graph
        :param x_label: X-axis label
        :param y_label: Y-axis label
        :param legend_loc: Location of the legend
        :param ext: Output file extension
        :param kwargs: Keyword arguments for the plot function
        """
        assert isinstance(key, str)
        assert isinstance(x_values, collections.abc.Sequence) or x_values is None
        assert isinstance(x_label, str) or x_label is None
        assert isinstance(y_label, str) or y_label is None
        assert isinstance(ext, str)

        # Get the probabilities associated with the provided key
        probabilities = [np.asarray(p) for p in self.probabilities[key]]

        if not len(probabilities):
            # No data to plot
            return
        elif x_values is None:
            # Generate generic X values
            x_values = np.arange(len(probabilities[0]))
        else:
            # Sort data based on the given x values
            x_values = np.asarray(x_values)
            ind = x_values.argsort()
            x_values = x_values[ind]
            probabilities = [p[ind] for p in probabilities]

        # Plotting defaults
        kwargs.setdefault('marker', 'o')

        # Plot
        fig, ax = plt.subplots()
        for i, y in enumerate(probabilities):
            ax.plot(x_values, y, label='Channel {:d}'.format(i), **kwargs)

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
        """Plot probability graphs for all keys available in the data.

        :param kwargs: Keyword arguments passed to :func:`plot_probability`
        """
        for key in self.histograms:
            self.plot_probability(key, **kwargs)


class HistogramContextError(RuntimeError):
    """Class for histogram context errors."""
    pass
