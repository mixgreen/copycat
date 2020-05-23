import typing
import collections
import numpy as np

from dax.experiment import *
from dax.interfaces.detection import DetectionInterface
from dax.util.ccb import get_ccb_tool

__all__ = ['HistogramContext', 'HistogramContextError']


class HistogramContext(DaxModule):
    """Context class for managing storage of PMT histogram data.

    This module can be used as a sub-module of a service providing state measurement abilities.
    The HistogramContext object can directly be passed to the user which can use it as a context
    or call its additional functions.

    The histogram context objects manages all result values, but the user is responsible for tracking
    "input parameters".
    """

    HISTOGRAM_PLOT_KEY = 'plot.dax.histogram_context.histogram'
    """Dataset name for plotting latest histogram."""
    PROBABILITY_PLOT_KEY = 'plot.dax.histogram_context.probability'
    """Dataset name for plotting latest probability graph."""
    PLOT_GROUP = 'dax.histogram_context'
    """Group to which the plot applets belong."""

    DEFAULT_DATASET_KEY = 'pmt_histogram'
    """The default name of the output dataset in archive."""

    DATASET_KEY_FORMAT = '{dataset_key:s}.{index:d}'
    """Format string for sub-dataset keys."""

    def build(self, default_dataset_key: typing.Optional[str] = None) -> None:  # type: ignore
        assert isinstance(default_dataset_key, str) or default_dataset_key is None, \
            'Provided default dataset key must be None or of type str'

        # Store default dataset key
        self._default_dataset_key = self.DEFAULT_DATASET_KEY if default_dataset_key is None else default_dataset_key

        # Get CCB tool
        self._ccb = get_ccb_tool(self)

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
        # Prepare the probability plot dataset
        self.set_dataset(self.PROBABILITY_PLOT_KEY, [], broadcast=True, archive=False)

    def post_init(self) -> None:
        # Obtain the state detection threshold
        detection = self.registry.find_interface(DetectionInterface)  # type: ignore
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

        :param data: A list of ints representing the PMT counts of different ions.
        """
        if not self._in_context:
            # Called out of context
            raise HistogramContextError('The histogram append function can only be called inside the histogram context')

        # Append the given element to the buffer
        self._buffer.append(data)

    @rpc(flags={'async'})
    def __call__(self, key=None,
                 clear_probability_plot=False):  # type: (typing.Optional[str], bool) -> HistogramContext
        """Optional configuration of the histogram context for the next run.

        Set the dataset base key used for the next histogram data.
        After every context exit, the dataset name resets to the default key.

        This function can not be used when already in context.

        :param key: Key for the result dataset
        :param clear_probability_plot: If true, clear the probability plot
        """
        assert isinstance(key, str) or key is None, 'Provided dataset key must be of type str or None'
        assert isinstance(clear_probability_plot, bool), 'Clear probability plot flag must be of type bool'

        if self._in_context:
            # Called in context
            raise HistogramContextError('Setting the target dataset can only be done when not in context')

        if key is not None:
            # Store the dataset key
            self._dataset_key = key

        if clear_probability_plot:
            # Clear the probability dataset
            self.set_dataset(self.PROBABILITY_PLOT_KEY, [], broadcast=True, archive=False)

        # Return self to use in a `with` statement
        return self

    @rpc(flags={'async'})
    def __enter__(self):  # type: () -> None
        """Enter the histogram context.

        Entering the histogram context will prepare the target dataset and clear the buffer.
        The earlier set target dataset will be used for the output.
        In case no target dataset was configured, the default dataset key will be used.
        """

        if self._in_context:
            # Prevent nested context
            raise HistogramContextError('The histogram context can not be nested')

        if self._dataset_key not in self._open_datasets:
            # Initialize counter to 0
            self._open_datasets[self._dataset_key] = 0

        # Clear histogram dataset
        self.set_dataset(self.HISTOGRAM_PLOT_KEY, [], broadcast=True, archive=False)
        # Create a new buffer (clearing it might result in data loss due to how the dataset manager works)
        self._buffer = []
        # Increment in context counter
        self._in_context += 1

    @rpc(flags={'async'})
    def __exit__(self, exc_type, exc_val, exc_tb):  # type: (typing.Any, typing.Any, typing.Any) -> None
        """Exit the histogram context."""

        if not self._in_context:
            # Called exit out of context
            raise HistogramContextError('The exit function can only be called from inside the histogram context')

        # Create a sub-dataset key for this result (HDF5 only supports fixed size elements in a list)
        sub_dataset_key = self.DATASET_KEY_FORMAT.format(dataset_key=self._dataset_key,
                                                         index=self._open_datasets[self._dataset_key])

        if len(self._buffer):
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
        # Reset dataset key to default
        self._dataset_key = self._default_dataset_key
        # Update context counter
        self._in_context -= 1

    def _histogram_to_probability(self, counter: collections.Counter,
                                  state_detection_threshold: typing.Optional[int] = None) -> float:
        """Convert a histogram to a state probability."""

        if state_detection_threshold is None:
            # Use default state_detection_threshold if not set
            state_detection_threshold = self._state_detection_threshold

        # One measurements (recognizes binary measurements and counts)
        one = sum(f for c, f in counter.items() if c is True or c > state_detection_threshold)
        # Total measurements
        total = sum(counter.values())
        # Return probability
        return one / total

    @rpc(flags={'async'})
    def open(self, key=None, clear_probability_plot=False):  # type: (typing.Optional[str], bool) -> None
        """Enter the histogram context manually.

        This function can be used to manually configure and enter the histogram context.
        We strongly recommend to use the `with` statement instead.

        :param key: Key for the result dataset
        :param clear_probability_plot: If true, clear the probability plot
        """
        self.__call__(key, clear_probability_plot)
        self.__enter__()

    @rpc(flags={'async'})
    def close(self):  # type: () -> None
        """Exit the histogram context manually.

        This function can be used to manually exit the histogram context.
        We strongly recommend to use the `with` statement instead.
        """
        self.__exit__(None, None, None)

    """Applet plotting functions"""

    @rpc(flags={'async'})
    def applet_plot_histogram(self, **kwargs):  # type: (typing.Any) -> None
        """Open the applet that shows a plot of the latest histogram.

        :param kwargs: Extra keyword arguments for the plot
        """

        # Set default arguments
        kwargs.setdefault('x_label', 'Number of counts')
        kwargs.setdefault('y_label', 'Frequency')
        # Plot
        self._ccb.plot_hist('histogram', self.HISTOGRAM_PLOT_KEY, group=self.PLOT_GROUP, **kwargs)

    @rpc(flags={'async'})
    def applet_plot_probability(self, **kwargs):  # type: (typing.Any) -> None
        """Open the applet that shows a plot of state probabilities over multiple histograms.

        :param kwargs: Extra keyword arguments for the plot
        """

        # Set default label
        kwargs.setdefault('y_label', 'State probability')
        # Plot
        self._ccb.plot_xy_nested('probability', self.PROBABILITY_PLOT_KEY, group=self.PLOT_GROUP, **kwargs)

    @rpc(flags={'async'})
    def applet_close_histogram(self):  # type: () -> None
        """Close the histogram plot."""
        self._ccb.disable_applet(self.HISTOGRAM_PLOT_KEY, self.PLOT_GROUP)

    @rpc(flags={'async'})
    def applet_close_probability(self):  # type: () -> None
        """Close the probability plot."""
        self._ccb.disable_applet(self.PROBABILITY_PLOT_KEY, self.PLOT_GROUP)

    @rpc(flags={'async'})
    def applet_close_all(self):  # type: () -> None
        """Close all histogram context plots."""
        self._ccb.disable_applet_group(self.PLOT_GROUP)

    """Data analysis functions"""

    @host_only
    def get_histograms(self,
                       dataset_key: typing.Optional[str] = None) -> typing.List[typing.Sequence[collections.Counter]]:
        """Obtain all histogram objects recorded by this histogram context for a specific key.

        The data is formatted as a list of histograms per channel.
        So to access histogram N of channel C: `get_histograms()[C][N]`.

        In case no dataset key is provided, the default dataset key is used.

        :param dataset_key: Key of the dataset to obtain the histograms of
        :return: All histogram data
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
        :return: All probability data
        """
        return [[self._histogram_to_probability(h, state_detection_threshold) for h in histograms]
                for histograms in self.get_histograms(dataset_key)]


class HistogramContextError(RuntimeError):
    """Class for histogram context errors."""
    pass
