import typing
import collections
import math
import numpy as np
import h5py  # type: ignore
import os
import natsort

import matplotlib.pyplot as plt  # type: ignore

from dax.experiment import *
from dax.util.ccb import get_ccb_tool
from dax.util.output import get_file_name_generator, dummy_file_name_generator
from dax.util.units import UnitsFormatter

__all__ = ['TimeResolvedContext', 'TimeResolvedAnalyzer', 'TimeResolvedContextError']


class TimeResolvedContext(DaxModule):
    """Context class for managing storage of time-resolved detection output.

    This module can be used as a sub-module of a service.
    """

    PLOT_RESULT_KEY_FORMAT: str = 'plot.{base}.time_resolved_context.result'
    """Dataset name for plotting latest result graph (Y-axis)."""
    PLOT_TIME_KEY_FORMAT: str = 'plot.{base}.time_resolved_context.time'
    """Dataset name for plotting latest result graph (X-axis)."""
    PLOT_NAME: str = 'time resolved detection'
    """Name of the plot applet."""
    PLOT_GROUP_FORMAT: str = '{base}.time_resolved_context'
    """Group to which the plot applets belong."""

    DATASET_GROUP: str = 'time_resolved_context'
    """The group name for archiving data."""
    DATASET_KEY_FORMAT: str = DATASET_GROUP + '/{dataset_key}/{index}/{column}'
    """Format string for sub-dataset keys."""
    DEFAULT_DATASET_KEY: str = 'time_resolved'
    """The default name of the output dataset in archive."""
    DATASET_COLUMNS: typing.Tuple[str, ...] = ('width', 'time', 'result')
    """Column names of data within each sub-dataset."""

    def build(self, *,  # type: ignore
              default_dataset_key: typing.Optional[str] = None, plot_base_key: str = 'dax') -> None:
        """Build the time resolved context module.

        The plot base key can be used to group plot datasets and applets as desired.
        The base key is formatted with the `scheduler` object which allows users to
        add experiment-specific information in the base key.

        :param default_dataset_key: Default dataset name used for storing trace data
        :param plot_base_key: Base key for plot dataset keys and applets
        """
        assert isinstance(default_dataset_key, str) or default_dataset_key is None, \
            'Provided default dataset key must be None or of type str'
        assert isinstance(plot_base_key, str), 'Plot base key must be of type str'

        # Store default dataset key
        if default_dataset_key is None:
            self._default_dataset_key: str = self.DEFAULT_DATASET_KEY
        else:
            self._default_dataset_key = default_dataset_key

        # Get CCB tool
        self._ccb = get_ccb_tool(self)
        # Units formatter
        self._units_fmt: UnitsFormatter = UnitsFormatter()

        # By default we are not in context
        self._in_context: np.int32 = np.int32(0)
        # The count buffer (buffer appending is a bit faster than dict operations)
        self._buffer_data: typing.List[typing.Tuple[typing.Sequence[typing.Sequence[int]], float]] = []
        self._buffer_meta: typing.List[typing.Tuple[float, float, float]] = []

        # Cache for processed data
        self._cache: typing.Dict[str, typing.List[typing.Dict[str, typing.Sequence[float]]]] = {}

        # Target dataset key
        self._dataset_key: str = self._default_dataset_key
        # Store plot base key
        self._plot_base_key: str = plot_base_key
        # Datasets that are initialized with a counter, which represents the length of the data
        self._open_datasets: typing.Counter[str] = collections.Counter()

    def init(self) -> None:
        # Generate plot keys
        base: str = self._plot_base_key.format(scheduler=self.get_device('scheduler'))
        self._plot_result_key: str = self.PLOT_RESULT_KEY_FORMAT.format(base=base)
        self._plot_time_key: str = self.PLOT_TIME_KEY_FORMAT.format(base=base)
        # Generate applet plot group
        self._plot_group: str = self.PLOT_GROUP_FORMAT.format(base=base)

    def post_init(self) -> None:
        pass

    """Helper functions"""

    @staticmethod
    @host_only
    def partition_bins(num_bins: int, max_partition_size: int,
                       bin_width: float, bin_spacing: float) -> typing.List[typing.Tuple[np.int32, float]]:
        """Partition a number of bins.

        This function returns a list of tuples that can be used at runtime for partitioning in a loop.
        The format of each element is (current_num_bins, current_offset) which can be used accordingly.

        This function returns the partition table as a list. Hence, it can only be called from the host.

        :param num_bins: The total number of bins desired
        :param max_partition_size: The maximum number of bins in one partition
        :param bin_width: The width of each bin
        :param bin_spacing: The spacing between bins
        :return: A list with tuples that can be used for automatic partitioning at runtime
        """
        assert isinstance(num_bins, (int, np.integer)), 'Number of bins must be an integer'
        assert isinstance(max_partition_size, (int, np.integer)), 'Max partition size must be an integer'
        assert isinstance(bin_width, float), 'Bin width must be of type float'
        assert isinstance(bin_spacing, float), 'Bin spacing must be of type float'

        # Check input values
        if num_bins < 0:
            raise ValueError('Number of bins must positive')
        if max_partition_size <= 0:
            raise ValueError('Maximum partition size must be greater than zero')
        if bin_width < 0.0:
            raise ValueError('Bin width can not be less than zero')
        if bin_spacing < 0.0:
            raise ValueError('Bin spacing can not be less than zero')

        # Calculate the offset of a single whole partition
        partition_offset: float = max_partition_size * (bin_width + bin_spacing)

        # List of whole partitions (maximum number of bins)
        partitions = [(np.int32(max_partition_size), i * partition_offset)
                      for i in range(num_bins // max_partition_size)]

        mod = num_bins % max_partition_size
        if mod > 0:
            # Add element with partial partition (less than the maximum number of bins)
            partitions.append((np.int32(mod), len(partitions) * partition_offset))

        # Return the partitions
        return partitions

    @staticmethod
    @host_only
    def partition_window(window_size: float, max_partition_size: int,
                         bin_width: float, bin_spacing: float) -> typing.List[typing.Tuple[np.int32, float]]:
        """Partition a time window.

        This function returns a list of tuples that can be used at runtime for partitioning in a loop.
        The format of each element is (current_num_bins, current_offset) which can be used accordingly.

        This function returns the partition table as a list. Hence, it can only be called from the host.

        :param window_size: The total window size
        :param max_partition_size: The maximum number of bins in one partition
        :param bin_width: The width of each bin
        :param bin_spacing: The spacing between bins
        :return: A list with tuples that can be used for automatic partitioning at runtime
        """
        assert isinstance(window_size, float), 'Window size must be of type float'
        assert isinstance(bin_width, float), 'Bin width must be of type float'
        assert isinstance(bin_spacing, float), 'Bin spacing must be of type float'

        # Check input values
        if window_size < 0.0:
            raise ValueError('Window size must be positive')
        if bin_width < 0.0:
            raise ValueError('Bin width can not be less than zero')
        if bin_spacing < 0.0:
            raise ValueError('Bin spacing can not be less than zero')
        if bin_width == 0.0 and bin_spacing == 0.0:
            raise ValueError('Bin width and bin spacing can not both be zero')

        # Calculate the number of bins that fit in the window, rounding the value up
        num_bins: int = int(math.ceil(window_size / (bin_width + bin_spacing)))
        # Call partition bins
        return TimeResolvedContext.partition_bins(num_bins, max_partition_size, bin_width, bin_spacing)

    """Data handling functions"""

    @portable
    def in_context(self) -> bool:
        """True if we are in context."""
        return bool(self._in_context)

    @rpc(flags={'async'})
    def append_meta(self, bin_width, bin_spacing, offset=0.0):  # type: (float, float, float) -> None
        """Store metadata that matches the next call to :func:`append_data`.

        This function is intended to be fast to allow high input data throughput.
        No type checking is performed on the data.

        :param bin_width: The width of the bins
        :param bin_spacing: The spacing between the bins
        :param offset: The known fixed offset of this trace in seconds, used for partitioning
        :raises TimeResolvedContextError: Raised if called out of context
        """
        if not self._in_context:
            # Called out of context
            raise TimeResolvedContextError('The append function can only be called in-context')

        # Append the given element to the buffer (using tuples for high performance)
        self._buffer_meta.append((bin_width, bin_spacing, offset))

    @rpc(flags={'async'})
    def remove_meta(self):  # type: () -> None
        """Remove metadata that was appended with the last call to :func:`append_meta`.

        This function is intended to remove or cancel the last call to :func:`append_meta`.
        It can be used when no useful data can be provided to the :func:`append_data`
        call that should have matched the last :func:`append_meta` call.

        It is up to the user to manage the stream of data. Hence, the user can choose if
        the data point is dropped, or if a retry will be performed.
        In any way, the buffers of the time resolved context can remain consistent.

        :raises TimeResolvedContextError: Raised if called out of context
        :raises IndexError: Raised if there was no metadata to remove
        """
        if not self._in_context:
            # Called out of context
            raise TimeResolvedContextError('The remove function can only be called in-context')

        # Append the given element to the buffer (using tuples for high performance)
        self._buffer_meta.pop()

    @rpc(flags={'async'})
    def append_data(self, data, offset_mu=0):  # type: (typing.Sequence[typing.Sequence[int]], np.int64) -> None
        """Append PMT data (async RPC).

        This function is intended to be fast to allow high input data throughput.
        No type checking is performed on the data.

        Note that corrections for delayed events should result in **negative offset**.
        The negative offset represents the fact that detection started before the event happened.

        :param data: A 2D list of ints representing the PMT counts of different ions
        :param offset_mu: An offset to correct any shifts of events in machine units (defaults to no offset)
        :raises TimeResolvedContextError: Raised if called out of context
        """
        if not self._in_context:
            # Called out of context
            raise TimeResolvedContextError('The append function can only be called in-context')

        # Append the given element to the buffer
        self._buffer_data.append((data, self.core.mu_to_seconds(offset_mu)))  # Convert machine units to seconds

    @rpc(flags={'async'})
    def append(self, data, bin_width, bin_spacing, offset=0.0,
               offset_mu=0):  # type: (typing.Sequence[typing.Sequence[int]], float, float, float, np.int64) -> None
        """Append metadata and PMT data (async RPC).

        This function calls :func:`append_meta` and :func:`append_data` in one call
        and can be used in case it is not required to separate the two subroutines.

        This function is intended to be fast to allow high input data throughput.
        No type checking is performed on the data.

        :param data: A 2D list of ints representing the PMT counts of different ions
        :param bin_width: The width of the bins
        :param bin_spacing: The spacing between the bins
        :param offset: An offset to correct any shifts of events in seconds (defaults to no offset)
        :param offset_mu: An offset to correct any shifts of events in machine units (defaults to no offset)
        :raises TimeResolvedContextError: Raised if called out of context
        """
        self.append_meta(bin_width, bin_spacing, offset=offset)
        self.append_data(data, offset_mu=offset_mu)

    @rpc(flags={'async'})
    def config_dataset(self, key=None, *args, **kwargs):  # type: (typing.Optional[str], typing.Any, typing.Any) -> None
        """Optional configuration of the context output dataset (async RPC).

        Set the dataset base key used for the following results.
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
        :raises TimeResolvedContextError: Raised if called in context
        """
        assert isinstance(key, str) or key is None, 'Provided dataset key must be of type str or None'

        if self._in_context:
            # Called in context
            raise TimeResolvedContextError('Setting the target dataset can only be done when out of context')

        # Update the dataset key
        self._dataset_key = self._default_dataset_key if key is None else self._units_fmt.vformat(key, args, kwargs)

    @portable
    def __enter__(self):  # type: () -> None
        """Enter the context.

        Entering the context will prepare the target dataset and clear the buffer.
        Optionally, this context can be configured using the :func:`config` function before entering the context.
        """
        self.open()

    @portable
    def __exit__(self, exc_type, exc_val, exc_tb):  # type: (typing.Any, typing.Any, typing.Any) -> None
        """Exit the context."""
        self.close()

    @rpc(flags={'async'})
    def open(self):  # type: () -> None
        """Enter the context manually.

        Optionally, this context can be configured using the :func:`config` function.

        This function can be used to manually enter the context.
        We strongly recommend to use the `with` statement instead.

        :raises TimeResolvedContextError: Raised if already in context (context is non-reentrant)
        """

        if self._in_context:
            # Prevent context reentry
            raise TimeResolvedContextError('The time resolved context is non-reentrant')

        # Create a new buffers (clearing it might result in data loss due to how the dataset manager works)
        self._buffer_data = []
        self._buffer_meta = []
        # Increment in context counter
        self._in_context += 1

    @rpc(flags={'async'})
    def close(self):  # type: () -> None
        """Exit the context manually.

        This function can be used to manually exit the context.
        We strongly recommend to use the `with` statement instead.

        :raises TimeResolvedContextError: Raised if called out of context
        """

        if not self._in_context:
            # Called exit out of context
            raise TimeResolvedContextError('The exit function can only be called from inside the context')

        # Create a sub-dataset keys for this result
        sub_dataset_keys = {column: self.DATASET_KEY_FORMAT.format(column=column, dataset_key=self._dataset_key,
                                                                   index=self._open_datasets[self._dataset_key])
                            for column in self.DATASET_COLUMNS}

        if len(self._buffer_data) or len(self._buffer_meta):
            # Check consistency of data in the buffers
            if len(self._buffer_data) != len(self._buffer_meta):
                raise RuntimeError('Length of the data and meta buffer are not consistent, data probably corrupt')
            if any(len(b) != len(self._buffer_data[0][0]) for b, _ in self._buffer_data):
                raise RuntimeError('Buffered data is not consistent, data probably corrupt')
            if any(len(s) != len(b[0]) for b, _ in self._buffer_data for s in b):
                raise RuntimeError('Buffered data (inner series) is not consistent, data probably corrupt')

            # Transform metadata and raw data
            buffer = [[(meta, d) for meta, d in zip(self._buffer_meta, data)]
                      for data in zip(*(b for b, _ in self._buffer_data))]
            result = [np.concatenate([d for _, d in channel]) for channel in buffer]
            # Width and time are only calculated once since we assume all data is homogeneous
            width = np.concatenate([np.full(len(d), w, dtype=float) for (w, _, _), d in buffer[0]])
            time = np.concatenate([np.arange(len(d), dtype=float) * (w + s) + (o + o_correction)
                                   for ((w, s, o), d), (_, o_correction) in zip(buffer[0], self._buffer_data)])

            # Format results in a dict for easier access
            result_dict = {'result': result, 'time': time, 'width': width}

            # Store results in the cache
            self._cache.setdefault(self._dataset_key, []).append(result_dict)

            # Write results to sub-dataset for archiving
            for column in self.DATASET_COLUMNS:
                self.set_dataset(sub_dataset_keys[column], result_dict[column], archive=True)
            # Write result to plotting dataset
            self.set_dataset(self._plot_time_key, time + (width * 0.5), broadcast=True, archive=False)
            self.set_dataset(self._plot_result_key, np.column_stack(result), broadcast=True, archive=False)

        else:
            # Add empty element to the cache (keeps indexing consistent)
            self._cache.setdefault(self._dataset_key, []).append({c: [] for c in self.DATASET_COLUMNS})
            # Write empty element to sub-dataset for archiving (keeps indexing consistent)
            for column in self.DATASET_COLUMNS:
                self.set_dataset(sub_dataset_keys[column], [], archive=True)

        # Update counter for this dataset key
        self._open_datasets[self._dataset_key] += 1
        # Update context counter
        self._in_context -= 1

    """Applet plotting functions"""

    @rpc(flags={'async'})
    def plot(self, **kwargs):  # type: (typing.Any) -> None
        """Open the applet that shows a plot of the latest results.

        :param kwargs: Extra keyword arguments for the plot
        """

        # Set default arguments
        kwargs.setdefault('x_label', 'Time')
        kwargs.setdefault('y_label', 'Number of counts')
        # Plot
        self._ccb.plot_xy_multi(self.PLOT_NAME, self._plot_result_key,
                                x=self._plot_time_key, group=self._plot_group, **kwargs)

    @rpc(flags={'async'})
    def disable_plot(self):  # type: () -> None
        """Close the plot."""
        self._ccb.disable_applet(self.PLOT_NAME, self._plot_group)

    @rpc(flags={'async'})
    def disable_all_plots(self):  # type: () -> None
        """Close all context related plots."""
        self._ccb.disable_applet_group(self._plot_group)

    """Data access functions"""

    @host_only
    def get_keys(self) -> typing.List[str]:
        """Get the keys for which results were recorded.

        The returned keys can be used for the :func:`get_traces` function.

        :return: A list with keys
        """
        return list(self._cache)

    @host_only
    def get_traces(self, dataset_key: typing.Optional[str] = None) \
            -> typing.List[typing.Dict[str, typing.Sequence[float]]]:
        """Obtain all trace objects recorded by this time-resolved context for a specific key.

        The data is formatted as a list of dictionaries with the self-explaining keys
        time, width, and results (see :attr:`DATASET_COLUMNS`).
        Time and width are one-dimensional arrays while results is a list with one-dimensional arrays
        where the list index corresponds to the channels.

        In case no dataset key is provided, the default dataset key is used.

        :param dataset_key: Key of the dataset to obtain the trace of
        :return: All trace data for the specified key
        """
        return self._cache[self._default_dataset_key if dataset_key is None else dataset_key]


class TimeResolvedAnalyzer:
    """Basic automated analysis and offline plotting of data obtained by the time resolved context.

    Various data sources can be provided and presented data should have a uniform format.
    Simple automated plotting functions are provided, but users can also access data directly
    for manual processing and analysis.

    :attr:`keys` is a list of keys for which data is available.

    :attr:`traces` is a dict which for each key contains a list of traces.
    Each trace is a dict with values for bin width, bin, time, and results.
    Results are stored in a 2D array of which the first dimension is the channel
    and the second dimension are the values.
    """

    PLOT_FILE_FORMAT: str = '{key}_{index}'
    """File name format for plot files."""

    def __init__(self, source: typing.Union[DaxSystem, TimeResolvedContext, str, h5py.File]):
        """Create a new time resolved analyzer object.

        :param source: The source of the trace data
        """

        # Input conversion
        if isinstance(source, DaxSystem):
            # Obtain time resolved context module
            source = source.registry.find_module(TimeResolvedContext)
        elif isinstance(source, str):
            # Open HDF5 file
            source = h5py.File(os.path.expanduser(source), mode='r')

        if isinstance(source, TimeResolvedContext):
            # Get data from module
            self.keys: typing.List[str] = source.get_keys()
            self.traces: typing.Dict[str, typing.List[typing.Dict[str, typing.Sequence[float]]]] = \
                {k: source.get_traces(k) for k in self.keys}

            # Obtain the file name generator
            self._file_name_generator = get_file_name_generator(source.get_device('scheduler'))

        elif isinstance(source, h5py.File):
            # Verify format of HDF5 file
            group_name = 'datasets/' + TimeResolvedContext.DATASET_GROUP
            if group_name not in source:
                raise KeyError('The HDF5 file does not contain time resolved data')

            # Get the group which contains all data
            group = source[group_name]

            # Read and convert data from HDF5 file
            self.keys = list(group)
            self.traces = {k: [{column: group[k][index][column][()] for column in TimeResolvedContext.DATASET_COLUMNS}
                               for index in natsort.natsorted(group[k])] for k in self.keys}

            # Get a file name generator
            self._file_name_generator = dummy_file_name_generator

        else:
            raise TypeError('Unsupported source type')

    """Plotting functions"""

    def plot_trace(self, key: str,
                   x_label: typing.Optional[str] = 'Time', y_label: typing.Optional[str] = 'Count',
                   labels: typing.Optional[typing.Sequence[str]] = None,
                   legend_loc: typing.Optional[typing.Union[str, typing.Tuple[float, float]]] = None,
                   ext: str = 'pdf',
                   **kwargs: typing.Any) -> None:
        """Plot the traces for a given key.

        :param key: The key of the data to plot
        :param x_label: X-axis label
        :param y_label: Y-axis label
        :param labels: List of plot labels
        :param legend_loc: Location of the legend
        :param ext: Output file extension
        :param kwargs: Keyword arguments for the plot function
        """
        assert isinstance(key, str)
        assert isinstance(x_label, str) or x_label is None
        assert isinstance(y_label, str) or y_label is None
        assert isinstance(labels, collections.abc.Sequence) or labels is None
        assert isinstance(ext, str)

        # Get the traces associated with the given key
        traces = self.traces[key]

        # Create figure
        fig, ax = plt.subplots()

        for index, t in enumerate(traces):
            # Obtain raw data
            time = np.asarray(t['time'])
            width = np.asarray(t['width'])
            results = t['result']  # List with result arrays

            # Create X values
            x_values = time + width / 2  # Points are plotted in the middle of the bin

            # Current labels
            current_labels = [f'Plot {i}' for i in range(len(results))] if labels is None else labels
            if len(current_labels) < len(results):
                # Not enough labels
                raise IndexError('Number of labels is less than the number of plots')

            # Plotting defaults
            kwargs.setdefault('marker', 'o')
            kwargs.setdefault('linestyle', '')

            # Plot
            ax.cla()  # Clear axes
            for y, label in zip(results, current_labels):
                ax.plot(x_values, y, label=label, **kwargs)

            # Plot formatting
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.ticklabel_format(axis='x', scilimits=(0, 10))
            ax.legend(loc=legend_loc)

            # Save figure
            file_name = self._file_name_generator(self.PLOT_FILE_FORMAT.format(key=key, index=index), ext)
            fig.savefig(file_name, bbox_inches='tight')

        # Close the figure
        plt.close(fig)

    def plot_all_traces(self, **kwargs: typing.Any) -> None:
        """Plot traces for all keys available in the data.

        :param kwargs: Keyword arguments passed to :func:`plot_trace`
        """
        for key in self.traces:
            self.plot_trace(key, **kwargs)


class TimeResolvedContextError(RuntimeError):
    """Class for time resolved context errors."""
    pass
