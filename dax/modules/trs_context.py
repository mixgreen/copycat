import typing
import collections
import numpy as np

import dax.util.matplotlib_backend  # Workaround for QT error  # noqa: F401
import matplotlib.pyplot as plt  # type: ignore

from dax.experiment import *
from dax.util.ccb import get_ccb_tool
from dax.util.output import get_file_name_generator
from dax.util.units import UnitsFormatter

__all__ = ['TimeResolvedSpectroscopyContext', 'TimeResolvedSpectroscopyAnalyzer',
           'TimeResolvedSpectroscopyContextError']


class TimeResolvedSpectroscopyContext(DaxModule):
    """Context class for managing storage of time-resolved spectroscopy output.

    This module can be used as a sub-module of a service.
    """

    PLOT_RESULT_KEY = 'plot.dax.trs_context.result'
    """Dataset name for plotting latest result graph (Y-axis)."""
    PLOT_TIME_KEY = 'plot.dax.trs_context.time'
    """Dataset name for plotting latest result graph (X-axis)."""
    PLOT_NAME = 'time resolved spectroscopy'
    """Name of the plot applet."""
    PLOT_GROUP = 'dax.trs_context'
    """Group to which the plot applets belong."""

    DEFAULT_DATASET_KEY = 'trs'
    """The default name of the output dataset in archive."""
    META_KEY_FORMAT = '_trs.{key:s}'
    """Dataset key format for archiving metadata."""

    DATASET_KEY_FORMAT = '{dataset_key:s}.{index:d}.{column:s}'
    """Format string for sub-dataset keys."""
    DATASET_COLUMNS = ['width', 'time', 'result']
    """Column names of data within each sub-dataset."""

    def build(self, default_dataset_key: typing.Optional[str] = None) -> None:  # type: ignore
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
        self._buffer = []  # type: typing.List[typing.Tuple[typing.Sequence[typing.Sequence[int]], float]]
        self._buffer_meta = []  # type: typing.List[typing.Tuple[float, float, float]]

        # Archive to analyze high level data at the end of the experiment
        self._archive = {}  # type: typing.Dict[str, typing.List[typing.Dict[str, typing.Sequence[float]]]]

        # Target dataset key
        self._dataset_key = self._default_dataset_key
        # Datasets that are initialized with a counter, which represents the length of the data
        self._open_datasets = collections.Counter()  # type: typing.Dict[str, int]

    def init(self) -> None:
        pass

    def post_init(self) -> None:
        pass

    """Helper functions"""

    # TODO: add helper functions for partitioning based on number of bins or duration (use benchmark measurement)

    """Data handling functions"""

    @portable
    def in_context(self) -> bool:
        """True if we are in context."""
        return bool(self._in_context)

    @rpc(flags={'async'})
    def append_meta(self, bin_width, bin_spacing, offset=0.0):  # type: (float, float, float) -> None
        """Store metadata that matches the next call to :func:`append`.

        This function is intended to be fast to allow high input data throughput.
        No type checking is performed on the data.

        :param bin_width: The width of the bins
        :param bin_spacing: The spacing between the bins
        :param offset: The known fixed offset of this trace, used for partitioning
        :raises TimeResolvedSpectroscopyContextError: Raised if called out of context
        """
        if not self._in_context:
            # Called out of context
            raise TimeResolvedSpectroscopyContextError('The append_meta() function can only be called in-context')

        # Append the given element to the buffer (using tuples for high performance)
        self._buffer_meta.append((bin_width, bin_spacing, offset))

    @rpc(flags={'async'})
    def append(self, data, offset=0.0):  # type: (typing.Sequence[typing.Sequence[int]], float) -> None
        """Append PMT data (async RPC).

        This function is intended to be fast to allow high input data throughput.
        No type checking is performed on the data.

        :param data: A 2D list of ints representing the PMT counts of different ions
        :param offset: An offset to correct any shifts of events (defaults to no offset)
        :raises TimeResolvedSpectroscopyContextError: Raised if called out of context
        """
        if not self._in_context:
            # Called out of context
            raise TimeResolvedSpectroscopyContextError('The append() function can only be called in-context')

        # Append the given element to the buffer
        self._buffer.append((data, offset))

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
        :raises TimeResolvedSpectroscopyContextError: Raised if called in context
        """
        assert isinstance(key, str) or key is None, 'Provided dataset key must be of type str or None'

        if self._in_context:
            # Called in context
            raise TimeResolvedSpectroscopyContextError(
                'Setting the target dataset can only be done when out of context')

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

        :raises TimeResolvedSpectroscopyContextError: Raised if already in context (context non-reentrant)
        """

        if self._in_context:
            # Prevent context reentry
            raise TimeResolvedSpectroscopyContextError('The time resolved spectroscopy context is non-reentrant')

        # Create a new buffers (clearing it might result in data loss due to how the dataset manager works)
        self._buffer = []
        self._buffer_meta = []
        # Increment in context counter
        self._in_context += 1

    @rpc(flags={'async'})
    def close(self):  # type: () -> None
        """Exit the context manually.

        This function can be used to manually exit the context.
        We strongly recommend to use the `with` statement instead.

        :raises TimeResolvedSpectroscopyContextError: Raised if called out of context
        """

        if not self._in_context:
            # Called exit out of context
            raise TimeResolvedSpectroscopyContextError('The exit function can only be called from inside the context')

        # Create a sub-dataset keys for this result
        sub_dataset_keys = {column: self.DATASET_KEY_FORMAT.format(column=column, dataset_key=self._dataset_key,
                                                                   index=self._open_datasets[self._dataset_key])
                            for column in self.DATASET_COLUMNS}

        if len(self._buffer) or len(self._buffer_meta):
            # Check consistency of data in the buffers
            if len(self._buffer) != len(self._buffer_meta):
                self.logger.error('Data in the buffer and meta buffer are not consistent, data probably corrupt')
            if any(len(b) != len(self._buffer[0][0]) for b, _ in self._buffer):
                self.logger.error('Data in the buffer is not consistent, incomplete data is dropped')
            if any(len(s) != len(b[0]) for b, _ in self._buffer for s in b):
                self.logger.error('Data in the buffer (inner series) is not consistent, incomplete data is dropped')

            # Transform metadata and raw data
            buffer = [[(meta, d) for meta, d in zip(self._buffer_meta, data)]
                      for data in zip(*(b for b, _ in self._buffer))]
            result = [np.concatenate([d for _, d in channel]) for channel in buffer]
            # Width and time are only calculated once since we assume all data is homogeneous
            width = np.concatenate([np.full(len(d), w, dtype=float) for (w, _, _), d in buffer[0]])
            time = np.concatenate([np.arange(len(d), dtype=float) * (w + s) + (o + o_correction)
                                   for ((w, s, o), d), (_, o_correction) in zip(buffer[0], self._buffer)])

            # Format results in a dict for easier access
            result_dict = dict(result=result, time=time, width=width)

            # Store results in the local archive
            self._archive.setdefault(self._dataset_key, []).append(result_dict)

            # Write results to sub-dataset for archiving
            for column in self.DATASET_COLUMNS:
                self.set_dataset(sub_dataset_keys[column], result_dict[column], archive=True)
            # Write result to plotting dataset
            self.set_dataset(self.PLOT_TIME_KEY, time + (width * 0.5), broadcast=True, archive=False)
            self.set_dataset(self.PLOT_RESULT_KEY, np.column_stack(result), broadcast=True, archive=False)

        else:
            # Add empty element to the archive (keeps indexing consistent)
            self._archive.setdefault(self._dataset_key, []).append({c: [] for c in self.DATASET_COLUMNS})
            # Write empty element to sub-dataset for archiving (keeps indexing consistent)
            for column in self.DATASET_COLUMNS:
                self.set_dataset(sub_dataset_keys[column], [], archive=True)

        # Update counter for this dataset key
        self._open_datasets[self._dataset_key] += 1
        # Archive number of sub-datasets for current key
        self.set_dataset(self.META_KEY_FORMAT.format(key=self._dataset_key),
                         self._open_datasets[self._dataset_key], archive=True)
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
        self._ccb.plot_xy_multi(self.PLOT_NAME, self.PLOT_RESULT_KEY, self.PLOT_TIME_KEY,
                                group=self.PLOT_GROUP, **kwargs)

    @rpc(flags={'async'})
    def disable_plot(self):  # type: () -> None
        """Close the plot."""
        self._ccb.disable_applet(self.PLOT_NAME, self.PLOT_GROUP)

    @rpc(flags={'async'})
    def disable_all_plots(self):  # type: () -> None
        """Close all context related plots."""
        self._ccb.disable_applet_group(self.PLOT_GROUP)

    """Data access functions"""

    @host_only
    def get_keys(self) -> typing.List[str]:
        """Get the keys for which results were recorded.

        The returned keys can be used for the :func:`get_traces` function.

        :return: A list with keys
        """
        return list(self._archive)

    @host_only
    def get_traces(self, dataset_key: typing.Optional[str] = None) \
            -> typing.List[typing.Dict[str, typing.Sequence[float]]]:
        """Obtain all trace objects recorded by this time-resolved spectroscopy context for a specific key.

        The data is formatted as a list of dictionaries with the self-explaining keys
        time, width, and results (see :attr:`DATASET_COLUMNS`).
        Time and width are one-dimensional arrays while results is a list with one-dimensional arrays
        where the list index corresponds to the channels.

        In case no dataset key is provided, the default dataset key is used.

        :param dataset_key: Key of the dataset to obtain the trace of
        :return: All trace data for the specified key
        """
        return self._archive[self._default_dataset_key if dataset_key is None else dataset_key]


class TimeResolvedSpectroscopyAnalyzer:
    """Basic automated analysis and offline plotting of data obtained by the time resolved spectroscopy context."""

    PLOT_FILE_FORMAT = '{key:s}_{index:d}'
    """File name format for plot files."""

    def __init__(self, source: typing.Union[DaxSystem, TimeResolvedSpectroscopyContext, str]):
        """Create a new time resolved spectroscopy analyzer object.

        :param source: The source of the trace data
        """

        if isinstance(source, DaxSystem):
            # Obtain time resolved spectroscopy context module
            source = source.registry.find_module(TimeResolvedSpectroscopyContext)

        if isinstance(source, TimeResolvedSpectroscopyContext):
            # Get data from module
            self.keys = source.get_keys()
            self.traces = {k: source.get_traces(k) for k in self.keys}

            # Obtain the file name generator
            self._file_name_generator = get_file_name_generator(source.get_device('scheduler'))

        elif isinstance(source, str):
            # Read and convert data from HDF5 file
            raise NotImplementedError('Analysis from HDF5 source file is not yet implemented')

        else:
            raise TypeError('Unsupported source type')

    def plot_trace(self, key: str,
                   x_label: typing.Optional[str] = 'Time', y_label: typing.Optional[str] = 'Count',
                   legend_loc: typing.Optional[typing.Union[str, typing.Tuple[float, float]]] = None,
                   ext: str = 'pdf',
                   **kwargs: typing.Any) -> None:
        """Plot the traces for a given key.

        :param key: The key of the data to plot
        :param x_label: X-axis label
        :param y_label: Y-axis label
        :param legend_loc: Location of the legend
        :param ext: Output file extension
        :param kwargs: Keyword arguments for the plot function
        """
        assert isinstance(key, str)
        assert isinstance(x_label, str) or x_label is None
        assert isinstance(y_label, str) or y_label is None
        assert isinstance(ext, str)

        # Get the traces associated with the given key
        traces = self.traces[key]

        for index, t in enumerate(traces):
            # Obtain raw data
            time = np.asarray(t['time'])
            width = np.asarray(t['width'])
            results = t['result']  # List with result arrays

            # Create X values
            x_values = time + width / 2  # Points are plotted in the middle of the bin

            # Plotting defaults
            kwargs.setdefault('marker', 'o')
            kwargs.setdefault('linestyle', '')

            # Plot
            fig, ax = plt.subplots()
            for i, r in enumerate(results):
                ax.plot(x_values, r, label='Channel {:d}'.format(i), **kwargs)

            # Plot formatting
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.ticklabel_format(axis='x', scilimits=(0, 10))
            ax.legend(loc=legend_loc)

            # Save figure
            file_name = self._file_name_generator(self.PLOT_FILE_FORMAT.format(key=key, index=index), ext)
            fig.savefig(file_name, bbox_inches='tight')
            plt.close(fig)

    def plot_all_traces(self, **kwargs: typing.Any) -> None:
        """Plot traces for all keys available in the data.

        :param kwargs: Keyword arguments passed to :func:`plot_trace`
        """
        for key in self.traces:
            self.plot_trace(key, **kwargs)


class TimeResolvedSpectroscopyContextError(RuntimeError):
    """Class for context errors."""
    pass
