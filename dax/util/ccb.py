import typing

from artiq.experiment import HasEnvironment

__all__ = ['get_ccb_tool']

_G_T = typing.Union[str, typing.List[str]]  # Type of a group


def _convert_group(group: typing.Optional[_G_T]) -> typing.Optional[_G_T]:
    """Convert a group string to the desired format for ARTIQ applet hierarchies.

    Enables users to define group hierarchies using the dot "." character,
    similar as with datasets.

    :param group: The group name as a single string or None
    """
    # Strings are split to enable applet group hierarchies in the dashboard
    return group.split('.') if isinstance(group, str) else group


def _generate_command(base_command: str, *args: str, **kwargs: typing.Any) -> str:
    """Generate a command string.

    :param base_command: The fixed part of the command
    :param args: Positional arguments
    :param kwargs: Optional arguments
    :return: The command string
    """
    # Convert underscores in arguments, remove single quotes from strings, discard None and False values
    kwargs = {a.replace('_', '-').replace("'", ""): v.replace("'", "") if isinstance(v, str) else v
              for a, v in kwargs.items() if v is not None and v is not False}
    # Convert to command argument format
    optional_arguments = (f'--{a}' if v is True else f"--{a} '{v}'" for a, v in kwargs.items())
    # Convert positional arguments
    positional_arguments = (f"'{a}'" for a in args)
    # Return final command
    return f"{base_command} {' '.join(positional_arguments)} {' '.join(optional_arguments)}"


class CcbTool:
    """Wrapper around ARTIQ CCB object providing more convenient functions."""

    ARTIQ_APPLET: typing.ClassVar[str] = '${artiq_applet}'
    """The ARTIQ applet variable which can be used in CCB commands."""
    DAX_APPLET: typing.ClassVar[str] = '${python} -m dax_applets.'
    """The DAX applet starting command which can be used in CCB commands."""

    __ccb: typing.Any

    def __init__(self, ccb: typing.Any):
        """Construct a new CCB tool.

        :param ccb: The CCB object
        """

        # Store the CCB object
        self.__ccb = ccb

    @property
    def ccb(self) -> typing.Any:
        """Return the underlying ARTIQ CCB object.

        :return: The ARTIQ CCB object
        """
        return self.__ccb

    def issue(self, action: str, *args: typing.Any, **kwargs: typing.Any) -> None:
        """Plain CCB issue command.

        :param action: The action to perform (i.e. name of the broadcast)
        :param args: Positional arguments for the broadcast
        :param kwargs: Keyword arguments for the broadcast
        """
        self.ccb.issue(action, *args, **kwargs)

    def create_applet(self, name: str, command: str, group: typing.Optional[_G_T] = None,
                      code: typing.Optional[str] = None) -> None:
        """Create an applet.

        :param name: Name of the applet
        :param command: Command to run the applet
        :param group: Optional group of the applet
        :param code: Optional source code of the applet
        """
        self.issue('create_applet', name, command, group=_convert_group(group), code=code)

    def disable_applet(self, name: str, group: typing.Optional[_G_T] = None) -> None:
        """Disable an applet.

        :param name: Name of the applet
        :param group: Optional group of the applet
        """
        self.issue('disable_applet', name, group=_convert_group(group))

    def disable_applet_group(self, group: _G_T) -> None:
        """Disable an applet group.

        :param group: Group name of the applets
        """
        self.issue('disable_applet_group', _convert_group(group))

    """Functions that directly create ARTIQ applets"""

    def big_number(self, name: str, dataset: str, *,
                   digit_count: typing.Optional[int] = None,
                   update_delay: typing.Optional[float] = None,
                   group: typing.Optional[_G_T] = None,
                   **kwargs: typing.Any) -> None:
        """Create a big number applet.

        :param name: Name of the applet
        :param dataset: Dataset to show
        :param digit_count: Total number of digits to show
        :param update_delay: Time to wait after a modification before updating graph
        :param group: Optional group of the applet
        :param kwargs: Other optional arguments for the applet
        """
        # Assemble command
        command = _generate_command(f'{self.DAX_APPLET}big_number', dataset,
                                    digit_count=digit_count, update_delay=update_delay, **kwargs)
        # Create applet
        self.create_applet(name, command, group=group)

    def image(self, name: str, img: str, *,
              update_delay: typing.Optional[float] = None,
              group: typing.Optional[_G_T] = None,
              **kwargs: typing.Any) -> None:
        """Create an image applet.

        :param name: Name of the applet
        :param img: Image data (2D numpy array) dataset
        :param update_delay: Time to wait after a modification before updating graph
        :param group: Optional group of the applet
        :param kwargs: Other optional arguments for the applet
        """
        if update_delay is None:
            # Set update delay explicit for ARTIQ applets
            update_delay = 0.1
        # Assemble command
        command = _generate_command(f'{self.ARTIQ_APPLET}image', img, update_delay=update_delay, **kwargs)
        # Create applet
        self.create_applet(name, command, group=group)

    def plot_xy(self, name: str, y: str, *,
                x: typing.Optional[str] = None,
                error: typing.Optional[str] = None,
                fit: typing.Optional[str] = None,
                v_lines: typing.Optional[str] = None,
                h_lines: typing.Optional[str] = None,
                sliding_window: typing.Optional[int] = None,
                crosshair: typing.Optional[bool] = None,
                last: typing.Optional[bool] = None,
                title: typing.Optional[str] = None,
                x_label: typing.Optional[str] = None,
                y_label: typing.Optional[str] = None,
                update_delay: typing.Optional[float] = None,
                group: typing.Optional[_G_T] = None,
                **kwargs: typing.Any) -> None:
        """Create a plot XY applet.

        :param name: Name of the applet
        :param y: Y-value dataset
        :param x: X-value dataset
        :param error: Error dataset
        :param fit: Fit dataset
        :param v_lines: Vertical lines dataset
        :param h_lines: Horizontal lines dataset
        :param sliding_window: Set size of the sliding window, or :const:`None` to disable
        :param crosshair: Enable crosshair feature
        :param last: Show the last value in the title
        :param title: Graph title
        :param x_label: X-axis label
        :param y_label: Y-axis label
        :param update_delay: Time to wait after a modification before updating graph
        :param group: Optional group of the applet
        :param kwargs: Other optional arguments for the applet
        """
        # Assemble command
        command = _generate_command(f'{self.DAX_APPLET}plot_xy', y,
                                    x=x, error=error, fit=fit, v_lines=v_lines, h_lines=h_lines,
                                    sliding_window=sliding_window, title=title, crosshair=crosshair, last=last,
                                    x_label=x_label, y_label=y_label, update_delay=update_delay, **kwargs)
        # Create applet
        self.create_applet(name, command, group=group)

    def plot_xy_multi(self, name: str, y: str, *,
                      x: typing.Optional[str] = None,
                      error: typing.Optional[str] = None,
                      v_lines: typing.Optional[str] = None,
                      h_lines: typing.Optional[str] = None,
                      index: typing.Optional[int] = None,
                      sliding_window: typing.Optional[int] = None,
                      plot_names: typing.Optional[str] = None,
                      markers_only: typing.Optional[bool] = None,
                      title: typing.Optional[str] = None,
                      x_label: typing.Optional[str] = None,
                      y_label: typing.Optional[str] = None,
                      update_delay: typing.Optional[float] = None,
                      group: typing.Optional[_G_T] = None,
                      **kwargs: typing.Any) -> None:
        """Create a plot XY applet with multiple plots.

        :param name: Name of the applet
        :param y: Y-values dataset (multiple graphs)
        :param x: X-value dataset
        :param error: Error dataset (multiple graphs)
        :param v_lines: Vertical lines dataset
        :param h_lines: Horizontal lines dataset
        :param index: The index of the results to plot (default plots all)
        :param sliding_window: Set size of the sliding window, or :const:`None` to disable
        :param plot_names: Base names of the plots (numbered automatically, formatting with ``'{index}'`` possible)
        :param markers_only: Only plot markers and no lines between them
        :param title: Graph title
        :param x_label: X-axis label
        :param y_label: Y-axis label
        :param update_delay: Time to wait after a modification before updating graph
        :param group: Optional group of the applet
        :param kwargs: Other optional arguments for the applet
        """
        # Assemble command
        command = _generate_command(f'{self.DAX_APPLET}plot_xy_multi', y,
                                    x=x, error=error, v_lines=v_lines, h_lines=h_lines, index=index,
                                    sliding_window=sliding_window, plot_names=plot_names, markers_only=markers_only,
                                    title=title, x_label=x_label, y_label=y_label, update_delay=update_delay, **kwargs)
        # Create applet
        self.create_applet(name, command, group=group)

    def plot_hist(self, name: str, y: str, *,
                  index: typing.Optional[int] = None,
                  plot_names: typing.Optional[str] = None,
                  title: typing.Optional[str] = None,
                  x_label: typing.Optional[str] = None,
                  y_label: typing.Optional[str] = None,
                  update_delay: typing.Optional[float] = None,
                  group: typing.Optional[_G_T] = None,
                  **kwargs: typing.Any) -> None:
        """Create a plot histogram applet using DAX specific data formatting.

        :param name: Name of the applet
        :param y: Histogram dataset
        :param index: The index of the results to plot (default plots all)
        :param plot_names: Base names of the plots (numbered automatically, formatting with ``'{index}'`` possible)
        :param title: Graph title
        :param x_label: X-axis label
        :param y_label: Y-axis label
        :param update_delay: Time to wait after a modification before updating graph
        :param group: Optional group of the applet
        :param kwargs: Other optional arguments for the applet
        """
        # Assemble command
        command = _generate_command(f'{self.DAX_APPLET}plot_hist', y,
                                    index=index, plot_names=plot_names, title=title,
                                    x_label=x_label, y_label=y_label, update_delay=update_delay, **kwargs)
        # Create applet
        self.create_applet(name, command, group=group)

    def plot_hist_artiq(self, name: str, y: str, *,
                        x: typing.Optional[str] = None,
                        title: typing.Optional[str] = None,
                        update_delay: typing.Optional[float] = None,
                        group: typing.Optional[_G_T] = None,
                        **kwargs: typing.Any) -> None:
        """Create an ARTIQ plot histogram applet.

        :param name: Name of the applet
        :param y: Y-value dataset
        :param x: Bin boundaries dataset
        :param title: Graph title
        :param update_delay: Time to wait after a modification before updating graph
        :param group: Optional group of the applet
        :param kwargs: Other optional arguments for the applet
        """
        if update_delay is None:
            # Set update delay explicit for ARTIQ applets
            update_delay = 0.1
        # Assemble command
        command = _generate_command(f'{self.ARTIQ_APPLET}plot_hist', y,
                                    x=x, title=title, update_delay=update_delay, **kwargs)
        # Create applet
        self.create_applet(name, command, group=group)

    def plot_xy_hist(self, name: str, xs: str, histogram_bins: str, histogram_counts: str, *,
                     update_delay: typing.Optional[float] = None,
                     group: typing.Optional[_G_T] = None,
                     **kwargs: typing.Any) -> None:
        """Create a 2D histogram applet.

        :param name: Name of the applet
        :param xs: 1D array of point abscissas dataset
        :param histogram_bins: 1D array of histogram bin boundaries dataset
        :param histogram_counts: 2D array of histogram counts (for each point) dataset
        :param update_delay: Time to wait after a modification before updating graph
        :param group: Optional group of the applet
        :param kwargs: Other optional arguments for the applet
        """
        if update_delay is None:
            # Set update delay explicit for ARTIQ applets
            update_delay = 0.1
        # Assemble command
        command = _generate_command(f'{self.ARTIQ_APPLET}plot_xy_hist', xs, histogram_bins, histogram_counts,
                                    update_delay=update_delay, **kwargs)
        # Create applet
        self.create_applet(name, command, group=group)


def get_ccb_tool(environment: HasEnvironment) -> CcbTool:
    """Obtain a CCB tool.

    The CCB tool is a wrapper around the ARTIQ CCB object that allows users to conveniently
    create standard applets using straight-forward functions.

    :param environment: An object which inherits ARTIQ :class:`HasEnvironment`, required to get the ARTIQ CCB object
    :return: The CCB tool object
    """
    assert isinstance(environment, HasEnvironment), 'The given environment must be of type HasEnvironment'
    return CcbTool(environment.get_device('ccb'))
