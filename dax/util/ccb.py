import typing

from artiq.experiment import HasEnvironment

__all__ = ['get_ccb_tool']


def _generate_command(base_command: str, **kwargs: typing.Any) -> str:
    """Generate a command string.

    Underscores in argument names are converted to hyphens.

    :param base_command: The fixed part of the command
    :param kwargs: Optional arguments
    :return: The command string
    """
    # Convert kwargs to string arguments if not None
    arguments = ('--{:s} "{}"'.format(a.replace('_', '-'), v) for a, v in kwargs.items() if v is not None)
    # Return final command
    return '{:s} {:s}'.format(base_command, ' '.join(arguments))


class CcbTool:
    """Wrapper around ARTIQ CCB object providing more convenient functions."""

    ARTIQ_APPLET = '${artiq_applet}'
    """The ARTIQ applet variable which can be used in CCB commands."""

    def __init__(self, ccb: typing.Any):
        """Construct a new CCB tool.

        :param ccb: The CCB object
        """

        # Store the CCB object
        self.ccb = ccb

    def issue(self, action: str, *args: typing.Any, **kwargs: typing.Any) -> None:
        """Plain CCB issue command.

        :param action: The action to perform (i.e. name of the broadcast)
        :param args: Positional arguments for the broadcast
        :param kwargs: Keyword arguments for the broadcast
        """
        self.ccb.issue(action, *args, **kwargs)

    def create_applet(self, name: str, command: str, group: typing.Optional[str] = None,
                      code: typing.Optional[str] = None) -> None:
        """Create an applet.

        :param name: Name of the applet
        :param command: Command to run the applet
        :param group: Optional group of the applet
        :param code: Optional source code of the applet
        """
        self.issue('create_applet', name, command, group=group, code=code)

    def disable_applet(self, name: str, group: typing.Optional[str] = None) -> None:
        """Disable an applet.

        :param name: Name of the applet
        :param group: Optional group of the applet
        """
        self.issue('disable_applet', name, group=group)

    def disable_applet_group(self, group: str) -> None:
        """Disable an applet group.

        :param group: Group name of the applets
        """
        self.issue('disable_applet_groups', group)

    """Functions that directly create standard ARTIQ applets"""

    def big_number(self, name: str, dataset: str, digit_count: typing.Optional[int] = None,
                   update_delay: typing.Optional[float] = None, group: typing.Optional[str] = None,
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
        command = '{:s}big_number {:s}'.format(self.ARTIQ_APPLET, dataset)
        command = _generate_command(command, digit_count=digit_count, update_delay=update_delay, **kwargs)
        # Create applet
        self.create_applet(name, command, group=group)

    def image(self, name: str, img: str,
              update_delay: typing.Optional[float] = None, group: typing.Optional[str] = None,
              **kwargs: typing.Any) -> None:
        """Create an image applet.

        :param name: Name of the applet
        :param img: Image data (2D numpy array) dataset
        :param update_delay: Time to wait after a modification before updating graph
        :param group: Optional group of the applet
        :param kwargs: Other optional arguments for the applet
        """
        # Assemble command
        command = '{:s}image {:s}'.format(self.ARTIQ_APPLET, img)
        command = _generate_command(command, update_delay=update_delay, **kwargs)
        # Create applet
        self.create_applet(name, command, group=group)

    def plot_xy(self, name: str, y: str, x: typing.Optional[str] = None,
                error: typing.Optional[str] = None, fit: typing.Optional[str] = None,
                title: typing.Optional[str] = None,
                update_delay: typing.Optional[float] = None, group: typing.Optional[str] = None,
                **kwargs: typing.Any) -> None:
        """Create a plot XY applet.

        :param name: Name of the applet
        :param y: Y-value dataset
        :param x: X-value dataset
        :param error: Error dataset
        :param fit: Fit dataset
        :param title: Graph title
        :param update_delay: Time to wait after a modification before updating graph
        :param group: Optional group of the applet
        :param kwargs: Other optional arguments for the applet
        """
        # Assemble command
        command = '{:s}plot_xy {:s}'.format(self.ARTIQ_APPLET, y)
        command = _generate_command(command, x=x, error=error, fit=fit, title=title, update_delay=update_delay,
                                    **kwargs)
        # Create applet
        self.create_applet(name, command, group=group)

    def plot_hist(self, name: str, y: str, x: typing.Optional[str] = None,
                  title: typing.Optional[str] = None,
                  update_delay: typing.Optional[float] = None, group: typing.Optional[str] = None,
                  **kwargs: typing.Any) -> None:
        """Create a plot histogram applet.

        :param name: Name of the applet
        :param y: Y-value dataset
        :param x: Bin boundaries dataset
        :param title: Graph title
        :param update_delay: Time to wait after a modification before updating graph
        :param group: Optional group of the applet
        :param kwargs: Other optional arguments for the applet
        """
        # Assemble command
        command = '{:s}plot_hist {:s}'.format(self.ARTIQ_APPLET, y)
        command = _generate_command(command, x=x, title=title, update_delay=update_delay, **kwargs)
        # Create applet
        self.create_applet(name, command, group=group)

    def plot_xy_hist(self, name: str, xs: str, histogram_bins: str, histogram_counts: str,
                     update_delay: typing.Optional[float] = None, group: typing.Optional[str] = None,
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
        # Assemble command
        command = '{:s}plot_xy_hist {:s} {:s} {:s}'.format(self.ARTIQ_APPLET, xs, histogram_bins, histogram_counts)
        command = _generate_command(command, update_delay=update_delay, **kwargs)
        # Create applet
        self.create_applet(name, command, group=group)


def get_ccb_tool(manager_or_parent: HasEnvironment) -> CcbTool:
    """Obtain a CCB tool.

    The CCB tool is a wrapper around the ARTIQ CCB object that allows users to conveniently
    create standard applets using straight-forward functions.

    :param manager_or_parent: Manager or parent object, required to get the ARTIQ CCB object
    :return: The CCB tool object
    """
    return CcbTool(manager_or_parent.get_device('ccb'))
