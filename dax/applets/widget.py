import typing

import PyQt5  # type: ignore  # noqa: F401
import pyqtgraph  # type: ignore

__all__ = ['PlotWidget']


class NoDefault:
    """Class used to indicate that no default was set."""
    pass


class SkipUpdateException(KeyError):
    """Raised when this update will be skipped."""
    pass


class PlotWidget(pyqtgraph.PlotWidget):
    """Minor extension over the regular PlotWidget with a few extra conveniences.

    Documentation about plotting functions can be found at:
    http://www.pyqtgraph.org/documentation/graphicsItems/plotitem.html#pyqtgraph.PlotItem.plot
    """

    def __init__(self, args: typing.Any):
        """The init function as it is called by ARTIQ.

        :param args: The arguments from argparse
        """

        # Call super
        pyqtgraph.PlotWidget.__init__(self)
        # Store args
        self.__args = args
        # Dict with data, acts as a buffer
        self.__data_buffer = {}  # type: typing.Dict[str, typing.Any]

    def update_applet(self, args):
        """This function replaces the :func:`data_changed` function and will be called whenever data changes.

        Originally the :func:`data_changed` function would be implemented for custom applets.
        Now the the :func:`data_changed` function provides generic functionality and instead
        custom applets should overwrite this method.

        The signature of this method changed and data can now be
        accessed using the :func:`get_data` function and the title is already set.

        :param args: The arguments object returned by argparse
        """
        raise NotImplementedError  # Not using abc to prevent metaclass conflict

    def get_dataset(self, key: str, default: typing.Any = NoDefault) -> typing.Any:
        """Get data from the latest buffer.

        If the data is not available and no default was set, the update function will gracefully return.

        :param key: The key
        :param default: A default value if desired
        :return: The requested value or the default if given
        """
        try:
            # Try to return the data
            return self.__data_buffer[key][1]  # Extra index required
        except KeyError:
            if default is NoDefault:
                # Raise if no default was set
                raise SkipUpdateException
            else:
                return default

    def data_changed(self, data: typing.Dict[str, typing.Any], mods: typing.List[typing.Any],
                     title: typing.Optional[str] = None) -> None:
        """This function is called when a subscribed dataset changes.

        It now provides some standard functionality and custom applets should override
        the :func:`update_applet` function.

        :param data: Raw data in the form of a dict
        :param mods: A list of unknown objects
        :param title: The title, if this is a TitleApplet and the title was set
        """
        assert isinstance(data, dict)
        assert isinstance(mods, list)
        assert isinstance(title, str) or title is None

        # Store data in the buffer
        self.__data_buffer = data

        # Set the title
        self.setTitle(title)  # Accepts None

        try:
            # Call the update function
            self.update_applet(self.__args)
        except SkipUpdateException:
            pass
