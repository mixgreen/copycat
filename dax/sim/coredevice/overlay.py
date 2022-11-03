import typing

__all__ = ['Overlay']


class Overlay:
    """Overlay an object to capture calls."""

    def __init__(self, parent: typing.Any, obj: typing.Any):
        self.__parent = parent
        self.__obj = obj

    @property
    def _parent(self) -> typing.Any:
        """The parent of this overlay object."""
        return self.__parent

    @property
    def _obj(self) -> typing.Any:
        """The internal object of the overlay, used to forward captured calls."""
        return self.__obj

    def __getattr__(self, item: typing.Any) -> typing.Any:
        # Forward call to internal object
        return getattr(self._obj, item)
