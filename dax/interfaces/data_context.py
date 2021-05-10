import abc
import typing

from artiq.language import portable

from dax.base.interface import DaxInterface
import dax.util.artiq

__all__ = ['DataContextError', 'DataContextInterface', 'validate_interface']


class DataContextError(RuntimeError):
    """Class for data context errors."""
    pass


class DataContextInterface(DaxInterface, abc.ABC):
    """The data context interface is used to define batches during data collection."""

    @abc.abstractmethod
    def open(self):  # type: () -> None
        """Enter the data context manually.

        This function can be used to manually enter the data context.
        We strongly recommend to use the ``with`` statement instead.

        :raises DataContextError: Raised if the data context was already entered (context is non-reentrant)
        """
        pass

    @abc.abstractmethod
    def close(self):  # type: () -> None
        """Exit the data context manually.

        This function can be used to manually exit the data context.
        We strongly recommend to use the ``with`` statement instead.

        :raises DataContextError: Raised if the data context was not entered
        """
        pass

    @portable
    def __enter__(self):  # type: () -> None
        """Enter the data context (see :func:`open`)."""
        self.open()

    @portable  # noqa:ATQ306
    def __exit__(self, exc_type, exc_val, exc_tb):  # type: (typing.Any, typing.Any, typing.Any) -> None  # noqa:ATQ306
        """Exit the data context (see :func:`close`)."""
        self.close()

    @abc.abstractmethod
    def get_raw(self) -> typing.Sequence[typing.Sequence[typing.Sequence[int]]]:
        """Obtain the raw data captured by the data context.

        **This function can only be called from the host.**

        Data is formatted as a 3-dimensional list.
        To access the measurement of data context N of data point P of qubit C: ``get_raw()[N][P][C]``.

        :return: Raw measurement data
        """
        pass


def validate_interface(data_context: DataContextInterface) -> bool:
    """Validate a data context interface object.

    :param data_context: The data context interface object
    :return: :const:`True`, to allow usage of this function in an ``assert`` statement
    :raises TypeError: Raised if validation failed
    """
    if not isinstance(data_context, DataContextInterface):
        raise TypeError('The provided interface is not of type DataContextInterface')

    # Validate not host only functions
    not_host_only_fn: typing.Set[str] = {'open', 'close', '__enter__', '__exit__'}
    if not all(not dax.util.artiq.is_host_only(getattr(data_context, fn, None)) for fn in not_host_only_fn):
        raise TypeError(f'The following functions can not be host only: {", ".join(not_host_only_fn)}')

    # Validate host only functions
    host_only_fn: typing.Set[str] = {'get_raw'}
    if not all(dax.util.artiq.is_host_only(getattr(data_context, fn, None)) for fn in host_only_fn):
        raise TypeError(f'The following functions must be host only: {", ".join(host_only_fn)}')

    # Return True
    return True
