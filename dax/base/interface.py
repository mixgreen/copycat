import abc
import typing
import inspect
import itertools

import artiq.language.core

__all__ = ['DaxInterface', 'optional', 'is_optional', 'get_optionals']

_FN_T = typing.TypeVar('_FN_T', bound=typing.Callable[..., typing.Any])  # Function type; type variable

_OPTIONAL_METHOD_KEY: str = '__dax_interface_optional'


class DaxInterface(abc.ABC):  # pragma: no cover
    """Base class for interfaces."""
    pass


def optional(fn: _FN_T) -> _FN_T:
    """A decorator indicating optional methods.

    Indicate a method is abstract, but optional to implement.
    Optional methods must raise a :class:`NotImplementedError`
    and can not be called directly or through ``super``.
    When an optional method is overridden, the optional decorator is not inherited.

    :param fn: The function to decorate
    :return: The decorated function
    """
    # Mark the function as optional
    setattr(fn, _OPTIONAL_METHOD_KEY, True)
    # Return the function decorated as portable (results in a sensible compile/runtime error)
    return artiq.language.core.portable(fn)


def is_optional(fn: typing.Any) -> bool:
    """Returns if a method is decorated as optional or not.

    :param fn: The function of interest
    :return: :const:`True` if the given function is optional
    """
    return getattr(fn, _OPTIONAL_METHOD_KEY, False) is True


def get_optionals(obj: typing.Any) -> typing.Set[str]:
    """Get a set of optional methods given an object.

    :param obj: The object to inspect, can be an instance or a class
    :return: The set of optional methods as strings
    :raises ValueError: Raised if the given class is invalid
    """
    # Inspect both functions and methods to be compatible with classes, instances, and static methods
    items = itertools.chain(inspect.getmembers(obj, inspect.isfunction), inspect.getmembers(obj, inspect.ismethod))
    return {k for k, m in items if is_optional(m)}
