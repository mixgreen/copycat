import logging

import artiq.language.core

import dax.util.artiq

__all__ = ['decorate_logger_class']


def decorate_logger_class(class_: type, *, force: bool = False) -> None:
    """Decorate the given logger class to make it async RPC capable.

    :param class_: The logger class to decorate
    :param force: Set to :const:`True` to also decorate functions that were already decorated
    """
    assert issubclass(class_, logging.Logger), 'The given class must be a subclass of logging.Logger'

    # Decorate functions (mutates existing functions)
    for fn in ['debug', 'info', 'warning', 'error', 'critical', 'exception', 'log']:
        logger_fn = getattr(class_, fn)
        if force or not dax.util.artiq.is_decorated(logger_fn):
            artiq.language.core.rpc(logger_fn, flags={'async'})
