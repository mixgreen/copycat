__all__ = ['BuildError', 'NonUniqueRegistrationError']


class BuildError(RuntimeError):
    """Raised when the original build error has already been logged.

    A new exception type is used to prevent that a build error is logged multiple times.
    """
    pass


class NonUniqueRegistrationError(LookupError):
    """Exception when a name is registered more then once.

    Raised when registering an object with a name that was already occupied.
    """
    pass
