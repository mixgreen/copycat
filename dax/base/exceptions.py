__all__ = ['BuildError', 'NonUniqueRegistrationError',
           'CalibrationError', 'OutOfSpecError', 'BadDataError', 'FailedCalibrationError']


class BuildError(RuntimeError):
    """Raised when the original build error has already been logged.

    A new exception type is used to prevent that a build error is logged multiple times.
    """
    pass


class NonUniqueRegistrationError(LookupError):
    """Exception when a name is registered more than once.

    Raised when registering an object with a name that was already occupied.
    """
    pass


class CalibrationError(RuntimeError):
    """Base class for calibration-related errors."""
    pass


class OutOfSpecError(CalibrationError):
    """Exception for `check_data` experiments to throw in the case that a parameter is out of spec."""
    pass


class BadDataError(CalibrationError):
    """Exception for `check_data` experiments to throw in the case of bad data."""
    pass


class FailedCalibrationError(CalibrationError):
    """Exception to throw when a calibration has failed without resolution."""
    pass
