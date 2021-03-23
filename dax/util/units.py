import typing
import string
import numbers
import numpy as np

import artiq.language.core
import artiq.language.units

__all__ = ['time_to_str', 'str_to_time',
           'freq_to_str', 'str_to_freq',
           'volt_to_str', 'str_to_volt',
           'ampere_to_str', 'str_to_ampere',
           'watt_to_str', 'str_to_watt',
           'UnitsFormatter']

_R_T = typing.Union[int, float, np.integer]  # Real number type


@artiq.language.core.host_only
def _value_to_str(value: _R_T, threshold: _R_T, precision: int, scales: typing.Sequence[str]) -> str:
    assert isinstance(value, numbers.Real), 'Input value must be a real'
    assert isinstance(threshold, numbers.Real), 'Threshold must be a real'
    assert isinstance(precision, int) and precision >= 0, 'Precision must be a positive int'

    # Take the abs of threshold
    threshold = abs(threshold)
    # The scaled value
    scaled_value = value

    for s in scales:
        # Scale time
        scaled_value = value / getattr(artiq.language.units, s)
        if abs(scaled_value) >= threshold:
            # Return value as string
            return f'{scaled_value:.{precision}f} {s}'

    # Using last scaling without rounding
    return f'{scaled_value} {scales[-1]}'


@artiq.language.core.host_only
def time_to_str(time: _R_T, *, threshold: _R_T = 1.0, precision: int = 2) -> str:
    """Convert a time to a string for pretty printing."""
    return _value_to_str(time, threshold, precision, ['s', 'ms', 'us', 'ns', 'ps'])


@artiq.language.core.host_only
def freq_to_str(frequency: _R_T, *, threshold: _R_T = 1.0, precision: int = 2) -> str:
    """Convert a frequency to a string for pretty printing."""
    return _value_to_str(frequency, threshold, precision, ['GHz', 'MHz', 'kHz', 'Hz', 'mHz'])


@artiq.language.core.host_only
def volt_to_str(volt: _R_T, *, threshold: _R_T = 1.0, precision: int = 2) -> str:
    """Convert a voltage to a string for pretty printing."""
    return _value_to_str(volt, threshold, precision, ['kV', 'V', 'mV', 'uV'])


@artiq.language.core.host_only
def ampere_to_str(ampere: _R_T, *, threshold: _R_T = 1.0, precision: int = 2) -> str:
    """Convert an amperage to a string for pretty printing."""
    return _value_to_str(ampere, threshold, precision, ['A', 'mA', 'uA'])


@artiq.language.core.host_only
def watt_to_str(watt: _R_T, *, threshold: _R_T = 1.0, precision: int = 2) -> str:
    """Convert a wattage to a string for pretty printing."""
    return _value_to_str(watt, threshold, precision, ['W', 'mW', 'uW'])


@artiq.language.core.host_only
def _str_to_value(string_: str, units: typing.Set[str]) -> float:
    assert isinstance(string_, str), 'Input must be of type str'

    try:
        # Split the string
        value, unit = string_.split()
    except ValueError as e:
        raise ValueError(f'String "{string_}" can not be tokenized correctly (missing a space?)') from e

    if unit not in units:
        raise ValueError(f'String "{string_}" does not contain a valid unit for this conversion')

    try:
        # Return the scaled value
        return float(value) * typing.cast(float, getattr(artiq.language.units, unit))
    except ValueError as e:
        raise ValueError(f'String "{string_}" does not contain a valid number') from e


@artiq.language.core.host_only
def str_to_time(string_: str) -> float:
    """Convert a string to a time."""
    return _str_to_value(string_, {'s', 'ms', 'us', 'ns', 'ps'})


@artiq.language.core.host_only
def str_to_freq(string_: str) -> float:
    """Convert a string to a frequency."""
    return _str_to_value(string_, {'GHz', 'MHz', 'kHz', 'Hz', 'mHz'})


@artiq.language.core.host_only
def str_to_volt(string_: str) -> float:
    """Convert a string to a voltage."""
    return _str_to_value(string_, {'kV', 'V', 'mV', 'uV'})


@artiq.language.core.host_only
def str_to_ampere(string_: str) -> float:
    """Convert a string to an amperage."""
    return _str_to_value(string_, {'A', 'mA', 'uA'})


@artiq.language.core.host_only
def str_to_watt(string_: str) -> float:
    """Convert a string to a wattage."""
    return _str_to_value(string_, {'W', 'mW', 'uW'})


class UnitsFormatter(string.Formatter):
    """String formatter supporting extended conversions.

    The following extra conversions are available:

    - ``'{!t}'``, conversion to time
    - ``'{!f}'``, conversion to frequency
    - ``'{!v}'``, conversion to volt
    - ``'{!a}'``, conversion to ampere (overrides default ascii conversion)
    - ``'{!w}'``, conversion to watt
    """

    def __init__(self, *, precision: int = 6):
        """Create a new units string formatter object.

        :param precision: The number of digits displayed after the decimal point
        """
        assert isinstance(precision, int) and precision >= 0, 'Precision must be equal or greater than zero'
        self._precision: int = precision

    def convert_field(self, value: typing.Any, conversion: str) -> typing.Any:
        if conversion == 't':
            return time_to_str(value, precision=self._precision)
        elif conversion == 'f':
            return freq_to_str(value, precision=self._precision)
        elif conversion == 'v':
            return volt_to_str(value, precision=self._precision)
        elif conversion == 'a':
            return ampere_to_str(value, precision=self._precision)
        elif conversion == 'w':
            return watt_to_str(value, precision=self._precision)
        else:
            return super(UnitsFormatter, self).convert_field(value, conversion)
