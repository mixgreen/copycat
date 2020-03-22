from artiq.language.core import *
from artiq.language.units import *


@host_only
def _value_to_str(value, threshold, precision, scales):
    assert isinstance(value, float), 'Input value must be of type float'
    assert isinstance(threshold, float), 'Threshold must be a float'
    assert isinstance(precision, int) and precision >= 0, 'Precision must be a positive int'

    # Take the abs of threshold
    threshold = abs(threshold)
    # Prepare the string template
    template = '{{:.{:d}f}} {{:s}}'.format(precision)
    # The scaled value
    scaled_value = value

    for s in scales:
        # Scale time
        scaled_value = value / globals()[s]
        if abs(scaled_value) >= threshold:
            # Return value as string
            return template.format(scaled_value, s)

    # Using last scaling without rounding
    return '{:f} {:s}'.format(scaled_value, scales[-1])


@host_only
def time_to_str(time, threshold=1.0, precision=2):
    """Convert a time to a string for pretty printing."""
    return _value_to_str(time, threshold, precision, ['s', 'ms', 'us', 'ns', 'ps'])


@host_only
def freq_to_str(frequency, threshold=1.0, precision=2):
    """Convert a frequency to a string for pretty printing."""
    return _value_to_str(frequency, threshold, precision, ['GHz', 'MHz', 'kHz', 'Hz', 'mHz'])


@host_only
def _str_to_value(string, units):
    assert isinstance(string, str), 'Input must be of type str'

    try:
        # Split the string
        value, unit = string.split()
    except ValueError as e:
        raise ValueError('String "{:s}" can not be tokenized correctly (missing a space?)'.format(string)) from e

    if unit not in units:
        raise ValueError('String "{:s}" does not contain a valid unit for this conversion'.format(string))

    try:
        # Return the scaled value
        return float(value) * globals()[unit]
    except ValueError as e:
        raise ValueError('String "{:s}" does not contain a valid number'.format(string)) from e


@host_only
def str_to_time(string):
    """Convert a string to a time."""
    return _str_to_value(string, {'s', 'ms', 'us', 'ns', 'ps'})


@host_only
def str_to_freq(string):
    """Convert a string to a frequency."""
    return _str_to_value(string, {'GHz', 'MHz', 'kHz', 'Hz', 'mHz'})
