import numpy as np

from artiq.language import portable, TList, TInt32, TInt64

__all__ = ['list_to_int32', 'list_to_int64']


@portable
def list_to_int32(measurements: TList(TInt32)) -> TInt32:  # type: ignore
    """Convert a list of measurement results to an ``int32``.

    **Note that the 32th element of the list can flip the sign due to the usage of a signed data type**.
    Element 0 in the list is the LSB.
    Measurements are of type ``int32`` and are interpreted as boolean values.
    If the list contains more than 32 elements, only the first 32 elements will be considered.

    :param measurements: List of measurements
    :return: An ``int32`` that packs measurements in binary form (``int32`` NumPy type on host)
    """
    # Initialize result and mask
    r = np.int32(0)
    mask = np.int32(0x1)

    for i in range(min(len(measurements), 32)):  # Process up to 32 elements
        if measurements[i]:  # Implicit conversion to bool
            r |= mask
        mask <<= np.int32(1)

    # Return the result
    return r


@portable
def list_to_int64(measurements: TList(TInt32)) -> TInt64:  # type: ignore
    """Convert a list of measurement results to an ``int64``.

    **Note that the 64th element of the list can flip the sign due to the usage of a signed data type**.
    Element 0 in the list is the LSB.
    Measurements are of type ``int32`` and are interpreted as boolean values.
    If the list contains more than 64 elements, only the first 64 elements will be considered.

    :param measurements: List of measurements
    :return: An ``int64`` that packs measurements in binary form (``int64`` NumPy type on host)
    """
    # Initialize result and mask
    r = np.int64(0)
    mask = np.int64(0x1)

    for i in range(min(len(measurements), 64)):  # Process up to 64 elements
        if measurements[i]:  # Implicit conversion to bool
            r |= mask
        mask <<= np.int32(1)

    # Return the result
    return r
