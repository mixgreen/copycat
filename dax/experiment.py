"""
This module is intended to be imported by users.
It will import the ARTIQ library (``from artiq.experiment import *``) and the system base components of DAX.

See :mod:`dax.base.system`.
"""

# Import DAX system base
from dax.base.system import *  # noqa: F401

# Import DAX calibration exceptions
from dax.base.exceptions import BadDataError, OutOfSpecError, FailedCalibrationError  # noqa: F401

# Import artiq.experiment
from artiq.experiment import *  # noqa: F401
