"""
This module is intended to be imported by users.
It will import the ARTIQ library (``from artiq.experiment import *``) and the base components of DAX.
"""

# Import DAX base
from dax.base.dax import *  # noqa: F401

# Import artiq.experiment
from artiq.experiment import *  # noqa: F401
