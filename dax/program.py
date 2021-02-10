"""
This module is intended to be imported by users.
It will import the ARTIQ library (``from artiq.experiment import *``) and the components for DAX.program.
"""

# Import DAX program base
from dax.base.program import *  # noqa: F401

# Import utility functions
from dax.util.convert import *  # noqa: F401

# Import artiq.experiment
from artiq.experiment import *  # noqa: F401
