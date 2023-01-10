import os.path
from ._version import get_version

__dax_dir__: str = str(os.path.dirname(os.path.abspath(__file__)))
"""Directory of the DAX library."""
__version__: str = str(get_version())
"""DAX version string."""

del os
del get_version
