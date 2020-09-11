import os
from ._version import get_versions

__dax_dir__: str = str(os.path.dirname(os.path.abspath(__file__)))
"""Directory of the DAX library."""
__version__: str = str(get_versions()['version']).replace('_', '+')
"""DAX version string."""

del os
del get_versions
