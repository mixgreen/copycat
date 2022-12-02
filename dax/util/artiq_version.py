import artiq

__all__ = ['ARTIQ_MAJOR_VERSION']

ARTIQ_MAJOR_VERSION: int = int(artiq.__version__[0])
"""The ARTIQ major version number."""
