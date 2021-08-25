import typing
import configparser
import os.path

import dax.util.git

__all__ = ['get_dax_config']


class DaxConfigParser(configparser.ConfigParser):
    """DAX configuration parser class."""

    CONFIG_FILES: typing.ClassVar[typing.List[str]] = ['.dax', 'setup.cfg']
    """The locations of the configuration files in order of precedence."""

    __used_config_files: typing.FrozenSet[str]

    def __init__(self, *args: typing.Any, **kwargs: typing.Any):
        """Create a new DAX configuration parser object.

        :param args: Positional arguments for the underlying :class:`ConfigParser` object.
        :param kwargs: Keyword arguments for the underlying :class:`ConfigParser` object.
        """
        # Call super
        super(DaxConfigParser, self).__init__(*args, **kwargs)
        # Read configuration files
        self._read_config_files()

        if not self.__used_config_files and dax.util.git.in_repository():
            # Fallback on root of the Git repository (see https://github.com/m-labs/artiq/issues/1747)
            self._read_config_files(base=dax.util.git.get_repository_info().path)

    def _read_config_files(self, *, base: str = '') -> None:
        """Read configuration files, relative from the base path."""
        self.__used_config_files = frozenset(self.read(os.path.join(base, f) for f in reversed(self.CONFIG_FILES)))

    @property
    def used_config_files(self) -> typing.FrozenSet[str]:
        """Return the set of configuration files that populated this object."""
        return self.__used_config_files

    def optionxform(self, optionstr: str) -> str:
        """No option conversion, makes options case-sensitive."""
        return optionstr


_dax_config: typing.Optional[DaxConfigParser] = None
"""The cached DAX configuration object."""


def get_dax_config(*, clear_cache: bool = False) -> DaxConfigParser:
    """Get the DAX configuration object.

    Note: the configuration object is cached and any mutations will be shared.

    :param clear_cache: Clear the cache, forces files to be read again from disk
    :return: Populated DAX configuration object
    """
    assert isinstance(clear_cache, bool), 'Clear cache flag must be of type bool'

    global _dax_config

    if _dax_config is None or clear_cache:
        _dax_config = DaxConfigParser()

    return _dax_config
