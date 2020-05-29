import unittest
import logging
import typing
import collections

from artiq.experiment import HasEnvironment
from artiq.master.databases import device_db_from_file  # type: ignore

from dax.util.artiq_helpers import get_manager_or_parent
from dax.sim import enable_dax_sim
from dax.sim.signal import get_signal_manager

__all__ = ['PeekTestCase']

_logger = logging.getLogger(__name__)
"""The module logger object."""


class PeekTestCase(unittest.TestCase):
    DEFAULT_DEVICE_DB = 'device_db.py'  # type: str
    """The path of the default device DB used to construct environments."""

    __E_T = typing.TypeVar('__E_T', bound=HasEnvironment)  # Type variable for environment

    def construct_env(self, env_class: typing.Type[__E_T],
                      device_db: typing.Optional[str] = None,
                      logging_level: typing.Union[int, str] = logging.WARNING,
                      build_args: typing.Sequence[typing.Any] = None,
                      build_kwargs: typing.Dict[str, typing.Any] = None,
                      **env_kwargs: typing.Any) -> __E_T:
        """Construct an ARTIQ environment based on the given class.

        :param env_class: The environment class to construct
        :param device_db: The device DB to use (defaults to :attr:`DEFAULT_DEVICE_DB`)
        :param logging_level: The desired logging level
        :param env_kwargs: Keyword arguments passed to the constructed environment
        :return: The constructed ARTIQ environment object
        """

        # Set default values
        if build_args is None:
            build_args = tuple()
        if build_kwargs is None:
            build_kwargs = dict()

        assert issubclass(env_class, HasEnvironment), 'The environment class must be a subclass of HasEnvironment'
        assert isinstance(device_db, str) or device_db is None, 'Device DB must be of type str or None'
        assert isinstance(logging_level, (int, str)), 'Logging level must be of type int or str'
        assert isinstance(build_args, collections.abc.Sequence), 'Build arguments must be a sequence'
        assert isinstance(build_kwargs, dict), 'Build keyword arguments must be a dict'

        # Set level of module logger
        _logger.setLevel(logging_level)

        # Construct an expid object
        expid = {'log_level': logging_level,
                 'class_name': env_class.__name__,
                 'repo_rev': 'N/A'}  # type: typing.Dict[str, typing.Any]

        # Obtain device DB from file
        ddb = device_db_from_file(self.DEFAULT_DEVICE_DB if device_db is None else device_db)
        # Convert and configure device DB
        enable_dax_sim(enable=True, ddb=ddb, logging_level=logging_level, output='peek')

        # Construct environment, which will also construct a new signal manager
        env = env_class(get_manager_or_parent(ddb, expid, **env_kwargs), *build_args, **build_kwargs)

        # Store the new signal manager
        self.__signal_manager = get_signal_manager()

        # Return the environment
        return env

    def peek(self) -> typing.Any:
        pass

    def expect(self) -> None:
        pass
