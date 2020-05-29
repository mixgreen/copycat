import unittest
import logging
import typing
import collections
import numpy as np

from artiq.experiment import HasEnvironment, now_mu
from artiq.master.databases import device_db_from_file  # type: ignore

from dax.util.artiq_helpers import get_manager_or_parent
from dax.sim import enable_dax_sim
from dax.sim.signal import get_signal_manager, PeekSignalManager, SignalNotSet

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
        :param build_args: Positional arguments passed to the build function of the environment
        :param build_kwargs: Keyword arguments passed to the build function of the environment
        :param env_kwargs: Keyword arguments passed to the argument parser of the environment
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
        self.__signal_manager = typing.cast(PeekSignalManager, get_signal_manager())
        assert isinstance(self.__signal_manager, PeekSignalManager), 'Did not obtained correct signal manager type'

        # Return the environment
        return env

    def peek(self, scope: typing.Any, signal: str) -> typing.Any:
        """Peek a signal of a device at the current time.

        :param scope: The scope (device) of the signal
        :param signal: The name of the signal
        :return: The value of the signal at the current time.
        """
        # TODO, check the scope type
        # TODO, value matching (e.g. x and X)
        # TODO, disable info message from ddb
        # TODO, handle signal not set
        # Return the value
        return self.__signal_manager.peek(scope, signal)

    def expect(self, scope: typing.Any, signal: str, value: typing.Any, msg: typing.Optional[str] = None) -> None:
        """Test if a signal holds a given value at the current time.

        :param scope: The scope (device) of the signal
        :param signal: The name of the signal
        :param value: The expected value
        :param msg: Message to show when this assertion fails
        """
        # Get the value and the type
        peek, type_ = self.__signal_manager.peek_and_type(scope, signal)

        if type_ in {object, str}:
            # Raise if the signal has an invalid type
            raise TypeError('Signal "{:s}.{:s}" type "{}" can not be tested'.format(scope.key, signal, type_))

        # Match with special values for supported types
        if type_ in {bool, int, np.int32, np.int64}:
            if any(value in s and peek in s for s in [{'x', 'X', SignalNotSet}, {'z', 'Z'}]):  # type: ignore
                return  # We have a match on a special value

        # Assert if the values are equal
        self.assertEqual(value, self.peek(scope, signal), msg=msg)
