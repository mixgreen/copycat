import unittest
import logging
import typing
import collections
import warnings

from artiq.experiment import HasEnvironment, now_mu
from artiq.master.databases import device_db_from_file

from dax.util.artiq import get_managers
from dax.sim import enable_dax_sim
from dax.sim.signal import get_signal_manager, PeekSignalManager, SignalNotSet
from dax.sim.device import DaxSimDevice

__all__ = ['PeekTestCase']

_logger: logging.Logger = logging.getLogger(__name__)
"""The module logger object."""


class PeekTestCase(unittest.TestCase):
    """An extension of the `unittest.TestCase` class with functions for peek-testing.

    Users can inherit from this class to create their own test cases,
    similar as someone would make test cases for normal software.
    This class behaves like a normal `unittest.TestCase` class but has a few extra functions
    that are useful for peek-testing of a simulated ARTIQ environment.
    Users are allowed to combine regular unittest constructs with peek-testing features.
    """

    DEFAULT_DEVICE_DB: str = 'device_db.py'
    """The path of the default device DB used to construct environments."""

    __E_T = typing.TypeVar('__E_T', bound=HasEnvironment)  # Type variable for environment

    def construct_env(self, env_class: typing.Type[__E_T], *,
                      device_db: typing.Union[str, typing.Dict[str, typing.Any], None] = None,
                      logging_level: typing.Union[int, str] = logging.NOTSET,
                      build_args: typing.Optional[typing.Sequence[typing.Any]] = None,
                      build_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None,
                      env_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None,
                      **kwargs: typing.Any) -> __E_T:
        """Construct an ARTIQ environment based on the given class.

        The constructed environment can be used as a Device Under Testing (DUT).

        :param env_class: The environment class to construct
        :param device_db: The device DB to use (defaults to file configured in :attr:`DEFAULT_DEVICE_DB`)
        :param logging_level: The desired logging level
        :param build_args: Positional arguments passed to the build function of the environment
        :param build_kwargs: Keyword arguments passed to the build function of the environment
        :param env_kwargs: Keyword arguments passed to the argument parser of the environment
        :param kwargs: Keyword arguments passed to the argument parser of the environment (updates `env_kwargs`)
        :return: The constructed ARTIQ environment object
        """

        # Set default values
        if build_args is None:
            build_args = ()
        if build_kwargs is None:
            build_kwargs = {}
        if env_kwargs is None:
            env_kwargs = {}

        assert issubclass(env_class, HasEnvironment), 'The environment class must be a subclass of HasEnvironment'
        assert isinstance(device_db, (str, dict)) or device_db is None, 'Device DB must be of type str, dict, or None'
        assert isinstance(logging_level, (int, str)), 'Logging level must be of type int or str'
        assert isinstance(build_args, collections.abc.Sequence), 'Build arguments must be a sequence'
        assert isinstance(build_kwargs, dict), 'Build keyword arguments must be a dict'

        # Set level of module logger
        _logger.setLevel(logging_level)

        # Construct an expid object
        expid: typing.Dict[str, typing.Any] = {'log_level': logging_level,
                                               'class_name': env_class.__name__,
                                               'repo_rev': 'N/A'}

        if not isinstance(device_db, dict):
            # Obtain device DB from file
            _logger.debug('Obtaining device DB from file')
            with warnings.catch_warnings():
                # Ignore resource warnings that could be raised from evaluating the device DB
                # These warnings appear when starting the MonInjDummyService
                warnings.simplefilter('ignore', category=ResourceWarning)
                device_db = device_db_from_file(self.DEFAULT_DEVICE_DB if device_db is None else device_db)

        # Convert and configure device DB
        _logger.debug('Converting device DB')
        enable_dax_sim(ddb=device_db, enable=True, logging_level=logging_level, output='peek', moninj_service=False)

        # Construct environment, which will also construct a new signal manager
        _logger.debug('Constructing environment')
        env = env_class(get_managers(device_db, expid=expid, arguments=env_kwargs, **kwargs),
                        *build_args, **build_kwargs)

        # Store the new signal manager
        _logger.debug('Retrieving peek signal manager')
        self.__signal_manager: PeekSignalManager = typing.cast(PeekSignalManager, get_signal_manager())
        assert isinstance(self.__signal_manager, PeekSignalManager), 'Did not obtained correct signal manager type'

        # Return the environment
        return env

    def peek(self, scope: typing.Any, signal: str) -> typing.Any:
        """Peek a signal of a device at the current time.

        :param scope: The scope (device) of the signal
        :param signal: The name of the signal
        :return: The value of the signal at the current time
        """
        # Peek the value using the signal manager
        value = self.__signal_manager.peek(scope, signal)
        _logger.info(f'PEEK {scope}.{signal} -> {value}')

        # Return the value
        return value

    def expect(self, scope: typing.Any, signal: str, value: typing.Any, msg: typing.Optional[str] = None) -> None:
        """Test if a signal holds a given value at the current time.

        If the signal does not match the value, the test will fail.

        :param scope: The scope (device) of the signal
        :param signal: The name of the signal
        :param value: The expected value
        :param msg: Message to show when this assertion fails
        """
        # Get the value and the type
        peek, type_ = self.__signal_manager.peek_and_type(typing.cast(DaxSimDevice, scope), signal)
        _logger.info(f'EXPECT {scope}.{signal} -> {value} == {peek}')

        if type_ not in {bool, int, float}:
            # Raise if the signal has an unsupported
            raise TypeError(f'Signal "{scope.key}.{signal}" of type "{type_}" can not be tested')

        # Match with special values
        if any(value in s and peek in s for s in [{'x', 'X', SignalNotSet}, {'z', 'Z'}]):  # type: ignore
            return  # We have a match on a special value
        # Special conversion for vector matching
        if type_ is bool and isinstance(value, str):
            value = value.lower()  # Apply conversion to allow string matching

        if msg is None:
            # Set default error message
            msg = f'at {now_mu()} mu'

        # Assert if values are not equal
        self.assertEqual(value, peek, msg=msg)
