import unittest
import subprocess
import typing
import collections.abc

from test.environment import NIX_ENV, HW_TEST_ENABLED
from test.hw_test.device_db import core_addr

import artiq.master.worker_db
from artiq.language.environment import HasEnvironment

import dax.util.artiq

import test.hw_test.device_db

__all__ = ['HardwareTestCase']


# noinspection PyPep8Naming
def setUpModule() -> None:
    if not NIX_ENV:
        raise unittest.SkipTest('Not in a Nix environment, skipping hardware tests')
    if not HW_TEST_ENABLED:
        raise unittest.SkipTest('Hardware tests not enabled, skipping hardware tests')
    if core_addr is None:
        raise unittest.SkipTest('No IP address configured for test hardware, skipping hardware tests')
    if subprocess.call(f'ping -c 1 -w 10 {core_addr}'.split()) != 0:
        raise unittest.SkipTest('Can not ping test hardware, skipping hardware tests')


@unittest.skipUnless(NIX_ENV, 'Not in a Nix environment')
@unittest.skipUnless(HW_TEST_ENABLED, 'Hardware test not enabled')
class HardwareTestCase(unittest.TestCase):
    """An extension of the :class:`unittest.TestCase` class which facilitates hardware testing.

    Users can inherit this class to create their own hardware test cases.
    It has predefined :func:`setUp` and :func:`tearDown` functions for resource handling.
    The :func:`construct_env` function can be used to construct test environment classes.
    """

    __E_T = typing.TypeVar('__E_T', bound=HasEnvironment)  # Type variable for environment

    def setUp(self) -> None:
        """Set up the ARTIQ manager objects."""
        self.__managers = dax.util.artiq.get_managers(device_db=test.hw_test.device_db.device_db)

    def tearDown(self) -> None:
        """Close the ARTIQ managers to free resources."""
        self.__managers.close()

    def construct_env(self, env_class: typing.Type[__E_T], *,
                      build_args: typing.Sequence[typing.Any] = (),
                      build_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None,
                      **kwargs: typing.Any) -> __E_T:
        """Construct an ARTIQ environment based on the given class.

        The constructed environment can be used for hardware testing.

        :param env_class: The environment class to construct
        :param build_args: Positional arguments passed to the build function of the environment
        :param build_kwargs: Keyword arguments passed to the build function of the environment
        :param kwargs: Keyword arguments passed to the build function of the environment (updates ``build_kwargs``)
        :return: The constructed ARTIQ environment object
        """

        if build_kwargs is None:
            # Set default value
            build_kwargs = {}
        else:
            assert isinstance(build_kwargs, dict), 'Build keyword arguments must be of type dict'
            assert all(isinstance(k, str) for k in build_kwargs), 'Keys of the build kwargs dict must be of type str'
            build_kwargs = build_kwargs.copy()  # Copy arguments to make sure the dict is not mutated

        assert issubclass(env_class, HasEnvironment), 'The environment class must be a subclass of HasEnvironment'
        assert isinstance(build_args, collections.abc.Sequence), 'Build arguments must be a sequence'
        assert isinstance(build_kwargs, dict), 'Build keyword arguments must be a dict'
        assert all(isinstance(k, str) for k in build_kwargs), 'Keys of the build kwargs dict must be of type str'

        # Merge arguments
        build_kwargs.update(kwargs)

        try:
            # Construct environment
            env = env_class(self.__managers, *build_args, **build_kwargs)
        except artiq.master.worker_db.DeviceError as e:
            # Skip test in case of device errors (raising instead of calling ``self.skipTest()`` for better typing)
            raise unittest.SkipTest(f'Device error: "{str(e)}"')
        else:
            # Return the environment
            return env
