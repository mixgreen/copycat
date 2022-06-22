import abc
import typing
import types
import time
import collections.abc

import numpy as np

from artiq.language.core import host_only, portable, rpc
from artiq.language.environment import NumberValue
from artiq.language.types import TBool

import dax.base.control_flow
import dax.util.ccb

__all__ = ['DaxServo']


class DaxServo(dax.base.control_flow.DaxControlFlow, abc.ABC):
    """Servo class for standardized servo functionality.

    A servo experiment implements a closed loop control system with feedback. At every step,
    one or more variables can be updated.

    Users can inherit this class to implement their own servo experiments. The first step is to
    build the servo by overriding the :func:`build_servo` function. Servo variables can be
    added via the :func:`add_servo_argument` function, which adds ARTIQ number values to the servo object.

    The servo class inherits from the :class:`dax.base.control_flow.DaxControlFlow` class for setup and cleanup
    procedures. The :func:`prepare` and :func:`analyze` functions are not implemented by the servo class, but users
    are free to provide their own implementations.

    The following functions can be overridden to define the servo behavior:

    1. :func:`dax.base.control_flow.DaxControlFlow.host_setup`
    2. :func:`dax.base.control_flow.DaxControlFlow.device_setup`
    3. :func:`run_point` (must be implemented)
    4. :func:`dax.base.control_flow.DaxControlFlow.device_cleanup`
    5. :func:`dax.base.control_flow.DaxControlFlow.host_cleanup`

    Finally, the :func:`dax.base.control_flow.DaxControlFlow.host_enter` and
    :func:`dax.base.control_flow.DaxControlFlow.host_exit` functions can be overridden to implement any
    functionality executed once at the start of or just before leaving the :func:`run` function.

    To exit a servo early, call the :func:`stop_servo` function.

    The end user can specify the number of servo runs by using the "Servo iterations" argument.
    Choosing the value ``0`` means that the number of servo iterations is infinite.
    If the servo runs infinitely, it can be stopped by either calling the :func:`stop_servo`
    function, or by using the "Terminate experiment" button on the dashboard.
    The default number of servo iterations can be set by overriding the :attr:`SERVO_ITERATIONS_DEFAULT` class variable.

    The :func:`run_point` function should be overridden by the user, and contains code that is executed on every
    run of the servo. This function has access to a point and an index argument. The point contains
    the current value of all servo variables that were added with the :func:`add_servo_argument` function. Users
    should update the servo variables by writing to the point variable.
    The index argument contains the index of the current servo iterations.

    In case the servo is performed in a kernel, users are responsible for setting up the
    right devices to actually run a kernel.

    Arguments passed to the :func:`build` function are passed to the super class for
    compatibility with other libraries and abstraction layers.
    It is still possible to pass arguments to the :func:`build_servo` function by using special
    keyword arguments of the :func:`build` function of which the keywords are defined
    in the :attr:`SERVO_ARGS_KEY` and :attr:`SERVO_KWARGS_KEY` attributes.

    :attr:`SERVO_PLOT_KEY_FORMAT` and :attr:`SERVO_PLOT_GROUP_FORMAT` can be used to group plot datasets
    and applets as desired. Both are formatted with the ARTIQ ``scheduler`` object which allows users to
    add experiment-specific information in the keys.
    The plot key format is additionally formatted with the key of each independent servo.

    By default, plot datasets are unique based on the experiment RID while applets are reused.
    Examples of common settings include:

     - Create unique plot datasets based on the experiment RID but reuse applets (default)
     - Create unique plot datasets and applets based on the experiment RID:
       ``SERVO_PLOT_GROUP_FORMAT="{scheduler.rid}.servo"``
    """

    SERVO_ITERATIONS_DEFAULT: typing.ClassVar[int] = 0
    """Default number of servo iterations."""

    SERVO_GROUP: typing.ClassVar[str] = 'servo'
    """The group name for archiving data."""
    SERVO_ARCHIVE_KEY_FORMAT: typing.ClassVar[str] = f'{SERVO_GROUP}/{{key}}'
    """Dataset key format for archiving independent servo values."""
    SERVO_ARCHIVE_EPOCH_KEY: typing.ClassVar[str] = f'{SERVO_GROUP}/epoch'
    """Dataset key for archiving epoch timestamps."""

    SERVO_PLOT_KEY_FORMAT: typing.ClassVar[str] = 'plot.{scheduler.rid}.servo.{key}'
    """Dataset key format for plotting independent servo values
    (formatted with the ARTIQ ``scheduler`` object and the ``key`` of each independent servo)."""
    SERVO_PLOT_GROUP_FORMAT: typing.ClassVar[str] = 'servo'
    """Group to which the plot applets belong (formatted with the ARTIQ ``scheduler`` object)."""

    SERVO_ARGS_KEY: typing.ClassVar[str] = 'servo_args'
    """:func:`build` keyword argument for positional arguments passed to :func:`build_servo`."""
    SERVO_KWARGS_KEY: typing.ClassVar[str] = 'servo_kwargs'
    """:func:`build` keyword argument for keyword arguments passed to :func:`build_servo`."""

    __in_build: bool
    __servo_values: typing.Dict[str, typing.Any]
    _dax_servo_iterations: np.int32
    _dax_servo_point: types.SimpleNamespace
    _dax_servo_index: np.int32

    def build(self, *args: typing.Any, **kwargs: typing.Any) -> None:
        """Build the servo object using the :func:`build_servo` function.

        Normally users would build their servo object by overriding the :func:`build_servo` function.
        In specific cases where this function might be overridden, do not forget to call ``super().build()``.

        :param args: Positional arguments forwarded to the superclass
        :param kwargs: Keyword arguments forwarded to the superclass (includes args and kwargs for :func:`build_servo`)
        """

        assert isinstance(self.SERVO_ITERATIONS_DEFAULT, int), 'Default servo iterations must be of type int'
        assert self.SERVO_ITERATIONS_DEFAULT >= 0, 'Default servo iterations must be 0 or greater'
        assert isinstance(self.SERVO_GROUP, str), 'Servo group must be of type str'
        assert isinstance(self.SERVO_ARCHIVE_KEY_FORMAT, str), 'Servo archive key format must be of type str'
        assert isinstance(self.SERVO_ARCHIVE_EPOCH_KEY, str), 'Servo archive epoch key must be of type str'
        assert isinstance(self.SERVO_PLOT_KEY_FORMAT, str), 'Servo plot key format must be of type str'
        assert isinstance(self.SERVO_PLOT_GROUP_FORMAT, str), 'Servo plot group format must be of type str'
        assert isinstance(self.SERVO_ARGS_KEY, str), 'Servo args keyword must be of type str'
        assert isinstance(self.SERVO_KWARGS_KEY, str), 'Servo kwargs keyword must be of type str'

        # Obtain the servo args and kwargs
        servo_args: typing.Sequence[typing.Any] = kwargs.pop(self.SERVO_ARGS_KEY, ())
        servo_kwargs: typing.Dict[str, typing.Any] = kwargs.pop(self.SERVO_KWARGS_KEY, {})
        assert isinstance(servo_args, collections.abc.Sequence), 'Servo args must be a sequence'
        assert isinstance(servo_kwargs, dict), 'Servo kwargs must be a dict'
        assert all(isinstance(k, str) for k in servo_kwargs), 'All servo kwarg keys must be of type str'

        # Make properties kernel invariant
        self.update_kernel_invariants('is_infinite_servo', 'is_terminated_servo')

        # Call super and forward arguments, for compatibility with other libraries
        # noinspection PyArgumentList
        super(DaxServo, self).build(*args, **kwargs)

        # Get CCB tool
        self.__ccb = dax.util.ccb.get_ccb_tool(self)
        # Collection of servos values
        self.__servo_values = {}

        # Build this servo (no args or kwargs available)
        self.logger.debug('Building servo')
        self.__in_build = True
        # noinspection PyArgumentList
        self.build_servo(*servo_args, **servo_kwargs)  # type: ignore[call-arg]
        self.__in_build = False

        # Add an argument for the number of servo iterations
        self._dax_servo_iterations = self.get_argument(
            'Servo iterations',
            NumberValue(self.SERVO_ITERATIONS_DEFAULT, min=0, ndecimals=0, step=1),
            group='DAX.servo',
            tooltip='Number of servo iterations, 0 for an infinite run'
        )
        self.update_kernel_invariants('_dax_servo_iterations')

    @abc.abstractmethod
    def build_servo(self) -> None:  # pragma: no cover
        """Users should override this method to build their servo.

        To build a servo, use the :func:`add_servo_argument` function. Additionally, users can
        also add normal arguments using the standard ARTIQ functions.

        It is possible to pass arguments from the constructor to this function using the
        keyword arguments defined in :attr:`SERVO_ARGS_KEY` and :attr:`SERVO_KWARGS_KEY` (see :class:`DaxServo`).
        """
        pass

    @property
    def is_infinite_servo(self) -> bool:
        """:const:`True` if the servo was set to be an infinite servo."""
        if hasattr(self, '_dax_servo_iterations'):
            return self._dax_servo_iterations == 0
        else:
            raise AttributeError('is_infinite_servo can only be obtained after build() was called')

    @host_only
    def add_servo(self, key: str, value: typing.Union[int, float, np.int32, np.int64]) -> None:
        """Register a servo with a fixed initial value.

        :param key: Unique key of the servo, used to obtain the value later
        :param value: The initial value of this servo
        """

        assert isinstance(key, str), 'Key must be of type str'

        # Verify this function was called in the build_servo() function
        if not self.__in_build:
            raise RuntimeError('add_servo_argument() can only be called in the build_servo() method')

        # Verify type of the given value
        if not isinstance(value, (int, float, np.int32, np.int64)):
            raise TypeError('The value of the servo must be numeric')

        # Verify the key is valid and not in use
        if not key.isidentifier():
            raise ValueError(f'Provided key "{key}" is not valid')
        if key in self.__servo_values:
            raise LookupError(f'Provided key "{key}" is already in use')

        # Add argument to servo values
        self.__servo_values[key] = value

    @host_only
    def add_servo_argument(self, key: str, name: str, value: NumberValue, *,
                           group: typing.Optional[str] = None, tooltip: typing.Optional[str] = None) -> None:
        """Register a servo with an argument for the initial value.

        The number provided should be an ARTIQ :class:`NumberValue`.

        This function can only be called in the :func:`build_servo` function.

        :param key: Unique key of the servo, used to obtain the value later
        :param name: The name of the argument
        :param value: The ARTIQ :class:`NumberValue` object used to obtain the initial value of this servo
        :param group: The argument group name
        :param tooltip: The shown tooltip
        """

        assert isinstance(key, str), 'Key must be of type str'
        assert isinstance(name, str), 'Name must be of type str'
        assert isinstance(group, str) or group is None, 'Group must be of type str or None'
        assert isinstance(tooltip, str) or tooltip is None, 'Tooltip must be of type str or None'

        # Verify this function was called in the build_servo() function
        if not self.__in_build:
            raise RuntimeError('add_servo_argument() can only be called in the build_servo() method')

        # Verify type of the given number value
        if not isinstance(value, NumberValue):
            raise TypeError('The value of the servo must be an ARTIQ NumberValue object')

        # Verify the key is valid and not in use
        if not key.isidentifier():
            raise ValueError(f'Provided key "{key}" is not valid')
        if key in self.__servo_values:
            raise LookupError(f'Provided key "{key}" is already in use')

        # Add argument to servo values
        self.__servo_values[key] = self.get_argument(name, value, group=group, tooltip=tooltip)

    @host_only
    def get_servo_values(self) -> typing.Dict[str, typing.Any]:
        """Get the values of all added servo variables.

        The return value is a dict whose keys and values are the servo key names and values, respectively.
        After :func:`run_point`, the servo values could have been updated and this function will return the new values.

        This function can only be used after the :func:`run` function was called
        which normally means it is not available during the build and prepare phase.
        See also :func:`init_servo_point`.

        :return: A dict containing all the servo points on a per-key basis
        """
        if hasattr(self, '_dax_servo_point'):
            return self._dax_servo_point.__dict__.copy()
        else:
            raise AttributeError('Servo values can only be obtained after servo points are initialized')

    @host_only
    def init_servo_point(self) -> None:
        """Initialize the servo point.

        By default, this is called at the beginning of :func:`run`, however it may be called in :func:`prepare` if the
        user desired the ability to call :func:`get_servo_values` before :func:`run`.
        """
        try:
            # Make the servo point
            self._dax_servo_point = types.SimpleNamespace(**self.__servo_values)
        except AttributeError:
            # build() was not called
            raise RuntimeError('DaxServo.build() was not called') from None

    """Internal control flow functions"""

    @portable
    def _dax_control_flow_while(self) -> TBool:
        return self._dax_servo_index < self._dax_servo_iterations

    def _dax_control_flow_is_kernel(self) -> bool:
        return dax.util.artiq.is_kernel(self.run_point)

    @portable
    def _dax_control_flow_run(self):  # type: () -> None
        # Run point
        self.run_point(self._dax_servo_point, self._dax_servo_index)
        # Store point
        self._dax_servo_store_point(self._dax_servo_point)

        if not self.is_infinite_servo:
            # Increment index
            self._dax_servo_index += np.int32(1)

    """Servo-specific functionality"""

    @host_only
    def run(self) -> None:
        # Initialize servo point if not already done
        if not hasattr(self, '_dax_servo_point'):
            self.init_servo_point()

        if not self.__servo_values:
            # There are no servo values
            self.logger.warning('No servo values found, aborting experiment')
            return

        # Report servo iterations
        if self.is_infinite_servo:
            self.logger.debug('Infinite servo')
        else:
            self.logger.debug(f'Servo with {self._dax_servo_iterations} iteration(s)')

        # Prepare dataset keys
        self.__archive_keys = {key: self.SERVO_ARCHIVE_KEY_FORMAT.format(key=key) for key in self.__servo_values}
        if self.SERVO_ARCHIVE_EPOCH_KEY in self.__archive_keys:
            raise ValueError('A servo key conflicts with the epoch dataset key')
        self.__plot_keys = {
            key: self.SERVO_PLOT_KEY_FORMAT.format(key=key, scheduler=self._dax_control_flow_scheduler)
            for key in self.__servo_values
        }
        # Create plot group
        self.__plot_group = self.SERVO_PLOT_GROUP_FORMAT.format(scheduler=self._dax_control_flow_scheduler)

        # Prepare datasets
        self.clear_servo_plot()
        self.set_dataset(self.SERVO_ARCHIVE_EPOCH_KEY, [], archive=True)
        for key in self.__servo_values:
            self.set_dataset(self.__archive_keys[key], [], archive=True)

        # Index of current servo iteration
        self._dax_servo_index = np.int32(-1 if self.is_infinite_servo else 0)
        # Store the initial point
        self._dax_servo_store_point(self._dax_servo_point)

        # Call super
        super(DaxServo, self).run()

    @portable
    def stop_servo(self):  # type: () -> None
        """Stop the servo after the current point.

        This function should only be called from the :func:`run_point` function.
        """
        self._dax_servo_index = np.int32(self._dax_servo_iterations)

    @property
    def is_terminated_servo(self) -> bool:
        """:const:`True` if the servo was terminated by the user."""
        return self._dax_control_flow_is_terminated

    @rpc(flags={'async'})
    def _dax_servo_store_point(self, point):  # type: (types.SimpleNamespace) -> None
        # Store timestamp
        self.append_to_dataset(self.SERVO_ARCHIVE_EPOCH_KEY, time.time())
        # Append values to archive
        for key, value in point.__dict__.items():
            self.append_to_dataset(self.__archive_keys[key], value)
            self.append_to_dataset(self.__plot_keys[key], value)

    """Plotting functions"""

    @rpc(flags={'async'})
    def plot_servo(self, *keys, normalize=False, **kwargs):  # type: (str, bool, typing.Any) -> None
        """Create one or more servo plots.

        This function can only be called in the ``run`` phase of the experiment.

        :param keys: One or more servo keys (all keys of none is given, non-existing keys are silently ignored)
        :param normalize: Normalize the values to the servo starting value (handled in the applet)
        """
        # Set defaults
        kwargs.setdefault('title', f'RID {self._dax_control_flow_scheduler.rid}')
        kwargs.setdefault('last', True)

        for key in (k for k in keys if k in self.__servo_values) if keys else self.__servo_values:
            if normalize:
                kwargs['multiplier'] = 1.0 / self.__servo_values[key]
            self.__ccb.plot_xy(key, self.__plot_keys[key], group=self.__plot_group, **kwargs)

    @rpc(flags={'async'})
    def clear_servo_plot(self, *keys):  # type: (str) -> None
        """Clear one or more servo plots.

        This function can only be called in the ``run`` phase of the experiment.

        :param keys: One or more servo keys (all keys of none is given, non-existing keys are silently ignored)
        """
        for key in (k for k in keys if k in self.__servo_values) if keys else self.__servo_values:
            self.set_dataset(self.__plot_keys[key], [], broadcast=True, archive=False)

    @rpc(flags={'async'})
    def disable_servo_plot(self, *keys):  # type: (str) -> None
        """Disable one or more servo plots.

        This function can only be called in the ``run`` phase of the experiment.

        :param keys: One or more servo keys (all keys of none is given, non-existing keys are silently ignored)
        """
        for key in (k for k in keys if k in self.__servo_values) if keys else self.__servo_values:
            self.__ccb.disable_applet(key, group=self.__plot_group)

    @rpc(flags={'async'})
    def disable_all_plots(self):  # type: () -> None
        """Close all servo plots by closing the applet group.

        This function can only be called in the ``run`` phase of the experiment.
        """
        self.__ccb.disable_applet_group(self.__plot_group)

    """End-user functions"""

    @abc.abstractmethod
    def run_point(self, point, index):  # type: (typing.Any, typing.Any) -> None  # pragma: no cover
        """Code to run for a single point, called infinitely, or as specified by the number of servo runs.

        :param point: Point object containing the current servo parameter values
        :param index: Index object containing the current iteration, or ``-1`` if the servo scan is infinite
        """
        pass
