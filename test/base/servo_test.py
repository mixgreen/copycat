import collections
import typing
import unittest

from artiq.experiment import *

from dax.base.servo import DaxServo
from dax.base.system import DaxSystem
from dax.util.artiq import get_managers

from test.environment import CI_ENABLED
import test.helpers


class _MockSystem(DaxSystem):
    SYS_ID = 'test_system'
    SYS_VER = 0
    CORE_LOG_KEY = None
    DAX_INFLUX_DB_KEY = None


class _MockServo1(DaxServo, _MockSystem):
    SERVO_ITERATIONS_DEFAULT = 1
    FOO = 20
    FOOBAR = 3.5

    def build_servo(self) -> None:
        # Counter
        self.counter: typing.Counter[str] = collections.Counter()

        # Servo
        self.add_servo_argument('foo', 'foo', NumberValue(default=self.FOO))
        self.add_servo('foobar', self.FOOBAR)

    def init_servo_point(self) -> None:
        super(_MockServo1, self).init_servo_point()
        self.counter['init_servo_point'] += 1

    def host_enter(self) -> None:
        self.counter['host_enter'] += 1

    def host_setup(self) -> None:
        self.counter['host_setup'] += 1

    def device_setup(self):  # type: () -> None
        self.counter['device_setup'] += 1

    def run_point(self, point, index):  # type: (typing.Any, typing.Any) -> None
        self.counter['run_point'] += 1

    def device_cleanup(self):  # type: () -> None
        self.counter['device_cleanup'] += 1

    def host_cleanup(self) -> None:
        self.counter['host_cleanup'] += 1

    def host_exit(self) -> None:
        self.counter['host_exit'] += 1


class _MockServoCallback(_MockServo1):

    def build_servo(self) -> None:
        self.callback()

    def callback(self):
        raise NotImplementedError


class _MockServoEmpty(_MockServo1):

    def build_servo(self) -> None:
        # Counter
        self.counter: typing.Counter[str] = collections.Counter()

    def run_point(self, point, index):  # type: (typing.Any, typing.Any) -> None
        raise NotImplementedError('The run_point() function should not have been reached')


class _MockServo2(_MockServo1):
    BAR = 30

    def build_servo(self) -> None:
        super(_MockServo2, self).build_servo()
        self._add_servo()

    def _add_servo(self):
        self.add_servo_argument('bar', 'bar', NumberValue(default=self.BAR))

    def run_point(self, point, index):  # type: (typing.Any, typing.Any) -> None
        super(_MockServo2, self).run_point(point, index)
        point.foo += 1
        point.bar += 1


class _MockServoTerminate(_MockServo1):
    SERVO_ITERATIONS_DEFAULT = 100
    TERMINATE = 5

    def run_point(self, point, index):  # type: (typing.Any, typing.Any) -> None
        if self.counter['run_point'] == self.TERMINATE:
            raise TerminationRequested
        self.counter['run_point'] += 1


class _MockServoStop(_MockServo1):
    SERVO_ITERATIONS_DEFAULT = 100
    STOP = 5

    def run_point(self, point, index):  # type: (typing.Any, typing.Any) -> None
        if self.counter['run_point'] == self.STOP:
            self.stop_servo()
        self.counter['run_point'] += 1


class _MockServoInfinite(_MockServo1):
    SERVO_ITERATIONS_DEFAULT = 0
    STOP = 100

    def run_point(self, point, index):  # type: (typing.Any, typing.Any) -> None
        if self.counter['run_point'] == self.STOP:
            self.stop_servo()
        self.counter['run_point'] += 1
        assert index == -1, 'Index was not -1 in an infinite scan'


class _MockServo2ValueCheck(_MockServo2):

    def build_can(self):
        super(_MockServo2ValueCheck, self).build_servo()

    def run_point(self, point, index):  # type: (typing.Any, typing.Any) -> None
        super(_MockServo2ValueCheck, self).run_point(point, index)

        _index = self.counter['run_point'] - 1,
        assert index == _index, f'{index} != {_index}'

        point_foo = index + self.FOO
        point_bar = index + self.BAR
        assert point.foo == point_foo, f'{point.foo} != {point_foo}'
        assert point.bar == point_bar, f'{point.bar} != {point_bar}'


class Servo1TestCase(unittest.TestCase):
    SERVO_CLASS = _MockServo1
    SERVO_ITERATIONS = 10 if CI_ENABLED else 1

    def setUp(self) -> None:
        self.managers = get_managers(arguments={'Servo iterations': self.SERVO_ITERATIONS})
        self.servo = self.SERVO_CLASS(self.managers)

    def tearDown(self) -> None:
        # Close managers
        self.managers.close()

    def test_kernel_invariants(self):
        test.helpers.test_system_kernel_invariants(self, self.servo)

    def test_is_infinite(self):
        self.assertFalse(self.servo.is_infinite_servo, 'Servo reported incorrectly it was infinite')

    def test_is_terminated(self):
        self.assertFalse(self.servo.is_terminated_servo)

    def test_get_servo_values(self):
        self.servo.init_servo_point()
        self.assertEqual(self.servo.get_servo_values(),
                         {'foo': self.SERVO_CLASS.FOO, 'foobar': self.SERVO_CLASS.FOOBAR})

    def test_call_counters(self):
        # Run the servo
        self.servo.run()

        # Verify counters
        counter_ref = {
            'init_servo_point': 1,
            'host_enter': 1,
            'host_setup': 1,
            'device_setup': 1,
            'run_point': self.SERVO_ITERATIONS,
            'device_cleanup': 1,
            'host_cleanup': 1,
            'host_exit': 1,
        }

        self.assertDictEqual(self.servo.counter, counter_ref, 'Function counters did not match expected values')

    def test_early_servo_point_init(self):
        # Call point init before run
        self.servo.init_servo_point()
        # Run the servo
        self.servo.run()
        # Verify init was only called once
        self.assertEqual(self.servo.counter['init_servo_point'], 1,
                         'init_servo_point counter did not match expected value')

    def test_raise_add_servo(self):
        with self.assertRaises(RuntimeError, msg='Adding servo outside build did not raise'):
            self.servo.add_servo_argument('bar', 'bar', NumberValue(1))

    def test_get_servo_values_early(self):
        with self.assertRaises(AttributeError, msg='Servo point request before run did not raise'):
            self.servo.get_servo_values()
        self.servo.run()
        self.servo.get_servo_values()

    def test_init_servo_point(self):
        with self.assertRaises(AttributeError, msg='Servo point request before run did not raise'):
            self.servo.get_servo_values()
        self.servo.init_servo_point()
        self.servo.get_servo_values()

    def test_plots(self):
        # We need to be in the run phase of the experiment to initialize attributes
        self.servo.run()

        # In simulation, we can only call these functions, but nothing will happen
        self.servo.plot_servo()
        self.servo.plot_servo('foo', 'bar')
        self.servo.disable_servo_plot()
        self.servo.disable_servo_plot('foo', 'bar')
        self.servo.disable_all_plots()

        # These functions will actually write datasets
        self.servo.clear_servo_plot()
        self.servo.clear_servo_plot('foo', 'bar')


class BuildServoTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.managers = get_managers()

    def tearDown(self) -> None:
        # Close managers
        self.managers.close()

    def test_raise_duplicate_servo_key(self):
        class MockServo(_MockServoCallback):
            # noinspection PyMethodParameters
            def callback(self_servo):
                self_servo.add_servo_argument('foo', 'foo', NumberValue(1))
                with self.assertRaises(LookupError, msg='Reusing servo key did not raise'):
                    self_servo.add_servo_argument('foo', 'foo', NumberValue(1))

        MockServo(self.managers)

    def test_raise_bad_servo_type(self):
        test_data = [
            Scannable(NoScan(1)),
            EnumerationValue('abc'),
            'aa',
        ]

        class MockServo(_MockServoCallback):
            # noinspection PyMethodParameters
            def callback(self_servo):
                for number_value in test_data:
                    with self.subTest(number_value=number_value):
                        with self.assertRaises(TypeError, msg='Bad servo type did not raise'):
                            # noinspection PyTypeChecker
                            self_servo.add_servo_argument('foo', 'foo', number_value)

        MockServo(self.managers)

    def test_raise_bad_servo_key(self):
        class MockServo(_MockServoCallback):
            # noinspection PyMethodParameters
            def callback(self_servo):
                keys = ['aa.', '9int', 'foo-bar']
                for k in keys:
                    with self.subTest(key=k):
                        with self.assertRaises(ValueError, msg='Bad servo key did not raise'):
                            self_servo.add_servo_argument(k, 'some name', NumberValue(1))

        MockServo(self.managers)

    def test_servo_build_arguments(self):
        test_args = (1, 2, 'd', 6.9)
        test_kwargs = {'foo': 4, 'foo bar': 6.6, 'bar': RangeScan(2, 5, 10)}

        class MockServo(_MockServo1):
            # noinspection PyMethodParameters
            def build_servo(self_servo, *args, **kwargs) -> None:
                self.assertTupleEqual(args, test_args, 'Positional arguments did not match')
                self.assertDictEqual(kwargs, test_kwargs, 'Keyword arguments did not match')

        MockServo(self.managers, servo_args=test_args, servo_kwargs=test_kwargs)


class EmptyServoTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.managers = get_managers()
        self.servo = _MockServoEmpty(self.managers)

    def tearDown(self) -> None:
        # Close managers
        self.managers.close()

    def test_function_calls(self):
        # Verify counters (object is an empty dict)
        self.assertDictEqual(self.servo.counter, {}, 'Function counters did not match expected values')

    def test_run_point_not_called(self):
        # The run function should exit early and run_point() is not called (which will raise if it does)
        self.servo.run()


class Servo2TestCase(Servo1TestCase):
    SERVO_CLASS = _MockServo2

    def test_call_counters(self):
        # Run the servo
        self.servo.run()

        # Verify counters
        counter_ref = {
            'init_servo_point': 1,
            'host_enter': 1,
            'host_setup': 1,
            'device_setup': 1,
            'run_point': self.SERVO_ITERATIONS,
            'device_cleanup': 1,
            'host_cleanup': 1,
            'host_exit': 1,
        }

        self.assertDictEqual(self.servo.counter, counter_ref, 'Function counters did not match expected values')

    def test_get_servo_values(self):
        self.servo.init_servo_point()
        start_values = self.servo.get_servo_values()
        self.servo.run()
        values = self.servo.get_servo_values()
        self.assertIn('foo', values)
        self.assertIn('foobar', values)
        self.assertIn('bar', values)

        self.assertEqual(values['foo'], self.servo.FOO + self.SERVO_ITERATIONS)
        self.assertEqual(values['foobar'], self.servo.FOOBAR)
        self.assertEqual(values['bar'], self.servo.BAR + self.SERVO_ITERATIONS)
        self.assertNotEqual(start_values, values)


class ServoTerminateTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.managers = get_managers()
        self.servo = _MockServoTerminate(self.managers)

    def tearDown(self) -> None:
        # Close managers
        self.managers.close()

    def test_call_counters(self):
        # Verify flag
        self.assertFalse(self.servo.is_terminated_servo)
        # Run the servo
        self.servo.run()
        # Verify the flag
        self.assertTrue(self.servo.is_terminated_servo)

        # Verify counters
        counter_ref = {
            'init_servo_point': 1,
            'host_enter': 1,
            'host_setup': 1,
            'device_setup': 1,
            'run_point': self.servo.TERMINATE,
            'device_cleanup': 1,
            'host_cleanup': 1,
            # 'host_exit': 1,
        }
        self.assertDictEqual(self.servo.counter, counter_ref, 'Function counters did not match expected value')


class ServoStopTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.managers = get_managers()
        self.servo = _MockServoStop(self.managers)

    def tearDown(self) -> None:
        # Close managers
        self.managers.close()

    def test_call_counters(self):
        # Run the servo
        self.servo.run()

        # Verify counters
        # Verify counters
        counter_ref = {
            'init_servo_point': 1,
            'host_enter': 1,
            'host_setup': 1,
            'device_setup': 1,
            'run_point': self.servo.STOP + 1,
            'device_cleanup': 1,
            'host_cleanup': 1,
            'host_exit': 1,
        }
        self.assertDictEqual(self.servo.counter, counter_ref, 'Function counters did not match expected values')


class InfiniteServoTestCase(unittest.TestCase):
    SERVO_CLASS = _MockServoInfinite

    def setUp(self) -> None:
        self.managers = get_managers()
        self.servo = self.SERVO_CLASS(self.managers)

    def tearDown(self) -> None:
        # Close managers
        self.managers.close()

    def test_is_infinite(self):
        self.assertTrue(self.servo.is_infinite_servo)

    def test_call_counters(self):
        # Run the servo
        self.servo.run()

        # Verify counters
        # Verify counters
        counter_ref = {
            'init_servo_point': 1,
            'host_enter': 1,
            'host_setup': 1,
            'device_setup': 1,
            'run_point': self.servo.STOP + 1,
            'device_cleanup': 1,
            'host_cleanup': 1,
            'host_exit': 1,
        }
        self.assertDictEqual(self.servo.counter, counter_ref, 'Function counters did not match expected values')
