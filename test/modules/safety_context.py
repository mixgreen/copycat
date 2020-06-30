import unittest
import collections

from dax.experiment import *
from dax.modules.safety_context import *
from dax.util.artiq_helpers import get_manager_or_parent


class _TestSystem(DaxSystem):
    SYS_ID = 'unittest_system'
    SYS_VER = 0

    def build(self) -> None:  # type: ignore
        super(_TestSystem, self).build()
        self.context = SafetyContext(self, 'context', enter_cb=self.enter, exit_cb=self.exit)
        self.counter = collections.Counter({'enter': 0, 'exit': 0})

    def enter(self):
        self.counter['enter'] += 1

    def exit(self):
        self.counter['exit'] += 1


class SafetyContextTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.s = _TestSystem(get_manager_or_parent())
        self.s.dax_init()
        self.counter = self.s.counter
        self.context = self.s.context

    def test_nested_context(self):
        self.assertDictEqual(self.counter, {'enter': 0, 'exit': 0}, 'Counters did not match expected values')

        with self.context:
            self.assertDictEqual(self.counter, {'enter': 1, 'exit': 0}, 'Counters did not match expected values')
            with self.assertRaises(SafetyContextError, msg='Reentering context did not raise'):
                with self.context:
                    self.fail()

        self.assertDictEqual(self.counter, {'enter': 1, 'exit': 1}, 'Counters did not match expected values')

    def test_nested_context_with(self):
        self.assertDictEqual(self.counter, {'enter': 0, 'exit': 0}, 'Counters did not match expected values')

        with self.assertRaises(SafetyContextError, msg='Reentering context did not raise'):
            with self.context, self.context:
                self.fail()

        self.assertDictEqual(self.counter, {'enter': 1, 'exit': 1}, 'Counters did not match expected values')

    def test_exit_mismatch(self):
        with self.assertRaises(SafetyContextError, msg='Out of sync exit did not raise'):
            # Call exit manually (which is bad)
            self.context.__exit__(None, None, None)

        self.assertDictEqual(self.counter, {'enter': 0, 'exit': 0}, 'Counters did not match expected values')

    def test_in_context(self):
        # Initially we are out of context
        self.assertFalse(self.context.in_context(), 'in_context() reported wrong value')
        with self.context:
            # In context
            self.assertTrue(self.context.in_context(), 'in_context() reported wrong value')

        # Out of context
        self.assertFalse(self.context.in_context(), 'in_context() reported wrong value')
        # Open context manually
        self.context.__enter__()
        # In context
        self.assertTrue(self.context.in_context(), 'in_context() reported wrong value')
        # Close context manually
        self.context.__exit__(None, None, None)
        # Out of context
        self.assertFalse(self.context.in_context(), 'in_context() reported wrong value')

        self.assertDictEqual(self.counter, {'enter': 2, 'exit': 2}, 'Counters did not match expected values')

    def test_enter_exception(self):
        def enter():
            raise ValueError

        self.context._enter_cb = enter

        with self.assertRaises(ValueError, msg='Enter exception did not raise'):
            with self.context:
                self.fail()

        # Expect 0 as enter failed and exit was never called
        self.assertEqual(self.context._in_context, 0, 'In context did not match expected value')
        self.assertDictEqual(self.counter, {'enter': 0, 'exit': 0}, 'Counters did not match expected values')

    def test_exit_exception(self):
        def exit_():
            raise ValueError

        self.context._exit_cb = exit_

        # noinspection PyUnusedLocal
        enter_flag = False

        with self.assertRaises(ValueError, msg='Exit exception did not raise'):
            with self.context:
                enter_flag = True
            self.fail()

        self.assertTrue(enter_flag, 'Context was never entered')
        # Expect 1 as enter passed successfully but exit failed
        self.assertEqual(self.context._in_context, 1, 'In context did not match expected value')
        self.assertDictEqual(self.counter, {'enter': 1, 'exit': 0}, 'Counters did not match expected values')

    def test_enter_exit_exception(self):
        def enter():
            raise ValueError

        def exit_():
            raise TypeError

        self.context._enter_cb = enter
        self.context._exit_cb = exit_

        with self.assertRaises(ValueError, msg='Enter and exit exception did not raise'):
            with self.context:
                self.fail()

        # Expect 0 as enter failed and exit was never called
        self.assertEqual(self.context._in_context, 0, 'In context did not match expected value')
        self.assertDictEqual(self.counter, {'enter': 0, 'exit': 0}, 'Counters did not match expected values')


if __name__ == '__main__':
    unittest.main()
