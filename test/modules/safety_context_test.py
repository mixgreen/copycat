import unittest
import collections

from dax.experiment import *
from dax.base.exceptions import BuildError
from dax.modules.safety_context import *
from dax.util.artiq import get_managers

import test.helpers


class _TestSystem(DaxSystem):
    SYS_ID = 'unittest_system'
    SYS_VER = 0
    CORE_LOG_KEY = None
    DAX_INFLUX_DB_KEY = None


class SafetyContextTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.managers = get_managers()
        self.s = _TestSystem(self.managers)

    def tearDown(self) -> None:
        # Close managers
        self.managers.close()

    def _test_callback_decorators(self, *, rpc_=False, cb, error):
        # Test if portable still works (should always work)
        ReentrantSafetyContext(_TestSystem(self.managers), 'context',
                               rpc_=rpc_, enter_cb=self._portable_fn, exit_cb=self._portable_fn)

        if error:
            # Expect error
            with self.assertRaises(BuildError):
                ReentrantSafetyContext(_TestSystem(self.managers), 'context',
                                       rpc_=rpc_, enter_cb=cb, exit_cb=self._portable_fn)
            with self.assertRaises(BuildError):
                ReentrantSafetyContext(_TestSystem(self.managers), 'context',
                                       rpc_=rpc_, enter_cb=self._portable_fn, exit_cb=cb)
        else:
            # No error
            ReentrantSafetyContext(_TestSystem(self.managers), 'context', rpc_=rpc_, enter_cb=cb, exit_cb=cb)

    def test_callback_decorators(self):
        error_set = {
            (False, self._host_only_fn),
            (True, self._kernel_fn),
        }
        for rpc_ in [False, True]:
            for cb in [self._kernel_fn, self._portable_fn, self._host_only_fn, self._rpc_fn]:
                self._test_callback_decorators(rpc_=rpc_, cb=cb, error=(rpc_, cb) in error_set)

    @kernel
    def _kernel_fn(self):
        pass

    @portable
    def _portable_fn(self):
        pass

    @host_only
    def _host_only_fn(self):
        pass

    @rpc
    def _rpc_fn(self):
        pass


class _ReentrantTestSystem(_TestSystem):
    SAFETY_CONTEXT_TYPE = ReentrantSafetyContext
    EXIT_ERROR = True
    RPC = False

    def build(self) -> None:  # type: ignore[override]
        super(_ReentrantTestSystem, self).build()
        self.context = self.SAFETY_CONTEXT_TYPE(self, 'context',
                                                enter_cb=self.enter, exit_cb=self.exit,
                                                exit_error=self.EXIT_ERROR, rpc_=self.RPC)
        self.counter = collections.Counter({'enter': 0, 'exit': 0})

    def enter(self):
        self.counter['enter'] += 1

    def exit(self):
        self.counter['exit'] += 1


class _ReentrantRpcTestSystem(_ReentrantTestSystem):
    RPC = True


class _ReentrantExitErrorTestSystem(_ReentrantTestSystem):
    EXIT_ERROR = False


class _ReentrantExitErrorRpcTestSystem(_ReentrantTestSystem):
    EXIT_ERROR = False
    RPC = True


class _NonReentrantTestSystem(_ReentrantTestSystem):
    SAFETY_CONTEXT_TYPE = SafetyContext


class _NonReentrantRpcTestSystem(_NonReentrantTestSystem):
    RPC = True


class _NonReentrantExitErrorTestSystem(_NonReentrantTestSystem):
    EXIT_ERROR = False


class _NonReentrantExitErrorRpcTestSystem(_NonReentrantTestSystem):
    EXIT_ERROR = False
    RPC = True


class ReentrantSafetyContextTestCase(unittest.TestCase):
    SYSTEM_TYPE = _ReentrantTestSystem

    def setUp(self) -> None:
        self.managers = get_managers()
        self.s = self.SYSTEM_TYPE(self.managers)
        self.s.dax_init()
        self.counter = self.s.counter
        self.context = self.s.context

    def tearDown(self) -> None:
        # Close managers
        self.managers.close()

    def test_exit_mismatch(self):
        if self.SYSTEM_TYPE.EXIT_ERROR:
            with self.assertRaises(SafetyContextError, msg='Out of sync exit did not raise'):
                # Call exit manually (which is bad)
                self.context.__exit__(None, None, None)
        else:
            # Call exit manually is allowed
            self.context.__exit__(None, None, None)
            self.context.__exit__(None, None, None)

        self.assertEqual(self.context._safety_context_entries, 0, 'In context counter is corrupted')
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

        self.context._safety_context_enter_cb = enter

        with self.assertRaises(ValueError, msg='Enter exception did not raise'):
            with self.context:
                self.fail()

        # Expect 0 as enter failed and exit was never called
        self.assertEqual(self.context._safety_context_entries, 0, 'In context did not match expected value')
        self.assertDictEqual(self.counter, {'enter': 0, 'exit': 0}, 'Counters did not match expected values')

    def test_exit_exception(self):
        def exit_():
            raise ValueError

        self.context._safety_context_exit_cb = exit_

        # noinspection PyUnusedLocal
        enter_flag = False

        with self.assertRaises(ValueError, msg='Exit exception did not raise'):
            with self.context:
                enter_flag = True
            self.fail()

        self.assertTrue(enter_flag, 'Context was never entered')
        self.assertEqual(self.context._safety_context_entries, 0, 'In context did not match expected value')
        self.assertDictEqual(self.counter, {'enter': 1, 'exit': 0}, 'Counters did not match expected values')

    def test_enter_exit_exception(self):
        def enter():
            raise ValueError

        def exit_():
            raise TypeError

        self.context._safety_context_enter_cb = enter
        self.context._safety_context_exit_cb = exit_

        with self.assertRaises(ValueError, msg='Enter and exit exception did not raise'):
            with self.context:
                self.fail()

        # Expect 0 as enter failed and exit was never called
        self.assertEqual(self.context._safety_context_entries, 0, 'In context did not match expected value')
        self.assertDictEqual(self.counter, {'enter': 0, 'exit': 0}, 'Counters did not match expected values')

    def test_multiple_context_objects(self):
        counter_b = collections.Counter()

        def enter():
            counter_b['enter'] += 1

        def exit_():
            counter_b['exit'] += 1

        context_b = SafetyContext(self.s, 'context_b', enter_cb=enter, exit_cb=exit_)
        # noinspection PyUnusedLocal
        enter_flag = False

        with self.context, context_b:
            enter_flag = True

        self.assertTrue(enter_flag, 'Context was never entered')
        self.assertDictEqual(self.counter, {'enter': 1, 'exit': 1}, 'Counters did not match expected values')
        self.assertDictEqual(counter_b, {'enter': 1, 'exit': 1}, 'Counters (b) did not match expected values')

    def test_kernel_invariants(self):
        # Test kernel invariants
        test.helpers.test_system_kernel_invariants(self, self.s)

    def test_nested_context(self):
        self.assertDictEqual(self.counter, {'enter': 0, 'exit': 0}, 'Counters did not match expected values')

        # noinspection PyUnusedLocal
        enter_flag = False

        with self.context:
            self.assertDictEqual(self.counter, {'enter': 1, 'exit': 0}, 'Counters did not match expected values')
            with self.context:
                enter_flag = True
            self.assertDictEqual(self.counter, {'enter': 1, 'exit': 0}, 'Counters did not match expected values')

        self.assertTrue(enter_flag, 'Context was never entered')
        self.assertDictEqual(self.counter, {'enter': 1, 'exit': 1}, 'Counters did not match expected values')

    def test_nested_context_single_with(self):
        self.assertDictEqual(self.counter, {'enter': 0, 'exit': 0}, 'Counters did not match expected values')

        # noinspection PyUnusedLocal
        enter_flag = False

        with self.context, self.context:
            enter_flag = True
            self.assertDictEqual(self.counter, {'enter': 1, 'exit': 0}, 'Counters did not match expected values')

        self.assertTrue(enter_flag, 'Context was never entered')
        self.assertDictEqual(self.counter, {'enter': 1, 'exit': 1}, 'Counters did not match expected values')


class ReentrantExitRpcContextTestCase(ReentrantSafetyContextTestCase):
    SYSTEM_TYPE = _ReentrantRpcTestSystem


class ReentrantExitErrorSafetyContextTestCase(ReentrantSafetyContextTestCase):
    SYSTEM_TYPE = _ReentrantExitErrorTestSystem


class ReentrantExitErrorRpcSafetyContextTestCase(ReentrantSafetyContextTestCase):
    SYSTEM_TYPE = _ReentrantExitErrorRpcTestSystem


class NonReentrantSafetyContextTestCase(ReentrantSafetyContextTestCase):
    SYSTEM_TYPE = _NonReentrantTestSystem

    def test_nested_context(self):
        self.assertDictEqual(self.counter, {'enter': 0, 'exit': 0}, 'Counters did not match expected values')

        with self.context:
            self.assertDictEqual(self.counter, {'enter': 1, 'exit': 0}, 'Counters did not match expected values')
            with self.assertRaises(SafetyContextError, msg='Reentering context did not raise'):
                with self.context:
                    self.fail()

        self.assertDictEqual(self.counter, {'enter': 1, 'exit': 1}, 'Counters did not match expected values')

    def test_nested_context_single_with(self):
        self.assertDictEqual(self.counter, {'enter': 0, 'exit': 0}, 'Counters did not match expected values')

        with self.assertRaises(SafetyContextError, msg='Reentering context did not raise'):
            with self.context, self.context:
                self.fail()

        self.assertDictEqual(self.counter, {'enter': 1, 'exit': 1}, 'Counters did not match expected values')


class NonReentrantRpcSafetyContextTestCase(NonReentrantSafetyContextTestCase):
    SYSTEM_TYPE = _NonReentrantRpcTestSystem


class NonReentrantExitErrorSafetyContextTestCase(NonReentrantSafetyContextTestCase):
    SYSTEM_TYPE = _NonReentrantExitErrorTestSystem


class NonReentrantExitErrorRpcSafetyContextTestCase(NonReentrantSafetyContextTestCase):
    SYSTEM_TYPE = _NonReentrantExitErrorRpcTestSystem
