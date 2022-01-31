import typing

from dax.experiment import *
from dax.modules.safety_context import SafetyContext, ReentrantSafetyContext, SafetyContextError

import test.hw_test


class _ReentrantTestSystem(DaxSystem):
    SYS_ID = 'unittest_system'
    SYS_VER = 0
    CORE_LOG_KEY = None
    DAX_INFLUX_DB_KEY = None

    SAFETY_CONTEXT_TYPE: typing.Any = ReentrantSafetyContext
    EXIT_ERROR = True
    RPC = False

    def build(self) -> None:  # type: ignore[override]
        super(_ReentrantTestSystem, self).build()
        self.context = self.SAFETY_CONTEXT_TYPE(self, 'context',
                                                enter_cb=self.enter, exit_cb=self.exit,
                                                exit_error=self.EXIT_ERROR, rpc_=self.RPC)
        self.update_kernel_invariants('context')

        self.enter_counter = 0
        self.exit_counter = 0

    @portable
    def enter(self):
        self.enter_counter += 1

    @portable
    def exit(self):
        self.exit_counter += 1

    @portable
    def get_portable(self) -> TTuple([TInt32, TInt32]):  # type: ignore[valid-type]
        return self.enter_counter, self.exit_counter

    @kernel
    def get_kernel(self) -> TTuple([TInt32, TInt32]):  # type: ignore[valid-type]
        return self.get_portable()

    @rpc
    def get_rpc(self) -> TTuple([TInt32, TInt32]):  # type: ignore[valid-type]
        return self.get_portable()


class _ReentrantRpcTestSystem(_ReentrantTestSystem):
    RPC = True


class _NonReentrantTestSystem(_ReentrantTestSystem):
    SAFETY_CONTEXT_TYPE = SafetyContext


class _NonReentrantRpcTestSystem(_NonReentrantTestSystem):
    RPC = True


class ReentrantSafetyContextTestCase(test.hw_test.HardwareTestCase):
    SYSTEM_TYPE = _ReentrantTestSystem

    def test_kernel_entry_kernel_check(self):
        class Env(self.SYSTEM_TYPE, Experiment):
            @kernel
            def run(self):
                with self.context:
                    self.core.reset()
                return self.get_kernel()

        self.assertEqual(self.construct_env(Env).run(), (0, 0) if self.SYSTEM_TYPE.RPC else (1, 1))

    def test_kernel_entry_host_check(self):
        class Env(self.SYSTEM_TYPE, Experiment):
            @kernel
            def run(self):
                with self.context:
                    self.core.reset()
                return self.get_rpc()

        self.assertEqual(self.construct_env(Env).run(), (1, 1) if self.SYSTEM_TYPE.RPC else (0, 0))

    def test_host_entry_kernel_check(self):
        class Env(self.SYSTEM_TYPE, Experiment):
            @rpc
            def rpc(self):
                with self.context:
                    pass

            @kernel
            def run(self):
                self.core.reset()
                self.rpc()
                return self.get_kernel()

        self.assertEqual(self.construct_env(Env).run(), (0, 0))

    def test_host_entry_host_check(self):
        class Env(self.SYSTEM_TYPE, Experiment):
            @rpc
            def rpc(self):
                with self.context:
                    pass

            @kernel
            def run(self):
                self.core.reset()
                self.rpc()
                return self.get_rpc()

        self.assertEqual(self.construct_env(Env).run(), (1, 1))

    def test_kernel_reentry(self):
        class Env(self.SYSTEM_TYPE, Experiment):
            @kernel
            def run(self):
                with self.context:
                    with self.context:
                        self.core.reset()

        env = self.construct_env(Env)
        env.run()

        # Even with reentry, enter and exit are only called once
        # With RPC enabled, the counters get overwritten by attribute synchronization at kernel exit
        self.assertEqual(env.get_portable(), (0, 0) if self.SYSTEM_TYPE.RPC else (1, 1))

    def test_host_reentry(self):
        class Env(self.SYSTEM_TYPE, Experiment):
            @rpc
            def rpc(self):
                with self.context:
                    with self.context:
                        pass

            @kernel
            def run(self):
                self.core.reset()
                self.rpc()

        env = self.construct_env(Env)
        env.run()

        # Even with reentry, enter and exit are only called once
        # `self.context` was never used, the counters do NOT get overwritten by attribute synchronization at kernel exit
        self.assertEqual(env.get_portable(), (1, 1))


class ReentrantRpcSafetyContextTestCase(ReentrantSafetyContextTestCase):
    SYSTEM_TYPE = _ReentrantRpcTestSystem


class NonReentrantSafetyContextTestCase(ReentrantSafetyContextTestCase):
    SYSTEM_TYPE = _NonReentrantTestSystem

    def test_kernel_reentry(self):
        with self.assertRaises(SafetyContextError):
            super(NonReentrantSafetyContextTestCase, self).test_kernel_reentry()

    def test_host_reentry(self):
        with self.assertRaises(SafetyContextError):
            super(NonReentrantSafetyContextTestCase, self).test_host_reentry()


class NonReentrantRpcSafetyContextTestCase(NonReentrantSafetyContextTestCase):
    SYSTEM_TYPE = _NonReentrantRpcTestSystem

    def test_kernel_reentry(self):
        super(NonReentrantRpcSafetyContextTestCase, self).test_kernel_reentry()
