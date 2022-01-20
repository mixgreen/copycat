from dax.services.safety_context import *
from test.modules.safety_context_test import *


class _TestSystem(DaxSystem):
    SYS_ID = 'unittest_system'
    SYS_VER = 0
    CORE_LOG_KEY = None
    DAX_INFLUX_DB_KEY = None


class SafetyContextServiceTestCase(SafetyContextTestCase):

    def _init_context(self, managers, **kwargs):
        ReentrantSafetyContextService(_TestSystem(managers), **kwargs)


class _ReentrantServiceTestSystem(_TestSystem):
    SAFETY_CONTEXT_TYPE = ReentrantSafetyContextService
    EXIT_ERROR = True
    RPC = False

    def build(self) -> None:  # type: ignore[override]
        super(_ReentrantServiceTestSystem, self).build()
        self.context = self._init_context(enter_cb=self.enter, exit_cb=self.exit,
                                          exit_error=self.EXIT_ERROR, rpc_=self.RPC)
        self.counter = collections.Counter({'enter': 0, 'exit': 0})

    def enter(self):
        self.counter['enter'] += 1

    def exit(self):
        self.counter['exit'] += 1

    def _init_context(self, **kwargs):
        return self.SAFETY_CONTEXT_TYPE(self, **kwargs)


class _ReentrantServiceRpcTestSystem(_ReentrantServiceTestSystem):
    RPC = True


class _ReentrantServiceExitErrorTestSystem(_ReentrantServiceTestSystem):
    EXIT_ERROR = False


class _ReentrantServiceExitErrorRpcTestSystem(_ReentrantServiceTestSystem):
    EXIT_ERROR = False
    RPC = True


class _NonReentrantServiceTestSystem(_ReentrantServiceTestSystem):
    SAFETY_CONTEXT_TYPE = SafetyContextService


class _NonReentrantServiceRpcTestSystem(_NonReentrantServiceTestSystem):
    RPC = True


class _NonReentrantServiceExitErrorTestSystem(_NonReentrantServiceTestSystem):
    EXIT_ERROR = False


class _NonReentrantServiceExitErrorRpcTestSystem(_NonReentrantServiceTestSystem):
    EXIT_ERROR = False
    RPC = True


class ReentrantSafetyContextServiceTestCase(ReentrantSafetyContextTestCase):
    SYSTEM_TYPE = _ReentrantServiceTestSystem
    pass


class ReentrantRpcSafetyContextServiceTestCase(ReentrantSafetyContextServiceTestCase):
    SYSTEM_TYPE = _ReentrantServiceRpcTestSystem


class ReentrantExitErrorSafetyContextServiceTestCase(ReentrantSafetyContextServiceTestCase):
    SYSTEM_TYPE = _ReentrantServiceExitErrorTestSystem


class ReentrantExitErrorRpcSafetyContextServiceTestCase(ReentrantSafetyContextServiceTestCase):
    SYSTEM_TYPE = _ReentrantServiceExitErrorRpcTestSystem


class NonReentrantSafetyContextServiceTestCase(NonReentrantSafetyContextTestCase,
                                               ReentrantSafetyContextServiceTestCase):
    SYSTEM_TYPE = _NonReentrantServiceTestSystem
    pass


class NonReentrantRpcSafetyContextServiceTestCase(NonReentrantSafetyContextServiceTestCase):
    SYSTEM_TYPE = _NonReentrantServiceRpcTestSystem


class NonReentrantExitErrorSafetyContextServiceTestCase(NonReentrantSafetyContextServiceTestCase):
    SYSTEM_TYPE = _NonReentrantServiceExitErrorTestSystem


class NonReentrantExitErrorRpcSafetyContextServiceTestCase(NonReentrantSafetyContextServiceTestCase):
    SYSTEM_TYPE = _NonReentrantServiceExitErrorRpcTestSystem
