import collections
import typing

from dax.experiment import *
from dax.services.safety_context import *

import test.modules.safety_context_test as _test


class _TestSystem(DaxSystem):
    SYS_ID = 'unittest_system'
    SYS_VER = 0
    CORE_LOG_KEY = None
    DAX_INFLUX_DB_KEY = None


class _ReentrantSafetyContextService(ReentrantSafetyContextService):
    SERVICE_NAME = 'safety_context_service'


class _SafetyContextService(SafetyContextService):
    SERVICE_NAME = 'safety_context_service'


class SafetyContextServiceTestCase(_test.SafetyContextTestCase):

    def _init_context(self, managers, **kwargs):
        _ReentrantSafetyContextService(_TestSystem(managers), **kwargs)


class _ReentrantServiceTestSystem(_TestSystem):
    SAFETY_CONTEXT_TYPE: typing.Any = _ReentrantSafetyContextService
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
    SAFETY_CONTEXT_TYPE = _SafetyContextService


class _NonReentrantServiceRpcTestSystem(_NonReentrantServiceTestSystem):
    RPC = True


class _NonReentrantServiceExitErrorTestSystem(_NonReentrantServiceTestSystem):
    EXIT_ERROR = False


class _NonReentrantServiceExitErrorRpcTestSystem(_NonReentrantServiceTestSystem):
    EXIT_ERROR = False
    RPC = True


class ReentrantSafetyContextServiceTestCase(_test.ReentrantSafetyContextTestCase):
    SYSTEM_TYPE: typing.Any = _ReentrantServiceTestSystem


class ReentrantRpcSafetyContextServiceTestCase(ReentrantSafetyContextServiceTestCase):
    SYSTEM_TYPE = _ReentrantServiceRpcTestSystem


class ReentrantExitErrorSafetyContextServiceTestCase(ReentrantSafetyContextServiceTestCase):
    SYSTEM_TYPE = _ReentrantServiceExitErrorTestSystem


class ReentrantExitErrorRpcSafetyContextServiceTestCase(ReentrantSafetyContextServiceTestCase):
    SYSTEM_TYPE = _ReentrantServiceExitErrorRpcTestSystem


class NonReentrantSafetyContextServiceTestCase(_test.NonReentrantSafetyContextTestCase,
                                               ReentrantSafetyContextServiceTestCase):
    SYSTEM_TYPE: typing.Any = _NonReentrantServiceTestSystem


class NonReentrantRpcSafetyContextServiceTestCase(NonReentrantSafetyContextServiceTestCase):
    SYSTEM_TYPE = _NonReentrantServiceRpcTestSystem


class NonReentrantExitErrorSafetyContextServiceTestCase(NonReentrantSafetyContextServiceTestCase):
    SYSTEM_TYPE = _NonReentrantServiceExitErrorTestSystem


class NonReentrantExitErrorRpcSafetyContextServiceTestCase(NonReentrantSafetyContextServiceTestCase):
    SYSTEM_TYPE = _NonReentrantServiceExitErrorRpcTestSystem
