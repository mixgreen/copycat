import unittest
import typing
import collections

from artiq.language import TBool, portable, kernel, rpc, host_only

from dax.base.control_flow import DaxControlFlow
from dax.base.exceptions import BuildError
from dax.base.system import DaxSystem
from dax.util.artiq import get_managers

from test.environment import CI_ENABLED
import test.helpers


class _MockSystem(DaxSystem):
    SYS_ID = 'test_system'
    SYS_VER = 0
    CORE_LOG_KEY = None
    DAX_INFLUX_DB_KEY = None


class _MockControlFlowReference(DaxControlFlow, _MockSystem):
    NUM_ITERATIONS = 10 if CI_ENABLED else 2

    def build(self) -> None:  # type: ignore[override]
        self.counter: typing.Counter[str] = collections.Counter()
        self.__iterations = 0
        super(_MockControlFlowReference, self).build()

    @portable
    def _dax_control_flow_while(self) -> TBool:
        return self.__iterations < self.NUM_ITERATIONS

    def _dax_control_flow_is_kernel(self) -> bool:
        return False

    def host_enter(self) -> None:
        self.counter['host_enter'] += 1

    def host_setup(self) -> None:
        self.counter['host_setup'] += 1

    def _dax_control_flow_setup(self) -> None:
        self.counter['_dax_control_flow_setup'] += 1

    def device_setup(self) -> None:
        self.counter['device_setup'] += 1

    @portable
    def _dax_control_flow_run(self) -> None:
        self.counter['_dax_control_flow_run'] += 1
        self.__iterations += 1

    def device_cleanup(self) -> None:
        self.counter['device_cleanup'] += 1

    def _dax_control_flow_cleanup(self) -> None:
        self.counter['_dax_control_flow_cleanup'] += 1

    def host_cleanup(self) -> None:
        self.counter['host_cleanup'] += 1

    def host_exit(self) -> None:
        self.counter['host_exit'] += 1


class ControlFlowTestCase(unittest.TestCase):
    CF_CLASS = _MockControlFlowReference

    def setUp(self) -> None:
        self.managers = get_managers()
        self.cf = self.CF_CLASS(self.managers)

    def tearDown(self) -> None:
        # Close managers
        self.managers.close()

    def test_kernel_invariants(self):
        test.helpers.test_system_kernel_invariants(self, self.cf)

    def test_call_counters(self):
        # Run the scan
        self.cf.run()

        # Verify counters
        counter_ref = {
            'host_enter': 1,
            'host_setup': 1,
            '_dax_control_flow_setup': 1,
            'device_setup': 1,
            '_dax_control_flow_run': self.CF_CLASS.NUM_ITERATIONS,
            'device_cleanup': 1,
            '_dax_control_flow_cleanup': 1,
            'host_cleanup': 1,
            'host_exit': 1,
        }
        self.assertDictEqual(self.cf.counter, counter_ref, 'Function counters did not match expected values')


class ControlFlowBuildTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.managers = get_managers()

    def tearDown(self) -> None:
        # Close managers
        self.managers.close()

    def test_run_portable(self):
        class CF0(_MockControlFlowReference):
            def _dax_control_flow_run(self) -> None:
                pass

        class CF1(_MockControlFlowReference):
            @rpc
            def _dax_control_flow_run(self) -> None:
                pass

        class CF2(_MockControlFlowReference):
            @kernel
            def _dax_control_flow_run(self) -> None:
                pass

        class CF3(_MockControlFlowReference):
            @host_only
            def _dax_control_flow_run(self) -> None:
                pass

        for C in [CF0, CF1, CF2, CF3]:
            with self.assertRaises(BuildError):
                C(self.managers)

    def test_host_fn_kernel(self):
        class CF0(_MockControlFlowReference):
            @kernel
            def host_enter(self) -> None:
                pass

        class CF1(_MockControlFlowReference):
            @kernel
            def host_setup(self) -> None:
                pass

        class CF2(_MockControlFlowReference):
            @kernel
            def host_cleanup(self) -> None:
                pass

        class CF3(_MockControlFlowReference):
            @kernel
            def host_exit(self) -> None:
                pass

        for C in [CF0, CF1, CF2, CF3]:
            with self.assertRaises(BuildError):
                C(self.managers)

    def test_device_fn_kernel(self):
        class CFIsKernel(_MockControlFlowReference):
            def _dax_control_flow_is_kernel(self) -> bool:
                return True

        class CFK(CFIsKernel):
            @kernel
            def device_setup(self) -> None:
                pass

            @portable
            def device_cleanup(self) -> None:
                pass

        class CF0(CFIsKernel):
            @rpc
            def device_setup(self) -> None:
                pass

        class CF1(CFIsKernel):
            @host_only
            def device_cleanup(self) -> None:
                pass

        # No errors
        CFK(self.managers)

        for C in [CF0, CF1]:
            with self.assertRaises(BuildError):
                C(self.managers)
