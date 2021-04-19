import unittest
import typing

import artiq.experiment

import dax.base.system
import dax.interfaces.operation
import dax.base.program
from dax.util.artiq import get_managers
from dax.sim import enable_dax_sim

import test.interfaces.test_operation
import test.interfaces.test_data_context
import test.helpers

_DEVICE_DB: typing.Dict[str, typing.Any] = {
    'core': {
        'type': 'local',
        'module': 'artiq.coredevice.core',
        'class': 'Core',
        'arguments': {'host': None, 'ref_period': 1e-9}
    },
    'core_cache': {
        'type': 'local',
        'module': 'artiq.coredevice.cache',
        'class': 'CoreCache'
    },
    'core_dma': {
        'type': 'local',
        'module': 'artiq.coredevice.dma',
        'class': 'CoreDMA'
    },
}


class _TestSystem(dax.base.system.DaxSystem,
                  test.interfaces.test_operation.OperationInstance,
                  test.interfaces.test_data_context.DataContextInstance):
    SYS_ID = 'unittest_system'
    SYS_VER = 0


class _TestProgram(dax.base.program.DaxProgram, artiq.experiment.Experiment):
    def build(self, *args, **kwargs) -> None:
        super(_TestProgram, self).build(*args, **kwargs)
        self.did_prepare = False
        self.did_run = False
        self.did_analyze = False

    def prepare(self):
        self.did_prepare = True

    def run(self):
        self.did_run = True

        # A random program
        self.core.reset()
        self.q.prep_0_all()
        self.q.h(0)
        self.q.m_z_all()
        self.q.store_measurements_all()

    def analyze(self):
        self.did_analyze = True


class DaxProgramTestCase(unittest.TestCase):
    EXPECTED_KERNEL_INVARIANTS = {'logger', 'core', 'q', 'data_context'}

    def setUp(self) -> None:
        self.managers = get_managers(enable_dax_sim(ddb=_DEVICE_DB, enable=True, logging_level=30,
                                                    output='null', moninj_service=False))

    def tearDown(self) -> None:
        self.managers.close()

    def test_link(self):
        # Create system
        system = _TestSystem(self.managers)
        # Dynamically link system to the program
        program = _TestProgram(system, core=system.core, operation=system, data_context=system)
        # Basic checks
        self.assertIsInstance(program, dax.base.program.DaxProgram)
        self.assertIs(program.core, system.core)
        self.assertIs(program.q, system)
        self.assertSetEqual(program.kernel_invariants, self.EXPECTED_KERNEL_INVARIANTS)
        self.assertIn(_TestProgram.__name__, program.get_identifier())
        dax.interfaces.operation.validate_interface(program.q)
        test.helpers.test_system_kernel_invariants(self, system)
        test.helpers.test_kernel_invariants(self, program)

    def test_isolated_link(self):
        # Create isolated managers
        isolated = dax.util.artiq.isolate_managers(self.managers)
        # Create system
        system = _TestSystem(self.managers)
        # Dynamically link system to the program in an isolated fashion
        program = _TestProgram(isolated, core=system.core, operation=system, data_context=system)
        # Basic checks
        self.assertIsInstance(program, dax.base.program.DaxProgram)
        self.assertIs(program.core, system.core)
        self.assertIs(program.q, system)
        self.assertSetEqual(program.kernel_invariants, self.EXPECTED_KERNEL_INVARIANTS)
        self.assertIn(_TestProgram.__name__, program.get_identifier())
        dax.interfaces.operation.validate_interface(program.q)
        test.helpers.test_system_kernel_invariants(self, system)
        test.helpers.test_kernel_invariants(self, program)

        # Return program for other tests
        return program

    def test_run(self):
        program = self.test_isolated_link()
        # Run the program
        self.assertFalse(program.did_prepare)
        program.prepare()
        self.assertTrue(program.did_prepare)
        self.assertFalse(program.did_run)
        program.run()
        self.assertTrue(program.did_run)
        self.assertFalse(program.did_analyze)
        program.analyze()
        self.assertTrue(program.did_analyze)


if __name__ == '__main__':
    unittest.main()
