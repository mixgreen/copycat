import unittest
import typing

import artiq.experiment

import dax.base.system
import dax.interfaces.operation
import dax.base.program
from dax.util.artiq import get_managers
from dax.sim import enable_dax_sim

import test.interfaces.test_operation

_DEVICE_DB: typing.Dict[str, typing.Any] = {
    'core': {
        'type': 'local',
        'module': 'artiq.coredevice.core',
        'class': 'Core',
        'arguments': {'host': '0.0.0.0', 'ref_period': 1e-9}
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


class _TestSystem(dax.base.system.DaxSystem, test.interfaces.test_operation.OperationInstance):
    SYS_ID = 'unittest_system'
    SYS_VER = 0


class _MinimalTestProgram(dax.base.program.DaxProgram, artiq.experiment.Experiment):
    def run(self):
        pass  # Run needs to be implemented


class _TestProgram(_MinimalTestProgram):
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
    def setUp(self) -> None:
        self.managers = get_managers(enable_dax_sim(ddb=_DEVICE_DB, enable=True, logging_level=30,
                                                    output='null', moninj_service=False))

    def tearDown(self) -> None:
        self.managers.close()

    def test_link(self):
        # Create the factory
        factory = dax.base.system.dax_client_factory(_TestProgram)
        # Link system to program
        linked_program_class = factory(_TestSystem)
        # Instantiate program
        return linked_program_class(self.managers)

    def test_run(self):
        program = self.test_link()
        # Run the program
        program.prepare()
        self.assertTrue(program.did_prepare)
        program.run()
        self.assertTrue(program.did_run)
        program.analyze()
        self.assertTrue(program.did_analyze)


if __name__ == '__main__':
    unittest.main()
