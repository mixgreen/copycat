import typing

from dax.experiment import *
from dax.modules.cpld_init import CpldInitModule
import dax.sim.test_case


class _TestSystem(DaxSystem):
    SYS_ID = 'unittest_system'
    SYS_VER = 0
    CORE_LOG_KEY = None
    DAX_INFLUX_DB_KEY = None

    def build(self, **kwargs) -> None:  # type: ignore[override]
        super(_TestSystem, self).build()
        self.cpld = CpldInitModule(self, 'cpld', **kwargs)


class CpldInitModuleTestCase(dax.sim.test_case.PeekTestCase):
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
        'spi_urukul0': {
            'type': 'local',
            'module': 'artiq.coredevice.spi2',
            'class': 'SPIMaster',
        },
        'cpld_0': {
            'type': 'local',
            'module': 'artiq.coredevice.urukul',
            'class': 'CPLD',
            'arguments': {
                'spi_device': 'spi_urukul0'
            }
        },
    }

    def _construct_env(self, **kwargs):
        env = self.construct_env(_TestSystem, device_db=self._DEVICE_DB, build_kwargs=kwargs)
        self.assertEqual(len(env.cpld.cpld), 1, 'Did not found the expected number of CPLD devices')
        return env

    def test_dax_init(self):
        s = self._construct_env(init_kernel=True)
        self.expect(s.cpld.cpld[0], 'init_att', 'x')
        s.dax_init()
        self.expect(s.cpld.cpld[0], 'init_att', 1)

    def test_dax_init_disabled(self):
        s = self._construct_env(init_kernel=False)
        self.expect(s.cpld.cpld[0], 'init_att', 'x')
        s.dax_init()
        self.expect(s.cpld.cpld[0], 'init_att', 'x')
        s.cpld.init_kernel()
        self.expect(s.cpld.cpld[0], 'init_att', 1)

    def test_manual_init(self):
        s = self._construct_env(init_kernel=False)
        self.expect(s.cpld.cpld[0], 'init_att', 'x')
        s.cpld.init()
        self.expect(s.cpld.cpld[0], 'init_att', 'x')
        s.cpld.init_kernel()
        self.expect(s.cpld.cpld[0], 'init_att', 1)

    def test_manual_init_2(self):
        s = self._construct_env(init_kernel=True)
        self.expect(s.cpld.cpld[0], 'init_att', 'x')
        s.cpld.init(force=True)
        self.expect(s.cpld.cpld[0], 'init_att', 1)

    def test_manual_init_force(self):
        s = self._construct_env(init_kernel=False)
        self.expect(s.cpld.cpld[0], 'init_att', 'x')
        s.cpld.init(force=True)
        self.expect(s.cpld.cpld[0], 'init_att', 1)
