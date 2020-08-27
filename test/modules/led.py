import unittest
import typing

from dax.experiment import *
from dax.modules.led import LedModule
import dax.sim.test_case


class _TestSystem(DaxSystem):
    SYS_ID = 'unittest_system'
    SYS_VER = 0

    def build(self, **kwargs) -> None:  # type: ignore
        super(_TestSystem, self).build()
        self.led = LedModule(self, 'led', 'led_0', **kwargs)


class LedModuleTestCase(dax.sim.test_case.PeekTestCase):
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
        'led_0': {
            'type': 'local',
            'module': 'artiq.coredevice.ttl',
            'class': 'TTLOut'
        },
    }

    def _construct_env(self, **kwargs):
        return self.construct_env(_TestSystem, device_db=self._DEVICE_DB, build_kwargs=kwargs)

    def test_dax_init(self):
        s = self._construct_env(init_kernel=True)
        self.expect(s.led.led[0], 'state', 'x')
        s.dax_init()
        self.expect(s.led.led[0], 'state', 0)

    def test_dax_init_disabled(self):
        s = self._construct_env(init_kernel=False)
        self.expect(s.led.led[0], 'state', 'x')
        s.dax_init()
        self.expect(s.led.led[0], 'state', 'x')
        s.led.init_kernel()
        self.expect(s.led.led[0], 'state', 0)

    def test_manual_init(self):
        s = self._construct_env(init_kernel=False)
        self.expect(s.led.led[0], 'state', 'x')
        s.led.init()
        self.expect(s.led.led[0], 'state', 'x')
        s.led.init_kernel()
        self.expect(s.led.led[0], 'state', 0)

    def test_manual_init_2(self):
        s = self._construct_env(init_kernel=True)
        self.expect(s.led.led[0], 'state', 'x')
        s.led.init(force=True)
        self.expect(s.led.led[0], 'state', 0)

    def test_manual_init_force(self):
        s = self._construct_env(init_kernel=False)
        self.expect(s.led.led[0], 'state', 'x')
        s.led.init(force=True)
        self.expect(s.led.led[0], 'state', 0)

    def test_set_o(self):
        s = self._construct_env()
        for o in [True, False]:
            s.led.set_o(o)
            self.expect(s.led.led[0], 'state', o)

    def test_on_off(self):
        s = self._construct_env()
        s.led.on()
        self.expect(s.led.led[0], 'state', 1)
        s.led.off()
        self.expect(s.led.led[0], 'state', 0)
        s.led.on_all()
        self.expect(s.led.led[0], 'state', 1)
        s.led.off_all()
        self.expect(s.led.led[0], 'state', 0)

    def test_pulse(self):
        s = self._construct_env()
        duration = 1.0
        with parallel:
            s.led.pulse(duration)
            with sequential:
                self.expect(s.led.led[0], 'state', 1)
                for d in [0.1] * 9:
                    delay(d)
                    self.expect(s.led.led[0], 'state', 1)
            with sequential:
                delay(duration)
                self.expect(s.led.led[0], 'state', 0)
        self.expect(s.led.led[0], 'state', 0)

    def test_set_code(self):
        s = self._construct_env()
        for code in [1, 8, 5, 7, 0, 3, 5, 77, 34, 789, 234, 59, 68]:
            delay_mu(1)
            s.led.set_code(code)
            self.expect(s.led.led[0], 'state', code % 2)


if __name__ == '__main__':
    unittest.main()
