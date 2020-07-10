import unittest
import typing

from dax.experiment import *
from dax.modules.rtio_benchmark import RtioBenchmarkModule, RtioLoopBenchmarkModule
import dax.sim.test_case


class _TestSystem(DaxSystem):
    SYS_ID = 'unittest_system'
    SYS_VER = 0

    def build(self, **kwargs) -> None:  # type: ignore
        super(_TestSystem, self).build()
        self.rtio = RtioBenchmarkModule(self, 'rtio', ttl_out='ttl_out', **kwargs)


class _LoopTestSystem(DaxSystem):
    SYS_ID = 'unittest_system'
    SYS_VER = 0

    def build(self, **kwargs) -> None:  # type: ignore
        super(_LoopTestSystem, self).build()
        self.rtio = RtioLoopBenchmarkModule(self, 'rtio', ttl_out='ttl_out', ttl_in='ttl_in', **kwargs)


class RtioBenchmarkModuleTestCase(dax.sim.test_case.PeekTestCase):
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
        'ttl_out': {
            'type': 'local',
            'module': 'artiq.coredevice.ttl',
            'class': 'TTLInOut'
        },
        'ttl_in': {
            'type': 'local',
            'module': 'artiq.coredevice.ttl',
            'class': 'TTLInOut'
        },
    }

    _SYS_TYPE: type = _TestSystem

    def _construct_env(self, **kwargs):
        return self.construct_env(self._SYS_TYPE, device_db=self._DEVICE_DB, build_kwargs=kwargs)

    def test_dax_init(self):
        s = self._construct_env(init_kernel=True)
        self.expect(s.rtio.ttl_out, 'state', 'x')
        self.expect(s.rtio.ttl_out, 'direction', 'x')
        s.dax_init()
        self.expect(s.rtio.ttl_out, 'state', 'x')
        self.expect(s.rtio.ttl_out, 'direction', 1)

    def test_dax_init_disabled(self):
        s = self._construct_env(init_kernel=False)
        self.expect(s.rtio.ttl_out, 'state', 'x')
        self.expect(s.rtio.ttl_out, 'direction', 'x')
        s.dax_init()
        self.expect(s.rtio.ttl_out, 'state', 'x')
        self.expect(s.rtio.ttl_out, 'direction', 'x')
        s.rtio.init_kernel()
        self.expect(s.rtio.ttl_out, 'state', 'x')
        self.expect(s.rtio.ttl_out, 'direction', 1)

    def test_manual_init(self):
        s = self._construct_env(init_kernel=False)
        self.expect(s.rtio.ttl_out, 'state', 'x')
        self.expect(s.rtio.ttl_out, 'direction', 'x')
        s.rtio.init()
        self.expect(s.rtio.ttl_out, 'state', 'x')
        self.expect(s.rtio.ttl_out, 'direction', 'x')
        s.rtio.init_kernel()
        self.expect(s.rtio.ttl_out, 'state', 'x')
        self.expect(s.rtio.ttl_out, 'direction', 1)

    def test_manual_init_2(self):
        s = self._construct_env(init_kernel=True)
        self.expect(s.rtio.ttl_out, 'state', 'x')
        self.expect(s.rtio.ttl_out, 'direction', 'x')
        s.rtio.init(force=True)
        self.expect(s.rtio.ttl_out, 'state', 'x')
        self.expect(s.rtio.ttl_out, 'direction', 1)

    def test_manual_init_force(self):
        s = self._construct_env(init_kernel=False)
        self.expect(s.rtio.ttl_out, 'state', 'x')
        self.expect(s.rtio.ttl_out, 'direction', 'x')
        s.rtio.init(force=True)
        self.expect(s.rtio.ttl_out, 'state', 'x')
        self.expect(s.rtio.ttl_out, 'direction', 1)


class RtioLoopBenchmarkModuleTestCase(RtioBenchmarkModuleTestCase):
    _SYS_TYPE = _LoopTestSystem

    def test_dax_init(self):
        s = self._construct_env(init_kernel=True)
        self.expect(s.rtio.ttl_out, 'state', 'x')
        self.expect(s.rtio.ttl_out, 'direction', 'x')
        self.expect(s.rtio.ttl_in, 'state', 'x')
        self.expect(s.rtio.ttl_in, 'sensitivity', 'x')
        self.expect(s.rtio.ttl_in, 'direction', 'x')
        s.dax_init()
        self.expect(s.rtio.ttl_out, 'state', 'x')
        self.expect(s.rtio.ttl_out, 'direction', 1)
        self.expect(s.rtio.ttl_in, 'state', 'z')
        self.expect(s.rtio.ttl_in, 'sensitivity', 0)
        self.expect(s.rtio.ttl_in, 'direction', 0)

    def test_dax_init_disabled(self):
        s = self._construct_env(init_kernel=False)
        self.expect(s.rtio.ttl_out, 'state', 'x')
        self.expect(s.rtio.ttl_out, 'direction', 'x')
        self.expect(s.rtio.ttl_in, 'state', 'x')
        self.expect(s.rtio.ttl_in, 'sensitivity', 'x')
        self.expect(s.rtio.ttl_in, 'direction', 'x')
        s.dax_init()
        self.expect(s.rtio.ttl_out, 'state', 'x')
        self.expect(s.rtio.ttl_out, 'direction', 'x')
        self.expect(s.rtio.ttl_in, 'state', 'x')
        self.expect(s.rtio.ttl_in, 'sensitivity', 'x')
        self.expect(s.rtio.ttl_in, 'direction', 'x')
        s.rtio.init_kernel()
        self.expect(s.rtio.ttl_out, 'state', 'x')
        self.expect(s.rtio.ttl_out, 'direction', 1)
        self.expect(s.rtio.ttl_in, 'state', 'z')
        self.expect(s.rtio.ttl_in, 'sensitivity', 0)
        self.expect(s.rtio.ttl_in, 'direction', 0)

    def test_manual_init(self):
        s = self._construct_env(init_kernel=False)
        self.expect(s.rtio.ttl_out, 'state', 'x')
        self.expect(s.rtio.ttl_out, 'direction', 'x')
        self.expect(s.rtio.ttl_in, 'state', 'x')
        self.expect(s.rtio.ttl_in, 'sensitivity', 'x')
        self.expect(s.rtio.ttl_in, 'direction', 'x')
        s.rtio.init()
        self.expect(s.rtio.ttl_out, 'state', 'x')
        self.expect(s.rtio.ttl_out, 'direction', 'x')
        self.expect(s.rtio.ttl_in, 'state', 'x')
        self.expect(s.rtio.ttl_in, 'sensitivity', 'x')
        self.expect(s.rtio.ttl_in, 'direction', 'x')
        s.rtio.init_kernel()
        self.expect(s.rtio.ttl_out, 'state', 'x')
        self.expect(s.rtio.ttl_out, 'direction', 1)
        self.expect(s.rtio.ttl_in, 'state', 'z')
        self.expect(s.rtio.ttl_in, 'sensitivity', 0)
        self.expect(s.rtio.ttl_in, 'direction', 0)

    def test_manual_init_2(self):
        s = self._construct_env(init_kernel=True)
        self.expect(s.rtio.ttl_out, 'state', 'x')
        self.expect(s.rtio.ttl_out, 'direction', 'x')
        self.expect(s.rtio.ttl_in, 'state', 'x')
        self.expect(s.rtio.ttl_in, 'sensitivity', 'x')
        self.expect(s.rtio.ttl_in, 'direction', 'x')
        s.rtio.init(force=True)
        self.expect(s.rtio.ttl_out, 'state', 'x')
        self.expect(s.rtio.ttl_out, 'direction', 1)
        self.expect(s.rtio.ttl_in, 'state', 'z')
        self.expect(s.rtio.ttl_in, 'sensitivity', 0)
        self.expect(s.rtio.ttl_in, 'direction', 0)

    def test_manual_init_force(self):
        s = self._construct_env(init_kernel=False)
        self.expect(s.rtio.ttl_out, 'state', 'x')
        self.expect(s.rtio.ttl_out, 'direction', 'x')
        self.expect(s.rtio.ttl_in, 'state', 'x')
        self.expect(s.rtio.ttl_in, 'sensitivity', 'x')
        self.expect(s.rtio.ttl_in, 'direction', 'x')
        s.rtio.init(force=True)
        self.expect(s.rtio.ttl_out, 'state', 'x')
        self.expect(s.rtio.ttl_out, 'direction', 1)
        self.expect(s.rtio.ttl_in, 'state', 'z')
        self.expect(s.rtio.ttl_in, 'sensitivity', 0)
        self.expect(s.rtio.ttl_in, 'direction', 0)


if __name__ == '__main__':
    unittest.main()
