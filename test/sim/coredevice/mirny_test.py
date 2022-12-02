import random

from artiq.experiment import *

import dax.sim.test_case
import dax.sim.coredevice.mirny
from dax.util.artiq_version import ARTIQ_MAJOR_VERSION

import test.sim.coredevice._compile_testcase as compile_testcase

_DEVICE_DB = {
    'core': {
        'type': 'local',
        'module': 'artiq.coredevice.core',
        'class': 'Core',
        'arguments': {'host': None, 'ref_period': 1e-9}
    },
    'spi_mirny0': {
        'type': 'local',
        'module': 'artiq.coredevice.spi2',
        'class': 'SPIMaster',
    },
    'ttl_mirny0_sw0': {
        'type': 'local',
        'module': 'artiq.coredevice.ttl',
        'class': 'TTLOut',
        'arguments': {},
    },
    "mirny0_cpld": {
        "type": "local",
        "module": "artiq.coredevice.mirny",
        "class": "Mirny",
        "arguments": {
            "spi_device": "spi_mirny0",
            "refclk": 100000000.0,
            "clk_sel": 0
        },
    },
    "mirny0_almazny": {
        "type": "local",
        "module": "artiq.coredevice.mirny",
        "class": "Almazny",
        "arguments": {
            "host_mirny": "mirny0_cpld",
        },
    }
}


class _Environment(HasEnvironment):
    def build(self, *, dut):
        self.core = self.get_device('core')
        self.dut = self.get_device(dut)


class MirnyTestCase(dax.sim.test_case.PeekTestCase):
    SEED = None

    def setUp(self) -> None:
        self.rng = random.Random(self.SEED)
        self.env = self.construct_env(_Environment, device_db=_DEVICE_DB, build_kwargs=dict(dut='mirny0_cpld'))

    def test_init(self):
        self.expect(self.env.dut, 'init', 'x')
        self.env.dut.init()
        self.expect(self.env.dut, 'init', 1)

    def test_set_att_mu(self):
        for channel in range(4):
            signal = f'att_{channel}'
            self.expect(self.env.dut, signal, 'x')
            self.env.dut.set_att_mu(channel, 255)
            self.expect(self.env.dut, signal, 0.0)

    if ARTIQ_MAJOR_VERSION >= 7:
        def test_att_to_mu(self):
            self.assertEqual(self.env.dut.att_to_mu(0.0), 255)

        def test_set_att(self):
            for channel in range(4):
                signal = f'att_{channel}'
                self.expect(self.env.dut, signal, 'x')
                self.env.dut.set_att(channel, 0.0)
                self.expect(self.env.dut, signal, 0.0)


class MirnyCompileTestCase(compile_testcase.CoredeviceCompileTestCase):
    DEVICE_CLASS = dax.sim.coredevice.mirny.Mirny
    DEVICE_KWARGS = {
        'spi_device': 'spi_mirny0',
    }
    FN_ARGS = {
        'set_att_mu': (0, 0),
        'set_att': (0, 0.0),
        'att_to_mu': (0.0,),
    }
    FN_EXCLUDE = {'write_reg', 'read_reg', 'write_ext'}
    DEVICE_DB = _DEVICE_DB


if ARTIQ_MAJOR_VERSION >= 7:  # noqa: C901

    class AlmaznyTestCase(dax.sim.test_case.PeekTestCase):
        SEED = None

        def setUp(self) -> None:
            self.rng = random.Random(self.SEED)
            self.env = self.construct_env(_Environment, device_db=_DEVICE_DB, build_kwargs=dict(dut='mirny0_almazny'))

        def test_init(self):
            self.expect(self.env.dut, 'init', 'x')
            self.assertFalse(self.env.dut.output_enable)
            self.env.dut.init()
            self.expect(self.env.dut, 'init', 1)
            self.assertFalse(self.env.dut.output_enable)

        def test_att_to_mu(self):
            for value, ref in [(0.0, 0), (31.5, 63)]:
                self.assertEqual(self.env.dut.att_to_mu(value), ref)

        def test_mu_to_att(self):
            for value, ref in [(0, 0.0), (63, 31.5)]:
                self.assertEqual(self.env.dut.mu_to_att(value), ref)

        def test_set_att_mu(self):
            for channel in range(4):
                att_signal = f'att_{channel}'
                sw_signal = f'sw_{channel}'
                self.expect(self.env.dut, att_signal, 'x')
                self.expect(self.env.dut, sw_signal, 'x')
                for sw in [True, False]:
                    self.env.dut.set_att_mu(channel, 0, rf_switch=sw)
                    self.expect(self.env.dut, att_signal, 0.0)
                    self.expect(self.env.dut, sw_signal, sw)

        def test_set_att(self):
            for channel in range(4):
                att_signal = f'att_{channel}'
                sw_signal = f'sw_{channel}'
                self.expect(self.env.dut, att_signal, 'x')
                self.expect(self.env.dut, sw_signal, 'x')
                for sw in [True, False]:
                    self.env.dut.set_att(channel, 0.0, rf_switch=sw)
                    self.expect(self.env.dut, att_signal, 0.0)
                    self.expect(self.env.dut, sw_signal, sw)

        def test_output_toggle(self):
            for oe in [True, False]:
                self.env.dut.output_toggle(oe)
                self.expect(self.env.dut, 'oe', oe)
                self.assertEqual(self.env.dut.output_enable, oe)


    class AlmaznyCompileTestCase(compile_testcase.CoredeviceCompileTestCase):  # noqa: E303
        DEVICE_CLASS = dax.sim.coredevice.mirny.Almazny
        DEVICE_KWARGS = {
            'host_mirny': 'mirny0_cpld',
        }
        FN_ARGS = {
            'set_att_mu': (0, 0),
            'set_att': (0, 0.0),
            'att_to_mu': (0.0,),
            'mu_to_att': (0,),
            'output_toggle': (True,),
            '_update_register': (0,),
        }
        DEVICE_DB = _DEVICE_DB
