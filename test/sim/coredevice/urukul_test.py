import unittest
import random
import numpy as np

from artiq.experiment import *

import dax.sim.test_case
import dax.sim.coredevice.urukul
from dax.util.artiq_version import ARTIQ_MAJOR_VERSION

import test.sim.coredevice._compile_testcase as compile_testcase
from test.environment import CI_ENABLED

_NUM_SAMPLES = 1000 if CI_ENABLED else 100

_DEVICE_DB = {
    'core': {
        'type': 'local',
        'module': 'artiq.coredevice.core',
        'class': 'Core',
        'arguments': {'host': None, 'ref_period': 1e-9}
    },
    'spi_urukul1': {
        'type': 'local',
        'module': 'artiq.coredevice.spi2',
        'class': 'SPIMaster',
    },
    'io_update': {
        'type': 'local',
        'module': 'artiq.coredevice.ttl',
        'class': 'TTLOut',
        'arguments': {},
    },
    "dut": {
        "type": "local",
        "module": "artiq.coredevice.urukul",
        "class": "CPLD",
        "arguments": {
            "spi_device": "spi_urukul1",
            "sync_device": None,
            "io_update_device": "io_update",
            "refclk": 1e9,
            "clk_sel": 1,
            "clk_div": 3
        }
    },
}


class UrukulTestCase(unittest.TestCase):
    SEED = None

    def setUp(self) -> None:
        self.rng = random.Random(self.SEED)

    def test_conversion(self):
        for _ in range(_NUM_SAMPLES):
            a = self.rng.randrange(2 ** 8)
            with self.subTest(att_mu=a):
                o = dax.sim.coredevice.urukul._att_to_mu(dax.sim.coredevice.urukul._mu_to_att(a))
                self.assertEqual(a, o)


class _Environment(HasEnvironment):
    def build(self):
        self.core = self.get_device('core')
        self.dut = self.get_device('dut')


class UrukulPeekTestCase(dax.sim.test_case.PeekTestCase):
    SEED = None

    def setUp(self) -> None:
        self.rng = random.Random(self.SEED)
        self.env = self.construct_env(_Environment, device_db=_DEVICE_DB)

    def test_init(self):
        self.expect(self.env.dut, 'init', 'x')
        self.env.dut.init()
        self.expect(self.env.dut, 'init', 1)
        self.expect(self.env.dut, 'profile', dax.sim.coredevice.urukul.DEFAULT_PROFILE)

    def test_get_att_mu(self):
        self.expect(self.env.dut, 'init_att', 'x')
        self.assertEqual(self.env.dut.get_att_mu(), self.env.dut.att_reg)
        self.expect(self.env.dut, 'init_att', 1)

    def test_set_all_att_mu(self):
        for _ in range(_NUM_SAMPLES):
            a = self.rng.randrange(2 ** 32)
            self.env.dut.set_all_att_mu(a)
            self.assertEqual(self.env.dut.get_att_mu(), self.env.dut.att_reg)
            for i in range(4):
                self.expect(self.env.dut, f'att_{i}', dax.sim.coredevice.urukul._mu_to_att((a >> (8 * i) & 0xFF)))

    def test_set_att_mu(self):
        att_reg = self.env.dut.get_att_mu()
        for _ in range(_NUM_SAMPLES):
            a = np.int32(self.rng.randrange(2 ** 8))
            c = self.rng.randrange(4)
            self.env.dut.set_att_mu(c, a)
            att_reg &= ~(0xFF << (c * 8))
            att_reg |= a << (c * 8)
            self.assertEqual(np.int32(att_reg), self.env.dut.att_reg)
            self.assertEqual(self.env.dut.get_att_mu(), self.env.dut.att_reg)
            self.expect(self.env.dut, f'att_{c}', dax.sim.coredevice.urukul._mu_to_att(a))

    def test_set_att(self):
        att_reg = self.env.dut.get_att_mu()
        for _ in range(_NUM_SAMPLES):
            a = self.rng.uniform(0 * dB, 31.5 * dB)
            a_mu = dax.sim.coredevice.urukul._att_to_mu(a)
            c = self.rng.randrange(4)
            self.env.dut.set_att(c, a)
            att_reg &= ~(0xFF << (c * 8))
            att_reg |= a_mu << (c * 8)
            self.assertEqual(np.int32(att_reg), self.env.dut.att_reg)
            self.assertEqual(self.env.dut.get_att_mu(), self.env.dut.att_reg)
            self.expect(self.env.dut, f'att_{c}', a)

    def test_cfg_sw(self):
        self.expect(self.env.dut, 'sw', 'x')
        ref = '0001000'
        for c in range(4):
            for state in [1, 0]:
                with self.subTest(c=c, state=state):
                    self.env.dut.cfg_sw(c, state)
                    value = ref[c:4 + c] if state else '0000'
                    assert value[-1 - c] == str(state)
                    self.expect(self.env.dut, 'sw', value)

    def test_cfg_switches(self):
        self.expect(self.env.dut, 'sw', 'x')
        for c in range(_NUM_SAMPLES):
            state = self.rng.randrange(2 ** 4)
            ref = f'{state:04b}'
            assert len(ref) == 4
            with self.subTest(state=state, ref=ref):
                self.env.dut.cfg_switches(state)
                self.expect(self.env.dut, 'sw', ref)

    def test_set_profile(self):
        self.expect(self.env.dut, 'profile', 'x')
        self.env.dut.set_profile(dax.sim.coredevice.urukul.DEFAULT_PROFILE)
        self.expect(self.env.dut, 'profile', dax.sim.coredevice.urukul.DEFAULT_PROFILE)

    def test_io_update_notify(self):
        notified = 0

        def fn():
            nonlocal notified
            notified += 1

        self.env.dut.io_update.pulse_subscribe(fn)
        self.assertEqual(notified, 0)
        self.env.dut.io_update.pulse(1 * us)
        self.assertEqual(notified, 1)


class UrukulRegIOUpdatePeekTestCase(UrukulPeekTestCase):
    DEVICE_DB = {
        'core': {
            'type': 'local',
            'module': 'artiq.coredevice.core',
            'class': 'Core',
            'arguments': {'host': None, 'ref_period': 1e-9}
        },
        'spi_urukul0': {
            'type': 'local',
            'module': 'artiq.coredevice.spi2',
            'class': 'SPIMaster',
        },
        "dut": {
            "type": "local",
            "module": "artiq.coredevice.urukul",
            "class": "CPLD",
            "arguments": {
                "spi_device": "spi_urukul0",
                "sync_device": None,
                "refclk": 1e9,
                "clk_sel": 1,
                "clk_div": 3
            }
        },
    }


class CompileTestCase(compile_testcase.CoredeviceCompileTestCase):
    DEVICE_CLASS = dax.sim.coredevice.urukul.CPLD
    DEVICE_KWARGS = {
        "spi_device": "spi_urukul1",
    }
    FN_KWARGS = {
        'cfg_sw': {'channel': 0, 'on': True},
        'cfg_switches': {'state': 0x0},
        'set_att_mu': {'channel': 0, 'att': 0},
        'set_all_att_mu': {'att_reg': 0},
        'set_att': {'channel': 0, 'att': 0.0},
    }
    if ARTIQ_MAJOR_VERSION >= 7:
        FN_KWARGS.update({
            'mu_to_att': {'att_mu': 0},
            'att_to_mu': {'att': 0.0},
            'get_channel_att_mu': {'channel': 0},
            'get_channel_att': {'channel': 0},
        })
    DEVICE_DB = _DEVICE_DB
