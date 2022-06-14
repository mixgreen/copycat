import random

from artiq.experiment import *

import dax.sim.test_case
import dax.sim.coredevice.ad9912

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
    "cpld": {
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
    "dut": {
        "type": "local",
        "module": "artiq.coredevice.ad9912",
        "class": "AD9912",
        "arguments": {
            "pll_n": 4,
            "chip_select": 5,
            "cpld_device": "cpld",
        }
    },
}


class _Environment(HasEnvironment):
    def build(self):
        self.core = self.get_device('core')
        self.dut = self.get_device('dut')


class AD9912TestCase(dax.sim.test_case.PeekTestCase):
    SEED = None

    def setUp(self) -> None:
        self.rng = random.Random(self.SEED)
        self.env = self.construct_env(_Environment, device_db=_DEVICE_DB)

    def _test_uninitialized(self):
        self.expect(self.env.dut, 'init', 'x')
        self.expect(self.env.dut, 'freq', 'x')
        self.expect(self.env.dut, 'phase', 'x')

    def test_init(self):
        self._test_uninitialized()
        self.env.dut.init()
        self.expect(self.env.dut, 'init', 1)

    def test_set_mu(self):
        self._test_uninitialized()
        for _ in range(_NUM_SAMPLES):
            f = self.rng.uniform(0 * MHz, 400 * MHz)
            p = self.rng.uniform(0.0, 0.99)
            self.env.dut.set_mu(self.env.dut.frequency_to_ftw(f), self.env.dut.turns_to_pow(p))
            self.expect_close(self.env.dut, 'freq', f, places=1)
            self.expect_close(self.env.dut, 'phase', p, places=3)

    def test_set(self):
        self._test_uninitialized()
        for _ in range(_NUM_SAMPLES):
            f = self.rng.uniform(0 * MHz, 400 * MHz)
            p = self.rng.uniform(0.0, 0.99)
            self.env.dut.set(f, phase=p)
            self.expect(self.env.dut, 'freq', f)
            self.expect(self.env.dut, 'phase', p)

    def test_set_att(self):
        signal = f'att_{self.env.dut.chip_select - 4}'
        self.expect(self.env.dut.cpld, signal, 'x')
        for _ in range(_NUM_SAMPLES):
            att = self.rng.uniform(0.0, 31.5)
            self.env.dut.set_att(att)
            self.expect(self.env.dut.cpld, signal, att)

    def test_set_att_mu(self):
        signal = f'att_{self.env.dut.chip_select - 4}'
        self.expect(self.env.dut.cpld, signal, 'x')
        self.env.dut.set_att_mu(255)
        self.expect(self.env.dut.cpld, signal, 0 * dB)

    def test_cfg_sw(self):
        ref = '0001000'
        index = self.env.dut.chip_select - 4
        for state in [0, 1]:
            self.env.dut.cfg_sw(state)
            value = ref[index:4 + index] if state else '0000'
            assert value[-1 - index] == str(state)
            self.expect(self.env.dut.cpld, 'sw', value)


class CompileTestCase(compile_testcase.CoredeviceCompileTestCase):
    DEVICE_CLASS = dax.sim.coredevice.ad9912.AD9912
    DEVICE_KWARGS = {
        'chip_select': 4,
        'cpld_device': 'cpld',
        'pll_n': 4,
    }
    FN_ARGS = {
        'set_mu': (0, 0),
        'set': (0.0, 0.0),
    }
    FN_KWARGS = {
        'frequency_to_ftw': {'frequency': 0.0},
        'ftw_to_frequency': {'ftw': 0},
        'turns_to_pow': {'phase': 0.0},
        'pow_to_turns': {'pow_': 0},
        'set_att_mu': {'att': 0},
        'set_att': {'att': 0.0},
        'cfg_sw': {'state': False},
    }
    FN_EXCLUDE = {'write', 'read'}
    DEVICE_DB = _DEVICE_DB
