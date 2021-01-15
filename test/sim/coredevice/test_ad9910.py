import unittest
import random

from artiq.experiment import *

import dax.sim.test_case
import dax.sim.coredevice.ad9910

from test.environment import CI_ENABLED

_NUM_SAMPLES = 1000 if CI_ENABLED else 100

_DEVICE_DB = {
    'core': {
        'type': 'local',
        'module': 'artiq.coredevice.core',
        'class': 'Core',
        'arguments': {'host': '1.2.3.4', 'ref_period': 1e-9}
    },
    "dut": {
        "type": "local",
        "module": "artiq.coredevice.ad9910",
        "class": "AD9910",
        "arguments": {
            "pll_en": 0,
            "chip_select": 6,
            "cpld_device": "cpld10",
        }
    },
    "cpld10": {
        "type": "local",
        "module": "artiq.coredevice.urukul",
        "class": "CPLD",
        "arguments": {
            "refclk": 1e9,
            "clk_div": 1
        }
    },
}


class _Environment(HasEnvironment):
    def build(self):
        self.core = self.get_device('core')
        self.dut = self.get_device('dut')


class AD9910TestCase(dax.sim.test_case.PeekTestCase):
    SEED = None

    def setUp(self) -> None:
        self.rng = random.Random(self.SEED)
        self.env = self.construct_env(_Environment, device_db=_DEVICE_DB)

    def _test_uninitialized(self):
        self.expect(self.env.dut, 'init', 'x')
        self.expect(self.env.dut, 'freq', 'x')
        self.expect(self.env.dut, 'phase', 'x')
        self.expect(self.env.dut, 'phase_mode', 'x')
        self.expect(self.env.dut, 'amp', 'x')

    def test_init(self):
        self._test_uninitialized()
        self.env.dut.init()
        self.expect(self.env.dut, 'init', 1)

    def test_phase_mode_timing(self):
        self._test_uninitialized()
        self.env.dut.set_phase_mode(dax.sim.coredevice.ad9910.PHASE_MODE_ABSOLUTE)
        self._test_uninitialized()
        self.env.dut.set(100 * MHz)
        self.expect(self.env.dut, 'phase_mode', '01')

    def test_default_phase_mode_timing(self):
        self._test_uninitialized()
        self.env.dut.set_mu(2 ** 30)
        self.expect(self.env.dut, 'phase_mode', '00')

    def test_set_mu(self):
        self._test_uninitialized()
        for _ in range(_NUM_SAMPLES):
            f = self.rng.uniform(0 * MHz, 400 * MHz)
            p = self.rng.randrange(2 ** 16)
            a = self.rng.randrange(2 ** 14)
            self.env.dut.set_mu(self.env.dut.frequency_to_ftw(f), pow_=p, asf=a)
            self.expect(self.env.dut, 'freq', f, places=0)
            self.expect(self.env.dut, 'phase', self.env.dut.pow_to_turns(p))
            self.expect(self.env.dut, 'amp', self.env.dut.asf_to_amplitude(a))

    def test_set_mu_separate(self):
        self._test_uninitialized()
        for _ in range(_NUM_SAMPLES):
            f = self.rng.uniform(0 * MHz, 400 * MHz)
            p = self.rng.randrange(2 ** 16)
            a = self.rng.randrange(2 ** 14)
            self.env.dut.set_ftw(self.env.dut.frequency_to_ftw(f))
            self.env.dut.set_pow(p)
            self.env.dut.set_asf(a)
            self.expect(self.env.dut, 'freq', f, places=0)
            self.expect(self.env.dut, 'phase', self.env.dut.pow_to_turns(p))
            self.expect(self.env.dut, 'amp', self.env.dut.asf_to_amplitude(a))

        # Phase mode is not set
        self.expect(self.env.dut, 'phase_mode', 'x')

    def test_set(self):
        self._test_uninitialized()
        for _ in range(_NUM_SAMPLES):
            f = self.rng.uniform(0 * MHz, 400 * MHz)
            p = self.rng.uniform(0.0, self.env.dut.pow_to_turns(2 ** 16))
            a = self.rng.uniform(0.0, 1.0)
            self.env.dut.set(f, phase=p, amplitude=a)
            self.expect(self.env.dut, 'freq', f)
            self.expect(self.env.dut, 'phase', p)
            self.expect(self.env.dut, 'amp', a)

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


if __name__ == '__main__':
    unittest.main()
