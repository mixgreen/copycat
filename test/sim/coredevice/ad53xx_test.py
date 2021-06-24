import unittest
import random

from artiq.experiment import *
import artiq.coredevice.ad53xx  # type: ignore

import dax.sim.test_case
import dax.sim.coredevice.ad53xx

from test.environment import CI_ENABLED

_NUM_SAMPLES = 1000 if CI_ENABLED else 100


class AD53xxTestCase(unittest.TestCase):
    SEED = None

    def setUp(self) -> None:
        self.rng = random.Random(self.SEED)

    def test_conversion(self):
        for v_ref in [2.0, 5.0]:
            for _ in range(_NUM_SAMPLES):
                v = self.rng.uniform(-2.0 * v_ref, 1.99 * v_ref)  # v < 2*v_ref
                with self.subTest(v_ref=v_ref, v_in=v):
                    o = dax.sim.coredevice.ad53xx._mu_to_voltage(
                        artiq.coredevice.ad53xx.voltage_to_mu(v, vref=v_ref), vref=v_ref, offset_dacs=0x2000)
                    self.assertAlmostEqual(v, o, places=3, msg='Input voltage does not match converted output voltage')


_DEVICE_DB = {
    'core': {
        'type': 'local',
        'module': 'artiq.coredevice.core',
        'class': 'Core',
        'arguments': {'host': None, 'ref_period': 1e-9}
    },
    "dut": {
        "type": "local",
        "module": "artiq.coredevice.ad53xx",
        "class": "AD53xx"
    },
}


class _Environment(HasEnvironment):
    def build(self):
        self.core = self.get_device('core')
        self.dut = self.get_device('dut')


class AD53xxPeekTestCase(dax.sim.test_case.PeekTestCase):
    SEED = None
    _NUM_CHANNELS = 32

    DDB = _DEVICE_DB

    def setUp(self) -> None:
        self.rng = random.Random(self.SEED)
        self.env = self.construct_env(_Environment, device_db=self.DDB)

    def _test_uninitialized(self):
        self.expect(self.env.dut, 'init', 'x')
        for i in range(self._NUM_CHANNELS):
            self.expect(self.env.dut, f'v_out_{i}', 'x')
            self.expect(self.env.dut, f'v_offset_{i}', 'x')
            self.expect(self.env.dut, f'gain_{i}', 'x')

    def test_init(self):
        self._test_uninitialized()
        self.env.dut.init()
        self.expect(self.env.dut, 'init', 1)

    def test_offset_timing(self):
        for c in range(self._NUM_CHANNELS):
            self.env.dut.write_offset(c, 0.0)
            self.env.dut.write_offset_mu(c, 0)
        self._test_uninitialized()

    def test_offset(self):
        self._test_uninitialized()
        for _ in range(_NUM_SAMPLES):
            v = self.rng.uniform(-2 * self.env.dut.vref, 2 * self.env.dut.vref)
            c = self.rng.randrange(self._NUM_CHANNELS)
            self.env.dut.write_offset(c, v)
            self.env.dut.load()
            self.expect(self.env.dut, f'v_offset_{c}', v)

    def test_offset_mu(self):
        self._test_uninitialized()
        for _ in range(_NUM_SAMPLES):
            v = self.rng.randrange(2 ** 16)
            c = self.rng.randrange(self._NUM_CHANNELS)
            self.env.dut.write_offset_mu(c, v)
            self.env.dut.load()
            self.expect(self.env.dut, f'v_offset_{c}',
                        dax.sim.coredevice.ad53xx._mu_to_voltage(v - 2 ** 15, vref=self.env.dut.vref))

    def test_gain_timing(self):
        for c in range(self._NUM_CHANNELS):
            self.env.dut.write_gain_mu(c)
        self._test_uninitialized()

    def test_gain(self):
        self._test_uninitialized()
        for _ in range(_NUM_SAMPLES):
            g = self.rng.randrange(0xFFFF)
            c = self.rng.randrange(self._NUM_CHANNELS)
            self.env.dut.write_gain_mu(c, g)
            self.env.dut.load()
            self.expect_close(self.env.dut, f'gain_{c}', (g + 1) / 2 ** 16, places=7)

    def test_v_out(self):
        self._test_uninitialized()
        for _ in range(_NUM_SAMPLES):
            # Generate parameters
            v = self.rng.uniform(0 * V, 3.9 * self.env.dut.vref)  # Multiplier < 4.0 to prevent overflows
            o = self.rng.randrange(2 ** 14)
            c = self.rng.randrange(self._NUM_CHANNELS)
            # Adjust voltage to make sure it is in range
            v += dax.sim.coredevice.ad53xx._mu_to_voltage(0 * V, vref=self.env.dut.vref, offset_dacs=o)
            with self.subTest(v=v, o=o):
                # Call functions
                self.env.dut.write_offset_dacs_mu(o)
                self.env.dut.set_dac([v], [c])
                # Test
                self.expect_close(self.env.dut, f'v_out_{c}', v, places=3)

    def test_write_dac_mu_timing(self):
        for c in range(self._NUM_CHANNELS):
            self.env.dut.write_dac_mu(c, 0)
            self.env.dut.write_dac(c, 0 * V)
        self._test_uninitialized()

    def test_offset_dacs_timing(self):
        self._test_uninitialized()
        self.env.dut.write_offset_dacs_mu(0)  # Should apply immediately
        for i in range(self._NUM_CHANNELS):
            self.expect_close(self.env.dut, f'v_out_{i}', 0.0, places=7)
