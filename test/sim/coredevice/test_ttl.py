import unittest
import random
import numpy as np

from artiq.experiment import *

import dax.sim.test_case
import dax.sim.coredevice.urukul

from test.environment import CI_ENABLED

_NUM_SAMPLES = 1000 if CI_ENABLED else 100

_INPUT_FREQ = 1 * kHz

_DEVICE_DB = {
    'core': {
        'type': 'local',
        'module': 'artiq.coredevice.core',
        'class': 'Core',
        'arguments': {'host': None, 'ref_period': 1e-9}
    },
    "TTLOut": {
        "type": "local",
        "module": "artiq.coredevice.ttl",
        "class": "TTLOut",
        "arguments": {}
    },
    "TTLInOut": {
        "type": "local",
        "module": "artiq.coredevice.ttl",
        "class": "TTLInOut",
        "arguments": {'input_freq': _INPUT_FREQ}
    },
    "TTLClockGen": {
        "type": "local",
        "module": "artiq.coredevice.ttl",
        "class": "TTLClockGen",
        "arguments": {}
    },
}


class _Environment(HasEnvironment):
    def build(self, *, dut):
        self.core = self.get_device('core')
        self.dut = self.get_device(dut)


class _BaseTTLTestCase(dax.sim.test_case.PeekTestCase):
    SEED = None
    DUT: str

    def setUp(self) -> None:
        assert isinstance(self.DUT, str)
        self.rng = random.Random(self.SEED)
        self.env = self.construct_env(_Environment, device_db=_DEVICE_DB, build_kwargs={'dut': self.DUT})


class TTLOutTestCase(_BaseTTLTestCase):
    DUT = 'TTLOut'

    def test_set_o(self):
        self.expect(self.env.dut, 'state', 'x')
        for o in [True, False, 0, 1, np.int32(0), np.int32(1), np.int64(0), np.int64(1)]:
            self.env.dut.set_o(o)
            self.expect(self.env.dut, 'state', 1 if o else 0)

    def test_on_off(self):
        self.expect(self.env.dut, 'state', 'x')
        self.env.dut.on()
        self.expect(self.env.dut, 'state', True)
        self.env.dut.off()
        self.expect(self.env.dut, 'state', False)

    def test_pulse(self):
        self.expect(self.env.dut, 'state', 'x')
        for _ in range(_NUM_SAMPLES):
            self.env.core.reset()
            duration = self.rng.uniform(0.5, 1.0)
            with parallel:
                with sequential:
                    self.env.dut.pulse(duration)
                    self.expect(self.env.dut, 'state', False)
                with sequential:
                    self.expect(self.env.dut, 'state', True)
                    delay(duration)
                    delay_mu(-1)
                    self.expect(self.env.dut, 'state', True)

    def test_pulse_mu(self):
        self.expect(self.env.dut, 'state', 'x')
        for _ in range(_NUM_SAMPLES):
            self.env.core.reset()
            duration = self.rng.randrange(1, 10000)
            with parallel:
                with sequential:
                    self.env.dut.pulse_mu(duration)
                    self.expect(self.env.dut, 'state', False)
                with sequential:
                    self.expect(self.env.dut, 'state', True)
                    delay_mu(duration - 1)
                    self.expect(self.env.dut, 'state', True)


class TTLInOutTestCase(TTLOutTestCase):
    DUT = 'TTLInOut'

    def test_input(self):
        self.expect(self.env.dut, 'direction', 'x')
        self.env.dut.input()
        self.expect(self.env.dut, 'direction', 0)
        self.expect(self.env.dut, 'sensitivity', 0)
        self.expect(self.env.dut, 'state', 'z')

    def test_output(self):
        self.expect(self.env.dut, 'direction', 'x')
        self.env.dut.output()
        self.expect(self.env.dut, 'direction', 1)
        self.expect(self.env.dut, 'sensitivity', 'z')
        self.expect(self.env.dut, 'state', 'x')
        self.env.dut.on()
        self.expect(self.env.dut, 'state', True)
        self.env.dut.output()
        self.expect(self.env.dut, 'state', 'x')

    def test_count(self):
        cases = [
            (self.env.dut.gate_rising, 1),
            (self.env.dut.gate_falling, 1),
            (self.env.dut.gate_both, 2),
        ]
        self.env.dut.input()

        for gate_fn, multiplier in cases:
            for _ in range(_NUM_SAMPLES):
                duration = self.rng.uniform(1 * ms, 10 * ms)

                with parallel:
                    with sequential:
                        self.expect(self.env.dut, 'sensitivity', 0)
                        t = gate_fn(duration)
                        self.expect(self.env.dut, 'sensitivity', 0)
                    with sequential:
                        delay_mu(1)
                        self.expect(self.env.dut, 'sensitivity', 1)
                    with sequential:
                        delay(duration)
                        delay_mu(-1)
                        self.expect(self.env.dut, 'sensitivity', 1)

                c = self.env.dut.count(t)
                self.assertAlmostEqual(c, _INPUT_FREQ * multiplier * duration, delta=1)

    def test_timestamp_mu(self):
        self.env.dut.input()
        self.env.core.reset()
        start_t = now_mu()
        for _ in range(_NUM_SAMPLES):
            duration = self.rng.randrange(1000000, 10000000)
            t = self.env.dut.gate_both_mu(duration)
            end_t = now_mu()

            # This calculation needs to be exactly equal to the simulated one for proper testing
            num_events = int(_INPUT_FREQ * 2 * self.env.core.mu_to_seconds(duration))

            for _ in range(num_events):
                r = self.env.dut.timestamp_mu(t)
                self.assertGreaterEqual(r, start_t)
                self.assertLessEqual(r, end_t)
            self.assertEqual(self.env.dut.timestamp_mu(t), -1)

    def test_sample_get(self):
        self.env.dut.input()
        self.env.core.break_realtime()
        for _ in range(_NUM_SAMPLES):
            n = self.rng.randrange(1, 10)
            for _ in range(n):
                self.env.dut.sample_input()
            for _ in range(n):
                self.assertIn(self.env.dut.sample_get(), {0, 1})
            with self.assertRaises(IndexError, msg='Obtaining more samples than number of samples did not raise'):
                self.env.dut.sample_get()

    def test_sample_get_nonrt(self):
        self.env.dut.input()
        self.env.core.break_realtime()
        for _ in range(_NUM_SAMPLES):
            s = self.env.dut.sample_get_nonrt()
            self.assertIn(s, {0, 1})

    class TTLClockGenTestCase(_BaseTTLTestCase):
        DUT = 'TTLClockGen'

        def test_conversion(self):
            for _ in range(_NUM_SAMPLES):
                ftw = self.rng.uniform(0.0, 1 * GHz)
                o = self.env.dut.frequency_to_ftw(self.env.dut.ftw_to_frequency(ftw))
                self.assertAlmostEqual(ftw, o, delta=1 * Hz)

        def test_set(self):
            for _ in range(_NUM_SAMPLES):
                f = self.rng.uniform(0.0, 1 * GHz)
                self.env.dut.set(f)
                self.expect(self.env.dut, 'freq', f)

        def test_set_mu(self):
            for _ in range(_NUM_SAMPLES):
                f = self.rng.uniform(0.0, 1 * GHz)
                self.env.dut.set_mu(self.env.dut.frequency_to_ftw(f))
                self.expect(self.env.dut, 'freq', f, places=-1)

        def test_stop(self):
            self.env.dut.stop()
            self.expect(self.env.dut, 'freq', 0 * Hz)

    if __name__ == '__main__':
        unittest.main()
