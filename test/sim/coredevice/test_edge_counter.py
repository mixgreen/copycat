import unittest
import random

from artiq.experiment import *

import dax.sim.test_case

from test.environment import CI_ENABLED

_NUM_SAMPLES = 1000 if CI_ENABLED else 100

_INPUT_FREQ = 200e3

_DEVICE_DB = {
    'core': {
        'type': 'local',
        'module': 'artiq.coredevice.core',
        'class': 'Core',
        'arguments': {'host': None, 'ref_period': 1e-9}
    },
    'ec': {
        "type": "local",
        "module": "artiq.coredevice.edge_counter",
        "class": "EdgeCounter",
        "arguments": {},
        "sim_args": {"input_freq": _INPUT_FREQ},
    }
}


class _Environment(HasEnvironment):
    def build(self):
        self.core = self.get_device('core')
        self.ec = self.get_device('ec')


class EdgeCounterTestCase(dax.sim.test_case.PeekTestCase):

    def setUp(self) -> None:
        self.env = self.construct_env(_Environment, device_db=_DEVICE_DB)

    def test_gate_rising_mu(self):
        for _ in range(_NUM_SAMPLES):
            duration = random.randrange(1000000, 1000000000)
            ref_time = now_mu()
            self.env.ec.gate_rising_mu(duration)

            self.assertEqual(now_mu(), ref_time + duration)
            self.assertEqual(self.env.ec.fetch_count(), int(self.env.core.mu_to_seconds(duration) * _INPUT_FREQ))

    def test_gate_falling_mu(self):
        for _ in range(_NUM_SAMPLES):
            duration = random.randrange(1000000, 1000000000)
            ref_time = now_mu()
            self.env.ec.gate_falling_mu(duration)

            self.assertEqual(now_mu(), ref_time + duration)
            self.assertEqual(self.env.ec.fetch_count(), int(self.env.core.mu_to_seconds(duration) * _INPUT_FREQ))

    def test_gate_both_mu(self):
        for _ in range(_NUM_SAMPLES):
            duration = random.randrange(1000000, 1000000000)
            ref_time = now_mu()
            self.env.ec.gate_both_mu(duration)

            self.assertEqual(now_mu(), ref_time + duration)
            self.assertEqual(self.env.ec.fetch_count(), int(self.env.core.mu_to_seconds(duration) * _INPUT_FREQ * 2))

    def test_set_config_gate_rising(self):
        for _ in range(_NUM_SAMPLES):
            duration = random.randrange(1000000, 1000000000)
            ref_time = now_mu()
            self.env.ec.set_config(True, False, False, True)  # Start gate
            delay_mu(duration)
            self.env.ec.set_config(False, False, True, False)  # Stop gate

            self.assertEqual(now_mu(), ref_time + duration)
            self.assertEqual(self.env.ec.fetch_count(), int(self.env.core.mu_to_seconds(duration) * _INPUT_FREQ))

    def test_set_config_gate_falling(self):
        for _ in range(_NUM_SAMPLES):
            duration = random.randrange(1000000, 1000000000)
            ref_time = now_mu()
            self.env.ec.set_config(False, True, False, True)  # Start gate
            delay_mu(duration)
            self.env.ec.set_config(False, False, True, False)  # Stop gate

            self.assertEqual(now_mu(), ref_time + duration)
            self.assertEqual(self.env.ec.fetch_count(), int(self.env.core.mu_to_seconds(duration) * _INPUT_FREQ))

    def test_set_config_gate_both(self):
        for _ in range(_NUM_SAMPLES):
            duration = random.randrange(1000000, 1000000000)
            ref_time = now_mu()
            self.env.ec.set_config(True, True, False, True)  # Start gate
            delay_mu(duration)
            self.env.ec.set_config(False, False, True, False)  # Stop gate

            self.assertEqual(now_mu(), ref_time + duration)
            self.assertEqual(self.env.ec.fetch_count(), int(self.env.core.mu_to_seconds(duration) * _INPUT_FREQ * 2))

    def test_fetch_timestamped_count(self):
        for _ in range(_NUM_SAMPLES):
            duration = random.randrange(1000000, 1000000000)
            self.env.ec.gate_rising_mu(duration)

            t, c = self.env.ec.fetch_timestamped_count()
            self.assertEqual(t, now_mu())
            self.assertEqual(c, int(self.env.core.mu_to_seconds(duration) * _INPUT_FREQ))

            t, _ = self.env.ec.fetch_timestamped_count()
            self.assertEqual(t, -1)


if __name__ == '__main__':
    unittest.main()
