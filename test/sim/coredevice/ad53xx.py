import unittest
import random

import artiq.coredevice.ad53xx  # type: ignore

import dax.sim.coredevice.ad53xx


class AD53xxTestCase(unittest.TestCase):
    NUM_SAMPLES = 20
    SEED = None

    def setUp(self) -> None:
        self.rng = random.Random(self.SEED)

    def test_conversion(self):
        for v_ref in [2.0, 5.0]:
            for _ in range(self.NUM_SAMPLES):
                v = self.rng.uniform(0.0, v_ref * 4)
                with self.subTest(v_ref=v_ref, v_in=v):
                    o = dax.sim.coredevice.ad53xx._mu_to_voltage(
                        artiq.coredevice.ad53xx.voltage_to_mu(v, vref=v_ref), vref=v_ref)
                    self.assertAlmostEqual(v, o, places=3, msg='Input voltage does not match converted output voltage')


if __name__ == '__main__':
    unittest.main()
