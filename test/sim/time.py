import unittest
import random

from artiq.language.core import now_mu, at_mu, delay, delay_mu, parallel, sequential, set_time_manager
from artiq.language.units import *

from dax.sim.time import DaxTimeManager


class TimeManagerTestCase(unittest.TestCase):
    # The reference period for this class
    REF_PERIOD = us
    # Number of samples to test
    NUM_SAMPLES = 100
    # Seed for random samples
    SEED = None

    def setUp(self) -> None:
        assert isinstance(self.REF_PERIOD, float)
        assert self.REF_PERIOD > 0.0
        set_time_manager(DaxTimeManager(self.REF_PERIOD))
        self.rnd = random.Random(self.SEED)

    def test_delay_mu(self):
        # Reference time
        ref_time = now_mu()

        for i in range(self.NUM_SAMPLES):
            # Random duration
            duration = self.rnd.randrange(1000000000)

            with self.subTest('delay_mu() sub test', i=i, duration=duration):
                # Insert the delay
                delay_mu(duration)
                # Compare to reference time
                ref_time += duration
                self.assertEqual(now_mu(), ref_time, 'Reference does not match now_mu()')

    def test_delay(self):
        # Reference time
        ref_time = now_mu()

        for i in range(self.NUM_SAMPLES):
            # Random duration
            duration = self.rnd.random()

            with self.subTest('delay() sub test', i=i, duration=duration):
                # Insert the delay
                delay(duration)
                # Compare to reference time
                ref_time += duration // self.REF_PERIOD  # floor div
                self.assertEqual(now_mu(), ref_time, 'Reference does not match now_mu()')

    def test_sequential(self):
        # Reference time
        ref_time = now_mu()

        for i in range(self.NUM_SAMPLES):
            with self.subTest('sequential sub test', i=i):
                with sequential:
                    for _ in range(10):
                        # Random duration
                        duration = self.rnd.randrange(1000000000)
                        # Delay
                        delay_mu(duration)
                        # Update ref time
                        ref_time += duration

                # Compare to reference time
                self.assertEqual(now_mu(), ref_time, 'Reference does not match now_mu()')

    def test_parallel(self):
        # Reference time
        ref_time = now_mu()

        for i in range(self.NUM_SAMPLES):
            with self.subTest('parallel sub test', i=i):
                # Block duration
                block_duration = 0
                with parallel:
                    for _ in range(10):
                        # Random duration
                        duration = self.rnd.randrange(1000000000)
                        # Delay
                        delay_mu(duration)
                        # Update block duration
                        block_duration = max(duration, block_duration)

                # Update ref time
                ref_time += block_duration
                # Compare to reference time
                self.assertEqual(now_mu(), ref_time, 'Reference does not match now_mu()')

    def test_at_mu(self):
        for i in range(self.NUM_SAMPLES):
            with self.subTest('at_mu() sub test', i=i):
                # Random time
                ref_time = self.rnd.randrange(1000000000)
                # Set time
                at_mu(ref_time)

                # Compare time
                self.assertEqual(now_mu(), ref_time, 'Reference does not match now_mu()')
