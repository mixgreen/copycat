import unittest
import random

from artiq.language.core import now_mu, at_mu, delay, delay_mu, parallel, sequential, set_time_manager
from artiq.language.core import watchdog
from artiq.coredevice.exceptions import WatchdogExpired
from artiq.language.units import *

from dax.sim.time import DaxTimeManager

from test.environment import CI_ENABLED

_NUM_SAMPLES = 1000 if CI_ENABLED else 100


class TimeManagerTestCase(unittest.TestCase):
    # The reference period for this class
    REF_PERIOD = us
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

        for i in range(_NUM_SAMPLES):
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

        for i in range(_NUM_SAMPLES):
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

        for i in range(_NUM_SAMPLES):
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

    def test_shallow_parallel(self):
        # Shallow parallel matches the timing model of ARTIQ, see https://github.com/m-labs/artiq/issues/1555

        range_stop = 1000000000

        for i in range(_NUM_SAMPLES):
            with self.subTest('parallel sub test', i=i):
                # Reference time
                ref_time = now_mu()

                # Block duration
                block_duration = 0
                with parallel:
                    # Random duration
                    duration = self.rnd.randrange(range_stop)
                    # Delay
                    delay_mu(duration)
                    # Update block duration
                    block_duration = max(duration, block_duration)

                    # Manually repeat test code to keep them top-level statements in the shallow parallel context
                    duration = self.rnd.randrange(range_stop)
                    delay_mu(duration)
                    block_duration = max(duration, block_duration)
                    duration = self.rnd.randrange(range_stop)
                    delay_mu(duration)
                    block_duration = max(duration, block_duration)
                    duration = self.rnd.randrange(range_stop)
                    delay_mu(duration)
                    block_duration = max(duration, block_duration)

                # Update ref time
                ref_time += block_duration
                # Compare to reference time
                self.assertEqual(now_mu(), ref_time, 'Reference does not match now_mu()')

    def test_explicit_shallow_parallel(self):
        # Shallow parallel matches the timing model of ARTIQ, see https://github.com/m-labs/artiq/issues/1555

        for i in range(_NUM_SAMPLES):
            with self.subTest('parallel sub test', i=i):
                # Reference time
                ref_time = now_mu()

                # Block duration
                block_duration = 0
                # Single random duration
                duration = self.rnd.randrange(1000000000)

                with parallel:
                    # Delay
                    delay_mu(duration)
                    # Update block duration
                    block_duration = max(duration, block_duration)  # Top-level statement, so parallel

                    if True:
                        # We are not a top level statement anymore, so timing is sequential
                        with sequential:  # Add an explicit sequential context!
                            sequential_duration = 0
                            for _ in range(10):
                                delay_mu(duration)
                                sequential_duration += duration

                            # Parallelize the sequential block with other parallel top-level statements
                            block_duration = max(sequential_duration, block_duration)

                # Update ref time
                ref_time += block_duration
                # Compare to reference time
                self.assertEqual(now_mu(), ref_time, 'Reference does not match now_mu()')

    def test_manual_deep_parallel(self):
        # Manual deep parallel matches the timing model of ARTIQ, see https://github.com/m-labs/artiq/issues/1555

        for i in range(_NUM_SAMPLES):
            with self.subTest('parallel sub test', i=i):
                # Reference time
                ref_time = now_mu()

                # Block duration
                block_duration = 0
                # Single random duration
                duration = self.rnd.randrange(1000000000)

                for _ in range(10):
                    at_mu(ref_time)  # Manually revert timeline for deep parallel emulation
                    # Delay
                    delay_mu(duration)
                    # Update block duration
                    block_duration = max(duration, block_duration)

                # Update ref time
                ref_time += block_duration
                # Compare to reference time
                self.assertEqual(now_mu(), ref_time, 'Reference does not match now_mu()')

    @unittest.expectedFailure
    def test_not_deep_parallel(self):
        # Deep parallel does not match the timing model of ARTIQ, see https://github.com/m-labs/artiq/issues/1555
        # But DAX.sim implements deep parallel! The parallel semantics of ARTIQ and DAX.sim differ here!

        for i in range(_NUM_SAMPLES):
            with self.subTest('parallel sub test', i=i):
                # Reference time
                ref_time = now_mu()

                # Block duration
                block_duration = 0
                with parallel:
                    for _ in range(10):  # Only the for-loop is a top-level statement, not the content inside the loop!
                        # with sequential:  # The implicit sequential context will be added here
                        # Random duration
                        duration = self.rnd.randrange(1000000000)
                        # Delay
                        delay_mu(duration)
                        # Update block duration
                        block_duration += duration  # Add duration to block duration, implicit sequential

                # Update ref time
                ref_time += block_duration
                # Compare to reference time
                self.assertEqual(now_mu(), ref_time, 'Reference does not match now_mu()')

    def test_deep_parallel(self):
        # Deep parallel does not match the timing model of ARTIQ, see https://github.com/m-labs/artiq/issues/1555
        # But DAX.sim implements deep parallel! The parallel semantics of ARTIQ and DAX.sim differ here!

        for i in range(_NUM_SAMPLES):
            with self.subTest('parallel sub test', i=i):
                # Reference time
                ref_time = now_mu()

                # Block duration
                block_duration = 0
                with parallel:
                    for _ in range(10):  # Deep parallel propagates beyond top-level statements
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
        for i in range(_NUM_SAMPLES):
            with self.subTest('at_mu() sub test', i=i):
                # Random time
                ref_time = self.rnd.randrange(1000000000)
                # Set time
                at_mu(ref_time)

                # Compare time
                self.assertEqual(now_mu(), ref_time, 'Reference does not match now_mu()')


class TimeManagerTestCase1ns(TimeManagerTestCase):
    REF_PERIOD = ns


class TimeManagerTestCase5ns(TimeManagerTestCase):
    REF_PERIOD = 5 * ns


class WatchdogTestCase(unittest.TestCase):
    # The reference period for this class
    REF_PERIOD = ns
    # Seed for random samples
    SEED = None
    # Window size
    WINDOW = 5 * us

    def setUp(self) -> None:
        assert isinstance(self.REF_PERIOD, float)
        assert self.REF_PERIOD > 0.0
        set_time_manager(DaxTimeManager(self.REF_PERIOD))
        self.rnd = random.Random(self.SEED)

    def test_watchdog_in_time(self):
        with watchdog(self.WINDOW):
            t = now_mu()
            for _ in range(_NUM_SAMPLES):
                # All delays are less than the window, so no exceptions should be raised
                at_mu(t)
                delay(self.WINDOW * self.rnd.random())

        for _ in range(_NUM_SAMPLES):
            with watchdog(self.WINDOW):
                # All delays are less than the window, so no exceptions should be raised
                delay(self.WINDOW * self.rnd.random())

        with watchdog(self.WINDOW):
            # Exact same time does not raise
            delay(self.WINDOW)

    def test_watchdog_expired(self):
        with watchdog(self.WINDOW):
            with self.assertRaises(WatchdogExpired, msg='Watchdog timeout did not raise'):
                # More than a window, so exception should be raised
                delay(self.WINDOW * (self.rnd.random() + 2))

        with watchdog(self.WINDOW):
            with self.assertRaises(WatchdogExpired, msg='Watchdog timeout did not raise'):
                # More than a window, so exception should be raised
                delay(self.WINDOW + self.REF_PERIOD)

    def test_nested_watchdog_in_time(self):
        with watchdog(self.WINDOW):
            with watchdog(self.WINDOW):
                with watchdog(self.WINDOW):
                    t = now_mu()
                    for _ in range(_NUM_SAMPLES):
                        # All delays are less than the window, so no exceptions should be raised
                        at_mu(t)
                        delay(self.WINDOW * self.rnd.random())

        for _ in range(_NUM_SAMPLES):
            with watchdog(self.WINDOW):
                with watchdog(self.WINDOW):
                    with watchdog(self.WINDOW):
                        # All delays are less than the window, so no exceptions should be raised
                        delay(self.WINDOW * self.rnd.random())

        with watchdog(self.WINDOW):
            with watchdog(self.WINDOW):
                with watchdog(self.WINDOW):
                    # Exact same time does not raise
                    delay(self.WINDOW)

    def test_nested_watchdog_expired(self):
        with watchdog(self.WINDOW):
            with watchdog(self.WINDOW):
                with watchdog(self.WINDOW):
                    with self.assertRaises(WatchdogExpired, msg='Watchdog timeout in inner nested watchdog context '
                                                                'did not raise'):
                        # More than a window, so exception should be raised
                        delay(self.WINDOW * (self.rnd.random() + 2))

        with self.assertRaises(WatchdogExpired, msg='Watchdog timeout in outer nested watchdog context did not raise'):
            with watchdog(self.WINDOW):
                with watchdog(self.WINDOW):
                    with watchdog(self.WINDOW):
                        # More than a window, so exception should be raised
                        delay(self.WINDOW * (self.rnd.random() + 2))

        with watchdog(self.WINDOW):
            with watchdog(self.WINDOW):
                with watchdog(self.WINDOW):
                    with self.assertRaises(WatchdogExpired, msg='Watchdog timeout in inner nested watchdog context '
                                                                'did not raise'):
                        # More than a window, so exception should be raised
                        delay(self.WINDOW + self.REF_PERIOD)

        with self.assertRaises(WatchdogExpired, msg='Watchdog timeout in outer nested watchdog context did not raise'):
            with watchdog(self.WINDOW):
                with watchdog(self.WINDOW):
                    with watchdog(self.WINDOW):
                        # More than a window, so exception should be raised
                        delay(self.WINDOW + self.REF_PERIOD)
