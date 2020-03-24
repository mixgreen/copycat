import unittest

from artiq.language.units import *

from dax.test.helpers.mypy import type_check


class UnitsStaticTyping(unittest.TestCase):

    def test_static_typing(self):
        # Type checking on DAX base
        import dax.util.units as module
        type_check(self, module, '--strict')


class UnitsTestCase(unittest.TestCase):

    def test_time_to_str(self):
        from dax.util.units import time_to_str
        self.assertEqual(time_to_str(1 * ns, precision=2), '1.00 ns')
        self.assertEqual(time_to_str(1 * ns, precision=0), '1 ns')
        self.assertEqual(time_to_str(9.99 * s, precision=2), '9.99 s')
        self.assertEqual(time_to_str(2300 * ns, precision=5), '2.30000 us')
        self.assertEqual(time_to_str(.97 * ns, precision=1), '970.0 ps')
        self.assertEqual(time_to_str(.64 * us, threshold=600.0, precision=0), '640 ns')

    def test_freq_to_str(self):
        from dax.util.units import freq_to_str
        self.assertEqual(freq_to_str(10 * GHz, precision=3), '10.000 GHz')
        self.assertEqual(freq_to_str(0.010 * GHz, precision=0), '10 MHz')
        self.assertEqual(freq_to_str(4 * MHz, threshold=1000.0, precision=1), '4000.0 kHz')

    def test_str_to_time(self):
        from dax.util.units import str_to_time
        self.assertEqual(str_to_time('10 ns'), 10 * ns)
        self.assertEqual(str_to_time('.10 s'), .1 * s)
        self.assertEqual(str_to_time('00050.001 ns'), 50.001 * ns)
        self.assertRaises(ValueError, str_to_time, 'foo')
        self.assertRaises(ValueError, str_to_time, '5ns')
        self.assertRaises(ValueError, str_to_time, '4.6 Us')

    def test_str_to_freq(self):
        from dax.util.units import str_to_freq
        self.assertEqual(str_to_freq('10 kHz'), 10 * kHz)
        self.assertEqual(str_to_freq('.10 Hz'), .1 * Hz)
        self.assertEqual(str_to_freq('00050.001 mHz'), 50.001 * mHz)
        self.assertRaises(ValueError, str_to_freq, 'bar')
        self.assertRaises(ValueError, str_to_freq, '5mHz')
        self.assertRaises(ValueError, str_to_freq, '4.6 khz')


if __name__ == '__main__':
    unittest.main()
