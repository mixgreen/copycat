import unittest

from artiq.language.units import *


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


class UnitsFormatterTestCase(unittest.TestCase):
    def setUp(self) -> None:
        from dax.util.units import UnitsFormatter
        self.f = UnitsFormatter()

    def test_time_format(self):
        test_data = [
            ('no formatting', [], {}, 'no formatting'),
            ('aa{}', [1], {}, 'aa1'),
            ('{!t}', [3 * ns], {}, '3.000000 ns'),
            ('{!t}', [3000 * us], {}, '3.000000 ms'),
            ('{!t}', [5.33333333333 * us], {}, '5.333333 us'),
            ('{a!t}-{b!t}', [], dict(a=5.33333333333 * us, b=4.5000066 * ns), '5.333333 us-4.500007 ns'),
        ]

        for fstring, args, kwargs, ref in test_data:
            with self.subTest(input=fstring):
                self.assertEqual(self.f.format(fstring, *args, **kwargs), ref)

    def test_freq_format(self):
        test_data = [
            ('no formatting', [], {}, 'no formatting'),
            ('aa{}', [1], {}, 'aa1'),
            ('{!f}', [3 * MHz], {}, '3.000000 MHz'),
            ('{!f}', [3000 * MHz], {}, '3.000000 GHz'),
            ('{!f}', [5.333333333333 * kHz], {}, '5.333333 kHz'),
            ('{!f}-{!f}', [5.33333333333 * kHz, 4.5000066 * MHz], {}, '5.333333 kHz-4.500007 MHz'),
            ('{a!f}-{b!f}', [], dict(a=5.3333333 * kHz, b=4.5000066 * MHz), '5.333333 kHz-4.500007 MHz'),
        ]

        for fstring, args, kwargs, ref in test_data:
            with self.subTest(input=fstring):
                self.assertEqual(self.f.format(fstring, *args, **kwargs), ref)


if __name__ == '__main__':
    unittest.main()
