import unittest
import numpy as np

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

        self.assertEqual(time_to_str(int(10), precision=2), '10.00 s')
        self.assertEqual(time_to_str(np.int32(10), precision=2), '10.00 s')
        self.assertEqual(time_to_str(np.int64(10), precision=2), '10.00 s')
        self.assertEqual(time_to_str(np.float_(.64 * us), threshold=600.0, precision=0), '640 ns')

    def test_freq_to_str(self):
        from dax.util.units import freq_to_str
        self.assertEqual(freq_to_str(10 * GHz, precision=3), '10.000 GHz')
        self.assertEqual(freq_to_str(0.010 * GHz, precision=0), '10 MHz')
        self.assertEqual(freq_to_str(4 * MHz, threshold=1000.0, precision=1), '4000.0 kHz')

        self.assertEqual(freq_to_str(int(10 * GHz), precision=3), '10.000 GHz')
        self.assertEqual(freq_to_str(np.int32(10 * Hz), precision=3), '10.000 Hz')  # GHz overflows 32 bit value
        self.assertEqual(freq_to_str(np.int64(10 * GHz), precision=3), '10.000 GHz')
        self.assertEqual(freq_to_str(np.float_(0.010 * GHz), precision=0), '10 MHz')

    def test_volt_to_str(self):
        from dax.util.units import volt_to_str
        self.assertEqual(volt_to_str(10 * mV, precision=3), '10.000 mV')
        self.assertEqual(volt_to_str(0.010 * kV, precision=0), '10 V')
        self.assertEqual(volt_to_str(1234567.8 * uV, precision=7), '1.2345678 V')
        self.assertEqual(volt_to_str(4 * V, threshold=1000.0, precision=1), '4000.0 mV')

    def test_ampere_to_str(self):
        from dax.util.units import ampere_to_str
        self.assertEqual(ampere_to_str(10 * mA, precision=3), '10.000 mA')
        self.assertEqual(ampere_to_str(0.010 * mA, precision=0), '10 uA')
        self.assertEqual(ampere_to_str(1234567 * uA, precision=7), '1.2345670 A')
        self.assertEqual(ampere_to_str(4 * A, threshold=1000.0, precision=1), '4000.0 mA')

    def test_watt_to_str(self):
        from dax.util.units import watt_to_str
        self.assertEqual(watt_to_str(46 * mW, precision=3), '46.000 mW')
        self.assertEqual(watt_to_str(0.089 * mW, precision=0), '89 uW')
        self.assertEqual(watt_to_str(0.089 * uW, precision=0), '89 nW')
        self.assertEqual(watt_to_str(1200567.8 * uW, precision=7), '1.2005678 W')
        self.assertEqual(watt_to_str(8 * W, threshold=1000.0, precision=1), '8000.0 mW')

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

    def test_str_to_volt(self):
        from dax.util.units import str_to_volt
        self.assertEqual(str_to_volt('10 kV'), 10 * kV)
        self.assertEqual(str_to_volt('.10 V'), .1 * V)
        self.assertEqual(str_to_volt('00050.001 mV'), 50.001 * mV)
        self.assertRaises(ValueError, str_to_volt, 'bar')
        self.assertRaises(ValueError, str_to_volt, '5V')
        self.assertRaises(ValueError, str_to_volt, '4.6 MV')

    def test_str_to_ampere(self):
        from dax.util.units import str_to_ampere
        self.assertEqual(str_to_ampere('10 uA'), 10 * uA)
        self.assertEqual(str_to_ampere('.10 A'), .1 * A)
        self.assertEqual(str_to_ampere('00050.001 mA'), 50.001 * mA)
        self.assertRaises(ValueError, str_to_ampere, 'foo')
        self.assertRaises(ValueError, str_to_ampere, '5mA')
        self.assertRaises(ValueError, str_to_ampere, '4.6 UV')

    def test_str_to_watt(self):
        from dax.util.units import str_to_watt
        self.assertEqual(str_to_watt('10 nW'), 10 * nW)
        self.assertEqual(str_to_watt('10 uW'), 10 * uW)
        self.assertEqual(str_to_watt('.10 W'), .1 * W)
        self.assertEqual(str_to_watt('00050.001 mW'), 50.001 * mW)
        self.assertRaises(ValueError, str_to_watt, 'foo')
        self.assertRaises(ValueError, str_to_watt, '5mW')
        self.assertRaises(ValueError, str_to_watt, '4.6 UW')


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

    def test_volt_format(self):
        test_data = [
            ('no formatting', [], {}, 'no formatting'),
            ('aa{}', [1], {}, 'aa1'),
            ('{!v}', [3 * kV], {}, '3.000000 kV'),
            ('{!v}', [3000 * mV], {}, '3.000000 V'),
            ('{!v}', [5.333333333333 * uV], {}, '5.333333 uV'),
            ('{!v}-{!v}', [5.33333333333 * kV, 4.5000066 * mV], {}, '5.333333 kV-4.500007 mV'),
            ('{a!v}-{b!v}', [], dict(a=5.3333333 * kV, b=4.5000066 * uV), '5.333333 kV-4.500007 uV'),
        ]

        for fstring, args, kwargs, ref in test_data:
            with self.subTest(input=fstring):
                self.assertEqual(self.f.format(fstring, *args, **kwargs), ref)

    def test_ampere_format(self):
        test_data = [
            ('no formatting', [], {}, 'no formatting'),
            ('aa{}', [1], {}, 'aa1'),
            ('{!a}', [3 * A], {}, '3.000000 A'),
            ('{!a}', [3000 * mA], {}, '3.000000 A'),
            ('{!a}', [5.333333333333 * uA], {}, '5.333333 uA'),
            ('{!a}-{!a}', [5.33333333333 * A, 4.5000066 * mA], {}, '5.333333 A-4.500007 mA'),
            ('{a!a}-{b!a}', [], dict(a=5.3333333 * mA, b=4.5000066 * uA), '5.333333 mA-4.500007 uA'),
        ]

        for fstring, args, kwargs, ref in test_data:
            with self.subTest(input=fstring):
                self.assertEqual(self.f.format(fstring, *args, **kwargs), ref)

    def test_watt_format(self):
        test_data = [
            ('no formatting', [], {}, 'no formatting'),
            ('aa{}', [1], {}, 'aa1'),
            ('{!w}', [3 * W], {}, '3.000000 W'),
            ('{!w}', [3000 * mW], {}, '3.000000 W'),
            ('{!w}', [5.333333333333 * uW], {}, '5.333333 uW'),
            ('{!w}-{!w}', [5.33333333333 * W, 4.5000066 * mW], {}, '5.333333 W-4.500007 mW'),
            ('{a!w}-{b!w}', [], dict(a=5.3333333 * mW, b=4.5000066 * uW), '5.333333 mW-4.500007 uW'),
        ]

        for fstring, args, kwargs, ref in test_data:
            with self.subTest(input=fstring):
                self.assertEqual(self.f.format(fstring, *args, **kwargs), ref)

    def test_other_format(self):
        test_data = [
            ('no formatting', [], {}, 'no formatting'),
            ('aa{}', [1], {}, 'aa1'),
            ('{!r}', ['foo'], {}, repr('foo')),
            ('{!s}', [30], {}, str(30)),
        ]

        for fstring, args, kwargs, ref in test_data:
            with self.subTest(input=fstring):
                self.assertEqual(self.f.format(fstring, *args, **kwargs), ref)
