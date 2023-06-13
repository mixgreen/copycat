import unittest
import numpy as np
import warnings

from dax.util.artiq import is_portable
import dax.util.convert


class ConvertTestCase(unittest.TestCase):

    def test_decorators(self):
        functions = [
            dax.util.convert.list_to_int32,
            dax.util.convert.list_to_int64,
        ]
        for fn in functions:
            with self.subTest(fn=fn):
                self.assertTrue(is_portable(fn))

    def test_list_to_int32(self):
        data = [
            ([], 0b0),
            ([0], 0b0),
            ([1], 0b1),
            ([0, 1], 0b10),
            ([0, 1, 2, 3, 0, 1], 0b101110),
            ([0] * 30 + [1], 0b1 << 30),
            ([0] * 31 + [1], 0b1 << 31),
            ([0] * 31 + [1], -(0b1 << 31)),
            ([1] * 32, 0xFFFFFFFF),
            ([1] * 32, -1),
            ([0] * 32 + [1], 0b0),
        ]

        for measurements, ref in data:
            with self.subTest(measurements=measurements):
                result = dax.util.convert.list_to_int32(measurements)
                self.assertIsInstance(result, np.int32)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', category=DeprecationWarning)
                    self.assertEqual(result, np.int32(ref))

    def test_list_to_int64(self):
        data = [
            ([], 0b0),
            ([0], 0b0),
            ([1], 0b1),
            ([0, 1], 0b10),
            ([0, 1, 2, 3, 0, 1], 0b101110),
            ([0] * 30 + [1], 0b1 << 30),
            ([0] * 31 + [1], 0b1 << 31),
            ([1] * 32, 0xFFFFFFFF),
            ([0] * 62 + [1], 0b1 << 62),
            ([0] * 63 + [1], -(0b1 << 63)),
            ([1] * 64, -1),
            ([0] * 64 + [1], 0b0),
        ]

        for measurements, ref in data:
            with self.subTest(measurements=measurements):
                result = dax.util.convert.list_to_int64(measurements)
                self.assertIsInstance(result, np.int64)
                self.assertEqual(result, np.int64(ref))
