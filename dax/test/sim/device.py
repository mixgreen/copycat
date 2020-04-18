import unittest

from dax.test.helpers.mypy import type_check


class DaxSimDeviceStaticTyping(unittest.TestCase):

    def test_device_static_typing(self):
        # Type checking on DAX sim device
        import dax.sim.device
        type_check(self, dax.sim.device, '--strict')


if __name__ == '__main__':
    unittest.main()
