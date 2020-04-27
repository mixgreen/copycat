import unittest

from dax.test.helpers.mypy import type_check


class DaxSimCoredeviceAd9912StaticTyping(unittest.TestCase):

    def test_static_typing(self):
        # Type checking on DAX sim coredevice AD9912
        import dax.sim.coredevice.ad9912
        type_check(self, dax.sim.coredevice.ad9912)


if __name__ == '__main__':
    unittest.main()
