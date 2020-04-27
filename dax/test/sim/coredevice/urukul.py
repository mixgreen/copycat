import unittest

from dax.test.helpers.mypy import type_check


class DaxSimCoredeviceUrukulStaticTyping(unittest.TestCase):

    def test_static_typing(self):
        # Type checking on DAX sim coredevice Urukul
        import dax.sim.coredevice.urukul
        type_check(self, dax.sim.coredevice.urukul)


if __name__ == '__main__':
    unittest.main()
