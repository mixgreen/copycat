import unittest

from dax.test.helpers.mypy import type_check


class DaxSimCoredeviceGenericStaticTyping(unittest.TestCase):

    def test_static_typing(self):
        # Type checking on DAX sim coredevice generic
        import dax.sim.coredevice.generic
        type_check(self, dax.sim.coredevice.generic)


if __name__ == '__main__':
    unittest.main()
