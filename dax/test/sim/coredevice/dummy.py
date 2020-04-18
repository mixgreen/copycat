import unittest

from dax.test.helpers.mypy import type_check


class DaxSimCoredeviceDummyStaticTyping(unittest.TestCase):

    def test_static_typing(self):
        # Type checking on DAX sim coredevice dummy
        import dax.sim.coredevice.dummy
        type_check(self, dax.sim.coredevice.dummy, '--strict')


if __name__ == '__main__':
    unittest.main()
