import unittest

from dax.test.helpers.mypy import type_check


class DaxSimCoredeviceCoreStaticTyping(unittest.TestCase):

    def test_static_typing(self):
        # Type checking on DAX sim coredevice core
        import dax.sim.coredevice.core
        type_check(self, dax.sim.coredevice.core, '--strict')


if __name__ == '__main__':
    unittest.main()
