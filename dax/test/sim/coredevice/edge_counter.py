import unittest

from dax.test.helpers.mypy import type_check


class DaxSimCoredeviceEdgeCounterStaticTyping(unittest.TestCase):

    def test_static_typing(self):
        # Type checking on DAX sim coredevice EdgeCounter
        import dax.sim.coredevice.edge_counter
        type_check(self, dax.sim.coredevice.edge_counter)


if __name__ == '__main__':
    unittest.main()
