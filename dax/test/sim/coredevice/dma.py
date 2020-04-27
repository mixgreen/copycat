import unittest

from dax.test.helpers.mypy import type_check


class DaxSimCoredeviceDmaStaticTyping(unittest.TestCase):

    def test_static_typing(self):
        # Type checking on DAX sim coredevice DMA
        import dax.sim.coredevice.dma
        type_check(self, dax.sim.coredevice.dma)


if __name__ == '__main__':
    unittest.main()
