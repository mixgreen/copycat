import unittest

from dax.test.helpers.mypy import type_check


class DaxSimCoredeviceTtlStaticTyping(unittest.TestCase):

    def test_static_typing(self):
        # Type checking on DAX sim coredevice TTL
        import dax.sim.coredevice.ttl
        type_check(self, dax.sim.coredevice.ttl)


if __name__ == '__main__':
    unittest.main()
