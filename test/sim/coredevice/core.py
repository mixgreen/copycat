import unittest

import dax.sim.coredevice.core
from dax.sim.signal import set_signal_manager, NullSignalManager


class CoreTestCase(unittest.TestCase):
    def setUp(self) -> None:
        set_signal_manager(NullSignalManager())

    def test_constructor_signature(self):
        # Make sure the signature is as expected
        with self.assertRaises(TypeError, msg='Core class constructor did not match expected signature'):
            # Not adding _key
            dax.sim.coredevice.core.Core(dmgr={}, ref_period=1e-9)
        with self.assertRaises(TypeError, msg='Core class constructor did not match expected signature'):
            # Not adding ref period
            # noinspection PyArgumentList
            dax.sim.coredevice.core.Core(dmgr={}, _key='core')

        # Test with correct arguments
        self.assertIsNotNone(dax.sim.coredevice.core.Core(dmgr={}, ref_period=1e-9, _key='core'))


class BaseCoreTestCase(CoreTestCase):
    def test_constructor_signature(self):
        # Should be able to construct base core without arguments
        self.assertIsNotNone(dax.sim.coredevice.core.BaseCore())


if __name__ == '__main__':
    unittest.main()
