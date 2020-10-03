import unittest
import logging

import dax.sim.coredevice.core
from dax.sim.signal import set_signal_manager, NullSignalManager
from dax.sim.ddb import enable_dax_sim
from dax.util.artiq import get_managers


class CoreTestCase(unittest.TestCase):
    _DEVICE_DB = {
        'core': {
            'type': 'local',
            'module': 'artiq.coredevice.core',
            'class': 'Core',
            'arguments': {'host': '0.0.0.0', 'ref_period': 1e-9}
        },
    }

    def setUp(self) -> None:
        set_signal_manager(NullSignalManager())

    def test_constructor_signature(self):
        dmgr, _, _, _ = get_managers(
            enable_dax_sim(self._DEVICE_DB, enable=True, logging_level=logging.WARNING,
                           moninj_service=False, output='null'))

        # Make sure the signature is as expected
        with self.assertRaises(TypeError, msg='Core class constructor did not match expected signature'):
            # Not adding _key
            dax.sim.coredevice.core.Core(dmgr=dmgr, ref_period=1e-9)
        with self.assertRaises(TypeError, msg='Core class constructor did not match expected signature'):
            # Not adding ref period
            # noinspection PyArgumentList
            dax.sim.coredevice.core.Core(dmgr=dmgr, _key='core')

        # Test with correct arguments
        self.assertIsNotNone(dax.sim.coredevice.core.Core(dmgr=dmgr, ref_period=1e-9, _key='core'))


class BaseCoreTestCase(CoreTestCase):
    def test_constructor_signature(self):
        # Should be able to construct base core without arguments
        self.assertIsNotNone(dax.sim.coredevice.core.BaseCore())


if __name__ == '__main__':
    unittest.main()
