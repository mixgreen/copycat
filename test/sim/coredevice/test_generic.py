import unittest

import dax.sim.coredevice.generic
from dax.sim import enable_dax_sim
from dax.util.artiq import get_managers


class GenericTestCase(unittest.TestCase):
    _DEVICE_DB = enable_dax_sim({
        'core': {
            'type': 'local',
            'module': 'artiq.coredevice.core',
            'class': 'Core',
            'arguments': {'host': '1.2.3.4', 'ref_period': 1e-9}
        },
        "generic": {
            "type": "local",
            "module": "non.existing.module",
            "class": "non.existing.class"
        },
    }, enable=True, output='null', moninj_service=False)

    def setUp(self) -> None:
        self.managers = get_managers(device_db=self._DEVICE_DB)
        device_mgr, _, _, _ = self.managers
        self.generic = dax.sim.coredevice.generic.Generic(device_mgr, _key='generic')

    def tearDown(self) -> None:
        # Close devices
        device_mgr, _, _, _ = self.managers
        device_mgr.close_devices()

    def test_direct_call(self):
        with self.assertRaises(TypeError, msg='Direct call on generic object did not raise'):
            # Note that the driver itself is not callable, which should raise an exception
            self.generic()

    def test_attributes(self):
        foo = self.generic.foo
        self.assertIsInstance(foo, dax.sim.coredevice.generic._GenericBase)
        self.assertEqual(foo, self.generic.foo)  # Test if same attribute is returned

        self.assertIsInstance(self.generic.foo.bar.baz, dax.sim.coredevice.generic._GenericBase)
        self.assertIsInstance(self.generic.bar.baz.foo, dax.sim.coredevice.generic._GenericBase)
        self.assertIsInstance(self.generic.baz.foo.bar, dax.sim.coredevice.generic._GenericBase)

    def test_method_call(self):
        foo = self.generic.foo
        self.assertIsInstance(foo, dax.sim.coredevice.generic._GenericBase)
        self.assertTrue(callable(foo), 'Generic object method/attribute is not callable')

        self.generic.bar()
        self.generic.foo.bar.baz()
        self.generic.bar.baz.foo()
        self.generic.baz.foo.bar()


if __name__ == '__main__':
    unittest.main()