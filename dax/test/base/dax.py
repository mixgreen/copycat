import unittest

import tempfile
import os.path

from dax.base.dax import *


def _get_manager_or_parent():
    """Returns an object that can function as a manager_or_parent for ARTIQ HasEnvironment."""

    from artiq.master.worker_db import DeviceManager, DatasetManager
    from artiq.master.databases import DeviceDB, DatasetDB
    from artiq.frontend.artiq_run import DummyScheduler, DummyCCB

    device_db_file_name = os.path.join(os.path.dirname(__file__), 'test_device_db.py')
    device_mgr = DeviceManager(DeviceDB(device_db_file_name),
                               virtual_devices={"scheduler": DummyScheduler(),
                                                "ccb": DummyCCB()})
    dataset_db_file_name = os.path.join(tempfile.gettempdir(), 'dax_test_dataset_db.pyon')
    dataset_db = DatasetDB(dataset_db_file_name)
    dataset_mgr = DatasetManager(dataset_db)

    # Return a tuple that is accepted as manager_or_parent
    # DeviceManager, DatasetManager, ArgumentParser.parse_args(), dict
    return device_mgr, dataset_mgr, object, {}


class _TestModule(DaxModule):
    """Testing module."""

    def load(self):
        pass

    def init(self):
        pass

    def config(self):
        pass


class _TestModuleChild(_TestModule):
    pass


class _TestService(DaxService):
    SERVICE_NAME = 'test_service'

    def load(self):
        pass

    def init(self):
        pass

    def config(self):
        pass


class _TestServiceChild(_TestService):
    SERVICE_NAME = 'test_service_child'


class _TestSystem(DaxSystem):
    """Testing system."""

    def build(self):
        # Call super
        super(_TestSystem, self).build()
        # Make module
        _TestModule(self, 'test_module')
        _TestModuleChild(self, 'test_module_child')


class DaxNameRegistryTestCase(unittest.TestCase):

    def test_init(self):
        from dax.base.dax import _DaxNameRegistry

        for key in ['foo', '_foo', 'FOO_', '0_foo', 'foo0._bar']:
            # Test valid names
            self.assertTrue(_DaxNameRegistry(key))

        for key in ['foo*', 'foo,', 'FOO+', 'foo-bar', 'foo/bar']:
            # Test illegal names
            self.assertRaises(ValueError, _DaxNameRegistry, key)

    def test_module(self):
        from dax.base.dax import _DaxNameRegistry

        # Test system
        s = DaxSystem(_get_manager_or_parent())
        # Registry
        r = s.registry

        # Test with no modules
        self.assertRaises(KeyError, r.get_module, 'not_existing_key')

        # Test with one module
        t0 = _TestModule(s, 'test_module')
        self.assertEqual(r.get_module(t0.get_system_key()), t0)
        self.assertRaises(TypeError, r.get_module, t0.get_system_key(), _TestModuleChild)
        self.assertEqual(r.search_module(_TestModule), t0)
        self.assertEqual(r.search_module(DaxModule), t0)
        self.assertRaises(KeyError, r.search_module, _TestModuleChild)
        self.assertEqual(r.get_module_key_list(), [m.get_system_key() for m in [s, t0]])
        self.assertRaises(_DaxNameRegistry._NonUniqueRegistrationError, r.add_module, t0)

        # Test with two modules
        t1 = _TestModuleChild(s, 'test_module_child')
        self.assertEqual(r.get_module(t1.get_system_key()), t1)
        self.assertEqual(r.get_module(t1.get_system_key(), _TestModuleChild), t1)
        self.assertEqual(r.search_module(_TestModuleChild), t1)
        self.assertRaises(_DaxNameRegistry._NonUniqueSearchError, r.search_module, _TestModule)
        self.assertEqual(r.get_module_key_list(), [m.get_system_key() for m in [s, t0, t1]])
        self.assertEqual(r.search_module_dict(_TestModule), {m.get_system_key(): m for m in [t0, t1]})

    def test_device(self):
        from dax.base.dax import _DaxNameRegistry

        # Test system
        s = DaxSystem(_get_manager_or_parent())
        t0 = _TestModule(s, 'test_module')
        # List of core devices
        core_devices = ['core', 'core_cache', 'core_dma']
        # Registry
        r = s.registry

        # Test core devices, which should be existing
        self.assertEqual(r.get_device_key_list(), core_devices)

        # Test adding other keys
        self.assertIsNone(r.add_device(t0, 'ttl0'))
        self.assertIsNone(r.add_device(t0, 'alias_2'))
        self.assertEqual(r.get_device_key_list(), core_devices + ['ttl0', 'ttl1'])
        self.assertRaises(_DaxNameRegistry._NonUniqueRegistrationError, r.add_device, t0, 'alias_1')

        # Test looped alias
        self.assertRaises(KeyError, r.add_device, t0, 'loop_alias_1')
        self.assertRaises(KeyError, r.add_device, t0, 'loop_alias_4')

    def test_service(self):
        from dax.base.dax import _DaxNameRegistry

        # Test system
        s = DaxSystem(_get_manager_or_parent())
        s0 = _TestService(s)
        # Registry
        r = s.registry

        # Test adding the service again
        self.assertRaises(_DaxNameRegistry._NonUniqueRegistrationError, r.add_service, s0)

        # Test with one service
        self.assertIsNone(r.has_service('foo'))
        self.assertIsNone(r.has_service(_TestServiceChild))
        self.assertEqual(r.has_service(_TestService.SERVICE_NAME), s0)
        self.assertEqual(r.has_service(_TestService), s0)
        self.assertEqual(r.get_service(s0.get_name()), s0)
        self.assertEqual(r.get_service(_TestService), s0)
        self.assertRaises(KeyError, r.get_service, _TestServiceChild)
        self.assertEqual(r.get_service_key_list(), [s.get_name() for s in [s0]])

        # Test with a second service
        s1 = _TestServiceChild(s)
        self.assertEqual(r.has_service(_TestServiceChild), s1)
        self.assertEqual(r.has_service(_TestServiceChild.SERVICE_NAME), s1)
        self.assertEqual(r.get_service_key_list(), [s.get_name() for s in [s0, s1]])


if __name__ == '__main__':
    unittest.main()
