import unittest

from dax.base.dax import *

from dax.test.helpers.artiq import get_manager_or_parent
from dax.test.helpers.mypy import type_check

from artiq.coredevice.edge_counter import EdgeCounter


class TestSystem(DaxSystem):
    SYS_ID = 'unittest_system'
    SYS_VER = 0


class TestModule(DaxModule):
    """Testing module."""

    def init(self):
        pass

    def post_init(self):
        pass


class TestModuleChild(TestModule):
    pass


class TestService(DaxService):
    SERVICE_NAME = 'test_service'

    def init(self):
        pass

    def post_init(self):
        pass


class TestServiceChild(TestService):
    SERVICE_NAME = 'test_service_child'


class DaxStaticTyping(unittest.TestCase):

    def test_static_typing(self):
        # Type checking on DAX base
        import dax.base.dax as dax_module
        type_check(self, dax_module)


class DaxHelpersTestCase(unittest.TestCase):

    def test_valid_name(self):
        from dax.base.dax import _is_valid_name

        for n in ['foo', '_0foo', '_', '0', '_foo', 'FOO_', '0_foo']:
            # Test valid names
            self.assertTrue(_is_valid_name(n))

        for n in ['', 'foo()', 'foo.bar', 'foo/', 'foo*', 'foo,', 'FOO+', 'foo-bar', 'foo/bar']:
            # Test illegal names
            self.assertFalse(_is_valid_name(n))

    def test_valid_key(self):
        from dax.base.dax import _is_valid_key

        for k in ['foo', '_0foo', '_', '0', 'foo.bar', 'foo.bar.baz', '_.0.A', 'foo0._bar']:
            # Test valid keys
            self.assertTrue(_is_valid_key(k))

        for k in ['', 'foo()', 'foo,bar', 'foo/', '.foo', 'bar.', 'foo.bar.baz.']:
            # Test illegal keys
            self.assertFalse(_is_valid_key(k))


class DaxNameRegistryTestCase(unittest.TestCase):

    def test_module(self):
        from dax.base.dax import _DaxNameRegistry

        # Test system
        s = TestSystem(get_manager_or_parent())
        # Registry
        r = s.registry

        # Test with no modules
        with self.assertRaises(KeyError, msg='Get non-existing module did not raise'):
            r.get_module('not_existing_key')

        # Test with one module
        t0 = TestModule(s, 'test_module')
        self.assertIs(r.get_module(t0.get_system_key()), t0, 'Returned module does not match expected module')
        with self.assertRaises(TypeError, msg='Type check in get_module() did not raise'):
            r.get_module(t0.get_system_key(), TestModuleChild)
        self.assertIs(r.search_module(TestModule), t0, 'Did not find the expected module')
        self.assertIs(r.search_module(DaxModule), t0, 'Did not find the expected module')
        with self.assertRaises(KeyError, msg='Search non-existing module did not raise'):
            r.search_module(TestModuleChild)
        self.assertListEqual(r.get_module_key_list(), [m.get_system_key() for m in [s, t0]],
                             'Module key list incorrect')
        with self.assertRaises(_DaxNameRegistry.NonUniqueRegistrationError, msg='Adding module twice did not raise'):
            r.add_module(t0)

        # Test with two modules
        t1 = TestModuleChild(s, 'test_module_child')
        self.assertIs(r.get_module(t1.get_system_key()), t1, 'Returned module does not match expected module')
        self.assertIs(r.get_module(t1.get_system_key(), TestModuleChild), t1,
                      'Type check in get_module() raised unexpectedly')
        self.assertIs(r.search_module(TestModuleChild), t1, 'Did not find expected module')
        with self.assertRaises(_DaxNameRegistry.NonUniqueSearchError, msg='Non-unique search did not raise'):
            r.search_module(TestModule)
        self.assertListEqual(r.get_module_key_list(), [m.get_system_key() for m in [s, t0, t1]],
                             'Module key list incorrect')
        self.assertDictEqual(r.search_module_dict(TestModule), {m.get_system_key(): m for m in [t0, t1]},
                             'Search result dict incorrect')

    def test_device(self):
        from dax.base.dax import _DaxNameRegistry

        # Test system
        s = TestSystem(get_manager_or_parent())
        t0 = TestModule(s, 'test_module')
        # List of core devices
        core_devices = ['core', 'core_cache', 'core_dma']
        # Registry
        r = s.registry

        # Test core devices, which should be existing
        self.assertEqual(r.get_device_key_list(), core_devices, 'Core devices were not found in device list')

        # Test adding other keys
        self.assertIsNone(r.add_device(t0, 'ttl0'), 'Device registration failed')
        self.assertIsNone(r.add_device(t0, 'alias_2'), 'Device registration with alias failed')
        self.assertIn('ttl1', r.get_device_key_list(),
                      'Device registration did not found correct unique key for device alias')
        self.assertListEqual(r.get_device_key_list(), core_devices + ['ttl0', 'ttl1'], 'Device key list incorrect')
        with self.assertRaises(_DaxNameRegistry.NonUniqueRegistrationError,
                               msg='Double device registration did not raise when registered by unique name and alias'):
            r.add_device(t0, 'alias_1')

        # Test looped alias
        with self.assertRaises(KeyError, msg='Looped key alias did not raise'):
            r.add_device(t0, 'loop_alias_1')
        with self.assertRaises(KeyError, msg='Looped key alias did not raise'):
            r.add_device(t0, 'loop_alias_4')

    def test_service(self):
        from dax.base.dax import _DaxNameRegistry

        # Test system
        s = TestSystem(get_manager_or_parent())
        s0 = TestService(s)
        # Registry
        r = s.registry

        # Test adding the service again
        with self.assertRaises(_DaxNameRegistry.NonUniqueRegistrationError,
                               msg='Double service registration did not raise'):
            r.add_service(s0)

        # Test with one service
        self.assertFalse(r.has_service('foo'), 'Non-existing service did not returned false')
        self.assertFalse(r.has_service(TestServiceChild), 'Non-existing service did not returned false')
        self.assertTrue(r.has_service(TestService.SERVICE_NAME), 'Did not returned true for existing service')
        self.assertTrue(r.has_service(TestService), 'Did not returned true for existing service')
        self.assertIs(r.get_service(s0.get_name()), s0, 'Did not returned expected service')
        self.assertIs(r.get_service(TestService), s0, 'Did not returned expected service')
        with self.assertRaises(KeyError, msg='Retrieving non-existing service did not raise'):
            r.get_service(TestServiceChild)
        self.assertListEqual(r.get_service_key_list(), [s.get_name() for s in [s0]],
                             'List of registered service keys incorrect')

        # Test with a second service
        s1 = TestServiceChild(s)
        self.assertTrue(r.has_service(TestServiceChild), 'Did not returned true for existing service')
        self.assertTrue(r.has_service(TestServiceChild.SERVICE_NAME), 'Did not returned true for existing service')
        self.assertListEqual(r.get_service_key_list(), [s.get_name() for s in [s0, s1]],
                             'List of registered service keys incorrect')


class DaxModuleBaseTestCase(unittest.TestCase):
    """Tests _DaxHasSystemBase, DaxModuleBase, DaxModule, and DaxSystem.

    The four mentioned modules are highly related and overlap mostly.
    Therefore they are all tested mutually.
    """

    def test_system_build(self):
        # A system that does not call super() in build()
        class BadTestSystem(TestSystem):
            def build(self):
                pass  # No call to super(), which is bad

        # Test if an error occurs when super() is not called in build()
        with self.assertRaises(AttributeError, msg='Not calling super.build() in user system did not raise'):
            BadTestSystem(get_manager_or_parent())

    def test_system_id(self):
        # Test if an error is raised when no ID is given to a system
        with self.assertRaises(AssertionError, msg='Not providing system id did not raise'):
            DaxSystem(get_manager_or_parent())

        # Systems with bad ID
        class TestSystemBadId1(DaxSystem):
            SYS_ID = 'wrong.name'
            SYS_VER = 0

        # Systems with bad ID
        class TestSystemBadId2(DaxSystem):
            SYS_ID = ''
            SYS_VER = 0

        # Systems with bad ID
        class TestSystemBadId3(DaxSystem):
            SYS_ID = '+wrong'
            SYS_VER = 0

        for BadSystem in [TestSystemBadId1, TestSystemBadId2, TestSystemBadId3]:
            # Test if an error is raised when a bad ID is given to a system
            with self.assertRaises(AssertionError, msg='Providing bad system id did not raise'):
                BadSystem(get_manager_or_parent())

    def test_system_ver(self):
        class TestSystemNoVer(DaxSystem):
            SYS_ID = 'unittest_system'

        # Test if an error is raised when no version is given to a system
        with self.assertRaises(AssertionError, msg='Not providing system version did not raise'):
            TestSystemNoVer(get_manager_or_parent())

        # System with bad version
        class TestSystemBadVer1(DaxSystem):
            SYS_ID = 'unittest_system'
            SYS_VER = '1'

        # System with bad version
        class TestSystemBadVer2(DaxSystem):
            SYS_ID = 'unittest_system'
            SYS_VER = -1

        # System with bad version
        class TestSystemBadVer3(DaxSystem):
            SYS_ID = 'unittest_system'
            SYS_VER = 1.1

        for BadSystem in [TestSystemBadVer1, TestSystemBadVer2, TestSystemBadVer3]:
            # Test if an error is raised when a bad version is given to a system
            with self.assertRaises(AssertionError, msg='Providing bad system version did not raise'):
                BadSystem(get_manager_or_parent())

        # System with version 0, which is fine
        class TestSystemVerZero(DaxSystem):
            SYS_ID = 'unittest_system'
            SYS_VER = 0

        # Test if it is possible to create a system with version 0
        TestSystemVerZero(get_manager_or_parent())

    def test_init(self):
        manager_or_parent = get_manager_or_parent()
        s = TestSystem(get_manager_or_parent())

        # Check constructor
        self.assertIsNotNone(s, 'Could not create DaxSystem')
        self.assertIsNotNone(TestModule(s, 'module_name'), 'Could not create a test module')
        with self.assertRaises(ValueError, msg='Invalid module name did not raise'):
            TestModule(s, 'wrong!')
        with self.assertRaises(ValueError, msg='Invalid module name did not raise'):
            TestModule(s, 'this.is.bad')
        with self.assertRaises(TypeError, msg='Providing non-DaxModuleBase parent to new module did not raise'):
            TestModule(manager_or_parent, 'module_name')

        # Check register
        s = TestSystem(get_manager_or_parent())
        t = TestModule(s, 'module_name')
        self.assertDictEqual(s.registry._modules, {m.get_system_key(): m for m in [s, t]},
                             'Dict with registered modules does not match expected content')

    def test_names_keys(self):
        s = TestSystem(get_manager_or_parent())

        self.assertEqual(s.get_name(), TestSystem.SYS_NAME, 'Returned name did not match expected name')
        self.assertEqual(s.get_system_key(), TestSystem.SYS_NAME, 'Returned key did not match expected key')

        n = 'test_module_name'
        t = TestModule(s, n)
        self.assertEqual(t.get_system_key(), '.'.join([TestSystem.SYS_NAME, n]),
                         'Key created for nested module did not match expected key')
        some_key = 'some_key'
        self.assertEqual(t.get_system_key(some_key), '.'.join([TestSystem.SYS_NAME, n, some_key]),
                         'System key creation derived from current module key failed')
        with self.assertRaises(ValueError, msg='Creating bad system key did not raise'):
            s.get_system_key('bad,key')
        with self.assertRaises(AssertionError, msg='Creating system key with wrong key input did not raise'):
            s.get_system_key(1)

    def test_devices(self):
        s = TestSystem(get_manager_or_parent())

        self.assertIsNone(s.setattr_device('ttl0'), 'setattr_device() did not return None')
        self.assertTrue(hasattr(s, 'ttl0'), 'setattr_device() did not set the attribute correctly')
        self.assertIsNone(s.setattr_device('alias_2', 'foo'), 'setattr_device() with attribute name failed')
        self.assertTrue(hasattr(s, 'foo'), 'setattr_device() with attribute name did not set attribute correctly')

        s = TestSystem(get_manager_or_parent())
        with self.assertRaises(TypeError, msg='get_device() type check did not raise'):
            s.get_device('ttl1', EdgeCounter)  # EdgeCounter does not match the device type of ttl1

    def test_dataset(self):
        s = TestSystem(get_manager_or_parent())

        key = 'key1'
        self.assertListEqual(s.get_dataset_sys(key, default=[11, 12, 13]), [11, 12, 13],
                             'get_dataset_sys() did not returned the provided default value')
        with self.assertRaises(KeyError,
                               msg='get_dataset_sys() wrote the default value to the dataset, which it should not'):
            s.get_dataset_sys(key)

        key = 'key3'
        self.assertIsNone(s.setattr_dataset_sys(key, 10), 'setattr_dataset_sys() failed')
        self.assertTrue(hasattr(s, key), 'setattr_dataset_sys() did not set the attribute corectly')
        self.assertEqual(getattr(s, key), 10, 'Returned system dataset value does not match expected result')
        self.assertIn(key, s.kernel_invariants,
                      'setattr_dataset_sys() did not added the attribute to kernel_invariants by default')

    def test_dataset_append(self):
        s = TestSystem(get_manager_or_parent())

        key = 'key2'
        self.assertIsNone(s.set_dataset_sys(key, []), 'Setting new system dataset failed')
        self.assertListEqual(s.get_dataset_sys(key), [],
                             'Returned system dataset value does not match expected result')
        self.assertIsNone(s.append_to_dataset_sys(key, 1), 'Appending to system dataset failed')
        self.assertListEqual(s.get_dataset_sys(key), [1], 'Appending to system dataset has incorrect behavior')

        key = 'key4'
        self.assertIsNone(s.set_dataset(key, [0]), 'Setting new dataset failed')
        self.assertListEqual(s.get_dataset(key), [0], 'Returned dataset value does not match expected result')
        self.assertIsNone(s.append_to_dataset(key, 1), 'Appending to dataset failed')
        self.assertListEqual(s.get_dataset(key), [0, 1], 'Appending to dataset has incorrect behavior')

        key = 'key5'
        self.assertIsNone(s.set_dataset(s.get_system_key(key), [0]), 'Setting new dataset failed')
        self.assertListEqual(s.get_dataset(s.get_system_key(key)), [0],
                             'Returned dataset value does not match expected result')
        self.assertIsNone(s.append_to_dataset(s.get_system_key(key), 1), 'Appending to dataset failed')
        self.assertListEqual(s.get_dataset(s.get_system_key(key)), [0, 1],
                             'Appending to dataset has incorrect behavior')

    def test_dataset_mutate(self):
        s = TestSystem(get_manager_or_parent())

        key = 'key2'
        self.assertIsNone(s.set_dataset_sys(key, [0, 0, 0, 0]), 'Setting new system dataset failed')
        self.assertListEqual(s.get_dataset_sys(key), [0, 0, 0, 0],
                             'Returned system dataset value does not match expected result')
        self.assertIsNone(s.mutate_dataset_sys(key, 1, 9), 'Mutating system dataset failed')
        self.assertIsNone(s.mutate_dataset_sys(key, 3, 99), 'Mutating system dataset failed')
        self.assertListEqual(s.get_dataset_sys(key), [0, 9, 0, 99], 'Mutating system dataset has incorrect behavior')

    def test_identifier(self):
        s = TestSystem(get_manager_or_parent())
        self.assertTrue(isinstance(s.get_identifier(), str), 'get_identifier() did not returned a string')


class DaxServiceTestCase(unittest.TestCase):

    def test_init(self):
        from dax.base.dax import _DaxNameRegistry

        s = TestSystem(get_manager_or_parent())

        class NoNameService(DaxService):
            def init(self) -> None:
                pass

            def post_init(self) -> None:
                pass

        with self.assertRaises(AssertionError, msg='Lack of class service name did not raise'):
            NoNameService(s)

        class WrongNameService(NoNameService):
            SERVICE_NAME = 3

        with self.assertRaises(AssertionError, msg='Wrong type service name did not raise'):
            WrongNameService(s)

        class GoodNameService(NoNameService):
            SERVICE_NAME = 'service_name'

        service = GoodNameService(s)
        self.assertIs(s.registry.get_service(GoodNameService), service,
                      'get_service() did not returned expected object')
        self.assertIn(GoodNameService.SERVICE_NAME, s.registry.get_service_key_list(),
                      'Could not find service name key in registry')

        class DuplicateNameService(NoNameService):
            SERVICE_NAME = 'service_name'

        with self.assertRaises(_DaxNameRegistry.NonUniqueRegistrationError,
                               msg='Duplicate service name registration did not raise'):
            DuplicateNameService(s)

        class GoodNameService2(NoNameService):
            SERVICE_NAME = 'service_name_2'

        self.assertTrue(GoodNameService2(service), 'Could not create new service with other service as parent')


class DaxClientTestCase(unittest.TestCase):

    def test_not_decorated(self):
        s = TestSystem(get_manager_or_parent())

        class Client(DaxClient):
            pass

        with self.assertRaises(AssertionError, msg='Using client without client factory decorator did not raise'):
            Client(s)

    def test_load_super(self):
        class System(TestSystem):
            def init(self) -> None:
                self.is_initialized = True

        @dax_client_factory
        class Client(DaxClient):
            pass

        class ImplementableClient(Client(System)):
            pass

        c = ImplementableClient(get_manager_or_parent())
        c.init()  # Is supposed to call the init() function of the system

        self.assertTrue(hasattr(c, 'is_initialized'), 'DAX system parent of client was not initialized correctly')


if __name__ == '__main__':
    unittest.main()
