import unittest
import numpy as np
import logging
import os
import pygit2  # type: ignore

from artiq.experiment import HasEnvironment
import artiq.coredevice.edge_counter
import artiq.coredevice.ttl  # type: ignore
import artiq.coredevice.core  # type: ignore

from dax.base.dax import *
import dax.base.dax
import dax.base.exceptions
import dax.base.interface
from dax.util.artiq import get_manager_or_parent

"""Device DB for testing"""

_device_db = {
    # Core device
    'core': {
        'type': 'local',
        'module': 'artiq.coredevice.core',
        'class': 'Core',
        'arguments': {'host': '0.0.0.0', 'ref_period': 1e-9}
    },
    'core_cache': {
        'type': 'local',
        'module': 'artiq.coredevice.cache',
        'class': 'CoreCache'
    },
    'core_dma': {
        'type': 'local',
        'module': 'artiq.coredevice.dma',
        'class': 'CoreDMA'
    },

    # Generic TTL
    'ttl0': {
        'type': 'local',
        'module': 'artiq.coredevice.ttl',
        'class': 'TTLInOut',
        'arguments': {'channel': 0},
    },
    'ttl1': {
        'type': 'local',
        'module': 'artiq.coredevice.ttl',
        'class': 'TTLOut',
        'arguments': {'channel': 1},
    },

    # Aliases
    'alias_0': 'ttl1',
    'alias_1': 'alias_0',
    'alias_2': 'alias_1',

    # Looped alias
    'loop_alias_0': 'loop_alias_0',
    'loop_alias_1': 'loop_alias_0',
    'loop_alias_2': 'loop_alias_4',
    'loop_alias_3': 'loop_alias_2',
    'loop_alias_4': 'loop_alias_3',

    # Dead aliases
    'dead_alias_0': 'this_key_does_not_exist_123',
    'dead_alias_1': 'dead_alias_0',
    'dead_alias_2': 'dead_alias_1',
}

"""Classes used for testing"""


class _TestSystem(DaxSystem):
    SYS_ID = 'unittest_system'
    SYS_VER = 0


class _TestModule(DaxModule):
    """Testing module."""

    def init(self):
        pass

    def post_init(self):
        pass


class _TestModuleChild(_TestModule):
    pass


class _TestService(DaxService):
    SERVICE_NAME = 'test_service'

    def init(self):
        pass

    def post_init(self):
        pass


class _TestInterface(dax.base.interface.DaxInterface):
    pass


class _TestServiceChild(_TestService, _TestInterface):
    SERVICE_NAME = 'test_service_child'


"""Actual test cases"""


class DaxHelpersTestCase(unittest.TestCase):

    def test_valid_name(self):
        for n in ['foo', '_0foo', '_', '0', '_foo', 'FOO_', '0_foo']:
            # Test valid names
            self.assertTrue(dax.base.dax._is_valid_name(n))

    def test_invalid_name(self):
        for n in ['', 'foo()', 'foo.bar', 'foo/', 'foo*', 'foo,', 'FOO+', 'foo-bar', 'foo/bar']:
            # Test illegal names
            self.assertFalse(dax.base.dax._is_valid_name(n))

    def test_valid_key(self):
        for k in ['foo', '_0foo', '_', '0', 'foo.bar', 'foo.bar.baz', '_.0.A', 'foo0._bar']:
            # Test valid keys
            self.assertTrue(dax.base.dax._is_valid_key(k))

    def test_invalid_key(self):
        for k in ['', 'foo()', 'foo,bar', 'foo/', '.foo', 'bar.', 'foo.bar.baz.']:
            # Test illegal keys
            self.assertFalse(dax.base.dax._is_valid_key(k))

    def test_unique_device_key(self):
        # Test system and device DB
        s = _TestSystem(get_manager_or_parent(_device_db))
        d = s.get_device_db()

        # Test against various keys
        self.assertEqual(dax.base.dax._get_unique_device_key(d, 'ttl0'), 'ttl0',
                         'Unique device key not returned correctly')
        self.assertEqual(dax.base.dax._get_unique_device_key(d, 'alias_0'), 'ttl1',
                         'Alias key key does not return correct unique key')
        self.assertEqual(dax.base.dax._get_unique_device_key(d, 'alias_1'), 'ttl1',
                         'Multi-alias key does not return correct unique key')
        self.assertEqual(dax.base.dax._get_unique_device_key(d, 'alias_2'), 'ttl1',
                         'Multi-alias key does not return correct unique key')

    def test_looped_device_key(self):
        # Test system and device DB
        s = _TestSystem(get_manager_or_parent(_device_db))
        d = s.get_device_db()

        # Test looped alias
        loop_aliases = ['loop_alias_1', 'loop_alias_4']
        for key in loop_aliases:
            with self.assertRaises(LookupError, msg='Looped key alias did not raise'):
                dax.base.dax._get_unique_device_key(d, key)

    def test_unavailable_device_key(self):
        # Test system and device DB
        s = _TestSystem(get_manager_or_parent(_device_db))
        d = s.get_device_db()

        # Test non-existing keys
        loop_aliases = ['not_existing_key_0', 'not_existing_key_1', 'dead_alias_2']
        for key in loop_aliases:
            with self.assertRaises(KeyError, msg='Non-existing key did not raise'):
                dax.base.dax._get_unique_device_key(d, key)

    def test_virtual_device_key(self):
        # Test system and device DB
        s = _TestSystem(get_manager_or_parent(_device_db))
        d = s.get_device_db()
        # Test virtual devices
        virtual_devices = {'scheduler', 'ccb'}
        self.assertSetEqual(virtual_devices, dax.base.dax._ARTIQ_VIRTUAL_DEVICES,
                            'List of virtual devices in test does not match DAX base virtual device list')
        for k in virtual_devices:
            self.assertEqual(dax.base.dax._get_unique_device_key(d, k), k, 'Virtual device key not returned correctly')

    def test_cwd_commit_hash(self):
        self.assertIsInstance(dax.base.dax._CWD_COMMIT, (str, type(None)), 'Unexpected type for CWD commit hash')

        # Discover repo path
        path = pygit2.discover_repository(os.getcwd())

        if path is None:
            # CWD is not in a git repository at this moment, skipping test
            self.skipTest('CWD currently not in a git repo')

        # Test if CWD commit hash was loaded
        self.assertIsNotNone(dax.base.dax._CWD_COMMIT, 'CWD commit hash was not loaded')
        self.assertIsInstance(dax.base.dax._CWD_COMMIT, str, 'Unexpected type for CWD commit hash')
        self.assertEqual(dax.base.dax._CWD_COMMIT, str(pygit2.Repository(path).head.target.hex),
                         'CWD commit hash did not match reference')


class DaxNameRegistryTestCase(unittest.TestCase):

    def test_module(self):
        # Test system
        s = _TestSystem(get_manager_or_parent(_device_db))
        # Registry
        r = s.registry

        # Test with no modules
        with self.assertRaises(KeyError, msg='Get non-existing module did not raise'):
            r.get_module('not_existing_key')
        with self.assertRaises(KeyError, msg='Find non-existing module did not raise'):
            r.find_module(_TestModule)
        self.assertDictEqual(r.search_modules(_TestModule), {}, 'Search result dict incorrect')

        # Test with one module
        t0 = _TestModule(s, 'test_module')
        self.assertIs(r.get_module(t0.get_system_key()), t0, 'Returned module does not match expected module')
        with self.assertRaises(TypeError, msg='Type check in get_module() did not raise'):
            r.get_module(t0.get_system_key(), _TestModuleChild)
        self.assertIs(r.find_module(_TestModule), t0, 'Did not find the expected module')
        self.assertIs(r.find_module(DaxModule), t0, 'Did not find the expected module')
        with self.assertRaises(KeyError, msg='Search non-existing module did not raise'):
            r.find_module(_TestModuleChild)
        self.assertListEqual(r.get_module_key_list(), [m.get_system_key() for m in [s, t0]],
                             'Module key list incorrect')
        self.assertSetEqual(set(r.get_module_list()), {s, t0}, 'Module list incorrect')
        with self.assertRaises(dax.base.exceptions.NonUniqueRegistrationError,
                               msg='Adding module twice did not raise'):
            r.add_module(t0)
        with self.assertRaises(LookupError, msg='Adding module twice did not raise a LookupError'):
            r.add_module(t0)

        # Test with two modules
        t1 = _TestModuleChild(s, 'test_module_child')
        self.assertIs(r.get_module(t1.get_system_key()), t1, 'Returned module does not match expected module')
        self.assertIs(r.get_module(t1.get_system_key(), _TestModuleChild), t1,
                      'Type check in get_module() raised unexpectedly')
        self.assertIs(r.find_module(_TestModuleChild), t1, 'Did not find expected module')
        with self.assertRaises(LookupError, msg='Non-unique search did not raise'):
            r.find_module(_TestModule)
        self.assertListEqual(r.get_module_key_list(), [m.get_system_key() for m in [s, t0, t1]],
                             'Module key list incorrect')
        self.assertSetEqual(set(r.get_module_list()), {s, t0, t1}, 'Module list incorrect')
        self.assertDictEqual(r.search_modules(_TestModule), {m.get_system_key(): m for m in [t0, t1]},
                             'Search result dict incorrect')

    def test_device(self):
        # Test system
        s = _TestSystem(get_manager_or_parent(_device_db))
        # List of core devices
        core_devices = ['core', 'core_cache', 'core_dma']
        # Registry
        r = s.registry

        # Test core devices, which should be existing
        self.assertListEqual(r.get_device_key_list(), core_devices, 'Core devices were not found in device list')
        self.assertSetEqual(r.search_devices(artiq.coredevice.core.Core), {'core'},
                            'Search devices did not returned the expected set of results')

    def test_service(self):
        # Test system
        s = _TestSystem(get_manager_or_parent(_device_db))
        s0 = _TestService(s)
        # Registry
        r = s.registry

        # Test adding the service again
        with self.assertRaises(dax.base.exceptions.NonUniqueRegistrationError,
                               msg='Double service registration did not raise'):
            r.add_service(s0)

        # Test with one service
        self.assertFalse(r.has_service('foo'), 'Non-existing service did not returned false')
        self.assertFalse(r.has_service(_TestServiceChild), 'Non-existing service did not returned false')
        self.assertTrue(r.has_service(_TestService.SERVICE_NAME), 'Did not returned true for existing service')
        self.assertTrue(r.has_service(_TestService), 'Did not returned true for existing service')
        self.assertIs(r.get_service(s0.get_name()), s0, 'Did not returned expected service')
        self.assertIs(r.get_service(_TestService), s0, 'Did not returned expected service')
        with self.assertRaises(KeyError, msg='Retrieving non-existing service did not raise'):
            r.get_service(_TestServiceChild)
        self.assertListEqual(r.get_service_key_list(), [s.get_name() for s in [s0]],
                             'List of registered service keys incorrect')
        self.assertSetEqual(set(r.get_service_list()), {s0}, 'List of registered services incorrect')

        # Test with a second service
        s1 = _TestServiceChild(s)
        self.assertTrue(r.has_service(_TestServiceChild), 'Did not returned true for existing service')
        self.assertTrue(r.has_service(_TestServiceChild.SERVICE_NAME), 'Did not returned true for existing service')
        self.assertListEqual(r.get_service_key_list(), [s.get_name() for s in [s0, s1]],
                             'List of registered service keys incorrect')
        self.assertSetEqual(set(r.get_service_list()), {s0, s1}, 'List of registered services incorrect')

    def test_interface(self):
        # Test system
        s = _TestSystem(get_manager_or_parent(_device_db))
        _TestService(s)
        # Registry
        r = s.registry

        # Confirm that interface can not be found before adding
        with self.assertRaises(KeyError, msg='Interface not available did not raise'):
            r.find_interface(_TestInterface)
        self.assertDictEqual(r.search_interfaces(_TestInterface), {},
                             'Interface not available did not return an empty dict')

        # Add and test interface features
        itf = _TestServiceChild(s)  # Class that implements the interface
        self.assertIs(r.find_interface(_TestInterface), itf, 'Find interface did not return expected object')
        self.assertDictEqual(r.search_interfaces(_TestInterface), {itf.get_system_key(): itf},
                             'Search interfaces did not return expected result')


class DaxDataStoreInfluxDbTestCase(unittest.TestCase):
    class MockDataStore(dax.base.dax.DaxDataStoreInfluxDb):
        """Data store connector that does not write but a callback instead."""

        def __init__(self, callback, *args, **kwargs):
            assert callable(callback), 'Callback must be a callable function'
            self.callback = callback
            super(DaxDataStoreInfluxDbTestCase.MockDataStore, self).__init__(*args, **kwargs)

            # List of points that reached the callback
            self.points = []

        def _get_driver(self, system: DaxSystem, key: str) -> None:
            pass  # Do not obtain the driver

        def _get_key(self, point):
            keys = [k for k in point['fields'] if k not in self._base_fields]
            if len(keys) != 1:
                raise LookupError('Could not find a unique data key in the point')
            else:
                return keys[0]

        def _write_points(self, points):
            # Filter out keys
            keys = (self._get_key(p) for p in points)
            # Add converted points to list of points
            self.points.extend((k, p['fields'][k], p['tags'].get('index')) for p, k in zip(points, keys))

            # Do not write points but do a callback instead
            self.callback(points)

    def setUp(self) -> None:
        # Callback function
        def callback(points):
            for d in points:
                # Check if the types of all field values are valid
                for value in d['fields'].values():
                    self.assertIsInstance(value, (int, str, float, bool), 'Field in point has invalid type')
                # Check if the index is correct (if existing)
                self.assertIsInstance(d['tags'].get('index', ''), str, 'Index has invalid type (expected str)')

        # Test system
        self.s = _TestSystem(get_manager_or_parent(_device_db))
        # Special data store that skips actual writing
        self.ds = self.MockDataStore(callback, self.s, 'dax_influx_db')

    def test_make_point(self):
        # Data to test against
        test_data = [
            ('k', 4),
            ('k', 0.1),
            ('k', True),
            ('k', 'value'),
            ('k.a', 7),
            ('k.b.c', 8),
            ('k.ddd', 9),
        ]

        for k, v in test_data:
            with self.subTest(k=k, v=v):
                # Test making point
                d = self.ds._make_point(k, v)

                # Split key
                split_key = k.rsplit('.', maxsplit=1)
                base = split_key[0] if len(split_key) == 2 else ''

                # Verify point object
                self.assertEqual(base, d['tags']['base'], 'Base of key does not match tag in point object')
                self.assertIn(k, d['fields'], 'Key is not an available field in the point object')
                self.assertEqual(v, d['fields'][k], 'Field value in point object is not equal to inserted value')

    def test_make_point_index(self):
        # Data to test against
        test_data = [
            ('k', 4, None),
            ('k', 0.1, 1),
            ('k', True, np.int32(5)),
            ('k', 'value', np.int64(88)),
        ]

        for k, v, i in test_data:
            with self.subTest(k=k, v=v, i=i):
                # Test making point
                d = self.ds._make_point(k, v, i)

                if i is not None:
                    # Verify point object
                    self.assertIn('index', d['tags'], 'Index is not an available tag in the point object')
                    self.assertEqual(str(i), d['tags']['index'], 'Index of point object is not equal to inserted value')
                else:
                    # Confirm index does not exist
                    self.assertNotIn('index', d['tags'], 'Index found as tag in the point object while None was given')

    def test_set(self):
        # Data to test against
        test_data = [
            ('k', 5),
            ('k', 5),
            ('k', 7.65),
            ('k', 3.55),
            ('k', 'value'),
            ('k', False),
        ]

        for count, (k, v) in enumerate(test_data, 1):
            with self.subTest(k=k, v=v):
                # Test using the callback function
                self.ds.set(k, v)
                # Test if the number of registered points matches
                self.assertEqual(count, len(self.ds.points),
                                 'Number of registered points does not match number of written elements')
                # Verify if the last point matches the data we wrote
                self.assertTupleEqual((k, v, None), self.ds.points[-1], 'Submitted data does not match written point')

    def test_set_bad(self):
        # Callback function
        def callback(*args, **kwargs):
            # This code is supposed to be unreachable
            self.fail('Bad type resulted in unwanted write (set) {} {}'.format(args, kwargs))

        # Replace callback function with a specific one for testing bad types
        self.ds.callback = callback

        # Data to test against
        test_data = [
            ('k', self),
            ('k', complex(3, 5)),
            ('k', complex(5)),
            ('k.a', self),
            ('kas', {1, 2, 6, 7}),
            ('kfd', {'i': 3}),
        ]

        for k, v in test_data:
            with self.subTest(k=k, v=v):
                with self.assertLogs(self.ds._logger, logging.WARNING):
                    # A warning will be given but no error is raised!
                    self.ds.set(k, v)

    def test_set_sequence(self):
        # Data to test against
        test_data = [
            ('k', [1, 2, 3]),
            ('k', list(range(9))),
            ('k', [str(i + 66) for i in range(5)]),
            ('k', [7.65 * i for i in range(4)]),
            ('k.a', np.arange(5)),
            ('k.a', np.full(6, 99.99)),  # Do not use np.empty() as it can result in unpredictable values
            ('k', [1, '2', True, 5.5]),
            ('k.a', range(7)),  # Ranges also work, though this is not specifically intended behavior
        ]

        for k, v in test_data:
            with self.subTest(k=k, v=v):
                # Test using the callback function
                self.ds.set(k, v)
                # Test if the number of points
                self.assertEqual(len(v), len(self.ds.points), 'Number of written points does not match sequence length')
                # Verify if the registered points are correct
                self.assertListEqual([(k, item, str(index)) for index, item in enumerate(v)],
                                     self.ds.points, 'Submitted data does not match point sequence')
                # Clear the list for the next sub-test
                self.ds.points.clear()

    def test_set_sequence_bad(self):
        # Callback function
        def callback(*args, **kwargs):
            # This code is supposed to be unreachable
            self.fail('Bad sequence resulted in unwanted write {} {}'.format(args, kwargs))

        # Replace callback function with a specific one for testing bad types
        self.ds.callback = callback

        # Data to test against
        test_data = [
            ('k', {bool(i % 2) for i in range(5)}),  # Set should not work
            ('k', {i: float(i) for i in range(5)}),  # Dict should not work
        ]

        for k, v in test_data:
            with self.subTest(k=k, v=v):
                with self.assertLogs(self.ds._logger, logging.WARNING):
                    # A warning will be given but no error is raised!
                    self.ds.set(k, v)

    def test_np_type_conversion(self):
        # Data to test against
        test_data = [
            ('k', np.int32(3)),
            ('k', np.int64(99999999)),
            ('k', np.float(4)),
        ]

        for count, (k, v) in enumerate(test_data, 1):
            with self.subTest(k=k, v=v):
                # Test using the callback function
                self.ds.set(k, v)
                # Test if the number of registered points matches
                self.assertEqual(count, len(self.ds.points),
                                 'Number of registered points does not match number of written elements')
                # Verify the last point matches the data we wrote
                self.assertTupleEqual((k, v, None), self.ds.points[-1], 'Submitted data does not match written point')

    def test_mutate(self):
        # Data to test against
        test_data = [
            ('k', 3, 5),
            ('k', 44, 23),
            ('k', 'np.float(4)', -99),  # Negative indices are valid, though this is not specifically intended behavior
        ]

        for count, (k, v, i) in enumerate(test_data, 1):
            with self.subTest(k=k, v=v, i=i):
                # Test using the callback function
                self.ds.mutate(k, i, v)
                # Test if the number of registered points matches
                self.assertEqual(count, len(self.ds.points),
                                 'Number of registered points does not match number of written elements')
                # Verify the last point matches the data we wrote
                self.assertTupleEqual((k, v, str(i)), self.ds.points[-1], 'Submitted data does not match written point')

    def test_mutate_index_np_type_conversion(self):
        # Data to test against
        test_data = [
            ('k', 3, np.int32(4)),
            ('k', 44, np.int64(-4)),
            ('k', True, np.int32(0)),
        ]

        for count, (k, v, i) in enumerate(test_data, 1):
            with self.subTest(k=k, v=v, i=i):
                # Test using the callback function
                self.ds.mutate(k, i, v)
                # Test if the number of registered points matches
                self.assertEqual(count, len(self.ds.points),
                                 'Number of registered points does not match number of written elements')
                self.assertTupleEqual((k, v, str(i)), self.ds.points[-1], 'Submitted data does not match written point')

    def test_mutate_bad(self):
        # Data to test against
        test_data = [
            ('k', complex(1, 3), 3),  # Unsupported value type
            ('k', 4, (44, 4)),  # Slicing not supported by influx
            ('k', 4, 1.1),  # Wrong index type
            ('k', 4, 1.0),  # Wrong index type
            ('k', 4, 4.0),  # Wrong index type
            ('k', 'np.float(4)', ((4, 5), (6, 7))),  # Multi-dimensional slicing not supported by influx
        ]

        for k, v, i in test_data:
            with self.subTest(k=k, v=v, i=i):
                with self.assertLogs(self.ds._logger, logging.WARNING):
                    # Test for warnings
                    self.ds.mutate(k, i, v)
                # Test if no writes happened
                self.assertEqual(0, len(self.ds.points), 'Unexpected write')

    def test_append(self):
        # Key
        key = 'k'
        # Data to test against
        test_data = [
            (key, 5),
            (key, 5),
            (key, 7.65),
            (key, 3.55),
            (key, 'value'),
            (key, False),
        ]

        # Initialize list for appending
        init_list = [4]
        self.ds.set(key, init_list)
        # Reset registered points
        self.ds.points.clear()
        # Track length of the list
        length = len(init_list)

        for count, (k, v) in enumerate(test_data, 1):
            with self.subTest(k=k, v=v):
                # Test using the callback function
                self.ds.append(k, v)
                # Test if the number of registered points matches
                self.assertEqual(count, len(self.ds.points),
                                 'Number of registered points does not match number of written elements')
                # Check written data
                self.assertTupleEqual((k, v, str(length)), self.ds.points[-1],
                                      'Submitted data does not match written point')
                # Test increment
                length += 1
                self.assertEqual(self.ds._index_table[key], length, 'Cached index was not updated correctly')

    def test_append_bad(self):
        # Callback function
        def callback(*args, **kwargs):
            # This code is supposed to be unreachable
            self.fail('Bad type resulted in unwanted write (append) {} {}'.format(args, kwargs))

        # Key
        key = 'k'
        # Data to test against
        test_data = [
            (key, self),
            (key, complex(3, 5)),
            (key, [2, 7, 4]),  # Can not append a list, only simple values
            (key, complex(5)),
        ]

        # Initialize list for appending
        self.ds.set(key, [4])

        # Replace callback function with a specific one for testing bad types
        self.ds.callback = callback

        for k, v in test_data:
            with self.subTest(k=k, v=v):
                with self.assertLogs(self.ds._logger, logging.WARNING):
                    # A warning will be given but no error is raised!
                    self.ds.append(k, v)

    def test_append_not_cached(self):
        # Callback function
        def callback(*args, **kwargs):
            # This code is supposed to be unreachable
            self.fail('Not-cached sequence append resulted in unexpected write {} {}'.format(args, kwargs))

        # Replace callback function with a specific one for testing bad types
        self.ds.callback = callback

        # Key
        key = 'k'
        # Data to test against
        test_data = [
            (key, 1),
            (key, 'complex(3, 5)'),
            (key, 5.5),
        ]

        for k, v in test_data:
            with self.subTest(k=k, v=v):
                with self.assertLogs(self.ds._logger, logging.WARNING):
                    # A warning will be given but no error is raised!
                    self.ds.append(k, v)

    def test_append_cache(self):
        # Data to test against
        test_data = [
            ('k.b.c', []),
            ('k.a', list()),
            ('fds.aaa', np.zeros(0)),
            ('kfh', range(0)),
            ('ka.a', [5, 7, 3, 2]),
            ('kh.rt', np.zeros(4)),
            ('kee', range(6)),
        ]

        # Test empty cache
        self.assertDictEqual(self.ds._index_table, {}, 'Expected empty index cache table')

        for k, v in test_data:
            with self.subTest(k=k, v=v):
                # Set
                self.ds.set(k, v)
                # Check if length is cached
                self.assertIn(k, self.ds._index_table, 'Expected entry in cache')
                self.assertEqual(len(v), self.ds._index_table[k], 'Cached length does not match actual list length')

    def test_empty_list(self):
        # Callback function
        def callback(*args, **kwargs):
            # This code is supposed to be unreachable
            self.fail('Set empty list resulted in unexpected write {} {}'.format(args, kwargs))

        # Replace callback function with a specific one for testing bad types
        self.ds.callback = callback

        # Data to test against
        test_data = [
            ('k', []),
            ('k.a', list()),
            ('k', np.zeros(0)),
            ('k', range(0)),
        ]

        for k, v in test_data:
            with self.subTest(k=k, v=v):
                # Store of empty sequence should never result in a write
                self.ds.set(k, v)
                # Check if length is cached
                self.assertIn(k, self.ds._index_table, 'Expected entry in cache')
                self.assertEqual(0, self.ds._index_table[k], 'Cached length does not match actual list length')


class DaxModuleBaseTestCase(unittest.TestCase):
    """Tests _DaxHasSystemBase, DaxModuleBase, DaxModule, and DaxSystem.

    The four mentioned modules are highly related and overlap mostly.
    Therefore they are all tested mutually.
    """

    def test_system_build(self):
        # A system that does not call super() in build()
        class BadTestSystem(_TestSystem):
            def build(self):
                pass  # No call to super(), which is bad

        # Test if an error occurs when super() is not called in build()
        with self.assertRaises(AttributeError, msg='Not calling super.build() in user system did not raise'):
            BadTestSystem(get_manager_or_parent(_device_db))

    def test_system_kernel_invariants(self):
        s = _TestSystem(get_manager_or_parent(_device_db))

        # No kernel invariants attribute yet
        self.assertTrue(hasattr(s, 'kernel_invariants'), 'Default kernel invariants not found')

        # Update kernel invariants
        invariant = 'foo'
        s.update_kernel_invariants(invariant)
        self.assertIn(invariant, s.kernel_invariants, 'Kernel invariants update not successful')

    def test_system_id(self):
        # Test if an error is raised when no ID is given to a system
        with self.assertRaises(AssertionError, msg='Not providing system id did not raise'):
            DaxSystem(get_manager_or_parent(_device_db))

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
                BadSystem(get_manager_or_parent(_device_db))

    def test_system_ver(self):
        class TestSystemNoVer(DaxSystem):
            SYS_ID = 'unittest_system'

        # Test if an error is raised when no version is given to a system
        with self.assertRaises(AssertionError, msg='Not providing system version did not raise'):
            TestSystemNoVer(get_manager_or_parent(_device_db))

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
                BadSystem(get_manager_or_parent(_device_db))

        # System with version 0, which is fine
        class TestSystemVerZero(DaxSystem):
            SYS_ID = 'unittest_system'
            SYS_VER = 0

        # Test if it is possible to create a system with version 0
        TestSystemVerZero(get_manager_or_parent(_device_db))

    def test_init(self):
        manager_or_parent = get_manager_or_parent(_device_db)
        s = _TestSystem(get_manager_or_parent(_device_db))

        # Check constructor
        self.assertIsNotNone(s, 'Could not create DaxSystem')
        self.assertIsNotNone(_TestModule(s, 'module_name'), 'Could not create a test module')
        with self.assertRaises(ValueError, msg='Invalid module name did not raise'):
            _TestModule(s, 'wrong!')
        with self.assertRaises(ValueError, msg='Invalid module name did not raise'):
            _TestModule(s, 'this.is.bad')
        with self.assertRaises(TypeError, msg='Providing non-DaxModuleBase parent to new module did not raise'):
            _TestModule(manager_or_parent, 'module_name')

    def test_module_registration(self):
        # Check register
        s = _TestSystem(get_manager_or_parent(_device_db))
        t = _TestModule(s, 'module_name')
        self.assertDictEqual(s.registry._modules, {m.get_system_key(): m for m in [s, t]},
                             'Dict with registered modules does not match expected content')

    def test_name(self):
        s = _TestSystem(get_manager_or_parent(_device_db))

        self.assertEqual(s.get_name(), _TestSystem.SYS_NAME, 'Returned name did not match expected name')

    def test_system_key(self):
        s = _TestSystem(get_manager_or_parent(_device_db))

        self.assertEqual(s.get_system_key(), _TestSystem.SYS_NAME, 'Returned key did not match expected key')

    def test_system_key_arguments(self):
        s = _TestSystem(get_manager_or_parent(_device_db))

        self.assertEqual(s.get_system_key('a', 'b'), '.'.join([_TestSystem.SYS_NAME, 'a', 'b']),
                         'Returned key did not match expected key based on multiple components')
        k = 'string_as_key_list'
        self.assertEqual(s.get_system_key(*k), '.'.join([_TestSystem.SYS_NAME, *k]),
                         'Returned key did not match expected key based on multiple components')

        n = 'test_module_name'
        t = _TestModule(s, n)
        self.assertEqual(t.get_system_key(), '.'.join([_TestSystem.SYS_NAME, n]),
                         'Key created for nested module did not match expected key')
        some_key = 'some_key'
        self.assertEqual(t.get_system_key(some_key), '.'.join([_TestSystem.SYS_NAME, n, some_key]),
                         'System key creation derived from current module key failed')

    def test_bad_system_key_arguments(self):
        s = _TestSystem(get_manager_or_parent(_device_db))

        with self.assertRaises(ValueError, msg='Creating bad system key did not raise'):
            s.get_system_key('bad,key')
        with self.assertRaises(ValueError, msg='Creating bad system key did not raise'):
            s.get_system_key('good_key', 'bad,key')
        with self.assertRaises(AssertionError, msg='Creating system key with wrong key input did not raise'):
            # Intentionally wrong argument type, disabling inspection
            # noinspection PyTypeChecker
            s.get_system_key(1)

    def test_setattr_device(self):
        s = _TestSystem(get_manager_or_parent(_device_db))

        self.assertIsNone(s.setattr_device('ttl0'), 'setattr_device() did not return None')
        self.assertTrue(hasattr(s, 'ttl0'), 'setattr_device() did not set the attribute correctly')
        self.assertIsNone(s.setattr_device('alias_2', 'foo'), 'setattr_device() with attribute name failed')
        self.assertTrue(hasattr(s, 'foo'), 'setattr_device() with attribute name did not set attribute correctly')

    def test_get_device(self):
        # Test system
        s = _TestSystem(get_manager_or_parent(_device_db))
        # List of core devices
        core_devices = ['core', 'core_cache', 'core_dma']
        # Registry
        r = s.registry

        # Test getting devices
        self.assertIsNotNone(s.get_device('ttl0'), 'Device request with unique key failed')
        self.assertIsNotNone(s.get_device('alias_2'), 'Device request with alias failed')
        self.assertIn('ttl1', r.get_device_key_list(),
                      'Device registration did not found correct unique key for device alias')
        self.assertListEqual(r.get_device_key_list(), core_devices + ['ttl0', 'ttl1'], 'Device key list incorrect')
        with self.assertRaises(dax.base.exceptions.NonUniqueRegistrationError,
                               msg='Double device registration did not raise when registered by unique name and alias'):
            s.get_device('alias_1')

    def test_get_device_type_check(self):
        s = _TestSystem(get_manager_or_parent(_device_db))

        with self.assertRaises(TypeError, msg='get_device() type check did not raise'):
            s.get_device('ttl1', artiq.coredevice.edge_counter.EdgeCounter)

        # Correct type, should not raise
        self.assertIsNotNone(s.get_device('ttl1', artiq.coredevice.ttl.TTLOut),
                             'get_device() type check raised unexpectedly')

    def test_search_devices(self):
        s = _TestSystem(get_manager_or_parent(_device_db))
        r = s.registry

        # Add devices
        self.assertIsNotNone(s.get_device('ttl0'), 'Device request with unique key failed')
        self.assertIsNotNone(s.get_device('alias_2'), 'Device request with alias failed')

        # Test if registry returns correct result
        self.assertSetEqual(r.search_devices(artiq.coredevice.ttl.TTLInOut), {'ttl0'},
                            'Search devices did not returned expected result')
        self.assertSetEqual(r.search_devices(artiq.coredevice.edge_counter.EdgeCounter), set(),
                            'Search devices did not returned expected result')
        self.assertSetEqual(r.search_devices((artiq.coredevice.ttl.TTLInOut, artiq.coredevice.ttl.TTLOut)),
                            {'ttl0', 'ttl1'}, 'Search devices did not returned expected result')

    def test_dataset(self):
        s = _TestSystem(get_manager_or_parent(_device_db))

        key = 'key1'
        value = [11, 12, 13]
        self.assertListEqual(s.get_dataset_sys(key, default=value), value,
                             'get_dataset_sys() did not returned the provided default value')
        self.assertListEqual(s.get_dataset_sys(key), value,
                             'get_dataset_sys() did not write the default value to the dataset')

    def test_setattr_dataset(self):
        s = _TestSystem(get_manager_or_parent(_device_db))

        key = 'key3'
        self.assertIsNone(s.setattr_dataset_sys(key, 10), 'setattr_dataset_sys() failed')
        self.assertTrue(hasattr(s, key), 'setattr_dataset_sys() did not set the attribute correctly')
        self.assertEqual(getattr(s, key), 10, 'Returned system dataset value does not match expected result')
        self.assertIn(key, s.kernel_invariants,
                      'setattr_dataset_sys() did not added the attribute to kernel_invariants by default')

        key = 'key5'
        s.set_dataset_sys(key, 5)
        self.assertIsNone(s.setattr_dataset_sys(key), 'setattr_dataset_sys() failed')
        self.assertTrue(hasattr(s, key), 'setattr_dataset_sys() did not set the attribute correctly')
        self.assertEqual(getattr(s, key), 5, 'Returned system dataset value does not match expected result')
        self.assertIn(key, s.kernel_invariants,
                      'setattr_dataset_sys() did not added the attribute to kernel_invariants by default')

        key = 'key4'
        self.assertIsNone(s.setattr_dataset_sys(key), 'setattr_dataset_sys() failed')
        self.assertFalse(hasattr(s, key), 'setattr_dataset_sys() set the attribute while it should not')
        self.assertNotIn(key, s.kernel_invariants,
                         'setattr_dataset_sys() did added the attribute to kernel_invariants while it should not')

    @unittest.expectedFailure
    def test_dataset_append(self):
        s = _TestSystem(get_manager_or_parent(_device_db))

        key = 'key2'
        self.assertIsNone(s.set_dataset_sys(key, []), 'Setting new system dataset failed')
        self.assertListEqual(s.get_dataset_sys(key), [],
                             'Returned system dataset value does not match expected result')
        self.assertIsNone(s.append_to_dataset_sys(key, 1), 'Appending to system dataset failed')
        self.assertListEqual(s.get_dataset_sys(key), [1], 'Appending to system dataset has incorrect behavior')
        # NOTE: This test fails for unknown reasons (ARTIQ library) while real-life tests show correct behavior

    def test_dataset_append_nonempty(self):
        s = _TestSystem(get_manager_or_parent(_device_db))

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
        s = _TestSystem(get_manager_or_parent(_device_db))

        key = 'key2'
        self.assertIsNone(s.set_dataset_sys(key, [0, 0, 0, 0]), 'Setting new system dataset failed')
        self.assertListEqual(s.get_dataset_sys(key), [0, 0, 0, 0],
                             'Returned system dataset value does not match expected result')
        self.assertIsNone(s.mutate_dataset_sys(key, 1, 9), 'Mutating system dataset failed')
        self.assertIsNone(s.mutate_dataset_sys(key, 3, 99), 'Mutating system dataset failed')
        self.assertListEqual(s.get_dataset_sys(key), [0, 9, 0, 99], 'Mutating system dataset has incorrect behavior')

    def test_identifier(self):
        s = _TestSystem(get_manager_or_parent(_device_db))
        self.assertTrue(isinstance(s.get_identifier(), str), 'get_identifier() did not returned a string')

    def test_repr(self):
        s = _TestSystem(get_manager_or_parent(_device_db))
        r = repr(s)
        self.assertTrue(isinstance(r, str), 'repr() did not returned a string')
        self.assertIn(s.get_system_key(), r)


class DaxSystemTestCase(unittest.TestCase):
    class InitTestSystem(DaxSystem):
        SYS_ID = 'unittest_init_system'
        SYS_VER = 0

        def build(self, test_case=None, *args, **kwargs) -> None:  # Default value required for type checking
            super(DaxSystemTestCase.InitTestSystem, self).build(*args, **kwargs)
            assert test_case is not None

            # Store reference to the test case
            self._test_case = test_case

            # Required for type checking
            self.children = getattr(self, 'children', [])

        def init(self) -> None:
            # Check if the call order is correct
            self._test_case.assertEqual(len(self.children), self._test_case.init_count,
                                        'System init was not called last')
            self._test_case.assertEqual(0, self._test_case.post_init_count,
                                        'Some post_init was called before the last init call')
            # Increment call counter
            self._test_case.init_count += 1

        def post_init(self) -> None:
            # Check if the call order is correct
            self._test_case.assertEqual(len(self.children) + 1, self._test_case.init_count,
                                        'Init count changed unexpectedly')
            self._test_case.assertEqual(len(self.children), self._test_case.post_init_count,
                                        'System post_init was not called last')
            # Increment call counter
            self._test_case.post_init_count += 1

    class InitTestModule(DaxModule):
        def build(self, index, test_case):
            # Remember own index
            self._index = index
            self._test_case = test_case

        def init(self) -> None:
            # Check if the call order is correct
            self._test_case.assertEqual(self._index, self._test_case.init_count,
                                        'Module init was not called in expected order')
            # Increment call counter
            self._test_case.init_count += 1

        def post_init(self) -> None:
            # Check if the call order is correct
            self._test_case.assertEqual(self._index, self._test_case.post_init_count,
                                        'Module init was not called in expected order')
            # Increment call counter
            self._test_case.post_init_count += 1

    def setUp(self) -> None:
        # Counters to track calls
        self.init_count = 0
        self.post_init_count = 0
        self.num_modules = 10

        # Assemble system and modules
        self.system = self.InitTestSystem(get_manager_or_parent(_device_db), self)
        for i in range(self.num_modules):
            self.InitTestModule(self.system, 'module_{}'.format(i), i, self)

    def test_dax_init(self):
        # Call DAX init
        self.system.dax_init()

        # Verify if the counters have the expected values
        self.assertEqual(self.init_count, self.num_modules + 1, 'Number of init calls does not match expected number')
        self.assertEqual(self.post_init_count, self.num_modules + 1,
                         'Number of post_init calls does not match expected number')

    def test_build_super(self):
        class SomeBaseClass(HasEnvironment):
            # noinspection PyUnusedLocal
            def build(self, *args, **kwargs):
                # super.build() here calls HasEnvironment, which is empty
                # Set flag
                self.base_build_was_called = True

        # SomeBaseClass also inherits from HasEnvironment, making it higher in the MRO than HasEnvironment itself
        class SuperTestSystem(_TestSystem, SomeBaseClass):
            def build(self, *args, **kwargs):
                # Call super
                super(SuperTestSystem, self).build(*args, **kwargs)
                # Set flag
                self.system_build_was_called = True

            def run(self):
                pass

        # Create system, which will call build()
        system = SuperTestSystem(get_manager_or_parent(_device_db))

        # Test if build of the super class and system class was called
        self.assertTrue(hasattr(system, 'base_build_was_called'), 'System did not called build() of super class')
        self.assertTrue(hasattr(system, 'system_build_was_called'), 'System did not called build() of itself')

        # Inheriting the other way, build of SomeBaseClass will be called first, which does not call super
        class SuperTestSystem(SomeBaseClass, _TestSystem):
            def run(self):
                pass

        with self.assertRaises(AttributeError, msg='build() of system was called unexpectedly'):
            # Create system, which will call build()
            # System build() will not be called, which raises an exception
            SuperTestSystem(get_manager_or_parent(_device_db))


class DaxServiceTestCase(unittest.TestCase):

    def test_init(self):
        s = _TestSystem(get_manager_or_parent(_device_db))

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
        self.assertIn(service, s.registry.get_service_list(), 'Could not find service in registry')

        class DuplicateNameService(NoNameService):
            SERVICE_NAME = 'service_name'

        with self.assertRaises(dax.base.exceptions.NonUniqueRegistrationError,
                               msg='Duplicate service name registration did not raise'):
            DuplicateNameService(s)

        class GoodNameService2(NoNameService):
            SERVICE_NAME = 'service_name_2'

        self.assertTrue(GoodNameService2(service), 'Could not create new service with other service as parent')


class DaxClientTestCase(unittest.TestCase):

    def test_not_decorated(self):
        class Client(DaxClient):
            def run(self) -> None:
                pass

        with self.assertRaises(TypeError, msg='Using client without client factory decorator did not raise'):
            Client(get_manager_or_parent(_device_db))

    def test_load_super(self):
        @dax_client_factory
        class Client(DaxClient):
            def init(self) -> None:
                self.is_initialized = True

            def run(self) -> None:
                pass

        class ImplementableClient(Client(_TestSystem)):
            pass

        # Disabled one inspection, inspection does not handle the decorator correctly
        # noinspection PyArgumentList
        c = ImplementableClient(get_manager_or_parent(_device_db))
        c.run()  # Is supposed to call the dax_init() function which will call the init() function of the client

        self.assertTrue(hasattr(c, 'is_initialized'), 'DAX system of client was not initialized correctly')

    def test_disable_dax_init(self):
        @dax_client_factory
        class Client(DaxClient):
            DAX_INIT = False

            def init(self) -> None:
                self.is_initialized = True

            def run(self) -> None:
                pass

        class ImplementableClient(Client(_TestSystem)):
            pass

        # Disabled one inspection, inspection does not handle the decorator correctly
        # noinspection PyArgumentList
        c = ImplementableClient(get_manager_or_parent(_device_db))
        c.run()  # Is not supposed to call the dax_init() function

        self.assertFalse(hasattr(c, 'is_initialized'), 'DAX system of client was initialized unexpectedly')


if __name__ == '__main__':
    unittest.main()
