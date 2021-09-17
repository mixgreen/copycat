import unittest
import numpy as np
import logging
import collections.abc
import typing
import itertools
from unittest.mock import Mock, call

from artiq.experiment import HasEnvironment, Experiment
import artiq.coredevice.edge_counter
import artiq.coredevice.ttl  # type: ignore[import]
import artiq.coredevice.core
from artiq import __version__ as _artiq_version

from dax.base.system import *
import dax.base.system
import dax.base.exceptions
import dax.base.interface
import dax.util.git
from dax.util.artiq import get_managers
from dax import __version__ as _dax_version

import test.util.logging_test
import test.helpers

"""Device DB for testing"""

_DEVICE_DB = {
    # Core device
    'core': {
        'type': 'local',
        'module': 'artiq.coredevice.core',
        'class': 'Core',
        'arguments': {'host': None, 'ref_period': 1e-9}
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


class _TestSystemWithControllers(DaxSystem):
    SYS_ID = 'unittest_system'
    SYS_VER = 0

    def __init__(self, *args, **kwargs):
        self._data_store = Mock(spec=dax.base.system.DaxDataStore)
        super(_TestSystemWithControllers, self).__init__(*args, **kwargs)

    @property
    def data_store(self):
        return self._data_store


class _TestSystem(_TestSystemWithControllers):
    CORE_LOG_KEY = None
    DAX_INFLUX_DB_KEY = None


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

    def setUp(self) -> None:
        self.managers = get_managers(_DEVICE_DB)

    def tearDown(self) -> None:
        # Close managers
        self.managers.close()

    def test_valid_name(self):
        for n in ['foo', '_0foo', '_', '0', '_foo', 'FOO_', '0_foo']:
            # Test valid names
            self.assertTrue(dax.base.system._is_valid_name(n))

    def test_invalid_name(self):
        for n in ['', 'foo()', 'foo.bar', 'foo/', 'foo*', 'foo,', 'FOO+', 'foo-bar', 'foo/bar']:
            # Test illegal names
            self.assertFalse(dax.base.system._is_valid_name(n))

    def test_valid_key(self):
        for k in ['foo', '_0foo', '_', '0', 'foo.bar', 'foo.bar.baz', '_.0.A', 'foo0._bar']:
            # Test valid keys
            self.assertTrue(dax.base.system._is_valid_key(k))

    def test_invalid_key(self):
        for k in ['', 'foo()', 'foo,bar', 'foo/', '.foo', 'bar.', 'foo.bar.baz.']:
            # Test illegal keys
            self.assertFalse(dax.base.system._is_valid_key(k))

    def test_unique_device_key(self):
        # Test system and device DB
        s = _TestSystem(self.managers)

        # Test against various keys
        self.assertEqual(s.registry.get_unique_device_key('ttl0'), 'ttl0',
                         'Unique device key not returned correctly')
        self.assertEqual(s.registry.get_unique_device_key('alias_0'), 'ttl1',
                         'Alias key key does not return correct unique key')
        self.assertEqual(s.registry.get_unique_device_key('alias_1'), 'ttl1',
                         'Multi-alias key does not return correct unique key')
        self.assertEqual(s.registry.get_unique_device_key('alias_2'), 'ttl1',
                         'Multi-alias key does not return correct unique key')

    def test_looped_device_key(self):
        # Test system and device DB
        s = _TestSystem(self.managers)

        # Test looped alias
        loop_aliases = ['loop_alias_1', 'loop_alias_4']
        for key in loop_aliases:
            with self.assertRaises(LookupError, msg='Looped key alias did not raise'):
                s.registry.get_unique_device_key(key)

    def test_unavailable_device_key(self):
        # Test system and device DB
        s = _TestSystem(self.managers)

        # Test non-existing keys
        loop_aliases = ['not_existing_key_0', 'not_existing_key_1', 'dead_alias_2']
        for key in loop_aliases:
            with self.assertRaises(KeyError, msg='Non-existing key did not raise'):
                s.registry.get_unique_device_key(key)

    def test_virtual_device_key(self):
        # Test system and device DB
        s = _TestSystem(self.managers)
        # Test virtual devices
        virtual_devices = {'scheduler', 'ccb'}
        self.assertSetEqual(virtual_devices, dax.base.system._ARTIQ_VIRTUAL_DEVICES,
                            'List of virtual devices in test does not match DAX base virtual device list')
        for k in virtual_devices:
            self.assertEqual(s.registry.get_unique_device_key(k), k, 'Virtual device key not returned correctly')

    def test_ndarray_isinstance_sequence(self):
        # See https://github.com/numpy/numpy/issues/2776 for more information
        a = np.zeros(4)
        self.assertIsInstance(a, collections.abc.Sequence, 'numpy ndarray is not considered an abstract sequence')

    def test_async_rpc_logger(self):
        # Test if logger is async rpc and kernel invariant
        s = _TestSystem(self.managers)
        self.assertTrue(test.util.logging_test.is_rpc_logger(s.logger))
        self.assertIn('logger', s.kernel_invariants)


class DaxNameRegistryTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.managers = get_managers(_DEVICE_DB)

    def tearDown(self) -> None:
        # Close managers
        self.managers.close()

    def test_module(self):
        # Test system
        s = _TestSystem(self.managers)
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
        s = _TestSystem(self.managers)
        # List of core devices
        core_devices = ['core', 'core_cache', 'core_dma']
        # Dict with device keys and parents
        device_parents = {k: s for k in core_devices}
        # Registry
        r = s.registry

        # Test core devices, which should be existing
        self.assertListEqual(r.get_device_key_list(), core_devices, 'Core devices were not found in device list')
        self.assertSetEqual(r.search_devices(artiq.coredevice.core.Core), {'core'},
                            'Search devices did not returned the expected set of results')
        self.assertDictEqual(r.get_device_parents(), device_parents,
                             'Device parents dict did not match expected result')

    def test_service(self):
        # Test system
        s = _TestSystem(self.managers)
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
        s = _TestSystem(self.managers)
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

    def test_device_db(self):
        # Test system
        s = _TestSystem(self.managers)
        # Verify cached device DB
        self.assertEqual(s.registry.device_db, s.get_device_db())

    def test_is_independent(self):
        # Test system
        s = _TestSystem(self.managers)
        # Registry
        r = s.registry
        # Add modules and services
        m0 = _TestModule(s, 'm0')
        m00 = _TestModule(m0, 'm')
        m000 = _TestModule(m00, 'm')
        m1 = _TestModule(s, 'm1')
        s0 = _TestService(s)
        component_list = [s, m0, m00, m000, m1, s0]

        zero_components = [((), True)]  # Zero components is always independent
        one_component = [((c,), True) for c in component_list]  # Single component is always independent
        sys_component = [((s, c), False) for c in component_list]  # Nothing is independent with the system
        service_component = [((s0, c), False) for c in component_list]  # Nothing is independent with a service
        m1_component = [((m1, c), True) for c in [m0, m00, m000]]  # m1 is independent from m0*
        m0_mix = [(c, False) for c in itertools.product([m0, m00, m000], [m0, m00, m000])]

        for components, ref in itertools.chain(
                zero_components, one_component, sys_component, service_component, m1_component, m0_mix):
            with self.subTest(components=components, ref=ref):
                self.assertEqual(r.is_independent(*components), ref)
                self.assertEqual(r.is_independent(*reversed(components)), ref)

    def test_is_independent_exceptions(self):
        # Test systems
        s0 = _TestSystem(self.managers)
        s1 = _TestSystem(self.managers)
        # Add modules and services
        m0 = _TestModule(s0, 'm0')
        m1 = _TestModule(s1, 'm1')

        for s, m in zip([s0, s1], [m1, m0]):
            with self.assertRaises(ValueError):
                s.registry.is_independent(m)


class DaxDataStoreInfluxDbTestCase(unittest.TestCase):
    class MockDataStore(dax.base.system.DaxDataStoreInfluxDb):
        """Data store connector that does not write but a callback instead."""

        def __init__(self, callback, *args, **kwargs):
            assert callable(callback), 'Callback must be a callable function'
            self.callback = callback
            super(DaxDataStoreInfluxDbTestCase.MockDataStore, self).__init__(*args, **kwargs)

            # List of points that reached the callback
            self.points = []

        def _get_driver(self, environment: artiq.experiment.HasEnvironment, key: str) -> None:
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
        self.managers = get_managers(_DEVICE_DB)
        self.s = _TestSystemWithControllers(self.managers)
        # Special data store that skips actual writing
        self.ds = self.MockDataStore(callback, self.s, type(self.s))

    def tearDown(self) -> None:
        # Close managers
        self.managers.close()

    def test_base_fields(self):
        # Test making point
        d = self.ds._make_point('k', 4)

        # Verify some base fields
        fields = {
            'artiq_version': _artiq_version,
            'dax_version': _dax_version,
        }
        if dax.util.git.in_repository():
            fields['git_commit'] = dax.util.git.get_repository_info().commit
        for k, v in fields.items():
            self.assertEqual(d['fields'][k], v)

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
            self.fail(f'Bad type resulted in unwanted write (set) {args} {kwargs}')

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

    def test_set_array(self):
        # Data to test against
        test_data = [
            ('k', np.zeros((2, 2))),
            ('k', np.ones((4, 2))),
            ('k.a', np.full((3, 6), 99.99)),  # Do not use np.empty() as it can result in unpredictable values
        ]

        for k, v in test_data:
            with self.subTest(k=k, v=v):
                # Test using the callback function
                self.ds.set(k, v)
                # Test if the number of points
                self.assertEqual(v.size, len(self.ds.points), 'Number of written points does not match sequence length')
                # Verify if the registered points are correct
                self.assertListEqual([(k, item, str(index)) for index, item in np.ndenumerate(v)],
                                     self.ds.points, 'Submitted data does not match point sequence')
                # Clear the list for the next sub-test
                self.ds.points.clear()

    def test_set_sequence_bad(self):
        # Callback function
        def callback(*args, **kwargs):
            # This code is supposed to be unreachable
            self.fail(f'Bad sequence resulted in unwanted write {args} {kwargs}')

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
            ('k', np.float_(5)),
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
            ('k', True, 5),
            ('k', 44, 23),
            ('k', 44.4, 3),
            ('k', np.int32(4), 3),
            ('k', np.int64(4), 3),
            ('k', np.float_(44.4), 3),
            ('k', 'foo', -99),  # Negative indices are valid, though this is not specifically intended behavior
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
            ('k', 4, np.float_(4)),  # Wrong index type
            ('k', 'foo', ((4, 5), (6, 7))),  # Multi-dimensional slicing not supported by influx
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
            self.fail(f'Bad type resulted in unwanted write (append) {args} {kwargs}')

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
            self.fail(f'Not-cached sequence append resulted in unexpected write {args} {kwargs}')

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
            self.fail(f'Set empty list resulted in unexpected write {args} {kwargs}')

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


class DaxBaseTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.managers = get_managers()

    def tearDown(self) -> None:
        # Close managers
        self.managers.close()

    def test_abstract_class(self):
        with self.assertRaises(TypeError, msg='Abstract class instantiation did not raise'):
            dax.base.system.DaxBase(None)

    def test_kernel_invariants(self):
        class Base(dax.base.system.DaxBase):
            def get_identifier(self) -> str:
                return 'identifier'

        b = Base(self.managers)
        base_kernel_invariants = {'logger'}
        self.assertSetEqual(b.kernel_invariants, base_kernel_invariants)

        # Test kernel invariant presence (before adding additional kernel invariants)
        test.helpers.test_kernel_invariants(self, b)

        keys = {'foo', 'bar', 'foobar'}
        b.update_kernel_invariants(*keys)
        self.assertSetEqual(b.kernel_invariants, keys | base_kernel_invariants)


class DaxHasKeyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.managers = get_managers()

    def tearDown(self) -> None:
        # Close managers
        self.managers.close()

    def test_constructor(self):
        with self.assertRaises(ValueError, msg='key not ending in name did not raise'):
            dax.base.system.DaxHasKey(self.managers, name='name', system_key='valid_key')

        with self.assertRaises(ValueError, msg='invalid name did not raise'):
            dax.base.system.DaxHasKey(self.managers, name='valid.key', system_key='valid.key')

    def test_key_attributes(self):
        with self.assertRaises(AttributeError, msg='Missing key attributes did not raise'):
            dax.base.system.DaxHasKey(self.managers, name='name', system_key='name')

        class HasKeyParent(dax.base.system.DaxHasKey):
            _data_store: dax.base.system.DaxDataStore = dax.base.system.DaxDataStore()

            @property
            def data_store(self) -> dax.base.system.DaxDataStore:
                return self._data_store

        class HasKey(dax.base.system.DaxHasKey):
            def build(self, *, parent):
                self._take_parent_key_attributes(parent)

        name = 'name'
        key = 'key.name'
        parent_ = HasKeyParent(self.managers, name=name, system_key=key)
        child = HasKey(parent_, name=name, system_key=key, parent=parent_)

        # Test kernel invariants
        test.helpers.test_kernel_invariants(self, parent_)

        for hk in [parent_, child]:
            with self.subTest(has_key_object=hk):
                self.assertIn('data_store', hk.kernel_invariants)
                self.assertEqual(hk.get_name(), name)
                self.assertEqual(hk.get_system_key(), key)
                self.assertTrue(hk.get_system_key('foo').endswith('.foo'))

    def test_has_attribute(self):
        class HasKeyParent(dax.base.system.DaxHasKey):
            _data_store: dax.base.system.DaxDataStore = dax.base.system.DaxDataStore()

            @property
            def data_store(self) -> dax.base.system.DaxDataStore:
                return self._data_store

        name = 'name'
        key = 'key.name'
        hk = HasKeyParent(self.managers, name=name, system_key=key)

        self.assertTrue(hk.hasattr('kernel_invariants'))
        self.assertFalse(hk.hasattr('foo'))
        self.assertFalse(hk.hasattr('kernel_invariants', 'foo'))

        key = 'bar'
        hk.setattr_dataset(key, 0)
        self.assertTrue(hk.hasattr(key))

        key = 'foobar'
        hk.setattr_dataset_sys(key)
        self.assertFalse(hk.hasattr(key))


class DaxHasSystemTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.managers = get_managers()

    def tearDown(self) -> None:
        # Close managers
        self.managers.close()

    def test_core_attributes(self):
        class HasSystemBase(dax.base.system.DaxHasSystem):

            def init(self) -> None:
                pass

            def post_init(self) -> None:
                pass

        with self.assertRaises(AttributeError, msg='Missing core attributes did not raise'):
            HasSystemBase(self.managers, name='name', system_key='name')

        class HasSystemParent(HasSystemBase):
            _data_store: dax.base.system.DaxDataStore = dax.base.system.DaxDataStore()
            _registry: dax.base.system.DaxNameRegistry = dax.base.system.DaxNameRegistry(
                _TestSystem(self.managers))

            def init(self) -> None:
                pass

            def post_init(self) -> None:
                pass

            @property
            def data_store(self) -> dax.base.system.DaxDataStore:
                return self._data_store

            @property
            def registry(self) -> dax.base.system.DaxNameRegistry:
                return self._registry

            @property
            def core(self):
                return None

            @property
            def core_cache(self):
                return None

            @property
            def core_dma(self):
                return None

        class HasSystem(HasSystemBase):
            def init(self) -> None:
                pass

            def post_init(self) -> None:
                pass

            def build(self, *, parent):
                self._take_parent_key_attributes(parent)
                self._take_parent_core_attributes(parent)

        name = 'name'
        key = 'key.name'
        parent_ = HasSystemParent(self.managers, name=name, system_key=key)
        child = HasSystem(parent_, name=name, system_key=key, parent=parent_)

        # Test kernel invariants
        test.helpers.test_system_kernel_invariants(self, parent_)

        for hk in [parent_, child]:
            with self.subTest(has_key_object=hk):
                for key in ['data_store', 'core', 'core_cache', 'core_dma']:
                    self.assertIn(key, hk.kernel_invariants)
                self.assertNotIn('registry', hk.kernel_invariants)


class DaxModuleBaseTestCase(unittest.TestCase):
    """Tests DaxHasKey, DaxHasSystem, DaxModuleBase, DaxModule, and DaxSystem.

    The mentioned modules are highly related and overlap mostly.
    Therefore they are all tested mutually.
    """

    def setUp(self) -> None:
        self.managers = get_managers(_DEVICE_DB)

    def tearDown(self) -> None:
        # Close managers
        self.managers.close()

    def test_system_build(self):
        # A system that does not call super() in build()
        class BadTestSystem(_TestSystem):
            def build(self):
                pass  # No call to super(), which is bad

        # Test if an error occurs when super() is not called in build()
        with self.assertRaises(AttributeError, msg='Not calling super().build() in user system did not raise'):
            BadTestSystem(self.managers)

    def test_system_kernel_invariants(self):
        s = _TestSystem(self.managers)

        # No kernel invariants attribute yet
        self.assertTrue(hasattr(s, 'kernel_invariants'), 'Default kernel invariants not found')

        # Update kernel invariants
        invariant = 'foo'
        s.update_kernel_invariants(invariant)
        self.assertIn(invariant, s.kernel_invariants, 'Kernel invariants update not successful')

    def test_system_id(self):
        # Test if an error is raised when no ID is given to a system
        with self.assertRaises(AssertionError, msg='Not providing system id did not raise'):
            DaxSystem(self.managers)

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
                BadSystem(self.managers)

    def test_system_ver(self):
        class TestSystemNoVer(DaxSystem):
            SYS_ID = 'unittest_system'

        # Test if an error is raised when no version is given to a system
        with self.assertRaises(AssertionError, msg='Not providing system version did not raise'):
            TestSystemNoVer(self.managers)

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
                BadSystem(self.managers)

        # System with version 0, which is fine
        class TestSystemVerZero(DaxSystem):
            SYS_ID = 'unittest_system'
            SYS_VER = 0

        # Test if it is possible to create a system with version 0
        TestSystemVerZero(self.managers)

    def test_build_controller_warnings(self):
        class TestSystem(_TestSystemWithControllers):
            # noinspection PyMethodParameters
            def build(self_, *args: typing.Any, **kwargs: typing.Any) -> None:
                self.assertIsNotNone(self_.CORE_LOG_KEY, 'Core log controller key was not configured')
                self.assertIsNotNone(self_.DAX_INFLUX_DB_KEY, 'Data store controller key was not configured')

                # There should be warnings due to the lack of controllers
                with self.assertLogs(self_.logger, logging.WARNING):
                    super(TestSystem, self_).build(*args, **kwargs)

        class NoCoreLogTestSystem(_TestSystemWithControllers):
            CORE_LOG_KEY = None

            # noinspection PyMethodParameters
            def build(self_, *args: typing.Any, **kwargs: typing.Any) -> None:
                self.assertIsNone(self_.CORE_LOG_KEY, 'Core log controller key was not disabled')
                self.assertIsNotNone(self_.DAX_INFLUX_DB_KEY, 'Data store controller key was not configured')

                # There should be warnings due to the lack of controllers
                with self.assertLogs(self_.logger, logging.WARNING):
                    super(NoCoreLogTestSystem, self_).build(*args, **kwargs)

        class NoDataStoreTestSystem(_TestSystemWithControllers):
            DAX_INFLUX_DB_KEY = None

            # noinspection PyMethodParameters
            def build(self_, *args: typing.Any, **kwargs: typing.Any) -> None:
                self.assertIsNotNone(self_.CORE_LOG_KEY, 'Core log controller key was not configured')
                self.assertIsNone(self_.DAX_INFLUX_DB_KEY, 'Data store controller key was not disabled')

                # There should be warnings due to the lack of controllers
                with self.assertLogs(self_.logger, logging.WARNING):
                    super(NoDataStoreTestSystem, self_).build(*args, **kwargs)

        class NoControllerTestSystem(_TestSystemWithControllers):
            CORE_LOG_KEY = None
            DAX_INFLUX_DB_KEY = None

            # noinspection PyMethodParameters
            def build(self_, *args: typing.Any, **kwargs: typing.Any) -> None:
                self.assertIsNone(self_.CORE_LOG_KEY, 'Core log controller key was not disabled')
                self.assertIsNone(self_.DAX_INFLUX_DB_KEY, 'Data store controller key was not disabled')

                # There should be no warnings, controllers are disabled
                with self.assertRaises(self.failureException), self.assertLogs(self_.logger, logging.WARNING):
                    super(NoControllerTestSystem, self_).build(*args, **kwargs)

        # Build test systems
        TestSystem(self.managers)
        NoCoreLogTestSystem(self.managers)
        NoDataStoreTestSystem(self.managers)
        NoControllerTestSystem(self.managers)

    def test_init(self):
        s = _TestSystem(self.managers)

        # Check constructor
        self.assertIsNotNone(s, 'Could not create DaxSystem')
        self.assertIsNotNone(_TestModule(s, 'module_name'), 'Could not create a test module')
        with self.assertRaises(ValueError, msg='Invalid module name did not raise'):
            _TestModule(s, 'wrong!')
        with self.assertRaises(ValueError, msg='Invalid module name did not raise'):
            _TestModule(s, 'this.is.bad')
        with self.assertRaises(TypeError, msg='Providing non-DaxModuleBase parent to new module did not raise'):
            # noinspection PyTypeChecker
            _TestModule(self.managers, 'module_name')

    def test_module_registration(self):
        # Check register
        s = _TestSystem(self.managers)
        t = _TestModule(s, 'module_name')
        self.assertDictEqual(s.registry._modules, {m.get_system_key(): m for m in [s, t]},
                             'Dict with registered modules does not match expected content')

    def test_name(self):
        s = _TestSystem(self.managers)

        self.assertEqual(s.get_name(), _TestSystem.SYS_NAME, 'Returned name did not match expected name')

    def test_system_key(self):
        s = _TestSystem(self.managers)

        self.assertEqual(s.get_system_key(), _TestSystem.SYS_NAME, 'Returned key did not match expected key')

    def test_system_key_arguments(self):
        s = _TestSystem(self.managers)

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
        s = _TestSystem(self.managers)

        with self.assertRaises(ValueError, msg='Creating bad system key did not raise'):
            s.get_system_key('bad,key')
        with self.assertRaises(ValueError, msg='Creating bad system key did not raise'):
            s.get_system_key('good_key', 'bad,key')
        with self.assertRaises(AssertionError, msg='Creating system key with wrong key input did not raise'):
            # Intentionally wrong argument type, disabling inspection
            # noinspection PyTypeChecker
            s.get_system_key(1)

    def test_setattr_device(self):
        s = _TestSystem(self.managers)

        self.assertIsNone(s.setattr_device('ttl0'), 'setattr_device() did not return None')
        self.assertTrue(hasattr(s, 'ttl0'), 'setattr_device() did not set the attribute correctly')

        with self.assertRaises(TypeError, msg='setattr_device() type check did not raise'):
            s.setattr_device('alias_2', artiq.coredevice.ttl.TTLInOut)
        self.assertFalse(hasattr(s, 'alias_2'), 'Failed setattr_device() did occupy attribute')

        self.assertIsNone(s.setattr_device('alias_2', artiq.coredevice.ttl.TTLOut),
                          'setattr_device() did not return None')
        self.assertTrue(hasattr(s, 'alias_2'), 'setattr_device() did not set the attribute correctly')
        self.assertIsInstance(s.alias_2, artiq.coredevice.ttl.TTLOut,
                              'setattr_device() did not returned correct device type')

    def test_get_device(self):
        # Test system
        s = _TestSystem(self.managers)
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
        s = _TestSystem(self.managers)

        with self.assertRaises(TypeError, msg='get_device() type check did not raise'):
            s.get_device('ttl1', artiq.coredevice.edge_counter.EdgeCounter)
        with self.assertRaises(TypeError, msg='get_device() type check did not raise'):
            s.get_device('ttl1', artiq.coredevice.ttl.TTLInOut)

        # Correct type, should not raise
        self.assertIsNotNone(s.get_device('ttl1', artiq.coredevice.ttl.TTLOut),
                             'get_device() type check raised unexpectedly')

    def test_search_devices(self):
        s = _TestSystem(self.managers)
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

    def test_get_dataset(self):
        s = _TestSystem(self.managers)

        key = 'key1'
        value = [11, 12, 13]
        self.assertListEqual(s.get_dataset_sys(key, default=value), value,
                             'get_dataset_sys() did not returned the provided default value')
        self.assertListEqual(s.get_dataset_sys(key), value,
                             'get_dataset_sys() did not write the default value to the dataset')

        key2 = 'key2'
        self.assertListEqual(s.get_dataset_sys(key2, default=value, data_store=False), value,
                             'get_dataset_sys() did not returned the provided default value')

        key3 = 'key3'
        self.assertListEqual(s.get_dataset_sys(key3, fallback=value), value,
                             'get_dataset_sys() did not return the provided fallback value')
        with self.assertRaises(KeyError, msg='get_dataset_sys() erroneously wrote fallback value to dataset'):
            s.get_dataset_sys(key3)

        value2 = [14, 15, 16]
        self.assertListEqual(s.get_dataset_sys(key3, default=value, fallback=value2, data_store=False), value,
                             'get_dataset_sys() did not return the provided default value')
        self.assertListEqual(s.get_dataset_sys(key3), value,
                             'get_dataset_sys() did not write the default value to the dataset')

        # Check data store calls
        self.assertListEqual(s.data_store.method_calls, [call.set(s.get_system_key(key), value)],
                             'Data store calls did not match expected pattern')

    def test_set_dataset(self):
        s = _TestSystem(self.managers)

        key = 'key1'
        value = [11, 12, 13]
        s.set_dataset_sys(key, value)
        self.assertListEqual(s.get_dataset_sys(key, 'not_used_default_value'), value,
                             'set_dataset_sys() did not write the value to the dataset')

        key2 = 'key2'
        s.set_dataset_sys(key2, value, data_store=False)
        self.assertListEqual(s.get_dataset_sys(key2), value,
                             'set_dataset_sys() did not write the value to the dataset')

        # Check data store calls
        self.assertListEqual(s.data_store.method_calls, [call.set(s.get_system_key(key), value)],
                             'Data store calls did not match expected pattern')

    def test_setattr_dataset(self):
        s = _TestSystem(self.managers)

        key = 'key3'
        self.assertIsNone(s.setattr_dataset_sys(key, 10, data_store=True), 'setattr_dataset_sys() failed')
        self.assertTrue(hasattr(s, key), 'setattr_dataset_sys() did not set the attribute correctly')
        self.assertEqual(getattr(s, key), 10, 'Returned system dataset value does not match expected result')
        self.assertIn(key, s.kernel_invariants,
                      'setattr_dataset_sys() did not added the attribute to kernel_invariants by default')

        key = 'key2'
        self.assertIsNone(s.setattr_dataset_sys(key, 12, data_store=False), 'setattr_dataset_sys() failed')
        self.assertTrue(hasattr(s, key), 'setattr_dataset_sys() did not set the attribute correctly')
        self.assertEqual(getattr(s, key), 12, 'Returned system dataset value does not match expected result')
        self.assertIn(key, s.kernel_invariants,
                      'setattr_dataset_sys() did not added the attribute to kernel_invariants by default')

        key = 'key1'
        self.assertIsNone(s.setattr_dataset_sys(key, 100, kernel_invariant=False), 'setattr_dataset_sys() failed')
        self.assertTrue(hasattr(s, key), 'setattr_dataset_sys() did not set the attribute correctly')
        self.assertEqual(getattr(s, key), 100, 'Returned system dataset value does not match expected result')
        self.assertNotIn(key, s.kernel_invariants,
                         'setattr_dataset_sys() added the attribute to kernel_invariants while it was not supposed to')

        key = 'key5'
        s.set_dataset_sys(key, 5, data_store=True)
        self.assertIsNone(s.setattr_dataset_sys(key, kernel_invariant=True, data_store=False),
                          'setattr_dataset_sys() failed')
        self.assertTrue(hasattr(s, key), 'setattr_dataset_sys() did not set the attribute correctly')
        self.assertEqual(getattr(s, key), 5, 'Returned system dataset value does not match expected result')
        self.assertIn(key, s.kernel_invariants,
                      'setattr_dataset_sys() did not added the attribute to kernel_invariants')

        key = 'key4'
        self.assertIsNone(s.setattr_dataset_sys(key), 'setattr_dataset_sys() failed')
        self.assertFalse(hasattr(s, key), 'setattr_dataset_sys() set the attribute while it should not')
        self.assertNotIn(key, s.kernel_invariants,
                         'setattr_dataset_sys() did added the attribute to kernel_invariants while it should not')

        key = 'key6'
        s.set_dataset_sys(key, 6, data_store=False)
        self.assertIsNone(s.setattr_dataset_sys(key, data_store=True), 'setattr_dataset_sys() failed')
        self.assertTrue(hasattr(s, key), 'setattr_dataset_sys() did not set the attribute correctly')
        self.assertEqual(getattr(s, key), 6, 'Returned system dataset value does not match expected result')
        self.assertIn(key, s.kernel_invariants,
                      'setattr_dataset_sys() did not added the attribute to kernel_invariants')

        key = 'key7'
        self.assertIsNone(s.setattr_dataset_sys(key, fallback=99, data_store=True))  # Will NOT write to data store
        self.assertTrue(hasattr(s, key), 'setattr_dataset_sys() did not set the attribute correctly')
        with self.assertRaises(KeyError, msg='setattr_dataset_sys() erroneously wrote fallback value to dataset'):
            s.get_dataset_sys(key)

        key = 'key8'
        self.assertIsNone(s.setattr_dataset_sys(key, default=80, fallback=81, data_store=False))
        self.assertEqual(s.get_dataset_sys(key), 80,
                         'setattr_dataset_sys() did not write the default value to the dataset')

        # Check data store calls
        self.assertListEqual(s.data_store.method_calls, [call.set(s.get_system_key('key3'), 10),
                                                         call.set(s.get_system_key('key1'), 100),
                                                         call.set(s.get_system_key('key5'), 5),
                                                         ], 'Data store calls did not match expected pattern')

    @unittest.expectedFailure
    def test_dataset_append(self):
        s = _TestSystem(self.managers)

        key = 'key2'
        self.assertIsNone(s.set_dataset_sys(key, []), 'Setting new system dataset failed')
        self.assertListEqual(s.get_dataset_sys(key), [],
                             'Returned system dataset value does not match expected result')
        self.assertIsNone(s.append_to_dataset_sys(key, 1), 'Appending to system dataset failed')
        self.assertListEqual(s.get_dataset_sys(key), [1], 'Appending to system dataset has incorrect behavior')
        # NOTE: This test fails for unknown reasons (ARTIQ library) while real-life tests show correct behavior

    def test_dataset_append_data_store(self):
        s = _TestSystem(self.managers)

        key = 'key2'
        self.assertIsNone(s.set_dataset_sys(key, []), 'Setting new system dataset failed')
        self.assertListEqual(s.get_dataset_sys(key), [],
                             'Returned system dataset value does not match expected result')

        # Check data store calls (test early due to mutating list values)
        self.assertEqual(s.data_store.method_calls[-1], call.set(s.get_system_key(key), []),
                         'Data store calls did not match expected pattern')

        self.assertIsNone(s.append_to_dataset_sys(key, 1), 'Appending to system dataset failed')
        self.assertIsNone(s.append_to_dataset_sys(key, 2, data_store=False), 'Appending to system dataset failed')

        # Check data store calls
        self.assertEqual(s.data_store.method_calls[-1], call.append(s.get_system_key(key), 1),
                         'Data store calls did not match expected pattern')

    def test_dataset_append_nonempty(self):
        s = _TestSystem(self.managers)

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
        s = _TestSystem(self.managers)

        key = 'key2'
        self.assertIsNone(s.set_dataset_sys(key, [0, 0, 0, 0]), 'Setting new system dataset failed')
        self.assertListEqual(s.get_dataset_sys(key), [0, 0, 0, 0],
                             'Returned system dataset value does not match expected result')

        # Check data store calls (test early due to mutating list values)
        self.assertListEqual(s.data_store.method_calls, [call.set(s.get_system_key(key), [0, 0, 0, 0])],
                             'Data store calls did not match expected pattern')

        self.assertIsNone(s.mutate_dataset_sys(key, 1, 9), 'Mutating system dataset failed')
        self.assertIsNone(s.mutate_dataset_sys(key, 3, 99, data_store=False), 'Mutating system dataset failed')
        self.assertListEqual(s.get_dataset_sys(key), [0, 9, 0, 99], 'Mutating system dataset has incorrect behavior')

        # Check data store calls
        self.assertEqual(len(s.data_store.method_calls), 2)
        self.assertEqual(s.data_store.method_calls[-1], call.mutate(s.get_system_key(key), 1, 9),
                         'Data store calls did not match expected pattern')

    def test_identifier(self):
        s = _TestSystem(self.managers)
        self.assertTrue(isinstance(s.get_identifier(), str), 'get_identifier() did not returned a string')

    def test_repr(self):
        s = _TestSystem(self.managers)
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
        self.managers = get_managers(_DEVICE_DB)
        self.system = self.InitTestSystem(self.managers, self)
        for i in range(self.num_modules):
            self.InitTestModule(self.system, f'module_{i}', i, self)

    def tearDown(self) -> None:
        # Close managers
        self.managers.close()

    def test_kernel_invariants(self):
        # Test kernel invariants
        test.helpers.test_system_kernel_invariants(self, self.system)

    def test_dax_init(self):
        # Call DAX init
        self.system.dax_init()

        # Verify if the counters have the expected values
        self.assertEqual(self.init_count, self.num_modules + 1, 'Number of init calls does not match expected number')
        self.assertEqual(self.post_init_count, self.num_modules + 1,
                         'Number of post_init calls does not match expected number')

    def test_system_info_archiving(self):
        keys = {
            'dax/system_id': self.system.SYS_ID,
            'dax/system_version': self.system.SYS_VER,
            'dax/dax_version': _dax_version,
            'dax/dax_sim_enabled': self.system.dax_sim_enabled,
        }
        if dax.util.git.in_repository():
            keys.update({f'dax/git_{k}': v for k, v in dax.util.git.get_repository_info().as_dict().items()})

        for k in keys:
            with self.assertRaises(KeyError):
                self.system.get_dataset(k)

        # Call DAX init
        self.system.dax_init()

        # Verify data exists in archive
        for k, v in keys.items():
            self.assertEqual(self.system.get_dataset(k), v)

    def test_build_super(self):
        class SomeBaseClass(HasEnvironment):
            # noinspection PyUnusedLocal
            def build(self, *args, **kwargs):
                # super().build() here calls HasEnvironment, which is empty
                # Set flag
                self.base_build_was_called = True

        # SomeBaseClass also inherits HasEnvironment, making it higher in the MRO than HasEnvironment itself
        class SuperTestSystem(_TestSystem, SomeBaseClass):
            def build(self, *args, **kwargs):
                # Call super
                super(SuperTestSystem, self).build(*args, **kwargs)
                # Set flag
                self.system_build_was_called = True

            def run(self):
                pass

        # Create system, which will call build()
        system = SuperTestSystem(self.managers)

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
            SuperTestSystem(self.managers)


class DaxServiceTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.managers = get_managers(_DEVICE_DB)

    def tearDown(self) -> None:
        # Close managers
        self.managers.close()

    def test_init(self):
        s = _TestSystem(self.managers)

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

    def setUp(self) -> None:
        self.managers = get_managers(_DEVICE_DB)

    def tearDown(self) -> None:
        # Close managers
        self.managers.close()

    def test_not_decorated(self):
        class Client(DaxClient, Experiment):
            def run(self) -> None:
                pass

        with self.assertRaises(TypeError, msg='Using client without client factory decorator did not raise'):
            # noinspection PyTypeChecker
            Client(self.managers)

    def test_not_experiment(self):
        class Client(DaxClient):
            def run(self) -> None:
                pass

        with self.assertRaises(TypeError, msg='Decorated client class does not inherit Experiment'):
            dax_client_factory(Client)

    def test_load_super(self):
        @dax_client_factory
        class Client(DaxClient, Experiment):
            def init(self) -> None:
                self.is_initialized = True

            def run(self) -> None:
                pass

        # noinspection PyTypeChecker
        class ImplementableClient(Client(_TestSystem)):
            pass

        # Disabled one inspection, inspection does not handle the decorator correctly
        # noinspection PyArgumentList
        c = ImplementableClient(self.managers)
        c.run()  # Is supposed to call the dax_init() function which will call the init() function of the client

        self.assertTrue(hasattr(c, 'is_initialized'), 'DAX system of client was not initialized correctly')

    def test_disable_dax_init(self):
        @dax_client_factory
        class Client(DaxClient, Experiment):
            DAX_INIT = False

            def init(self) -> None:
                self.is_initialized = True

            def run(self) -> None:
                pass

        # noinspection PyTypeChecker
        class ImplementableClient(Client(_TestSystem)):
            pass

        # Disabled one inspection, inspection does not handle the decorator correctly
        # noinspection PyArgumentList
        c = ImplementableClient(self.managers)
        c.run()  # Is not supposed to call the dax_init() function

        self.assertFalse(hasattr(c, 'is_initialized'), 'DAX system of client was initialized unexpectedly')

    def test_manager_kwarg(self):
        kwarg = 'managers'

        @dax_client_factory
        class Client(DaxClient, Experiment):
            MANAGERS_KWARG = kwarg

            def build(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

            def run(self) -> None:
                pass

        # noinspection PyTypeChecker
        class ImplementableClient(Client(_TestSystem)):
            pass

        # Disabled one inspection, inspection does not handle the decorator correctly
        # noinspection PyArgumentList
        c = ImplementableClient(self.managers)

        self.assertFalse(c.args)
        self.assertDictEqual(c.kwargs, {kwarg: self.managers})
