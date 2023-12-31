import unittest
import numpy as np

import dax.sim.coredevice.cache
from dax.sim import enable_dax_sim
from dax.util.artiq import get_managers

import test.sim.coredevice._compile_testcase as compile_testcase


class CoreCacheTestCase(unittest.TestCase):
    _CACHE = None

    _DEVICE_DB = enable_dax_sim({
        'core': {
            'type': 'local',
            'module': 'artiq.coredevice.core',
            'class': 'Core',
            'arguments': {'host': None, 'ref_period': 1e-9}
        },
        "core_cache": {
            "type": "local",
            "module": "artiq.coredevice.cache",
            "class": "CoreCache"
        },
    }, enable=True, output='null', moninj_service=False)

    def setUp(self) -> None:
        self.managers = get_managers(device_db=self._DEVICE_DB)
        self.cache = dax.sim.coredevice.cache.CoreCache(self.managers.device_mgr, cache=self._CACHE, _key='core_cache')

    def tearDown(self) -> None:
        # Close managers
        self.managers.close()

    def test_cache(self):
        data = {
            'foo': [0],
            'bar': [np.int32(3), 0, 4],
            'baz': [4, 6, 3, np.int32(99), 99],
        }

        for k, v in data.items():
            with self.subTest(key=k, value=v):
                # Put data
                self.cache.put(k, v)
                # Get data
                self.assertListEqual(self.cache.get(k), v, 'Data does not match earlier added data')

    def test_cache_bad_types(self):
        data = {
            'foo': [0.0],
            'bar': [np.int64(3), 0, 4],
            'baz': [4, 6, 3, 'a'],
            'foobar': [4, 6, [3]],
        }

        for k, v in data.items():
            with self.subTest(key=k, value=v):
                with self.assertRaises(TypeError):
                    # Put data
                    self.cache.put(k, v)

    def test_non_existing_key(self):
        self.assertListEqual(self.cache.get('non_existing_key'), [], 'Non-existing key did not return empty list')

    def test_erase(self):
        key = 'foo'
        # Add
        self.cache.put(key, [2, 3, 4])
        # Erase
        self.cache.put(key, [])
        self.assertListEqual(self.cache.get(key), [], 'Extracting erased key did not return empty list')


class CompileTestCase(compile_testcase.CoredeviceCompileTestCase):
    DEVICE_CLASS = dax.sim.coredevice.cache.CoreCache
    DEVICE_KWARGS = {'cache': {'k': [1]}}
    FN_KWARGS = {
        'get': {'key': 'k'},
        'put': {'key': 'k', 'value': [2]},
    }
