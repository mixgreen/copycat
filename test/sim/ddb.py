import unittest
import logging
import copy

from dax.sim.ddb import enable_dax_sim, DAX_SIM_CONFIG_KEY


class DdbTestCase(unittest.TestCase):
    DEVICE_DB = {
        'core': {
            'type': 'local',
            'module': 'artiq.coredevice.core',
            'class': 'Core',
            'arguments': {'host': '0.0.0.0', 'ref_period': 1e-9}
        },
        'ttl0': {
            'type': 'local',
            'module': 'artiq.coredevice.ttl',
            'class': 'TTLInOut',
            'arguments': {'channel': 0},
        },
        'controller': {
            'type': 'controller',
            'command': 'some_command'
        },
    }

    DEVICE_DB_GENERIC_0 = {
        'generic': {
            'type': 'local',
            'module': 'artiq.coredevice.non_existing_module',
            'class': 'NotExistingClass',
            'arguments': {'channel': 1},
        },
    }

    DEVICE_DB_GENERIC_1 = {
        'generic': {
            'type': 'local',
            'module': 'artiq.coredevice.ttl',
            'class': 'NotExistingClass',
        },
    }

    def test_disable(self):
        self.assertDictEqual(enable_dax_sim(False, copy.deepcopy(self.DEVICE_DB), logging_level=logging.WARNING),
                             self.DEVICE_DB, 'Disabled DAX sim did alter device DB')

    def test_application_styles(self):
        # Substitute style
        d0 = enable_dax_sim(True, copy.deepcopy(self.DEVICE_DB), logging_level=logging.WARNING)

        # In-place style
        d1 = copy.deepcopy(self.DEVICE_DB)
        enable_dax_sim(True, d1, logging_level=logging.WARNING)

        # Compare if the dicts are the same
        self.assertDictEqual(d0, d1, 'Substitute and in-place usage of enable_dax_sim() did not yielded same result')

    def test_ddb_shallow_copy(self):
        # Modify a shallow copy of the device db
        ddb = enable_dax_sim(True, self.DEVICE_DB.copy(), logging_level=logging.WARNING)
        # Verify the object attribute is still the same as the class attribute
        self.assertDictEqual(self.DEVICE_DB, DdbTestCase.DEVICE_DB, 'Shallow copy does not protect test usage of ddb')
        # Verify the modified device db is not the same as the object attribute
        self.assertNotEqual(ddb, self.DEVICE_DB, 'Shallow copy was not different from object attribute reference')

    def test_sim_config_device(self):
        # Signal manager kwargs
        sm_kwargs = {'_some_random': 1, '_random_random': 2, '_keyword_random': 3, '_arguments_random': 4}

        ddb = enable_dax_sim(True, self.DEVICE_DB.copy(), logging_level=logging.WARNING, **sm_kwargs)
        self.assertIn(DAX_SIM_CONFIG_KEY, ddb, 'Sim config device not found')
        self.assertDictEqual(sm_kwargs, ddb[DAX_SIM_CONFIG_KEY]['arguments']['signal_mgr_kwargs'],
                             'Signal manager kwargs do not match expected dict')

    def test_module_substituted(self):
        ddb = enable_dax_sim(True, self.DEVICE_DB.copy(), logging_level=logging.WARNING)
        for k, v in ddb.items():
            type_ = v['type']
            if type_ == 'local':
                module = v['module']
                self.assertTrue(module.startswith('dax.sim.coredevice.') or module == 'dax.sim.config',
                                'Device module was not correctly updated: {:s}'.format(module))
            elif type_ == 'controller':
                self.assertTrue('--simulation' in v['command'], 'Controller command was not correctly updated')
            else:
                self.fail('Internal exception, this statement should not have been reached')

    def test_generic(self):
        for ddb in [self.DEVICE_DB_GENERIC_0.copy(), self.DEVICE_DB_GENERIC_1.copy()]:
            ddb = enable_dax_sim(True, ddb, logging_level=logging.WARNING)
            self.assertEqual(ddb['generic']['module'], 'dax.sim.coredevice.generic')
            self.assertEqual(ddb['generic']['class'], 'Generic')
