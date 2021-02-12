import unittest
import logging
import copy
import textwrap

import dax.sim.ddb
from dax.sim.ddb import enable_dax_sim, DAX_SIM_CONFIG_KEY
from dax.util.output import temp_dir


class DdbTestCase(unittest.TestCase):
    DEVICE_DB = {
        'core': {
            'type': 'local',
            'module': 'artiq.coredevice.core',
            'class': 'Core',
            'arguments': {'host': None, 'ref_period': 1e-9}
        },
        'core_log': {
            'type': 'controller',
            'host': '::1',
            'port': 1,
            'command': 'some_command'
        },
        'ttl0': {
            'type': 'local',
            'module': 'artiq.coredevice.ttl',
            'class': 'TTLInOut',
            'arguments': {'channel': 0},
        },
        'controller': {
            'type': 'controller',
            'host': 'some_host',
            'port': 2,
            'command': 'some_command'
        },
        'controller2': {
            'type': 'controller',
            'host': 'some_host',
            'port': 3,
            'command': 'some_command'
        },
    }

    DEVICE_DB_CONFLICTING_PORTS = {
        'core': {
            'type': 'local',
            'module': 'artiq.coredevice.core',
            'class': 'Core',
            'arguments': {'host': None, 'ref_period': 1e-9}
        },
        'controller': {
            'type': 'controller',
            'host': 'some_host',
            'port': 1,
            'command': 'some_command'
        },
        'controller2': {
            'type': 'controller',
            'host': 'some_host',
            'port': 1,
            'command': 'some_command'
        },
    }

    DEVICE_DB_GENERIC_0 = {
        'core': {
            'type': 'local',
            'module': 'artiq.coredevice.core',
            'class': 'Core',
            'arguments': {'host': None, 'ref_period': 1e-9}
        },
        'generic': {
            'type': 'local',
            'module': 'artiq.coredevice.non_existing_module',
            'class': 'NotExistingClass',
            'arguments': {'channel': 1},
        },
    }

    DEVICE_DB_GENERIC_1 = {
        'core': {
            'type': 'local',
            'module': 'artiq.coredevice.core',
            'class': 'Core',
            'arguments': {'host': None, 'ref_period': 1e-9}
        },
        'generic': {
            'type': 'local',
            'module': 'artiq.coredevice.ttl',
            'class': 'NotExistingClass',
        },
    }

    DEVICE_DB_FOO_CORE = {
        'foo': {
            'type': 'local',
            'module': 'artiq.coredevice.core',
            'class': 'Core',
            'arguments': {'host': None, 'ref_period': 1e-9}
        },
    }

    def test_disable(self):
        self.assertDictEqual(enable_dax_sim(copy.deepcopy(self.DEVICE_DB), enable=False, logging_level=logging.WARNING,
                                            moninj_service=False),
                             self.DEVICE_DB, 'Disabled DAX sim did alter device DB')

    def test_application_styles(self):
        # Substitute style
        d0 = enable_dax_sim(copy.deepcopy(self.DEVICE_DB), enable=True, logging_level=logging.WARNING,
                            moninj_service=False)

        # In-place style
        d1 = copy.deepcopy(self.DEVICE_DB)
        enable_dax_sim(d1, enable=True, logging_level=logging.WARNING, moninj_service=False)

        # Compare if the dicts are the same
        self.assertDictEqual(d0, d1, 'Substitute and in-place usage of enable_dax_sim() did not yielded same result')

    def test_double_application(self):
        # Apply once
        d0 = enable_dax_sim(copy.deepcopy(self.DEVICE_DB), enable=True, logging_level=logging.WARNING,
                            moninj_service=False)

        # Apply twice on deep copy
        d1 = enable_dax_sim(copy.deepcopy(d0), enable=True, logging_level=logging.WARNING, moninj_service=False)

        # Compare if the dicts are the same
        self.assertDictEqual(d0, d1, 'Second application did modify ddb while it should not')

    def test_ddb_shallow_copy(self):
        # Modify a shallow copy of the device db
        ddb = enable_dax_sim(self.DEVICE_DB.copy(), enable=True, logging_level=logging.WARNING, moninj_service=False)
        # Verify the object attribute is still the same as the class attribute
        self.assertDictEqual(self.DEVICE_DB, DdbTestCase.DEVICE_DB, 'Shallow copy does not protect test usage of ddb')
        # Verify the modified device db is not the same as the object attribute
        self.assertNotEqual(ddb, self.DEVICE_DB, 'Shallow copy was not different from object attribute reference')

    def test_sim_config_device(self, *, config_module='dax.sim.config', config_class='DaxSimConfig'):
        # Signal manager kwargs
        sm_kwargs = {'_some_random': 1, '_random_random': 2, '_keyword_random': 3, '_arguments_random': 4}

        ddb = enable_dax_sim(self.DEVICE_DB.copy(), enable=True, logging_level=logging.WARNING, **sm_kwargs,
                             moninj_service=False)
        self.assertIn(DAX_SIM_CONFIG_KEY, ddb, 'Sim config device not found')
        self.assertEqual(ddb[DAX_SIM_CONFIG_KEY]['type'], 'local')
        self.assertEqual(ddb[DAX_SIM_CONFIG_KEY]['module'], config_module)
        self.assertEqual(ddb[DAX_SIM_CONFIG_KEY]['class'], config_class)
        self.assertDictEqual(sm_kwargs, ddb[DAX_SIM_CONFIG_KEY]['arguments']['signal_mgr_kwargs'],
                             'Signal manager kwargs do not match expected dict')

    def test_mutate_entries(self, *, localhost='::1'):
        ddb = enable_dax_sim(self.DEVICE_DB.copy(), enable=True, logging_level=logging.WARNING, moninj_service=False)
        for k, v in ddb.items():
            type_ = v['type']
            if type_ == 'local':
                module = v['module']
                self.assertTrue(module.startswith('dax.sim.coredevice.') or module == 'dax.sim.config',
                                f'Device module was not correctly updated: {module}')
            elif type_ == 'controller':
                self.assertTrue(v['host'] == localhost, 'Controller host was not correctly updated')
                command = v['command']
                for arg in dax.sim.ddb._SIMULATION_ARGS:
                    self.assertIn(arg, command, 'Controller command arguments were not correctly updated')
            else:
                self.fail('Internal exception, this statement should not have been reached')

    def test_conflicting_ports(self):
        with self.assertRaises(ValueError, msg='Conflicting ports did not raise'):
            enable_dax_sim(self.DEVICE_DB_CONFLICTING_PORTS.copy(), enable=True,
                           logging_level=logging.CRITICAL, moninj_service=False)

    def test_core_address(self, *, ddb=None, core_device='core', localhost='::1'):
        if ddb is None:
            ddb = self.DEVICE_DB
        ddb = enable_dax_sim(ddb.copy(), enable=True, logging_level=logging.WARNING, moninj_service=False)
        # Core host address should be mutated to localhost
        self.assertIn(core_device, ddb)
        self.assertEqual(ddb[core_device]['arguments']['host'], localhost)

    def test_generic(self):
        for ddb in [self.DEVICE_DB_GENERIC_0.copy(), self.DEVICE_DB_GENERIC_1.copy()]:
            ddb = enable_dax_sim(ddb, enable=True, logging_level=logging.WARNING, moninj_service=False)
            self.assertEqual(ddb['generic']['module'], 'dax.sim.coredevice.generic')
            self.assertEqual(ddb['generic']['class'], 'Generic')

    def test_cfg_enable(self):
        cfg = """
        [dax.sim]
        enable = {}
        """
        for cfg_file in ['.dax', 'setup.cfg']:
            for dax_enable in [True, False]:
                with temp_dir():
                    with open(cfg_file, mode='w') as f:
                        f.write(textwrap.dedent(cfg.format(str(dax_enable).lower())))

                    ddb = enable_dax_sim(self.DEVICE_DB.copy(), logging_level=logging.WARNING, moninj_service=False)
                    if dax_enable:
                        self.assertIn(DAX_SIM_CONFIG_KEY, ddb, 'DAX.sim was unintentionally enabled')
                    else:
                        self.assertNotIn(DAX_SIM_CONFIG_KEY, ddb, 'DAX.sim was not enabled')

    def test_cfg_precedence(self):
        cfg = """
        [dax.sim]
        enable = {}
        """
        with temp_dir():
            for dax_enable, setup_enable in ((a, b) for a in range(2) for b in range(2)):
                with open('.dax', mode='w') as f:
                    f.write(textwrap.dedent(cfg.format(dax_enable)))
                with open('setup.cfg', mode='w') as f:
                    f.write(textwrap.dedent(cfg.format(setup_enable)))

                ddb = enable_dax_sim(self.DEVICE_DB.copy(), logging_level=logging.WARNING, moninj_service=False)
                if dax_enable:
                    self.assertIn(DAX_SIM_CONFIG_KEY, ddb, 'DAX.sim was unintentionally enabled')
                else:
                    self.assertNotIn(DAX_SIM_CONFIG_KEY, ddb, 'DAX.sim was not enabled')

    def test_cfg_localhost(self):
        localhost = '127.0.0.1'
        cfg = f"""
        [dax.sim]
        localhost = {localhost}
        """
        with temp_dir():
            with open('.dax', mode='w') as f:
                f.write(textwrap.dedent(cfg))
            self.test_mutate_entries(localhost=localhost)
            self.test_core_address(localhost=localhost)

    def test_cfg_core_device(self):
        core_device = 'foo'
        cfg = f"""
        [dax.sim]
        core_device = {core_device}
        """
        with temp_dir():
            with open('.dax', mode='w') as f:
                f.write(textwrap.dedent(cfg))
            self.test_core_address(ddb=self.DEVICE_DB_FOO_CORE, core_device=core_device)

            with self.assertRaises(KeyError, msg='Non-existing core device key did not raise'):
                self.test_core_address(core_device=core_device)

    def test_cfg_config_module_class(self):
        config_module = 'my.config.module'
        config_class = 'MyConfigClass'
        cfg = f"""
        [dax.sim]
        config_module = {config_module}
        config_class = {config_class}
        """
        with temp_dir():
            with open('.dax', mode='w') as f:
                f.write(textwrap.dedent(cfg))

            self.test_sim_config_device(config_module=config_module, config_class=config_class)
