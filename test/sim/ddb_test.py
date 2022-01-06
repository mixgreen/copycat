import unittest
import logging
import copy
import textwrap
import pathlib

import dax.sim.ddb
from dax.sim.ddb import enable_dax_sim, DAX_SIM_CONFIG_KEY
from dax.util.output import temp_dir
from dax.util.configparser import get_dax_config
import dax.util.git


def _clear_cache():
    # Clear DAX config parser cache
    dax.util.git._REPO_INFO = None
    get_dax_config(clear_cache=True)


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
            'command': 'foo -p {port} --bind {bind}'
        },
        'ttl0': {
            'type': 'local',
            'module': 'artiq.coredevice.ttl',
            'class': 'TTLInOut',
            'arguments': {'channel': 0},
            'sim_args': {'input_prob': 0.9}
        },
        'controller': {
            'type': 'controller',
            'host': 'some_host',
            'port': 2,
            'command': 'foo -p {port} --bind {bind}'
        },
        'controller2': {
            'type': 'controller',
            'host': 'some_host',
            'port': 3,
            'command': 'bar -p {port} --bind {bind}'
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
        },
        'controller2': {
            'type': 'controller',
            'host': 'some_host',
            'port': 1,
        },
    }

    DEVICE_DB_MISSING_HOST = {
        'core': {
            'type': 'local',
            'module': 'artiq.coredevice.core',
            'class': 'Core',
            'arguments': {'host': None, 'ref_period': 1e-9}
        },
        'controller': {
            'type': 'controller',
            'port': 1,
        },
    }

    DEVICE_DB_MISSING_PORT = {
        'core': {
            'type': 'local',
            'module': 'artiq.coredevice.core',
            'class': 'Core',
            'arguments': {'host': None, 'ref_period': 1e-9}
        },
        'controller': {
            'type': 'controller',
            'host': 'some_host',
        },
    }

    DEVICE_DB_MISSING_BIND_ARG = {
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
            'command': 'foo -p {port}'
        },
    }

    DEVICE_DB_MISSING_PORT_ARG = {
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
            'command': 'foo --bind {bind}'
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

    def setUp(self) -> None:
        # Always make a deep copy at the start to make sure we do not mutate the dict
        self.DEVICE_DB = copy.deepcopy(self.DEVICE_DB)
        _clear_cache()

    def test_no_config(self):
        with temp_dir():
            _clear_cache()
            with self.assertRaises(FileNotFoundError):
                enable_dax_sim(copy.deepcopy(self.DEVICE_DB), logging_level=logging.WARNING, moninj_service=False)

    def test_missing_config(self):
        with temp_dir():
            _clear_cache()
            pathlib.Path('.dax').touch()  # Create empty config file
            with self.assertRaises(Exception):
                enable_dax_sim(copy.deepcopy(self.DEVICE_DB), logging_level=logging.WARNING, moninj_service=False)

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

    def test_sim_config_device(self, *, config_module='dax.sim.config', config_class='DaxSimConfig'):
        # Signal manager kwargs
        sm_kwargs = {'_some_random': 1, '_random_random': 2, '_keyword_random': 3, '_arguments_random': 4}

        ddb = enable_dax_sim(self.DEVICE_DB, enable=True, logging_level=logging.WARNING, **sm_kwargs,
                             moninj_service=False)
        self.assertIn(DAX_SIM_CONFIG_KEY, ddb, 'Sim config device not found')
        self.assertEqual(ddb[DAX_SIM_CONFIG_KEY]['type'], 'local')
        self.assertEqual(ddb[DAX_SIM_CONFIG_KEY]['module'], config_module)
        self.assertEqual(ddb[DAX_SIM_CONFIG_KEY]['class'], config_class)
        self.assertDictEqual(sm_kwargs, ddb[DAX_SIM_CONFIG_KEY]['arguments']['signal_mgr_kwargs'],
                             'Signal manager kwargs do not match expected dict')

    def test_mutate_entries(self, *, localhost='::1'):
        ddb = enable_dax_sim(self.DEVICE_DB, enable=True, logging_level=logging.WARNING, moninj_service=False)
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
            enable_dax_sim(copy.deepcopy(self.DEVICE_DB_CONFLICTING_PORTS), enable=True,
                           logging_level=logging.CRITICAL, moninj_service=False)

    def test_missing_host(self):
        with self.assertRaises(KeyError, msg='Missing host did not raise'):
            enable_dax_sim(copy.deepcopy(self.DEVICE_DB_MISSING_HOST), enable=True,
                           logging_level=logging.CRITICAL, moninj_service=False)

    def test_missing_port(self):
        with self.assertRaises(KeyError, msg='Missing port did not raise'):
            enable_dax_sim(copy.deepcopy(self.DEVICE_DB_MISSING_PORT), enable=True,
                           logging_level=logging.CRITICAL, moninj_service=False)

    def test_missing_bind_arg(self):
        with self.assertRaises(ValueError, msg='Missing bind argument in command did not raise'):
            enable_dax_sim(copy.deepcopy(self.DEVICE_DB_MISSING_BIND_ARG), enable=True,
                           logging_level=logging.CRITICAL, moninj_service=False)

    def test_missing_port_arg(self):
        with self.assertLogs(dax.sim.ddb._logger, logging.WARNING):
            enable_dax_sim(copy.deepcopy(self.DEVICE_DB_MISSING_PORT_ARG), enable=True,
                           logging_level=logging.WARNING, moninj_service=False)

    def test_core_address(self, *, ddb=None, core_device='core', localhost='::1'):
        if ddb is None:
            ddb = self.DEVICE_DB
        ddb = enable_dax_sim(copy.deepcopy(ddb), enable=True, logging_level=logging.WARNING, moninj_service=False)
        # Core host address should be mutated to localhost
        self.assertIn(core_device, ddb)
        self.assertEqual(ddb[core_device]['arguments']['host'], localhost)

    def test_core_compile_flag(self, *, core_device='core', compile_flag=None):
        ddb = enable_dax_sim(copy.deepcopy(self.DEVICE_DB), enable=True, logging_level=logging.WARNING,
                             moninj_service=False)
        # Core compile flag should be added
        self.assertIn(core_device, ddb)
        self.assertEqual(ddb[core_device]['arguments'].get('compile'), compile_flag)

    def test_generic(self):
        for ddb in [copy.deepcopy(self.DEVICE_DB_GENERIC_0), copy.deepcopy(self.DEVICE_DB_GENERIC_1)]:
            ddb = enable_dax_sim(ddb, enable=True, logging_level=logging.WARNING, moninj_service=False)
            self.assertEqual(ddb['generic']['module'], 'dax.sim.coredevice.generic')
            self.assertEqual(ddb['generic']['class'], 'Generic')

    def test_cfg_enable(self):
        cfg = """
        [dax.sim]
        enable = {}
        """
        for cfg_file in ['.dax', 'setup.cfg']:
            with temp_dir():
                for dax_enable in [True, False]:
                    # Write configuration
                    with open(cfg_file, mode='w') as f:
                        f.write(textwrap.dedent(cfg.format(str(dax_enable).lower())))
                    # Clear DAX config parser cache
                    get_dax_config(clear_cache=True)

                    ddb = enable_dax_sim(copy.deepcopy(self.DEVICE_DB),
                                         logging_level=logging.WARNING, moninj_service=False)
                    if dax_enable:
                        self.assertIn(DAX_SIM_CONFIG_KEY, ddb, 'DAX.sim was not enabled')
                    else:
                        self.assertNotIn(DAX_SIM_CONFIG_KEY, ddb, 'DAX.sim was unintentionally enabled')

    def test_cfg_precedence(self):
        cfg = """
        [dax.sim]
        enable = {}
        """
        with temp_dir():
            for dax_enable, setup_enable in ((a, b) for a in range(2) for b in range(2)):
                # Write configuration
                with open('.dax', mode='w') as f:
                    f.write(textwrap.dedent(cfg.format(dax_enable)))
                with open('setup.cfg', mode='w') as f:
                    f.write(textwrap.dedent(cfg.format(setup_enable)))
                # Clear DAX config parser cache
                get_dax_config(clear_cache=True)

                ddb = enable_dax_sim(copy.deepcopy(self.DEVICE_DB), logging_level=logging.WARNING, moninj_service=False)
                if dax_enable:
                    self.assertIn(DAX_SIM_CONFIG_KEY, ddb, 'DAX.sim was not enabled')
                else:
                    self.assertNotIn(DAX_SIM_CONFIG_KEY, ddb, 'DAX.sim was unintentionally enabled')

    def test_cfg_localhost(self):
        localhost = '127.0.0.1'
        cfg = f"""
        [dax.sim]
        localhost = {localhost}
        """
        with temp_dir():
            # Write configuration
            with open('.dax', mode='w') as f:
                f.write(textwrap.dedent(cfg))
            # Clear DAX config parser cache
            get_dax_config(clear_cache=True)

            self.test_mutate_entries(localhost=localhost)
            self.test_core_address(localhost=localhost)

    def test_cfg_core_device(self):
        core_device = 'foo'
        cfg = f"""
        [dax.sim]
        core_device = {core_device}
        """
        with temp_dir():
            # Write configuration
            with open('.dax', mode='w') as f:
                f.write(textwrap.dedent(cfg))
            # Clear DAX config parser cache
            get_dax_config(clear_cache=True)

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
            # Write configuration
            with open('.dax', mode='w') as f:
                f.write(textwrap.dedent(cfg))
            # Clear DAX config parser cache
            get_dax_config(clear_cache=True)

            self.test_sim_config_device(config_module=config_module, config_class=config_class)

    def test_cfg_sim_args(self):
        with temp_dir():
            device = 'ttl0'
            args = {'channel': '77', 'input_prob': '0.0',
                    'float': '9.9', 'int': '1', 'str': '"foo"', 'bool': 'true', 'None': 'None', 'cAsE': 'None',
                    '_key': '"foobar"'}
            ref = {'channel': 77, 'input_prob': 0.0,
                   'float': 9.9, 'int': 1, 'str': 'foo', 'bool': True, 'None': None, 'cAsE': None,
                   '_key': device}

            args_str = '\n'.join(f'{k}={v}' for k, v in args.items())
            cfg = f"""
            [dax.sim.{device}]
            {args_str}
            """
            # Write configuration
            with open('.dax', mode='w') as f:
                f.write(cfg)
            # Clear DAX config parser cache
            get_dax_config(clear_cache=True)

            ddb = enable_dax_sim(copy.deepcopy(self.DEVICE_DB), enable=True, logging_level=logging.WARNING,
                                 moninj_service=False)
            # Check if configuration args are correctly overwritten, both 'arguments' values and 'sim_args'
            self.assertDictEqual(ddb[device]['arguments'], ref)

    def test_cfg_sim_args_compile(self):
        with temp_dir():
            for compile_flag in [True, False]:
                cfg = f"""
                [dax.sim.core]
                compile = {compile_flag}
                """
                # Write configuration
                with open('.dax', mode='w') as f:
                    f.write(textwrap.dedent(cfg))
                # Clear DAX config parser cache
                get_dax_config(clear_cache=True)

                self.test_core_compile_flag(compile_flag=compile_flag)
