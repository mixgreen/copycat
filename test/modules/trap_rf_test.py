import typing
import logging
import unittest

from dax.experiment import *
from dax.modules.cpld_init import CpldInitModule
from dax.modules.trap_rf import TrapRfModule
import dax.sim.test_case

from test.environment import CI_ENABLED


class _TestSystem(DaxSystem):
    SYS_ID = 'unittest_system'
    SYS_VER = 0
    CORE_LOG_KEY = None
    DAX_INFLUX_DB_KEY = None

    def build(self, **kwargs) -> None:  # type: ignore[override]
        super(_TestSystem, self).build()
        CpldInitModule(self, 'cpld')  # Also add a CPLD init module
        self.trap_rf = TrapRfModule(self, 'trap_rf', key='urukul0_ch0', **kwargs)


class TrapRfTestCase(dax.sim.test_case.PeekTestCase):
    RESONANCE_FREQ = 50.03 * MHz
    RESONANCE_DIFF = 0.2 * MHz

    _DEVICE_DB: typing.Dict[str, typing.Any] = {
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

        'eeprom_urukul0': {
            "type": "local",
            "module": "artiq.coredevice.kasli_i2c",
            "class": "KasliEEPROM",
            "arguments": {"port": "EEM5"}
        },

        'spi_urukul0': {
            "type": "local",
            "module": "artiq.coredevice.spi2",
            "class": "SPIMaster",
            "arguments": {"channel": 0x000043}
        },

        'ttl_urukul0_io_update': {
            "type": "local",
            "module": "artiq.coredevice.ttl",
            "class": "TTLOut",
            "arguments": {"channel": 0x000044}
        },

        'ttl_urukul0_sw0': {
            "type": "local",
            "module": "artiq.coredevice.ttl",
            "class": "TTLOut",
            "arguments": {"channel": 0x000045}
        },

        'ttl_urukul0_sw1': {
            "type": "local",
            "module": "artiq.coredevice.ttl",
            "class": "TTLOut",
            "arguments": {"channel": 0x000046}
        },

        'ttl_urukul0_sw2': {
            "type": "local",
            "module": "artiq.coredevice.ttl",
            "class": "TTLOut",
            "arguments": {"channel": 0x000047}
        },

        'ttl_urukul0_sw3': {
            "type": "local",
            "module": "artiq.coredevice.ttl",
            "class": "TTLOut",
            "arguments": {"channel": 0x000048}
        },

        'urukul0_cpld': {
            "type": "local",
            "module": "artiq.coredevice.urukul",
            "class": "CPLD",
            "arguments": {
                "spi_device": "spi_urukul0",
                "sync_device": None,
                "io_update_device": "ttl_urukul0_io_update",
                "refclk": 1000000000.0,
                "clk_sel": 1,
                "clk_div": 1,
            }
        },

        'urukul0_ch0': {
            "type": "local",
            "module": "artiq.coredevice.ad9910",
            "class": "AD9910",
            "arguments": {
                "pll_en": 0,
                "chip_select": 4,
                "cpld_device": "urukul0_cpld",
                "sw_device": "ttl_urukul0_sw0"
            }
        }
    }

    def setUp(self, **kwargs):
        kwargs['default_resonance_freq'] = self.RESONANCE_FREQ
        kwargs['default_resonance_diff'] = self.RESONANCE_DIFF
        self.env = self.construct_env(_TestSystem, device_db=self._DEVICE_DB, build_kwargs=kwargs)

    @unittest.skipUnless(CI_ENABLED, 'Not in a CI environment, skipping long test')
    def test_ramp(self):
        self.env.dax_init()

        for amp in [10.0, -4.5, -60.8, 5.6, -90.0]:
            with self.subTest(amplitude=amp):
                # Ramp
                self.env.trap_rf.ramp(amp, self.RESONANCE_FREQ)
                self.assertAlmostEqual(self.peek(self.env.trap_rf._trap_rf, 'freq'), self.RESONANCE_FREQ, places=1)
                self.expect(self.env.trap_rf._trap_rf.cpld, f'att_{self.env.trap_rf._trap_rf.chip_select - 4}', 0 * dB)
                device_amp = 10 ** (self.env.trap_rf.dbm_to_db(amp) / 20)
                self.assertAlmostEqual(self.peek(self.env.trap_rf._trap_rf, 'amp'), device_amp, places=4)
                self.assertEqual(self.env.trap_rf.get_amp(), amp)
                self.assertEqual(self.env.trap_rf.get_freq(), self.RESONANCE_FREQ)

    @unittest.skipUnless(CI_ENABLED, 'Not in a CI environment, skipping long test')
    def test_ramp_only_module_init(self):
        # Initialize only trap DAC module
        self.env.trap_rf.init()

        # Ramp
        self.expect(self.env.trap_rf._trap_rf.cpld, 'init_att', 'x')
        self.env.trap_rf.ramp(10 * dB, self.RESONANCE_FREQ)
        self.expect(self.env.trap_rf._trap_rf.cpld, 'init_att', True)
        self.assertAlmostEqual(self.peek(self.env.trap_rf._trap_rf, 'freq'), self.RESONANCE_FREQ, places=1)
        self.expect(self.env.trap_rf._trap_rf.cpld, f'att_{self.env.trap_rf._trap_rf.chip_select - 4}', 0 * dB)

    @unittest.skipUnless(CI_ENABLED, 'Not in a CI environment, skipping long test')
    def test_re_ramp(self):
        self.env.dax_init()

        def check():
            self.assertAlmostEqual(self.peek(self.env.trap_rf._trap_rf, 'freq'), self.RESONANCE_FREQ, places=1)
            self.expect(self.env.trap_rf._trap_rf.cpld, f'att_{self.env.trap_rf._trap_rf.chip_select - 4}', 0 * dB)
            device_amp = 10 ** (self.env.trap_rf.dbm_to_db(amp) / 20)
            self.assertAlmostEqual(self.peek(self.env.trap_rf._trap_rf, 'amp'), device_amp, places=4)
            self.assertEqual(self.env.trap_rf.get_amp(), amp)
            self.assertEqual(self.env.trap_rf.get_freq(), self.RESONANCE_FREQ)

        for amp in [-4.5, -60.8]:
            with self.subTest(amplitude=amp):
                self.env.trap_rf.ramp(amp, self.RESONANCE_FREQ)
                check()
                self.env.trap_rf.re_ramp()
                check()

    def test_dbm_db_conversion(self):
        self.env.dax_init()

        data = [3.5, 6.0, -60.9, -50.7, -26.5, -10.0]
        converted = [self.env.trap_rf.db_to_dbm(self.env.trap_rf.dbm_to_db(d)) for d in data]
        for d, c in zip(data, converted):
            with self.subTest(data=d, converted=c):
                self.assertAlmostEqual(d, c)

    def test_illegal_values(self):
        self.env.dax_init()

        with self.assertRaises(AssertionError, msg='Did not raise for too high amplitude'):
            self.env.trap_rf.ramp(10.1, self.RESONANCE_FREQ)
        with self.assertRaises(ValueError, msg='Did not raise for off resonance frequency'):
            self.env.trap_rf.ramp(0.1, self.RESONANCE_FREQ + 2 * self.RESONANCE_DIFF)
        with self.assertRaises(ValueError, msg='Did not raise for off resonance frequency'):
            self.env.trap_rf.ramp(0.1, self.RESONANCE_FREQ - 2 * self.RESONANCE_DIFF)
        with self.assertLogs(self.env.trap_rf.logger, level=logging.WARNING):
            self.env.trap_rf.ramp(-85.0, self.RESONANCE_FREQ)

    @unittest.skipUnless(CI_ENABLED, 'Not in a CI environment, skipping long test')
    def test_max_amplitude(self):
        for max_amp in [9.0, 9.9, -60.8, -5.6, 2.7]:  # Not using 10.0 as it will give an assertion error first!
            with self.subTest(max_amp=max_amp):
                # Set max amp
                self.env.trap_rf.update_max_amp(max_amp)
                self.env.trap_rf.init()  # Needs initialization to pick up new configuration
                # Test
                with self.assertRaises(ValueError, msg='Did not raise for too high amplitude'):
                    self.env.trap_rf.ramp(max_amp + 0.01, self.RESONANCE_FREQ)
                self.env.trap_rf.ramp(max_amp, self.RESONANCE_FREQ)

    def test_is_enabled(self):
        self.env.dax_init()

        with self.assertRaises(KeyError, msg='Did not raise on trap RF enabled flag after reboot '
                                             'and without initialization'):
            # This key error happens due to simplified boot simulation behavior
            self.env.trap_rf.is_enabled()
        self.env.trap_rf.ramp(-70.0, self.RESONANCE_FREQ, enabled=True)
        self.assertTrue(self.env.trap_rf.is_enabled(), 'Trap RF enabled flag not set correctly')
        self.env.trap_rf.ramp(-70.0, self.RESONANCE_FREQ, enabled=False)
        self.assertFalse(self.env.trap_rf.is_enabled(), 'Trap RF enabled flag not set correctly')

    @unittest.skipUnless(CI_ENABLED, 'Not in a CI environment, skipping long test')
    def test_safety_ramp(self):
        self.env.dax_init()

        for amp in [-10.0, -4.5, -60.8]:
            with self.subTest(amplitude=amp):
                # Ramp
                self.env.trap_rf.ramp(amp, self.RESONANCE_FREQ)
                # Safety ramp down
                self.env.trap_rf.safety_ramp_down()
                self.assertFalse(self.env.trap_rf.is_enabled(), 'Trap RF enabled flag not set correctly')

    def test_update_functions(self):
        self.env.trap_rf.update_max_amp(-20.0)
        self.env.trap_rf.update_ramp_slope(2 * dB)
        self.env.trap_rf.update_resonance_freq(40 * MHz)
        self.env.trap_rf.update_resonance_diff(1 * MHz)
        self.env.trap_rf.update_ramp_compression(True)
