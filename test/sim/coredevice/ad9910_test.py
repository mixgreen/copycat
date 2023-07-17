import random

from artiq.experiment import *
from artiq.coredevice.ad9910 import RAM_DEST_ASF  # type: ignore[import]

import dax.sim.test_case
import dax.sim.coredevice.urukul
import dax.sim.coredevice.ad9910

import test.sim.coredevice._compile_testcase as compile_testcase
from test.environment import CI_ENABLED

_NUM_SAMPLES = 1000 if CI_ENABLED else 100

_DEVICE_DB = {
    'core': {
        'type': 'local',
        'module': 'artiq.coredevice.core',
        'class': 'Core',
        'arguments': {'host': None, 'ref_period': 1e-9}
    },
    "dut": {
        "type": "local",
        "module": "artiq.coredevice.ad9910",
        "class": "AD9910",
        "arguments": {
            "pll_en": 0,
            "chip_select": 6,
            "cpld_device": "cpld",
        }
    },
    "dut1": {
        "type": "local",
        "module": "artiq.coredevice.ad9910",
        "class": "AD9910",
        "arguments": {
            "chip_select": 4,
            "cpld_device": "cpld",
            "pll_en": 0,
            "sync_delay_seed": 15,
            "io_update_delay": 3
        }
    },
    "dut2": {
        "type": "local",
        "module": "artiq.coredevice.ad9910",
        "class": "AD9910",
        "arguments": {
            "chip_select": 4,
            "cpld_device": "cpld",
            "pll_en": 0,
            "sync_delay_seed": "eeprom_urukul0:00",
            "io_update_delay": "eeprom_urukul0:00"
        }
    },
    "cpld": {
        "type": "local",
        "module": "artiq.coredevice.urukul",
        "class": "CPLD",
        "arguments": {
            "spi_device": "spi_urukul0",
            "refclk": 1e9,
            "clk_div": 1
        }
    },
    'spi_urukul0': {
        'type': 'local',
        'module': 'artiq.coredevice.spi2',
        'class': 'SPIMaster',
    },
}


class _Environment(HasEnvironment):
    def build(self):
        self.core = self.get_device('core')
        self.dut = self.get_device('dut')
        self.dut1 = self.get_device('dut1')
        self.dut2 = self.get_device('dut2')


class AD9910TestCase(dax.sim.test_case.PeekTestCase):
    SEED = None

    def setUp(self) -> None:
        self.rng = random.Random(self.SEED)
        self.env = self.construct_env(_Environment, device_db=_DEVICE_DB)

    def _test_uninitialized(self):
        self.expect(self.env.dut, 'init', 'x')
        self.expect(self.env.dut, 'freq', 'x')
        self.expect(self.env.dut, 'phase', 'x')
        self.expect(self.env.dut, 'phase_mode', 'x')
        self.expect(self.env.dut, 'amp', 'x')
        self.expect(self.env.dut, 'ram_enable', 'x')
        self.expect(self.env.dut, 'ram_dest', 'x')
        self.assertEqual(self.env.dut.sync_data.sync_delay_seed, -1)
        self.assertEqual(self.env.dut.sync_data.io_update_delay, 0)

    def test_init(self):
        self._test_uninitialized()
        self.env.dut.init()
        self.expect(self.env.dut, 'init', 1)
        self.expect(self.env.dut, 'ram_enable', 0)
        self.expect(self.env.dut, 'ram_dest', '00')
        self.assertEqual(self.env.dut.sync_data.sync_delay_seed, -1)
        self.assertEqual(self.env.dut.sync_data.io_update_delay, 0)

    def test_init_sync_data(self):
        self._test_uninitialized()
        self.env.dut1.init()
        self.env.dut2.init()
        for d in self.env.dut1, self.env.dut2:
            self.expect(d, 'init', 1)
            self.expect(d, 'ram_enable', 0)
            self.expect(d, 'ram_dest', '00')

        self.assertEqual(self.env.dut1.sync_data.sync_delay_seed, 15)
        self.assertEqual(self.env.dut1.sync_data.io_update_delay, 3)
        self.assertEqual(self.env.dut2.sync_data.sync_delay_seed, 0)
        self.assertEqual(self.env.dut2.sync_data.io_update_delay, 0)

    def test_phase_mode_timing(self):
        self._test_uninitialized()
        self.env.dut.set_phase_mode(dax.sim.coredevice.ad9910.PHASE_MODE_ABSOLUTE)
        self._test_uninitialized()
        self.env.dut.set(100 * MHz)
        self.expect(self.env.dut, 'phase_mode', '01')

    def test_default_phase_mode_timing(self):
        self._test_uninitialized()
        self.env.dut.set_mu(2 ** 30)
        self.expect(self.env.dut, 'phase_mode', '00')

    def test_set_mu(self):
        self._test_uninitialized()
        for _ in range(_NUM_SAMPLES):
            f = self.rng.uniform(0 * MHz, 400 * MHz)
            p = self.rng.randrange(2 ** 16)
            a = self.rng.randrange(2 ** 14)
            self.env.dut.set_mu(self.env.dut.frequency_to_ftw(f), pow_=p, asf=a)
            self.expect_close(self.env.dut, 'freq', f, places=0)
            self.expect(self.env.dut, 'phase', self.env.dut.pow_to_turns(p))
            self.expect(self.env.dut, 'amp', self.env.dut.asf_to_amplitude(a))

    def test_set(self):
        self._test_uninitialized()
        for _ in range(_NUM_SAMPLES):
            f = self.rng.uniform(0 * MHz, 400 * MHz)
            p = self.rng.uniform(0.0, self.env.dut.pow_to_turns(2 ** 16))
            a = self.rng.uniform(0.0, 1.0)
            self.env.dut.set(f, phase=p, amplitude=a)
            self.expect(self.env.dut, 'freq', f)
            self.expect(self.env.dut, 'phase', p)
            self.expect(self.env.dut, 'amp', a)

    def test_set_att(self):
        signal = f'att_{self.env.dut.chip_select - 4}'
        self.expect(self.env.dut.cpld, signal, 'x')
        for _ in range(_NUM_SAMPLES):
            att = self.rng.uniform(0.0, 31.5)
            self.env.dut.set_att(att)
            self.expect(self.env.dut.cpld, signal, att)

    def test_set_att_mu(self):
        signal = f'att_{self.env.dut.chip_select - 4}'
        self.expect(self.env.dut.cpld, signal, 'x')
        self.env.dut.set_att_mu(255)
        self.expect(self.env.dut.cpld, signal, 0 * dB)

    def test_cfg_sw(self):
        ref = '0001000'
        index = self.env.dut.chip_select - 4
        for state in [0, 1]:
            self.env.dut.cfg_sw(state)
            value = ref[index:4 + index] if state else '0000'
            assert value[-1 - index] == str(state)
            self.expect(self.env.dut.cpld, 'sw', value)

    def test_ram_conversion(self):
        self.env.dut.frequency_to_ram([100 * MHz], [0])
        self.env.dut.turns_to_ram([0.1], [0])
        self.env.dut.amplitude_to_ram([0.1], [0])
        self.env.dut.turns_amplitude_to_ram([0.1], [0.5], [0])

    def test_ram_mode(self, *, num_steps=100):
        amps = [i / num_steps for i in range(num_steps)]
        amps_ram = [0] * num_steps
        self.env.dut.amplitude_to_ram(amps, amps_ram)

        self.env.dut.cpld.init()
        self.env.dut.init()
        self.env.dut.set_cfr1(ram_enable=0)
        self.env.dut.cpld.io_update.pulse_mu(8)
        self.expect(self.env.dut, 'ram_enable', 0)

        self.env.dut.cpld.set_profile(dax.sim.coredevice.urukul.DEFAULT_PROFILE)
        self.env.dut.set_profile_ram(start=0, end=100)
        self.env.dut.cpld.io_update.pulse_mu(8)
        self.env.dut.write_ram(amps_ram)
        self.env.dut.set_frequency(200 * MHz)
        self.env.dut.set_phase(0.0)
        self.env.dut.set_amplitude(1.0)  # Just added for testing coverage
        self.env.dut.set_cfr1(ram_enable=1, ram_destination=RAM_DEST_ASF)
        self.expect(self.env.dut, 'ram_enable', 0)
        self.env.dut.cpld.io_update.pulse_mu(8)
        self.expect(self.env.dut, 'ram_enable', 1)
        self.expect(self.env.dut, 'ram_dest', f'{RAM_DEST_ASF:02b}')


class CompileTestCase(compile_testcase.CoredeviceCompileTestCase):
    DEVICE_CLASS = dax.sim.coredevice.ad9910.AD9910
    DEVICE_KWARGS = {
        'chip_select': 4,
        'cpld_device': 'cpld',
        'pll_en': 0,
    }
    FN_KWARGS = {
        'set_phase_mode': {'phase_mode': 0},
        'set_mu': {'ftw': 0},
        'frequency_to_ftw': {'frequency': 0.0},
        'ftw_to_frequency': {'ftw': 0},
        'turns_to_pow': {'turns': 0.0},
        'pow_to_turns': {'pow_': 0},
        'amplitude_to_asf': {'amplitude': 0.0},
        'asf_to_amplitude': {'asf': 0},
        'set': {'frequency': 0.0},
        'set_att_mu': {'att': 0},
        'set_att': {'att': 0.0},
        'cfg_sw': {'state': False},
    }
    FN_EXCLUDE = {'write16', 'write32', 'write64', 'read16', 'read32', 'read64',
                  'write_ram', 'read_ram', 'set_profile_ram',
                  'frequency_to_ram', 'turns_to_ram', 'amplitude_to_ram', 'turns_amplitude_to_ram',
                  'set_ftw', 'set_pow', 'set_asf', 'set_frequency', 'set_phase', 'set_amplitude',
                  'set_sync', 'measure_io_update_alignment'}
    DEVICE_DB = _DEVICE_DB


class SyncDelayCompileTestCase(CompileTestCase):
    DEVICE_KWARGS = {
        "chip_select": 5,
        'cpld_device': 'cpld',
        "pll_en": 0,
        "sync_delay_seed": 15,
        "io_update_delay": 3
    }


class SyncDelayEEPROMCompileTestCase(SyncDelayCompileTestCase):
    DEVICE_KWARGS = {
        "chip_select": 5,
        'cpld_device': 'cpld',
        "pll_en": 0,
        "sync_delay_seed": "eeprom_urukul0:00",
        "io_update_delay": "eeprom_urukul0:00"
    }
