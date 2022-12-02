from artiq.experiment import *

import dax.sim.test_case
import dax.sim.coredevice.phaser
import dax.sim.signal
from dax.util.artiq_version import ARTIQ_MAJOR_VERSION
import test.sim.coredevice._compile_testcase as compile_testcase


class _Environment(HasEnvironment):
    def build(self):
        self.core = self.get_device('core')
        self.dut: dax.sim.coredevice.phaser.Phaser = self.get_device('dut')


class PhaserBasebandTestCase(dax.sim.test_case.PeekTestCase):
    def setUp(self):
        device_db = {
            'core': {
                'type': 'local',
                'module': 'artiq.coredevice.core',
                'class': 'Core',
                'arguments': {'host': None, 'ref_period': 1e-9}
            },
            'dut': {
                'type': 'local',
                'module': 'artiq.coredevice.phaser',
                'class': 'Phaser',
                'arguments': {
                    'channel_base': 0,
                    'miso_delay': 1,
                },
            },
        }
        self.env = self.construct_env(_Environment, device_db=device_db)

    def _test_uninitialized(self):
        self.expect(self.env.dut, 'leds', 'x')
        self.expect(self.env.dut, 'fan', 'x')
        self.expect(self.env.dut, 'dac_sleep', 0)
        self.expect(self.env.dut, 'dac_cmix', 'x')
        for i in range(len(self.env.dut.channel)):
            self.expect(self.env.dut, f'att{i}_rstn', 1)

            ch_key_base = f'ch_{i}'
            ch_keys = ['_duc_freq', '_duc_phase', '_nco_freq', '_nco_phase', '_att']
            for key in ch_keys:
                self.expect(self.env.dut, f'{ch_key_base}{key}', 'x')

            for j in range(len(self.env.dut.channel[i].oscillator)):
                osc_key_base = f'{ch_key_base}_osc_{j}'
                osc_keys = ['_freq', '_amp', '_phase', '_clr']
                for key in osc_keys:
                    self.expect(self.env.dut, f'{osc_key_base}{key}', 'x')

    def test_init(self):
        self._test_uninitialized()
        self.env.dut.init()
        self.expect(self.env.dut, 'leds', '000000')
        self.expect(self.env.dut, 'fan', 0.)
        self.expect(self.env.dut, 'dac_sleep', 0)
        self.expect(self.env.dut, 'dac_cmix', 'x')
        for i in range(len(self.env.dut.channel)):
            self.expect(self.env.dut, f'att{i}_rstn', 1)
            ch_key_base = f'ch_{i}'
            self.expect(self.env.dut, f'{ch_key_base}_duc_freq', 'x')
            self.expect(self.env.dut, f'{ch_key_base}_duc_phase', 0x6000 / (1 << 16))
            self.expect(self.env.dut, f'{ch_key_base}_nco_freq', 'x')
            self.expect(self.env.dut, f'{ch_key_base}_nco_phase', 'x')
            self.expect(self.env.dut, f'{ch_key_base}_att', 0xff / 8)
            for j in range(len(self.env.dut.channel[i].oscillator)):
                osc_key_base = f'{ch_key_base}_osc_{j}'
                self.expect(self.env.dut, f'{osc_key_base}_freq', 'x')
                self.expect(self.env.dut, f'{osc_key_base}_amp', 0.)
                self.expect(self.env.dut, f'{osc_key_base}_phase', 0xc000 / (1 << 16))
                self.expect(self.env.dut, f'{osc_key_base}_clr', 1)

    def test_set_leds(self):
        self._test_uninitialized()
        self.env.dut.set_leds(0x2a)
        self.expect(self.env.dut, 'leds', '101010')

    def test_set_fan_mu(self):
        self._test_uninitialized()
        self.env.dut.set_fan_mu(128)
        self.expect(self.env.dut, 'fan', 128 / 255)

    def test_set_fan(self):
        self._test_uninitialized()
        self.env.dut.set_fan(0.5)
        self.expect(self.env.dut, 'fan', 0.5)
        with self.assertRaises(ValueError):
            self.env.dut.set_fan(2.)
        self.expect(self.env.dut, 'fan', 0.5)

    # separate tests for set_cfg -> dac_sleep and att_rst
    def test_dac_sleep(self):
        self._test_uninitialized()
        self.env.dut.set_cfg(dac_sleep=1)
        self.expect(self.env.dut, 'dac_sleep', 1)

    def test_att_rst(self):
        self._test_uninitialized()
        self.env.dut.set_cfg(att0_rstn=0, att1_rstn=0)
        for i in range(len(self.env.dut.channel)):
            self.expect(self.env.dut, f'att{i}_rstn', 0)
            with self.assertRaises(AssertionError):
                self.env.dut.channel[i].set_att(1 * dB)

        self.env.dut.set_cfg(att0_rstn=1, att1_rstn=1)
        for i in range(len(self.env.dut.channel)):
            self.expect(self.env.dut, f'att{i}_rstn', 1)
            self.expect(self.env.dut, f'ch_{i}_att', 0 * dB)

    def test_get_dac_temperature(self):
        self._test_uninitialized()
        self.assertEqual(self.env.dut.get_dac_temperature(), 30)

    def test_get_dac_alarms(self):
        self._test_uninitialized()
        self.assertEqual(self.env.dut.get_dac_alarms(), 0)

    # PhaserChannel functions
    def test_set_duc_cfg(self):
        self._test_uninitialized()
        for i in range(len(self.env.dut.channel)):
            with self.assertRaises(AssertionError):
                self.env.dut.channel[i].set_duc_cfg(select=1)

    def set_duc_frequency_mu(self):
        self._test_uninitialized()
        for i in range(len(self.env.dut.channel)):
            self.env.dut.channel[i].set_duc_frequency_mu(0xaaaaaaaa)
            self.expect(self.env.dut, f'ch_{i}_duc_freq', 0xaaaaaaaa * 125 * MHz / (1 << 30))

    def test_set_duc_frequency(self):
        self._test_uninitialized()
        for i in range(len(self.env.dut.channel)):
            self.env.dut.channel[i].set_duc_frequency(100 * MHz)
            self.expect(self.env.dut, f'ch_{i}_duc_freq', 100 * MHz)

    def test_set_duc_phase_mu(self):
        self._test_uninitialized()
        for i in range(len(self.env.dut.channel)):
            self.env.dut.channel[i].set_duc_phase_mu(0xaaaa)
            self.expect(self.env.dut, f'ch_{i}_duc_phase', 0xaaaa / (1 << 16))

    def test_set_duc_phase(self):
        self._test_uninitialized()
        for i in range(len(self.env.dut.channel)):
            self.env.dut.channel[i].set_duc_phase(0.5)
            self.expect(self.env.dut, f'ch_{i}_duc_phase', 0.5)

    def test_set_nco_frequency_mu(self):
        self._test_uninitialized()
        for i in range(len(self.env.dut.channel)):
            self.env.dut.channel[i].set_nco_frequency_mu(0xaaaaaaaa)
            self.expect(self.env.dut, f'ch_{i}_nco_freq', 0xaaaaaaaa * 250 * MHz / (1 << 30))

    def test_set_nco_frequency(self):
        self._test_uninitialized()
        for i in range(len(self.env.dut.channel)):
            self.env.dut.channel[i].set_nco_frequency(100 * MHz)
            self.expect(self.env.dut, f'ch_{i}_nco_freq', 100 * MHz)

    def test_set_nco_phase_mu(self):
        self._test_uninitialized()
        for i in range(len(self.env.dut.channel)):
            self.env.dut.channel[i].set_nco_phase_mu(0xaaaa)
            self.expect(self.env.dut, f'ch_{i}_nco_phase', 0xaaaa / (1 << 16))

    def test_set_nco_phase(self):
        self._test_uninitialized()
        for i in range(len(self.env.dut.channel)):
            self.env.dut.channel[i].set_nco_phase(0.5)
            self.expect(self.env.dut, f'ch_{i}_nco_phase', 0.5)

    def test_set_att_mu(self):
        self._test_uninitialized()
        for i in range(len(self.env.dut.channel)):
            self.env.dut.channel[i].set_att_mu(0xaa)
            self.expect(self.env.dut, f'ch_{i}_att', (0xff - 0xaa) / 8)

    def test_set_att(self):
        self._test_uninitialized()
        for i in range(len(self.env.dut.channel)):
            self.env.dut.channel[i].set_att(5 * dB)
            self.expect(self.env.dut, f'ch_{i}_att', 5 * dB)
            with self.assertRaises(ValueError):
                self.env.dut.channel[i].set_att(100 * dB)

    def test_get_att_mu(self):
        self._test_uninitialized()
        for i in range(len(self.env.dut.channel)):
            with self.assertRaises(dax.sim.signal.SignalNotSetError):
                self.env.dut.channel[i].get_att_mu()
            self.env.dut.channel[i].set_att(5 * dB)
            self.assertEqual(self.env.dut.channel[i].get_att_mu(), 0xff - 5 * dB * 8)

    # PhaserOscillator functions
    def test_set_frequency_mu(self):
        self._test_uninitialized()
        for i in range(len(self.env.dut.channel)):
            for j in range(len(self.env.dut.channel[i].oscillator)):
                self.env.dut.channel[i].oscillator[j].set_frequency_mu(0xaaaaaaaa)
                self.expect(self.env.dut, f'ch_{i}_osc_{j}_freq', 0xaaaaaaaa * 6.25 * MHz / (1 << 30))

    # PhaserOscillator functions
    def test_set_frequency(self):
        self._test_uninitialized()
        for i in range(len(self.env.dut.channel)):
            for j in range(len(self.env.dut.channel[i].oscillator)):
                self.env.dut.channel[i].oscillator[j].set_frequency(1 * MHz)
                self.expect(self.env.dut, f'ch_{i}_osc_{j}_freq', 1 * MHz)

    def test_set_amplitude_phase_mu(self):
        self._test_uninitialized()
        for i in range(len(self.env.dut.channel)):
            for j in range(len(self.env.dut.channel[i].oscillator)):
                self.env.dut.channel[i].oscillator[j].set_amplitude_phase_mu(asf=0x6aaa, pow=0xaaaa, clr=1)
                self.expect(self.env.dut, f'ch_{i}_osc_{j}_amp', 0x6aaa / 0x7fff)
                self.expect(self.env.dut, f'ch_{i}_osc_{j}_phase', 0xaaaa / (1 << 16))
                self.expect(self.env.dut, f'ch_{i}_osc_{j}_clr', 1)

    def test_set_amplitude_phase(self):
        self._test_uninitialized()
        for i in range(len(self.env.dut.channel)):
            for j in range(len(self.env.dut.channel[i].oscillator)):
                self.env.dut.channel[i].oscillator[j].set_amplitude_phase(0.5, phase=0.5, clr=1)
                self.expect(self.env.dut, f'ch_{i}_osc_{j}_amp', 0.5)
                self.expect(self.env.dut, f'ch_{i}_osc_{j}_phase', 0.5)
                self.expect(self.env.dut, f'ch_{i}_osc_{j}_clr', 1)


class PhaserUpconverterTestCase(dax.sim.test_case.PeekTestCase):
    def setUp(self):
        device_db = {
            'core': {
                'type': 'local',
                'module': 'artiq.coredevice.core',
                'class': 'Core',
                'arguments': {'host': None, 'ref_period': 1e-9}
            },
            'dut': {
                'type': 'local',
                'module': 'artiq.coredevice.phaser',
                'class': 'Phaser',
                'arguments': {
                    'channel_base': 0,
                    'miso_delay': 1,
                },
                'sim_args': {
                    'is_baseband': False,
                },
            },
        }
        self.env = self.construct_env(_Environment, device_db=device_db)

    def _test_uninitialized(self):
        for i in range(len(self.env.dut.channel)):
            self.expect(self.env.dut, f'trf{i}_ps', 0)
            self.expect(self.env.dut, f'ch_{i}_trf_freq', 'x')

    def test_init(self):
        self._test_uninitialized()
        self.env.dut.init()
        for i in range(len(self.env.dut.channel)):
            self.expect(self.env.dut, f'trf{i}_ps', 0)
            self.expect(self.env.dut, f'ch_{i}_trf_freq', 2.875 * GHz if ARTIQ_MAJOR_VERSION >= 7 else 1.25 * GHz)


class CompileTestCase(compile_testcase.CoredeviceCompileTestCase):
    DEVICE_CLASS = dax.sim.coredevice.phaser.Phaser
    DEVICE_KWARGS = {
        'channel_base': 0,
        'miso_delay': 1,
        'is_baseband': False,
    }
    FN_KWARGS = {
        'set_leds': {'leds': 0},
        'set_fan_mu': {'pwm': 0},
        'set_fan': {'duty': 0.0},
    }
    if ARTIQ_MAJOR_VERSION >= 7:
        FN_KWARGS.update({
            'set_dac_cmix': {'fs_8_step': 0},
        })
    FN_EXCLUDE = {'write8', 'read8', 'write32', 'read32', 'set_sync_dly', 'spi_cfg', 'spi_write', 'dac_write',
                  'dac_read', 'dac_iotest'}
