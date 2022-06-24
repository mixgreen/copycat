# mypy: disallow_untyped_defs = False
# mypy: disallow_incomplete_defs = False
# mypy: check_untyped_defs = False

from typing import List
from numpy import int32, int64

from artiq.language.core import kernel, delay_mu, delay, now_mu
from artiq.language.units import ns, us, ms, MHz, dB
from artiq.language.types import TInt32

from dax.sim.device import DaxSimDevice, ARTIQ_MAJOR_VERSION
from dax.sim.signal import get_signal_manager


class Phaser(DaxSimDevice):

    # noinspection PyUnusedLocal
    def __init__(self, dmgr, channel_base, miso_delay=1, tune_fifo_offset=True,
                 clk_sel=0, sync_dly=0, dac=None, trf0=None, trf1=None, **kwargs):
        super(Phaser, self).__init__(dmgr, **kwargs)
        self.channel_base = channel_base
        self.miso_delay = miso_delay
        assert self.core.ref_period == 1 * ns
        self.t_frame = 10 * 8 * 4
        if ARTIQ_MAJOR_VERSION >= 7:
            self.frame_tstamp = int64(0)
        self.clk_sel = clk_sel
        self.tune_fifo_offset = tune_fifo_offset
        self.sync_dly = sync_dly
        self.dac_mmap = None  # todo...?
        self.channel: List[PhaserChannel] = [
            PhaserChannel(self, ch, trf)
            for ch, trf in enumerate([trf0, trf1])
        ]

        # register signals
        signal_manager = get_signal_manager()
        self._leds = signal_manager.register(self, 'leds', str)  # bit/bool vector
        self._fan = signal_manager.register(self, 'fan', float)  # fan duty cycle
        # configuration pins
        self._dac_sleep = signal_manager.register(self, 'dac_sleep', bool, init=0)
        self._trf0_ps = signal_manager.register(self, 'trf0_ps', bool, init=0)
        self._trf1_ps = signal_manager.register(self, 'trf1_ps', bool, init=0)
        self._att_rstn = [signal_manager.register(self, f'att{i}_rstn', bool, init=1) for i in range(2)]  # active low
        # DAC coarse mixer frequency
        self._dac_cmix = signal_manager.register(self, 'dac_cmix', float)

    @kernel
    def init(self, debug=False):
        pass  # todo

    @kernel
    def write8(self, addr, data):
        raise NotImplementedError

    @kernel
    def read8(self, addr) -> TInt32:
        raise NotImplementedError

    @kernel
    def write32(self, addr, data: TInt32):
        raise NotImplementedError

    @kernel
    def read32(self, addr) -> TInt32:
        raise NotImplementedError

    @kernel
    def set_leds(self, leds):
        self._leds.push(f'{leds & 0x3f:06b}')  # 6 bits

    @kernel
    def set_fan_mu(self, pwm):
        self.set_fan(pwm / 255)

    @kernel
    def set_fan(self, duty):
        if duty < 0 or duty > 1:
            raise ValueError("duty cycle out of bounds")
        self._fan.push(duty)

    @kernel
    def set_cfg(self, clk_sel=0, dac_resetb=1, dac_sleep=0, dac_txena=1,
                trf0_ps=0, trf1_ps=0, att0_rstn=1, att1_rstn=1):
        self._dac_sleep.push(dac_sleep)
        self._trf0_ps.push(trf0_ps)
        self._trf1_ps.push(trf1_ps)
        for ch, att_rstn in enumerate([att0_rstn, att1_rstn]):
            # set channel attenuation to zero on release
            # todo: it's unclear if the reset should be on falling or rising (release) edge
            released = self._att_rstn[ch].pull() == 0 and att_rstn == 1
            self._att_rstn[ch].push(att_rstn)
            if released:
                self.channel[ch].set_att(0)

    @kernel
    def get_sta(self):
        raise NotImplementedError

    @kernel
    def get_crc_err(self):
        raise NotImplementedError

    if ARTIQ_MAJOR_VERSION >= 7:
        @kernel
        def measure_frame_timestamp(self):
            raise NotImplementedError

        @kernel
        def get_next_frame_mu(self):
            raise NotImplementedError

    @kernel
    def set_sync_dly(self, dly):
        raise NotImplementedError

    @kernel
    def duc_stb(self):
        # todo: don't actually update DUC settings until this gets called
        raise NotImplementedError

    @kernel
    def spi_cfg(self, select, div, end, clk_phase=0, clk_polarity=0,
                half_duplex=0, lsb_first=0, offline=0, length=8):
        raise NotImplementedError

    @kernel
    def spi_write(self, data):
        raise NotImplementedError

    @kernel
    def spi_read(self):
        raise NotImplementedError

    @kernel
    def dac_write(self, addr, data):
        raise NotImplementedError

    @kernel
    def dac_read(self, addr, div=34) -> TInt32:
        raise NotImplementedError

    @kernel
    def get_dac_temperature(self) -> TInt32:
        raise NotImplementedError

    if ARTIQ_MAJOR_VERSION >= 7:
        @kernel
        def dac_sync(self):
            raise NotImplementedError

        @kernel
        def set_dac_cmix(self, fs_8_step):
            raise NotImplementedError

    @kernel
    def get_dac_alarms(self):
        raise NotImplementedError

    @kernel
    def clear_dac_alarms(self):
        raise NotImplementedError

    @kernel
    def dac_iotest(self, pattern) -> TInt32:
        raise NotImplementedError

    @kernel
    def dac_tune_fifo_offset(self):
        raise NotImplementedError


class PhaserChannel:
    def __init__(self, phaser, index, trf):
        self.phaser = phaser
        self.index = index
        self.trf_mmap = None  # todo?
        self.oscillator = [PhaserOscillator(self, osc) for osc in range(5)]

        # register signals
        signal_manager = get_signal_manager()
        key_base = f'ch_{index}'
        self._duc_freq = signal_manager.register(phaser, f'{key_base}_duc_freq', float)
        self._duc_phase = signal_manager.register(phaser, f'{key_base}_duc_phase', float)
        self._nco_freq = signal_manager.register(phaser, f'{key_base}_nco_freq', float)
        self._nco_phase = signal_manager.register(phaser, f'{key_base}_nco_phase', float)
        self._att = signal_manager.register(phaser, f'{key_base}_att', float)

    @kernel
    def get_dac_data(self) -> TInt32:
        raise NotImplementedError

    @kernel
    def set_dac_test(self, data: TInt32):
        raise NotImplementedError

    @kernel
    def set_duc_cfg(self, clr=0, clr_once=0, select=0):
        raise NotImplementedError

    @kernel
    def set_duc_frequency_mu(self, ftw):
        self.set_duc_frequency(ftw * 125 * MHz / (1 << 30))

    @kernel
    def set_duc_frequency(self, frequency):
        self._duc_freq.push(frequency)

    @kernel
    def set_duc_phase_mu(self, pow):
        self.set_duc_phase(pow / (1 << 16))

    @kernel
    def set_duc_phase(self, phase):
        self._duc_phase.push(phase)

    @kernel
    def set_nco_frequency_mu(self, ftw):
        self.set_nco_frequency(ftw * 250 * MHz / (1 << 30))

    @kernel
    def set_nco_frequency(self, frequency):
        self._nco_freq.push(frequency)

    @kernel
    def set_nco_phase_mu(self, pow):
        self.set_nco_phase(pow / (1 << 16))

    @kernel
    def set_nco_phase(self, phase):
        self._nco_phase.push(phase)

    @kernel
    def set_att_mu(self, data):
        self.set_att((0xff - data) / 8)

    @kernel
    def set_att(self, att):
        # noinspection PyProtectedMember
        assert self.phaser._att_rstn[self.index] == 1, 'Tried to set channel attenuation with reset pin active'
        if att < 0 or att > 31.5 * dB:
            # technically the ARTIQ set_att will allow you to provide an att up to
            # 31.875 dB, but the docstring says it's only a 31.5 dB attenuator
            raise ValueError("attenuation out of bounds")
        # delay from ARTIQ for SPI transfer
        div = 34  # 30 ns min period
        t_xfer = self.phaser.core.seconds_to_mu((8 + 1) * div * 4 * ns)
        delay_mu(t_xfer)
        self._att.push(att)

    @kernel
    def get_att_mu(self) -> TInt32:
        # todo: could implement this, but the only use case in which it would be useful
        #  (crossover between experiments) wouldn't work
        raise NotImplementedError

    @kernel
    def trf_write(self, data, readback=False):
        raise NotImplementedError

    @kernel
    def trf_read(self, addr, cnt_mux_sel=0) -> TInt32:
        raise NotImplementedError

    if ARTIQ_MAJOR_VERSION >= 7:
        @kernel
        def cal_trf_vco(self):
            raise NotImplementedError

        @kernel
        def en_trf_out(self, rf=1, lo=0):
            raise NotImplementedError


class PhaserOscillator:
    def __init__(self, channel, index):
        # attributes from ARTIQ
        self.channel = channel
        self.base_addr = ((self.channel.phaser.channel_base + 1
                           + 2 * self.channel.index) << 8) | index
        # register signals
        signal_manager = get_signal_manager()
        key_base = f'ch_{channel.index}_osc_{index}'
        self._freq = signal_manager.register(channel.phaser, f'{key_base}_freq', float)
        self._amp = signal_manager.register(channel.phaser, f'{key_base}_amp', float)
        self._phase = signal_manager.register(channel.phaser, f'{key_base}_phase', float)
        self._clr = signal_manager.register(channel.phaser, f'{key_base}_clr', bool)

    @kernel
    def set_frequency_mu(self, ftw):
        self.set_frequency(ftw * 6.25 * MHz / (1 << 30))

    @kernel
    def set_frequency(self, frequency):
        self._freq.push(frequency)

    @kernel
    def set_amplitude_phase_mu(self, asf=0x7fff, pow=0, clr=0):
        self.set_amplitude_phase(asf / 0x7fff, phase=(pow / (1 << 16)), clr=(clr & 1))

    @kernel
    def set_amplitude_phase(self, amplitude, phase=0., clr=0):
        self._amp.push(amplitude)
        self._phase.push(phase)
        self._clr.push(clr)
