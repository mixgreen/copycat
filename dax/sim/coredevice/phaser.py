# mypy: disallow_untyped_defs = False
# mypy: disallow_incomplete_defs = False
# mypy: check_untyped_defs = False

from typing import List
from numpy import int32, int64

from artiq.language.core import kernel, delay_mu, delay, now_mu
from artiq.language.units import ns, us, ms, MHz, dB
from artiq.language.types import TInt32
from artiq.coredevice.dac34h84 import DAC34H84  # type: ignore
from artiq.coredevice.trf372017 import TRF372017  # type: ignore

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
        self.dac_mmap = DAC34H84(dac).get_mmap()
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
    def init(self, debug=False):  # noqa: C901
        # copied from ARTIQ with unsupported methods commented out
        # board_id = self.read8(PHASER_ADDR_BOARD_ID)
        # if board_id != PHASER_BOARD_ID:
        #     raise ValueError("invalid board id")
        delay(.1 * ms)  # slack

        # hw_rev = self.read8(PHASER_ADDR_HW_REV)
        delay(.1 * ms)  # slack
        # todo: if this is the only way to distinguish between baseband/upconverter variants then I'm not sure
        #  how to decide how long this method takes (there's a `continue` based on variant in the loop below)
        # is_baseband = hw_rev & PHASER_HW_REV_VARIANT

        # gw_rev = self.read8(PHASER_ADDR_GW_REV)
        if debug:
            # print("gw_rev:", gw_rev)
            self.core.break_realtime()
        delay(.1 * ms)  # slack

        # allow a few errors during startup and alignment since boot
        # if self.get_crc_err() > 20:
        #     raise ValueError("large number of frame CRC errors")
        delay(.1 * ms)  # slack

        # determine the origin for frame-aligned timestamps
        self.measure_frame_timestamp()
        if self.frame_tstamp < 0:
            raise ValueError("frame timestamp measurement timed out")
        delay(.1 * ms)

        # reset
        self.set_cfg(dac_resetb=0, dac_sleep=1, dac_txena=0,
                     trf0_ps=1, trf1_ps=1,
                     att0_rstn=0, att1_rstn=0)
        self.set_leds(0x00)
        self.set_fan_mu(0)
        # bring dac out of reset, keep tx off
        self.set_cfg(clk_sel=self.clk_sel, dac_txena=0,
                     trf0_ps=1, trf1_ps=1,
                     att0_rstn=0, att1_rstn=0)
        delay(.1 * ms)  # slack

        # crossing dac_clk (reference) edges with sync_dly
        # changes the optimal fifo_offset by 4
        # self.set_sync_dly(self.sync_dly)

        # 4 wire SPI, sif4_enable
        # self.dac_write(0x02, 0x0080)
        self._dac_write_delay()
        # if self.dac_read(0x7f) != 0x5409:
        #     raise ValueError("DAC version readback invalid")
        self._dac_read_delay()
        delay(.1 * ms)
        # if self.dac_read(0x00) != 0x049c:
        #     raise ValueError("DAC config0 reset readback invalid")
        self._dac_read_delay()
        delay(.1 * ms)

        t = self.get_dac_temperature()
        delay(.1 * ms)
        if t < 10 or t > 90:
            raise ValueError("DAC temperature out of bounds")

        for data in self.dac_mmap:
            # self.dac_write(data >> 16, data)
            self._dac_write_delay()
            delay(40 * us)
        self.dac_sync()
        delay(40 * us)

        # pll_ndivsync_ena disable
        # config18 = self.dac_read(0x18)
        self._dac_read_delay()
        delay(.1 * ms)
        # self.dac_write(0x18, config18 & ~0x0800)
        self._dac_write_delay()

        patterns = [
            [0xf05a, 0x05af, 0x5af0, 0xaf05],  # test channel/iq/byte/nibble
            [0x7a7a, 0xb6b6, 0xeaea, 0x4545],  # datasheet pattern a
            [0x1a1a, 0x1616, 0xaaaa, 0xc6c6],  # datasheet pattern b
        ]
        # A data delay of 2*50 ps heuristically and reproducibly matches
        # FPGA+board+DAC skews. There is plenty of margin (>= 250 ps
        # either side) and no need to tune at runtime.
        # Parity provides another level of safety.
        for i in range(len(patterns)):
            delay(.5 * ms)
            # errors = self.dac_iotest(patterns[i])
            # if errors:
            #     raise ValueError("DAC iotest failure")

        delay(2 * ms)  # let it settle
        # lvolt = self.dac_read(0x18) & 7
        self._dac_read_delay()
        delay(.1 * ms)
        # if lvolt < 2 or lvolt > 5:
        #     raise ValueError("DAC PLL lock failed, check clocking")

        if self.tune_fifo_offset:
            # fifo_offset = self.dac_tune_fifo_offset()
            if debug:
                # print("fifo_offset:", fifo_offset)
                self.core.break_realtime()

        # self.dac_write(0x20, 0x0000)  # stop fifo sync
        # alarm = self.get_sta() & 1
        # delay(.1*ms)
        self.clear_dac_alarms()
        delay(2 * ms)  # let it run a bit
        alarms = self.get_dac_alarms()
        delay(.1 * ms)  # slack
        if alarms & ~0x0040:  # ignore PLL alarms (see DS)
            if debug:
                print("alarms:", alarms)
                self.core.break_realtime()
                # ignore alarms
            else:
                raise ValueError("DAC alarm")

        # avoid malformed output for: mixer_ena=1, nco_ena=0 after power up
        # self.dac_write(self.dac_mmap[2] >> 16, self.dac_mmap[2] | (1 << 4))
        self._dac_write_delay()
        delay(40 * us)
        self.dac_sync()
        delay(100 * us)
        # self.dac_write(self.dac_mmap[2] >> 16, self.dac_mmap[2])
        self._dac_write_delay()
        delay(40 * us)
        self.dac_sync()
        delay(100 * us)

        # power up trfs, release att reset
        self.set_cfg(clk_sel=self.clk_sel, dac_txena=0)

        for ch in range(2):
            channel = self.channel[ch]
            # test attenuator write and readback
            channel.set_att_mu(0x5a)
            if channel.get_att_mu() != 0x5a:
                raise ValueError("attenuator test failed")
            delay(.1 * ms)
            channel.set_att_mu(0x00)  # minimum attenuation

            # test oscillators and DUC
            for i in range(len(channel.oscillator)):
                oscillator = channel.oscillator[i]
                asf = 0
                if i == 0:
                    asf = 0x7fff
                # 6pi/4 phase
                oscillator.set_amplitude_phase_mu(asf=asf, pow=0xc000, clr=1)
                delay(1 * us)
            # 3pi/4
            channel.set_duc_phase_mu(0x6000)
            channel.set_duc_cfg(select=0, clr=1)
            self.duc_stb()
            delay(.1 * ms)  # settle link, pipeline and impulse response
            # data = channel.get_dac_data()
            delay(1 * us)
            channel.oscillator[0].set_amplitude_phase_mu(asf=0, pow=0xc000,
                                                         clr=1)
            delay(.1 * ms)
            # sqrt2 = 0x5a81  # 0x7fff/sqrt(2)
            # data_i = data & 0xffff
            # data_q = (data >> 16) & 0xffff
            # allow ripple
            # if data_i < sqrt2 - 30 or data_i > sqrt2 or abs(data_i - data_q) > 2:
            #     raise ValueError("DUC+oscillator phase/amplitude test failed")

            # todo: re: todo from above
            # if is_baseband:
            #     continue

            # if channel.trf_read(0) & 0x7f != 0x68:
            #     raise ValueError("TRF identification failed")
            delay(.1 * ms)

            delay(.2 * ms)
            # for data in channel.trf_mmap:
            #     channel.trf_write(data)
            # channel.cal_trf_vco()

            delay(2 * ms)  # lock
            # if not (self.get_sta() & (PHASER_STA_TRF0_LD << ch)):
            #     raise ValueError("TRF lock failure")
            delay(.1 * ms)
            # if channel.trf_read(0) & 0x1000:
            #     raise ValueError("TRF R_SAT_ERR")
            delay(.1 * ms)
            # channel.en_trf_out()

        # enable dac tx
        self.set_cfg(clk_sel=self.clk_sel)

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
            # todo: basically assumes zero latency, not sure if that's okay.
            #  but the only alternative is picking an arbitrary latency to add
            self.frame_tstamp = now_mu() + 4 * self.t_frame
            delay(100 * us)

        @kernel
        def get_next_frame_mu(self):
            n = int64((now_mu() - self.frame_tstamp) / self.t_frame)
            return self.frame_tstamp + (n + 1) * self.t_frame

    @kernel
    def set_sync_dly(self, dly):
        # todo: this doesn't really affect any of the signals, could just `pass`
        raise NotImplementedError

    @kernel
    def duc_stb(self):
        # supporting PhaserChannel.set_duc_cfg means allowing this to be called as well
        pass

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
    def _dac_write_delay(self):
        # mimic the delay of a dac_write
        div = 34  # 100 ns min period
        t_xfer = self.core.seconds_to_mu((8 + 1) * div * 4 * ns)
        delay_mu(3 * t_xfer)

    @kernel
    def dac_read(self, addr, div=34) -> TInt32:
        raise NotImplementedError

    @kernel
    def _dac_read_delay(self):
        # mimic the delay of a dac_read
        div = 34  # 100 ns min period
        t_xfer = self.core.seconds_to_mu((8 + 1) * div * 4 * ns)
        delay_mu(2 * t_xfer)
        delay(20 * us)
        delay_mu(t_xfer)

    @kernel
    def get_dac_temperature(self) -> TInt32:
        self._dac_read_delay()  # type: ignore
        return int32(30)  # 30 degrees C seems reasonable?

    if ARTIQ_MAJOR_VERSION >= 7:
        @kernel
        def dac_sync(self):
            # just mimic the delay
            self._dac_read_delay()
            delay(1 * ms)
            self._dac_write_delay()
            self._dac_write_delay()

        @kernel
        def set_dac_cmix(self, fs_8_step):
            vals = [0, 125, 250, 375, 500, -375, -250, -125]
            cmix = vals[fs_8_step % 8] * MHz
            self._dac_read_delay()
            delay(0.1 * ms)
            self._dac_write_delay()
            self._dac_cmix.push(cmix)

    @kernel
    def get_dac_alarms(self):
        self._dac_read_delay()
        return 0

    @kernel
    def clear_dac_alarms(self):
        self._dac_write_delay()

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
        self.trf_mmap = TRF372017(trf).get_mmap()
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

    # noinspection PyUnusedLocal
    @kernel
    def set_duc_cfg(self, clr=0, clr_once=0, select=0):
        # extremely trivial implementation, but want to allow this to be called
        assert select == 0, 'DUC test data not supported'
        # ignore phase accumulator

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
        div = 34
        t_xfer = self.phaser.core.seconds_to_mu((8 + 1) * div * 4 * ns)
        delay_mu(t_xfer)
        delay(20 * us)
        delay_mu(t_xfer)
        # allow SignalNotSetError to be raised if not set
        return int32(0xff - self._att.pull() * 8)

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
