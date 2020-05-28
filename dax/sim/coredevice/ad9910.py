# mypy: disallow_untyped_defs = False
# mypy: disallow_incomplete_defs = False
# mypy: check_untyped_defs = False

import numpy as np

from artiq.language.core import *
from artiq.language.units import *

from dax.sim.device import DaxSimDevice
from dax.sim.signal import get_signal_manager

# Phase modes
_PHASE_MODE_DEFAULT = -1
PHASE_MODE_CONTINUOUS = 0
PHASE_MODE_ABSOLUTE = 1
PHASE_MODE_TRACKING = 2


class AD9910(DaxSimDevice):

    def __init__(self, dmgr, cpld_device, sw_device=None, pll_n=40, pll_en=1, **kwargs):
        # Call super
        super(AD9910, self).__init__(dmgr, **kwargs)

        # CPLD device
        self.cpld = dmgr.get(cpld_device)
        # Switch device
        if sw_device:
            self.sw = dmgr.get(sw_device)

        # Store attributes (from ARTIQ code)
        clk = self.cpld.refclk / [4, 1, 2, 4][self.cpld.clk_div]
        if pll_en:
            sysclk = clk * pll_n
            assert clk <= 60e6
        else:
            sysclk = clk
        assert sysclk <= 1e9
        self.ftw_per_hz = (1 << 32) / sysclk
        self.sysclk_per_mu = int(round(sysclk * self.core.ref_period))
        self.phase_mode = PHASE_MODE_CONTINUOUS

        # Register signals
        self._signal_manager = get_signal_manager()
        self._init = self._signal_manager.register(self, 'init', bool, size=1)
        self._freq = self._signal_manager.register(self, 'freq', float)
        self._phase = self._signal_manager.register(self, 'phase', float)
        self._phase_mode = self._signal_manager.register(self, 'phase_mode', bool, size=2, init=self.phase_mode)
        self._att = self._signal_manager.register(self, 'att', float)
        self._amp = self._signal_manager.register(self, 'amp', float)
        self._sw = self._signal_manager.register(self, 'sw', bool, size=1)

    @kernel
    def set_phase_mode(self, phase_mode):
        # From ARTIQ code
        self.phase_mode = phase_mode

        # Update signal
        self._signal_manager.event(self._phase_mode, phase_mode)

    @kernel
    def write32(self, addr, data):
        raise NotImplementedError

    @kernel
    def read32(self, addr):
        raise NotImplementedError

    @kernel
    def read64(self, addr):
        raise NotImplementedError

    @kernel
    def write64(self, addr, data_high, data_low):
        raise NotImplementedError

    @kernel
    def write_ram(self, data):
        raise NotImplementedError

    @kernel
    def read_ram(self, data):
        raise NotImplementedError

    @kernel
    def set_cfr1(self, power_down=0b0000, phase_autoclear=0,
                 drg_load_lrr=0, drg_autoclear=0,
                 internal_profile=0, ram_destination=0, ram_enable=0):
        raise NotImplementedError

    @kernel
    def init(self, blind=False):
        # Delays from ARTIQ
        delay(50 * ms)  # slack
        delay(1 * ms)
        if not blind:
            delay(50 * us)  # slack
        delay(1 * ms)
        delay(10 * us)  # slack
        delay(1 * ms)

        # Update signal
        self._signal_manager.event(self._init, True)

    @kernel
    def power_down(self, bits=0b1111):
        raise NotImplementedError

    @kernel
    def set_mu(self, ftw, pow_=0, asf=0x3fff, phase_mode=_PHASE_MODE_DEFAULT,
               ref_time_mu=np.int64(-1), profile=0):
        self.set(self.ftw_to_frequency(ftw), self.pow_to_turns(pow_), self.asf_to_amplitude(asf),
                 phase_mode, ref_time_mu, profile)
        # Returns pow
        return self._get_pow(ftw, pow_, phase_mode, ref_time_mu)

    def _get_pow(self, ftw, pow_, phase_mode, ref_time_mu):
        # From ARTIQ
        if phase_mode != PHASE_MODE_CONTINUOUS:
            if phase_mode == PHASE_MODE_TRACKING and ref_time_mu < 0:
                ref_time_mu = 0
            if ref_time_mu >= 0:
                dt = np.int32(now_mu()) - np.int32(ref_time_mu)
                pow_ += dt * ftw * self.sysclk_per_mu >> 16
        return pow_

    @kernel
    def set_profile_ram(self, start, end, step=1, profile=0, nodwell_high=0,
                        zero_crossing=0, mode=1):
        raise NotImplementedError

    @kernel
    def set_ftw(self, ftw):
        self.set_frequency(self.ftw_to_frequency(ftw))

    @kernel
    def set_asf(self, asf):
        self.set_amplitude(self.asf_to_amplitude(asf))

    @kernel
    def set_pow(self, pow_):
        self.set_phase(self.pow_to_turns(pow_))

    @portable(flags={"fast-math"})
    def frequency_to_ftw(self, frequency):
        return np.int32(round(self.ftw_per_hz * frequency))

    @portable(flags={"fast-math"})
    def ftw_to_frequency(self, ftw):
        return ftw / self.ftw_per_hz

    @portable(flags={"fast-math"})
    def turns_to_pow(self, turns):
        return np.int32(round(turns * 0x10000))

    @portable(flags={"fast-math"})
    def pow_to_turns(self, pow_):
        return pow_ / 0x10000

    @portable(flags={"fast-math"})
    def amplitude_to_asf(self, amplitude):
        return np.int32(round(amplitude * 0x3ffe))

    @portable(flags={"fast-math"})
    def asf_to_amplitude(self, asf):
        return asf / float(0x3ffe)

    @portable(flags={"fast-math"})
    def frequency_to_ram(self, frequency, ram):
        raise NotImplementedError

    @portable(flags={"fast-math"})
    def turns_to_ram(self, turns, ram):
        raise NotImplementedError

    @portable(flags={"fast-math"})
    def amplitude_to_ram(self, amplitude, ram):
        raise NotImplementedError

    @portable(flags={"fast-math"})
    def turns_amplitude_to_ram(self, turns, amplitude, ram):
        raise NotImplementedError

    @kernel
    def set_frequency(self, frequency):
        self._signal_manager.event(self._freq, frequency)

    @kernel
    def set_amplitude(self, amplitude):
        self._signal_manager.event(self._amp, amplitude)

    @kernel
    def set_phase(self, turns):
        self._signal_manager.event(self._phase, turns)

    @kernel
    def set(self, frequency, phase=0.0, amplitude=1.0,
            phase_mode=_PHASE_MODE_DEFAULT, ref_time_mu=np.int64(-1), profile=0):
        if profile != 0:
            raise NotImplementedError('AD9910 simulation does not support profiles at this moment')

        # From ARTIQ
        if phase_mode == _PHASE_MODE_DEFAULT:
            phase_mode = self.phase_mode

        # Manage signals
        self.set_frequency(frequency)
        self.set_phase(phase)
        self.set_amplitude(amplitude)

        # Returns pow
        return self.pow_to_turns(self._get_pow(
            self.frequency_to_ftw(frequency), self.turns_to_pow(phase), phase_mode, ref_time_mu))

    @kernel
    def set_att_mu(self, att):
        att = (255 - att) / 8  # Inverted att to att_mu
        self.set_att(att)

    @kernel
    def set_att(self, att):
        self._signal_manager.event(self._att, att)

    @kernel
    def cfg_sw(self, state):
        self._signal_manager.event(self._sw, state)

    @kernel
    def set_sync(self, in_delay, window):
        raise NotImplementedError

    @kernel
    def clear_smp_err(self):
        raise NotImplementedError

    @kernel
    def tune_sync_delay(self, search_seed=15):
        raise NotImplementedError

    @kernel
    def measure_io_update_alignment(self, delay_start, delay_stop):
        raise NotImplementedError

    @kernel
    def tune_io_update_delay(self):
        raise NotImplementedError
