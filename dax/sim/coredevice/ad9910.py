# mypy: disallow_untyped_defs = False
# mypy: disallow_incomplete_defs = False
# mypy: check_untyped_defs = False

from numpy import int32, int64

from artiq.language.core import kernel, delay, portable, now_mu
from artiq.language.units import us, ms, MHz
from artiq.language.types import TBool, TInt32, TInt64, TFloat, TList, TTuple
from artiq.coredevice.ad9910 import PHASE_MODE_CONTINUOUS, PHASE_MODE_ABSOLUTE, PHASE_MODE_TRACKING  # type: ignore

from dax.sim.device import DaxSimDevice, ARTIQ_MAJOR_VERSION
from dax.sim.signal import get_signal_manager
from dax.sim.coredevice.urukul import CPLD, DEFAULT_PROFILE

_NUM_PROFILES = 8
_PHASE_MODE_DEFAULT = -1
_PHASE_MODE_DICT = {m: f'{m:02b}' for m in [PHASE_MODE_CONTINUOUS, PHASE_MODE_ABSOLUTE, PHASE_MODE_TRACKING]}
"""Phase mode conversion dict."""


class AD9910(DaxSimDevice):

    def __init__(self, dmgr, chip_select, cpld_device, sw_device=None,
                 pll_n=40, pll_cp=7, pll_vco=5, pll_en=1, **kwargs):
        # Call super
        super(AD9910, self).__init__(dmgr, **kwargs)

        # CPLD device
        self.cpld: CPLD = dmgr.get(cpld_device)
        # Chip select
        assert 4 <= chip_select <= 7
        self.chip_select = chip_select
        # Switch device
        if sw_device:
            self.sw = dmgr.get(sw_device)

        # Store attributes (from ARTIQ code)
        clk = self.cpld.refclk / [4, 1, 2, 4][self.cpld.clk_div]
        self.pll_en = pll_en
        self.pll_n = pll_n
        self.pll_vco = pll_vco
        self.pll_cp = pll_cp
        if pll_en:
            sysclk = clk * pll_n
            assert clk <= 60e6
            assert 12 <= pll_n <= 127
            assert 0 <= pll_vco <= 5
            vco_min, vco_max = [(370, 510), (420, 590), (500, 700),
                                (600, 880), (700, 950), (820, 1150)][pll_vco]
            assert vco_min <= sysclk / 1e6 <= vco_max
            assert 0 <= pll_cp <= 7
        else:
            sysclk = clk
        assert sysclk <= 1e9
        self.ftw_per_hz: float = (1 << 32) / sysclk
        self.sysclk_per_mu = int(round(float(sysclk * self.core.ref_period)))
        self.sysclk = sysclk

        self.phase_mode = PHASE_MODE_CONTINUOUS

        # Register signals
        self._signal_manager = get_signal_manager()
        self._init = self._signal_manager.register(self, 'init', bool, size=1)
        self._freq = self._signal_manager.register(self, 'freq', float)
        self._phase = self._signal_manager.register(self, 'phase', float)
        self._phase_mode = self._signal_manager.register(self, 'phase_mode', bool, size=2)
        self._amp = self._signal_manager.register(self, 'amp', float)

    @kernel
    def set_phase_mode(self, phase_mode: TInt32):
        # From ARTIQ code
        self.phase_mode = phase_mode

    @kernel
    def write16(self, addr: TInt32, data: TInt32):
        raise NotImplementedError

    @kernel
    def write32(self, addr: TInt32, data: TInt32):
        raise NotImplementedError

    @kernel
    def read16(self, addr: TInt32) -> TInt32:
        raise NotImplementedError

    @kernel
    def read32(self, addr: TInt32) -> TInt32:
        raise NotImplementedError

    @kernel
    def read64(self, addr: TInt32) -> TInt64:
        raise NotImplementedError

    @kernel
    def write64(self, addr: TInt32, data_high: TInt32, data_low: TInt32):
        raise NotImplementedError

    @kernel
    def write_ram(self, data: TList(TInt32)):  # type: ignore[valid-type]
        raise NotImplementedError

    @kernel
    def read_ram(self, data: TList(TInt32)):  # type: ignore[valid-type]
        raise NotImplementedError

    @kernel
    def set_cfr1(self, power_down: TInt32 = 0b0000,
                 phase_autoclear: TInt32 = 0,
                 drg_load_lrr: TInt32 = 0, drg_autoclear: TInt32 = 0,
                 internal_profile: TInt32 = 0, ram_destination: TInt32 = 0,
                 ram_enable: TInt32 = 0, manual_osk_external: TInt32 = 0,
                 osk_enable: TInt32 = 0, select_auto_osk: TInt32 = 0):
        raise NotImplementedError

    @kernel
    def init(self, blind: TBool = False):
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
    def power_down(self, bits: TInt32 = 0b1111):
        raise NotImplementedError

    @kernel
    def set_mu(self, ftw: TInt32, pow_: TInt32 = 0, asf: TInt32 = 0x3fff,
               phase_mode: TInt32 = _PHASE_MODE_DEFAULT,
               ref_time_mu: TInt64 = int64(-1), profile: TInt32 = DEFAULT_PROFILE) -> TInt32:
        self.set(self.ftw_to_frequency(ftw), self.pow_to_turns(pow_), self.asf_to_amplitude(asf),
                 phase_mode, ref_time_mu, profile)
        # Returns pow
        return self._get_pow(ftw, pow_, phase_mode, ref_time_mu)

    def _get_pow(self, ftw: TInt32, pow_: TInt32, phase_mode: TInt32, ref_time_mu: TInt64) -> TInt32:
        # From ARTIQ
        if phase_mode != PHASE_MODE_CONTINUOUS:
            if phase_mode == PHASE_MODE_TRACKING and ref_time_mu < 0:
                ref_time_mu = 0
            if ref_time_mu >= 0:
                dt = int32(now_mu()) - int32(ref_time_mu)
                pow_ += dt * ftw * self.sysclk_per_mu >> 16
        return pow_

    @kernel
    def set_profile_ram(self, start: TInt32, end: TInt32, step: TInt32 = 1,
                        profile: TInt32 = DEFAULT_PROFILE, nodwell_high: TInt32 = 0,
                        zero_crossing: TInt32 = 0, mode: TInt32 = 1):
        raise NotImplementedError

    @kernel
    def set_ftw(self, ftw: TInt32):
        self.set_frequency(self.ftw_to_frequency(ftw))

    @kernel
    def set_asf(self, asf: TInt32):
        self.set_amplitude(self.asf_to_amplitude(asf))

    @kernel
    def set_pow(self, pow_: TInt32):
        self.set_phase(self.pow_to_turns(pow_))

    @portable(flags={"fast-math"})
    def frequency_to_ftw(self, frequency: TFloat) -> TInt32:
        return int32(round(float(self.ftw_per_hz * frequency)))

    @portable(flags={"fast-math"})
    def ftw_to_frequency(self, ftw: TInt32) -> TFloat:
        return ftw / self.ftw_per_hz

    @portable(flags={"fast-math"})
    def turns_to_pow(self, turns: TFloat) -> TInt32:
        return int32(round(float(turns * 0x10000))) & int32(0xffff)

    @portable(flags={"fast-math"})
    def pow_to_turns(self, pow_: TInt32) -> TFloat:
        return pow_ / 0x10000

    @portable(flags={"fast-math"})
    def amplitude_to_asf(self, amplitude: TFloat) -> TInt32:
        code = int32(round(float(amplitude * 0x3fff)))
        if code < 0 or code > 0x3fff:
            raise ValueError("Invalid AD9910 fractional amplitude!")
        return code

    @portable(flags={"fast-math"})
    def asf_to_amplitude(self, asf: TInt32) -> TFloat:
        return asf / float(0x3fff)

    @portable(flags={"fast-math"})
    def frequency_to_ram(self, frequency: TList(TFloat), ram: TList(TInt32)):  # type: ignore[valid-type]
        raise NotImplementedError

    @portable(flags={"fast-math"})
    def turns_to_ram(self, turns: TList(TFloat), ram: TList(TInt32)):  # type: ignore[valid-type]
        raise NotImplementedError

    @portable(flags={"fast-math"})
    def amplitude_to_ram(self, amplitude: TList(TFloat), ram: TList(TInt32)):  # type: ignore[valid-type]
        raise NotImplementedError

    @portable(flags={"fast-math"})
    def turns_amplitude_to_ram(self, turns: TList(TFloat),  # type: ignore[valid-type]
                               amplitude: TList(TFloat), ram: TList(TInt32)):  # type: ignore[valid-type]
        raise NotImplementedError

    @kernel
    def set_frequency(self, frequency: TFloat):
        raise NotImplementedError

    @kernel
    def set_amplitude(self, amplitude: TFloat):
        raise NotImplementedError

    @kernel
    def set_phase(self, turns: TFloat):
        raise NotImplementedError

    @kernel
    def set(self, frequency: TFloat, phase: TFloat = 0.0,
            amplitude: TFloat = 1.0, phase_mode: TInt32 = _PHASE_MODE_DEFAULT,
            ref_time_mu: TInt64 = int64(-1), profile: TInt32 = DEFAULT_PROFILE) -> TFloat:
        assert 0 * MHz <= frequency <= 400 * MHz, 'Frequency out of range'
        assert 0.0 <= phase < 1.0, 'Phase out of range'
        assert 0.0 <= amplitude <= 1.0, 'Amplitude out of range'
        assert 0 <= profile < _NUM_PROFILES, 'Profile out of range'

        if profile != DEFAULT_PROFILE:
            raise NotImplementedError('AD9910 simulation does not support profiles at this moment')

        # From ARTIQ
        if phase_mode == _PHASE_MODE_DEFAULT:
            phase_mode = self.phase_mode

        # Manage signals
        self._signal_manager.event(self._freq, float(frequency))
        self._signal_manager.event(self._phase, float(phase))
        self._signal_manager.event(self._amp, float(amplitude))
        self._signal_manager.event(self._phase_mode, _PHASE_MODE_DICT[self.phase_mode])

        # Returns pow
        return self.pow_to_turns(self._get_pow(
            self.frequency_to_ftw(frequency), self.turns_to_pow(phase), phase_mode, ref_time_mu))

    @kernel
    def set_att_mu(self, att: TInt32):
        self.cpld.set_att_mu(self.chip_select - 4, att)

    @kernel
    def set_att(self, att: TFloat):
        self.cpld.set_att(self.chip_select - 4, att)

    @kernel
    def cfg_sw(self, state: TBool):
        self.cpld.cfg_sw(self.chip_select - 4, state)

    @kernel
    def set_sync(self, in_delay: TInt32, window: TInt32):
        raise NotImplementedError

    @kernel
    def clear_smp_err(self):
        raise NotImplementedError

    @kernel
    def tune_sync_delay(self,
                        search_seed: TInt32 = 15) -> TTuple([TInt32, TInt32]):  # type: ignore[valid-type]
        raise NotImplementedError

    @kernel
    def measure_io_update_alignment(self, delay_start: TInt64,
                                    delay_stop: TInt64) -> TInt32:
        raise NotImplementedError

    @kernel
    def tune_io_update_delay(self) -> TInt32:
        raise NotImplementedError

    if ARTIQ_MAJOR_VERSION >= 7:  # pragma: no cover
        @kernel
        def get_ftw(self) -> TInt32:
            return self.frequency_to_ftw(self.get_frequency())

        @kernel
        def get_asf(self) -> TInt32:
            return self.amplitude_to_asf(self.get_amplitude())

        @kernel
        def get_pow(self) -> TInt32:
            return self.turns_to_pow(self.get_phase())

        @kernel
        def get_frequency(self) -> TFloat:
            raise NotImplementedError

        @kernel
        def get_amplitude(self) -> TFloat:
            raise NotImplementedError

        @kernel
        def get_phase(self) -> TFloat:
            raise NotImplementedError

        @kernel
        def get_mu(self, profile: TInt32 = DEFAULT_PROFILE) \
                -> TTuple([TInt32, TInt32, TInt32]):  # type: ignore[valid-type]
            freq, phase, amp = self.get(profile)
            return self.frequency_to_ftw(freq), self.turns_to_pow(phase), self.amplitude_to_asf(amp)

        @kernel
        def get(self, profile: TInt32 = DEFAULT_PROFILE) \
                -> TTuple([TFloat, TFloat, TFloat]):  # type: ignore[valid-type]
            raise NotImplementedError

        @kernel
        def get_att_mu(self) -> TInt32:
            return self.cpld.get_channel_att_mu(self.chip_select - 4)

        @kernel
        def get_att(self) -> TFloat:
            return self.cpld.get_channel_att(self.chip_select - 4)
