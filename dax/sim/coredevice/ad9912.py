# mypy: disallow_untyped_defs = False
# mypy: disallow_incomplete_defs = False
# mypy: check_untyped_defs = False

from numpy import int32, int64

from artiq.language.types import TInt32, TInt64, TFloat, TTuple, TBool
from artiq.language.core import kernel, delay, portable
from artiq.language.units import ms, us, MHz

from dax.sim.device import DaxSimDevice, ARTIQ_MAJOR_VERSION
from dax.sim.signal import get_signal_manager
from dax.sim.coredevice.urukul import CPLD


class AD9912(DaxSimDevice):

    def __init__(self, dmgr, chip_select, cpld_device, sw_device=None, pll_n=10, **kwargs):
        # Call super
        super(AD9912, self).__init__(dmgr, **kwargs)

        # Register signals
        self._signal_manager = get_signal_manager()
        self._init = self._signal_manager.register(self, 'init', bool, size=1)
        self._freq = self._signal_manager.register(self, 'freq', float)
        self._phase = self._signal_manager.register(self, 'phase', float)

        # CPLD device
        self.cpld: CPLD = dmgr.get(cpld_device)
        # Chip select
        assert 4 <= chip_select <= 7
        self.chip_select = chip_select
        # Switch device
        if sw_device:
            self.sw = dmgr.get(sw_device)

        # Store attributes (from ARTIQ code)
        sysclk = self.cpld.refclk / [1, 1, 2, 4][self.cpld.clk_div] * pll_n
        assert sysclk <= 1e9
        self.ftw_per_hz: float = 1 / sysclk * (int64(1) << 48)

    @kernel
    def write(self, addr: TInt32, data: TInt32, length: TInt32):
        raise NotImplementedError

    @kernel
    def read(self, addr: TInt32, length: TInt32) -> TInt32:
        raise NotImplementedError

    def _init_(self):
        self._signal_manager.event(self._init, 1)

    @kernel
    def init(self):
        # Delays from ARTIQ code
        delay(50 * us)
        delay(1 * ms)
        self._init_()

    @kernel
    def set_att_mu(self, att: TInt32):
        self.cpld.set_att_mu(self.chip_select - 4, att)

    @kernel
    def set_att(self, att: TFloat):
        self.cpld.set_att(self.chip_select - 4, att)

    if ARTIQ_MAJOR_VERSION < 7:
        # noinspection PyShadowingBuiltins
        @kernel
        def set_mu(self, ftw, pow):
            phase = pow / (1 << 14)  # Inverted turns_to_pow()
            self.set(self.ftw_to_frequency(ftw), phase)
    else:  # pragma: no cover
        @kernel
        def set_mu(self, ftw: TInt64, pow_: TInt32):
            self.set(self.ftw_to_frequency(ftw), self.pow_to_turns(pow_))

    @portable(flags={"fast-math"})
    def frequency_to_ftw(self, frequency: TFloat) -> TInt64:
        return int64(round(float(self.ftw_per_hz * frequency))) & ((int64(1) << 48) - 1)

    @portable(flags={"fast-math"})
    def ftw_to_frequency(self, ftw: TInt64) -> TFloat:
        return float(ftw / self.ftw_per_hz)

    @portable(flags={"fast-math"})
    def turns_to_pow(self, phase: TFloat) -> TInt32:
        return int32(round(float((1 << 14) * phase))) & int32(0xffff)

    @kernel
    def set(self, frequency: TFloat, phase: TFloat = 0.0):
        assert 0 * MHz <= frequency <= 400 * MHz, 'Frequency out of range'
        assert 0.0 <= phase < 1.0, 'Phase out of range'
        self._signal_manager.event(self._freq, float(frequency))
        self._signal_manager.event(self._phase, float(phase))

    @kernel
    def cfg_sw(self, state: TBool):
        self.cpld.cfg_sw(self.chip_select - 4, state)

    if ARTIQ_MAJOR_VERSION >= 7:  # pragma: no cover
        @kernel
        def get_att_mu(self) -> TInt32:
            return self.cpld.get_channel_att_mu(self.chip_select - 4)

        @kernel
        def get_att(self) -> TFloat:
            return self.cpld.get_channel_att(self.chip_select - 4)

        @kernel
        def get_mu(self) -> TTuple([TInt64, TInt32]):  # type: ignore[valid-type]
            freq, phase = self.get()
            return self.frequency_to_ftw(freq), self.turns_to_pow(phase)

        @portable(flags={"fast-math"})
        def pow_to_turns(self, pow_: TInt32) -> TFloat:
            return float(pow_ / (1 << 14))

        @kernel
        def get(self) -> TTuple([TFloat, TFloat]):  # type: ignore[valid-type]
            raise NotImplementedError
