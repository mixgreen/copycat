# mypy: disallow_untyped_defs = False
# mypy: disallow_incomplete_defs = False
# mypy: check_untyped_defs = False

import typing
import numpy as np

from artiq.language.core import kernel, delay, delay_mu, portable
from artiq.language.units import us, ms, dB
from artiq.language.types import TInt32, TFloat, TBool

from dax.sim.device import DaxSimDevice, ARTIQ_MAJOR_VERSION
from dax.sim.signal import get_signal_manager

DEFAULT_PROFILE = 0
NUM_PROFILES = 8

_NUM_CHANNELS = 4


def _mu_to_att(att_mu: TInt32) -> TFloat:
    return float((255 - (att_mu & 0xFF)) / 8)


def _att_to_mu(att: TFloat) -> TInt32:
    code = 255 - np.int32(round(att * 8))
    if code < 0 or code > 255:
        raise ValueError("Invalid urukul.CPLD attenuation!")
    return code


def _state_to_sw_reg(state: typing.Union[int, np.int32]) -> typing.List[str]:
    return ['1' if (state >> i) & 0x1 else '0' for i in range(4)]


class _IOUpdate:
    """Dummy IO update device to capture updates and notify subscribers."""

    def __init__(self):
        self._subscribers = []

    def pulse(self, t):
        delay(t)
        self._notify()

    def pulse_mu(self, t):
        delay_mu(t)
        self._notify()

    def _notify(self):
        """Notify subscribers."""
        for fn in self._subscribers:
            fn()

    def subscribe(self, fn):
        """Subscribe to this IO update.

        For internal simulation usage only.
        """
        assert callable(fn)
        self._subscribers.append(fn)


class CPLD(DaxSimDevice):

    def __init__(self, dmgr, clk_div=0, rf_sw=0, refclk=125e6, att=0x00000000, **kwargs):
        # Call super
        super(CPLD, self).__init__(dmgr, **kwargs)

        # Store attributes (from ARTIQ code)
        self.refclk = refclk
        assert 0 <= clk_div <= 3
        self.clk_div = clk_div
        self.att_reg: np.int32 = np.int32(np.int64(att))

        # Add a dummy IO update device for subscribers
        self.io_update = _IOUpdate()

        # Register signals
        signal_manager = get_signal_manager()
        self._init = signal_manager.register(self, 'init', bool, size=1)
        self._init_att = signal_manager.register(self, 'init_att', bool, size=1)
        self._att = [signal_manager.register(self, f'att_{i}', float) for i in range(4)]
        self._sw = signal_manager.register(self, 'sw', bool, size=4)
        self._profile = signal_manager.register(self, 'profile', int, init='x')

        # Internal registers
        self._att_reg = [_mu_to_att(att >> (i * 8)) for i in range(4)]
        self._sw_reg = _state_to_sw_reg(rf_sw)
        self._profile_reg = DEFAULT_PROFILE

    @kernel
    def cfg_write(self, cfg: TInt32):
        raise NotImplementedError

    @kernel
    def sta_read(self) -> TInt32:
        raise NotImplementedError

    # noinspection PyUnusedLocal
    @kernel
    def init(self, blind: TBool = False):
        # Delays from ARTIQ code
        delay(100 * us)  # reset, slack
        delay(1 * ms)  # DDS wake up
        self._profile.push(self._profile_reg)
        self._init.push(True)

    @kernel
    def io_rst(self):
        raise NotImplementedError

    @kernel
    def cfg_sw(self, channel: TInt32, on: TBool):
        assert 0 <= channel < _NUM_CHANNELS, 'Channel out of range'
        self._sw_reg[channel] = '1' if on else '0'
        self._update_switches()

    def _cfg_switches(self, state: TInt32):
        self._sw_reg = _state_to_sw_reg(state)
        self._update_switches()

    @kernel
    def cfg_switches(self, state: TInt32):
        self._cfg_switches(state)

    def _update_switches(self):  # type: () -> None
        self._sw.push(''.join(reversed(self._sw_reg)))

    @kernel
    def set_att_mu(self, channel: TInt32, att: TInt32):
        assert 0 <= channel < _NUM_CHANNELS, 'Channel out of range'
        assert 0 <= att <= 255, 'Attenuation mu out of range'
        a = self.att_reg & ~(0xff << (channel * 8))
        a |= att << (channel * 8)
        self.set_all_att_mu(a)

    def _set_all_att_mu(self, att_reg: TInt32):
        self.att_reg = np.int32(att_reg)
        self._att_reg = [_mu_to_att(att_reg >> (i * 8)) for i in range(4)]
        self._update_att()

    @kernel
    def set_all_att_mu(self, att_reg: TInt32):
        self._set_all_att_mu(att_reg)

    @kernel
    def set_att(self, channel: TInt32, att: TFloat):
        assert 0 <= channel < _NUM_CHANNELS, 'Channel out of range'
        # Update register
        a = self.att_reg & ~(0xff << (channel * 8))
        a |= _att_to_mu(att) << (channel * 8)
        self.att_reg = np.int32(a)
        # Handle signals
        self._att_reg[channel] = float(att)
        self._update_att()

    def _update_att(self):  # type: () -> None
        for s, a in zip(self._att, self._att_reg):
            assert 0 * dB <= a <= (255 / 8) * dB, 'Attenuation out of range'
            s.push(a)

    @kernel
    def get_att_mu(self) -> TInt32:
        # Returns the value in the register instead of the device value
        delay(10 * us)  # Delay from ARTIQ code
        self._init_att.push(True)
        return self.att_reg

    @kernel
    def set_sync_div(self, div: TInt32):
        raise NotImplementedError

    @kernel
    def set_profile(self, profile: TInt32):
        assert 0 <= profile < NUM_PROFILES, 'Profile out of range'

        if profile != DEFAULT_PROFILE:
            raise NotImplementedError('CPLD simulation only supports the default profile at this moment')

        self._profile_reg = profile
        self._profile.push(self._profile_reg)

    if ARTIQ_MAJOR_VERSION >= 7:  # pragma: no cover
        @portable(flags={"fast-math"})
        def mu_to_att(self, att_mu: TInt32) -> TFloat:
            return _mu_to_att(att_mu)

        @portable(flags={"fast-math"})
        def att_to_mu(self, att: TFloat) -> TInt32:
            return _att_to_mu(att)

        @kernel
        def get_channel_att_mu(self, channel: TInt32) -> TInt32:
            return np.int32((self.get_att_mu() >> (channel * 8)) & 0xff)

        @kernel
        def get_channel_att(self, channel: TInt32) -> TFloat:
            return self.mu_to_att(self.get_channel_att_mu(channel))
