# mypy: disallow_untyped_defs = False
# mypy: disallow_incomplete_defs = False
# mypy: check_untyped_defs = False

import numpy as np

from artiq.language.core import *
from artiq.language.units import *

from dax.sim.device import DaxSimDevice
from dax.sim.signal import get_signal_manager


def _mu_to_att(att_mu):
    return (255 - (att_mu & 0xFF)) / 8


def _att_to_mu(att):
    code = 255 - np.int32(round(att * 8))
    if code < 0 or code > 255:
        raise ValueError("Invalid urukul.CPLD attenuation!")
    return code


def _state_to_sw_reg(state):
    return ['1' if (state >> i) & 0x1 else '0' for i in range(4)]


class CPLD(DaxSimDevice):
    """Minimal implementation of CPLD to initialize AD99xx devices correctly."""

    def __init__(self, dmgr, clk_div=0, rf_sw=0, refclk=125e6, att=0x00000000, **kwargs):
        # Call super
        super(CPLD, self).__init__(dmgr, **kwargs)

        # Store attributes (from ARTIQ code)
        self.refclk = refclk
        assert 0 <= clk_div <= 3
        self.clk_div = clk_div
        self.att_reg = np.int32(np.int64(att))

        # Register signals
        self._signal_manager = get_signal_manager()
        self._init = self._signal_manager.register(self, 'init', bool, size=1)
        self._init_att = self._signal_manager.register(self, 'init_att', bool, size=1)
        self._att = [self._signal_manager.register(self, f'att_{i}', float) for i in range(4)]
        self._sw = self._signal_manager.register(self, 'sw', bool, size=4)

        # Internal registers
        self._att_reg = [_mu_to_att(att >> (i * 8)) for i in range(4)]
        self._sw_reg = _state_to_sw_reg(rf_sw)

    @kernel
    def cfg_write(self, cfg):
        raise NotImplementedError

    @kernel
    def sta_read(self):
        raise NotImplementedError

    # noinspection PyUnusedLocal
    @kernel
    def init(self, blind=False):
        # Delays from ARTIQ code
        delay(100 * us)  # reset, slack
        delay(1 * ms)  # DDS wake up
        self._signal_manager.event(self._init, 1)

    @kernel
    def io_rst(self):
        raise NotImplementedError

    @kernel
    def cfg_sw(self, channel, on):
        assert 0 <= channel < 4, 'Channel out of range'
        self._sw_reg[channel] = '1' if on else '0'
        self._update_switches()

    @kernel
    def cfg_switches(self, state):
        self._sw_reg = _state_to_sw_reg(state)
        self._update_switches()

    def _update_switches(self):
        self._signal_manager.event(self._sw, ''.join(reversed(self._sw_reg)))

    @kernel
    def set_att_mu(self, channel, att):
        assert 0 <= channel < 4, 'Channel out of range'
        assert 0 <= att <= 255, 'Attenuation mu out of range'
        a = self.att_reg & ~(0xff << (channel * 8))
        a |= att << (channel * 8)
        self.set_all_att_mu(a)

    @kernel
    def set_all_att_mu(self, att_reg):
        self.att_reg = att_reg
        self._att_reg = [_mu_to_att(att_reg >> (i * 8)) for i in range(4)]
        self._update_att()

    @kernel
    def set_att(self, channel, att):
        assert 0 <= channel < 4, 'Channel out of range'
        # Update register
        a = self.att_reg & ~(0xff << (channel * 8))
        a |= _att_to_mu(att) << (channel * 8)
        self.att_reg = a
        # Handle signals
        self._att_reg[channel] = float(att)
        self._update_att()

    def _update_att(self):
        for s, a in zip(self._att, self._att_reg):
            assert 0 * dB <= a <= (255 / 8) * dB, 'Attenuation out of range'
            self._signal_manager.event(s, a)

    @kernel
    def get_att_mu(self):
        # Returns the value in the register instead of the device value
        delay(10 * us)  # Delay from ARTIQ code
        self._signal_manager.event(self._init_att, 1)
        return self.att_reg

    @kernel
    def set_sync_div(self, div):
        raise NotImplementedError

    @kernel
    def set_profile(self, profile):
        raise NotImplementedError
