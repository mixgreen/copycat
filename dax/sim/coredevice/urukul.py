import numpy as np

from dax.sim.coredevice import *


class CPLD(DaxSimDevice):
    """Minimal implementation of CPLD to initialize AD99xx devices correctly."""

    def __init__(self, dmgr, clk_div=0, refclk=125e6, att=0x00000000, **kwargs):
        # Call super
        super(CPLD, self).__init__(dmgr, **kwargs)

        # Store attributes (from ARTIQ code)
        self.refclk = refclk
        assert 0 <= clk_div <= 3
        self.clk_div = clk_div
        self.att_reg = np.int32(np.int64(att))

        # Register signals
        self._signal_manager = get_signal_manager()
        self._init = self._signal_manager.register(self.key, 'init', bool, size=1)
        self._init_att = self._signal_manager.register(self.key, 'init_att', bool, size=1)

    @kernel
    def cfg_write(self, cfg):
        raise NotImplementedError

    @kernel
    def sta_read(self):
        raise NotImplementedError

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
        raise NotImplementedError

    @kernel
    def cfg_switches(self, state):
        raise NotImplementedError

    @kernel
    def set_att_mu(self, channel, att):
        raise NotImplementedError

    @kernel
    def set_all_att_mu(self, att_reg):
        raise NotImplementedError

    @kernel
    def set_att(self, channel, att):
        raise NotImplementedError

    @kernel
    def get_att_mu(self):
        # Does not return the actual value, but the default value
        delay(10 * us)  # Delay from ARTIQ code
        self._signal_manager.event(self._init_att, 1)
        return self.att_reg

    @kernel
    def set_sync_div(self, div):
        raise NotImplementedError

    @kernel
    def set_profile(self, profile):
        raise NotImplementedError
