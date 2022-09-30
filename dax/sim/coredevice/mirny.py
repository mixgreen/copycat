# mypy: disallow_untyped_defs = False
# mypy: disallow_incomplete_defs = False
# mypy: check_untyped_defs = False

from numpy import int32

from artiq.language.core import kernel, delay, portable
from artiq.language.types import TInt32, TFloat
from artiq.language.units import us

from dax.sim.device import DaxSimDevice, ARTIQ_MAJOR_VERSION
from dax.sim.signal import get_signal_manager

# From ARTIQ code
SPIT_WR = 4


@portable(flags={'fast-math'})
def _mu_to_att(att_mu: TInt32) -> TFloat:
    return float((255 - (att_mu & 0xFF)) / 8)


class Mirny(DaxSimDevice):
    kernel_invariants = {"bus", "core", "refclk"}

    def __init__(self, dmgr, spi_device, refclk=100e6, **kwargs):
        # Call super
        super(Mirny, self).__init__(dmgr, **kwargs)

        # From ARTIQ code
        self.bus = dmgr.get(spi_device)

        # reference clock frequency
        self.refclk = refclk
        if not (10 <= self.refclk / 1e6 <= 600):
            raise ValueError("Invalid refclk")

        # Register signals
        signal_manager = get_signal_manager()
        self._init = signal_manager.register(self, 'init', bool, size=1)
        self._att = [signal_manager.register(self, f'att_{i}', float) for i in range(4)]

    @kernel
    def read_reg(self, addr):
        raise NotImplementedError

    @kernel
    def write_reg(self, addr, data):
        raise NotImplementedError

    # noinspection PyUnusedLocal
    @kernel
    def init(self, blind=False):
        # From ARTIQ code
        delay(1000 * us)

        # Update signal
        self._init.push(True)

    def _set_att(self, channel, att):
        self._att[channel].push(att)

    @kernel
    def set_att_mu(self, channel, att):
        self._set_att(channel, _mu_to_att(att))

    if ARTIQ_MAJOR_VERSION >= 7:

        @portable(flags={"fast-math"})
        def att_to_mu(self, att):
            code = int32(255) - int32(round(att * 8))
            if code < 0 or code > 255:
                raise ValueError("Invalid Mirny attenuation!")
            return code

        @kernel
        def set_att(self, channel, att):
            self._set_att(channel, att)

        @kernel
        def write_ext(self, addr, length, data, ext_div=SPIT_WR):
            raise NotImplementedError

    else:

        @kernel
        def write_ext(self, addr, length, data):
            raise NotImplementedError


if ARTIQ_MAJOR_VERSION >= 7:

    class Almazny(DaxSimDevice):

        def __init__(self, dmgr, host_mirny, **kwargs):
            # Call super
            super(Almazny, self).__init__(dmgr, **kwargs)

            # From ARTIQ code
            self.mirny_cpld = dmgr.get(host_mirny)
            self.att_mu = [0x3f] * 4
            self.channel_sw = [0] * 4
            self.output_enable = False

            # Register signals
            signal_manager = get_signal_manager()
            self._init = signal_manager.register(self, 'init', bool, size=1)
            self._att = [signal_manager.register(self, f'att_{i}', float) for i in range(4)]
            self._output_enable = signal_manager.register(self, 'oe', bool, size=1)

        @kernel
        def init(self):
            # From ARTIQ code
            self.output_toggle(self.output_enable)

            # Update signals
            self._init.push(True)

        @kernel
        def att_to_mu(self, att):
            mu = round(att * 2.0)
            if mu > 63 or mu < 0:
                raise ValueError("Invalid Almazny attenuator settings!")
            return mu

        @kernel
        def mu_to_att(self, att_mu):
            return att_mu / 2

        @kernel
        def _set_att_helper(self, channel, rf_switch):
            self.channel_sw[channel] = 1 if rf_switch else 0
            self._update_register(channel)

        @kernel
        def set_att(self, channel, att, rf_switch=True):
            self.att_mu[channel] = self.att_to_mu(att)
            self._set_att_helper(channel, rf_switch)
            self._att[channel].push(att)

        @kernel
        def set_att_mu(self, channel, att_mu, rf_switch=True):
            self.att_mu[channel] = att_mu
            self._set_att_helper(channel, rf_switch)
            self._att[channel].push(self.mu_to_att(att_mu))

        @kernel
        def output_toggle(self, oe):
            """
            Toggles output on all shift registers on or off.
            :param oe - toggle output enable (bool)
            """
            # From ARTIQ code
            self.output_enable = oe
            delay(100 * us)
            delay(100 * us)

            # Update signals
            self._output_enable.push(oe)

        # noinspection PyUnusedLocal
        @kernel
        def _update_register(self, ch):
            # From ARTIQ code
            delay(100 * us)
