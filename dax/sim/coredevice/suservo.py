import typing

from artiq.language.core import kernel, host_only, delay, dealy_mu, portable
from artiq.language.units import us, ns
from artiq.coredevice import spi2 as spi
from artiq.coredevice.suservo import y_mu_to_full_scale, STATE_SEL

from dax.sim.device import DaxSimDevice
from dax.sim.signal import get_signal_manager
from dax.sim.coredevice import urukul, sampler

# NOTE: import from artiq core device if needed
# COEFF_WIDTH = 18
# Y_FULL_SCALE_MU = (1 << (COEFF_WIDTH - 1)) - 1
# COEFF_DEPTH = 10 + 1
# WE = 1 << COEFF_DEPTH + 1
# STATE_SEL = 1 << COEFF_DEPTH
# CONFIG_SEL = 1 << COEFF_DEPTH - 1
# CONFIG_ADDR = CONFIG_SEL | STATE_SEL
# T_CYCLE = (2 * (8 + 64) + 2) * 8 * ns  # Must match gateware Servo.t_cycle.
# COEFF_SHIFT = 11


@portable
def adc_mu_to_volts(x, gain):
    """Convert servo ADC data from machine units to Volt."""
    val = (x >> 1) & 0xffff
    mask = 1 << 15
    val = -(val & mask) + (val & ~mask)
    return sampler.adc_mu_to_volt(val, gain)


class SUServo(DaxSimDevice):
    """Simulation SUServo device"""

    def __init__(self, dmgr, channel, pgia_device,
                 cpld0_device, cpld1_device,
                 dds0_device, dds1_device,
                 gains=0x0000, **kwargs):

        # Call super
        super(SUServo, self).__init__(dmgr, **kwargs)

        # TODO: sampler sim driver
        # self.pgia = dmgr.get(pgia_device)
        # self.pgia.update_xfer_duration_mu(div=4, length=16)
        self.dds0 = dmgr.get(dds0_device)
        self.dds1 = dmgr.get(dds1_device)
        self.cpld0 = dmgr.get(cpld0_device)
        self.cpld1 = dmgr.get(cpld1_device)
        self.channel = channel
        self.gains = gains

        # TODO: how to sim this? use sim.core or core functions?
        self.ref_period_mu = self.core.seconds_to_mu(
            self.core.coarse_ref_period)
        assert self.ref_period_mu == self.core.ref_multiplier

        # Register signals
        signal_manager = get_signal_manager()
        self._init = signal_manager.register(self, 'init', bool, size=1)
        self._config = signal_manager.register(self, 'config', int)
        self._state_sel = signal_manager.register(self, 'state_sel', int)

        # Internal registers

    @kernel
    def init(self):
        self.set_config(enable=0)
        delay(3 * us)  # pipeline flush
        self.pgia.set_config_mu(
            sampler.SPI_CONFIG | spi.SPI_END,
            16, 4, sampler.SPI_CS_PGIA)
        self.cpld0.init(blind=True)
        self.dds0.init(blind=True)
        self.cpld1.init(blind=True)
        self.dds1.init(blind=True)

    @kernel
    def write(self, addr, value):
        raise NotImplementedError

    @kernel
    def read(self, addr):
        raise NotImplementedError

    @kernel
    def set_config(self, enable):
        self._config.push(enable)

    @kernel
    def get_status(self):
        return self._config.pull()

    @kernel
    def get_adc_mu(self, adc):
        return self._state_sel.pull(STATE_SEL | (adc << 1) | (1 << 8))

    @kernel
    def set_pgia_mu(self, channel, gain):
        gains = self.gains
        gains &= ~(0b11 << (channel * 2))
        gains |= gain << (channel * 2)
        self.gains = gains

    @kernel
    def get_adc(self, channel):
        val = self.get_adc_mu(channel)
        gain = (self.gains >> (channel * 2)) & 0b11
        return adc_mu_to_volts(val, gain)


class Channel(DaxSimDevice):
    """Simulation SUServo Channel device"""
