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

    # kernel_invariants = {"channel", "core", "servo", "servo_channel"}

    def __init__(self, dmgr, channel, servo_device):
        # TODO:
        self.servo = dmgr.get(servo_device)
        self.core = self.servo.core
        self.channel = channel
        # FIXME: this assumes the mem channel is right after the control
        # channels
        self.servo_channel = self.channel + 8 - self.servo.channel

    @kernel
    def set(self, en_out, en_iir=0, profile=0):
        # TODO:
        rtio_output(self.channel << 8,
                    en_out | (en_iir << 1) | (profile << 2))

    @kernel
    def set_dds_mu(self, profile, ftw, offs, pow_=0):
        # TODO:
        base = (self.servo_channel << 8) | (profile << 3)
        self.servo.write(base + 0, ftw >> 16)
        self.servo.write(base + 6, (ftw & 0xffff))
        self.set_dds_offset_mu(profile, offs)
        self.servo.write(base + 2, pow_)

    @kernel
    def set_dds(self, profile, frequency, offset, phase=0.):
        # TODO:
        if self.servo_channel < 4:
            dds = self.servo.dds0
        else:
            dds = self.servo.dds1
        ftw = dds.frequency_to_ftw(frequency)
        pow_ = dds.turns_to_pow(phase)
        offs = self.dds_offset_to_mu(offset)
        self.set_dds_mu(profile, ftw, offs, pow_)

    @kernel
    def set_dds_offset_mu(self, profile, offs):
        # TODO:
        base = (self.servo_channel << 8) | (profile << 3)
        self.servo.write(base + 4, offs)

    @kernel
    def set_dds_offset(self, profile, offset):
        # TODO:
        self.set_dds_offset_mu(profile, self.dds_offset_to_mu(offset))

    @portable
    def dds_offset_to_mu(self, offset):
        # TODO:
        return int(round(offset * (1 << COEFF_WIDTH - 1)))

    @kernel
    def set_iir_mu(self, profile, adc, a1, b0, b1, dly=0):
        # TODO:
        base = (self.servo_channel << 8) | (profile << 3)
        self.servo.write(base + 3, adc | (dly << 8))
        self.servo.write(base + 1, b1)
        self.servo.write(base + 5, a1)
        self.servo.write(base + 7, b0)

    @kernel
    def set_iir(self, profile, adc, kp, ki=0., g=0., delay=0.):
        # TODO:
        B_NORM = 1 << COEFF_SHIFT + 1
        A_NORM = 1 << COEFF_SHIFT
        COEFF_MAX = 1 << COEFF_WIDTH - 1

        kp *= B_NORM
        if ki == 0.:
            # pure P
            a1 = 0
            b1 = 0
            b0 = int(round(kp))
        else:
            # I or PI
            ki *= B_NORM * T_CYCLE / 2.
            if g == 0.:
                c = 1.
                a1 = A_NORM
            else:
                c = 1. / (1. + ki / (g * B_NORM))
                a1 = int(round((2. * c - 1.) * A_NORM))
            b0 = int(round(kp + ki * c))
            b1 = int(round(kp + (ki - 2. * kp) * c))
            if b1 == -b0:
                raise ValueError("low integrator gain and/or gain limit")

        if (b0 >= COEFF_MAX or b0 < -COEFF_MAX
                or b1 >= COEFF_MAX or b1 < -COEFF_MAX):
            raise ValueError("high gains")

        dly = int(round(delay / T_CYCLE))
        self.set_iir_mu(profile, adc, a1, b0, b1, dly)

    @kernel
    def get_profile_mu(self, profile, data):
        # TODO:
        base = (self.servo_channel << 8) | (profile << 3)
        for i in range(len(data)):
            data[i] = self.servo.read(base + i)
            delay(4 * us)

    @kernel
    def get_y_mu(self, profile):
        # TODO:
        return self.servo.read(STATE_SEL | (self.servo_channel << 5) | profile)

    @kernel
    def get_y(self, profile):
        # TODO:
        return y_mu_to_full_scale(self.get_y_mu(profile))

    @kernel
    def set_y_mu(self, profile, y):
        # TODO:
        # State memory is 25 bits wide and signed.
        # Reads interact with the 18 MSBs (coefficient memory width)
        self.servo.write(STATE_SEL | (self.servo_channel << 5) | profile, y)

    @kernel
    def set_y(self, profile, y):
        # TODO
        y_mu = int(round(y * Y_FULL_SCALE_MU))
        if y_mu < 0 or y_mu > (1 << 17) - 1:
            raise ValueError("Invalid SUServo y-value!")
        self.set_y_mu(profile, y_mu)
        return y_mu
