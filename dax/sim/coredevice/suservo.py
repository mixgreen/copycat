from copy import deepcopy

from artiq.language.core import kernel, portable
from artiq.language.units import us
from artiq.coredevice.sampler import adc_mu_to_volt
# from artiq.coredevice.urukul import turns_to_pow, frequency_to_ftw
# from artiq.coredevice.core import coarse_ref_period, ref_multiplier, seconds_to_mu
from artiq.coredevice.suservo import y_mu_to_full_scale, STATE_SEL, COEFF_SHIFT, COEFF_SHIFT, COEFF_WIDTH, T_CYCLE, Y_FULL_SCALE_MU

from dax.sim.device import DaxSimDevice
from dax.sim.signal import get_signal_manager


# SUServo defaults
DEFAULT_CONFIG = None

# Channel defaults
DEFAULT_OUT = None
DEFAULT_PROFILE = None
DEFAULT_Y = None
DEFAULT_IIR = None
DDS_DEFAULTS = {"profile": DEFAULT_PROFILE,
                "ftw": 0,
                "offs": 0,
                "pow_": 0,
                }


@portable
def adc_mu_to_volts(x, gain):
    """Convert servo ADC data from machine units to Volt."""
    val = (x >> 1) & 0xffff
    mask = 1 << 15
    val = -(val & mask) + (val & ~mask)
    return adc_mu_to_volt(val, gain)


class SUServo(DaxSimDevice):
    """Simulation SUServo device"""

    def __init__(self, dmgr, channel, pgia_device,
                 cpld0_device, cpld1_device,
                 dds0_device, dds1_device,
                 gains=0x0000, **kwargs):

        # Call super
        super(SUServo, self).__init__(dmgr, **kwargs)

        # TODO: need sampler sim driver
        # self.pgia = dmgr.get(pgia_device)
        # self.pgia.update_xfer_duration_mu(div=4, length=16)

        self.dds0 = dmgr.get(dds0_device)
        self.dds1 = dmgr.get(dds1_device)
        self.cpld0 = dmgr.get(cpld0_device)
        self.cpld1 = dmgr.get(cpld1_device)
        self.channel = channel
        self.gains = gains
        self.ref_period_mu = seconds_to_mu(coarse_ref_period)
        assert self.ref_period_mu == ref_multiplier

        # Register signals
        signal_manager = get_signal_manager()
        self._init = signal_manager.register(self, 'init', bool, size=1)

        # Internal registers
        self._config = DEFAULT_CONFIG

    @kernel
    def init(self):
        self.set_config(enable=0)
        delay(3 * us)  # pipeline flush

        # TODO: need sampler sim driver
        # self.pgia.set_config_mu(
        #     sampler.SPI_CONFIG | spi.SPI_END,
        #     16, 4, sampler.SPI_CS_PGIA)

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
        self._config = enable

    @kernel
    def get_status(self):
        return self._config

    @kernel
    def get_adc_mu(self, adc):
        raise NotImplementedError

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

    def __init__(self, dmgr, channel, servo_device, **kwargs):

        # Call super
        super(Channel, self).__init__(dmgr, **kwargs)

        self.servo = dmgr.get(servo_device)
        self.channel = channel
        # channels
        self.servo_channel = self.channel + 8 - self.servo.channel

        # Register signals
        signal_manager = get_signal_manager()
        self._init = signal_manager.register(self, 'init', bool, size=1)

        # Internal registers
        self._dds = deepcopy(DDS_DEFAULTS)
        self._profile = DEFAULT_PROFILE
        self._out = DEFAULT_OUT
        self._iir = DEFAULT_IIR
        self._y = DEFAULT_Y

    @kernel
    def set(self, en_out, en_iir=0, profile=0):
        self._profile = profile  # NOTE: currently, profiles do not store seperate sets of settings in sim
        self._out = en_out
        self._iir = en_iir

    @kernel
    def set_dds_mu(self, profile, ftw, offs, pow_=0):
        self._dds["ftw"] = ftw
        self.set_dds_offset_mu(profile, offs)
        self._dds["pow_"] = pow_

    @kernel
    def set_dds(self, profile, frequency, offset, phase=0.):
        ftw = frequency_to_ftw(frequency)
        pow_ = turns_to_pow(phase)
        offs = self.dds_offset_to_mu(offset)
        self.set_dds_mu(profile, ftw, offs, pow_)

    @kernel
    def set_dds_offset_mu(self, profile, offs):
        self._dds["offs"] = offs

    @kernel
    def set_dds_offset(self, profile, offset):
        self.set_dds_offset_mu(profile, self.dds_offset_to_mu(offset))

    @portable
    def dds_offset_to_mu(self, offset):
        return int(round(offset * (1 << COEFF_WIDTH - 1)))

    @kernel
    def set_iir_mu(self, profile, adc, a1, b0, b1, dly=0):
        # NOTE: currently no function to return iir
        pass

    @kernel
    def set_iir(self, profile, adc, kp, ki=0., g=0., delay=0.):
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
        return self._profile

    @kernel
    def get_y_mu(self, profile):
        return self._y

    @kernel
    def get_y(self, profile):
        return y_mu_to_full_scale(self.get_y_mu(profile))

    @kernel
    def set_y_mu(self, profile, y):
        self._y = y

    @kernel
    def set_y(self, profile, y):
        y_mu = int(round(y * Y_FULL_SCALE_MU))
        if y_mu < 0 or y_mu > (1 << 17) - 1:
            raise ValueError("Invalid SUServo y-value!")
        self.set_y_mu(profile, y_mu)
        return y_mu
