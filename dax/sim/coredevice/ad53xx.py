# mypy: disallow_untyped_defs = False
# mypy: disallow_incomplete_defs = False
# mypy: check_untyped_defs = False

from artiq.language.core import *
from artiq.language.units import *
from artiq.coredevice.ad53xx import voltage_to_mu  # type: ignore
from artiq.coredevice.ad53xx import ad53xx_cmd_write_ch, ad53xx_cmd_read_ch  # noqa: F401
from artiq.coredevice.ad53xx import AD53XX_CMD_DATA, AD53XX_CMD_OFFSET  # noqa: F401
from artiq.coredevice.ad53xx import AD53XX_CMD_GAIN, AD53XX_CMD_SPECIAL  # noqa: F401
from artiq.coredevice.ad53xx import AD53XX_SPECIAL_NOP, AD53XX_SPECIAL_CONTROL, AD53XX_SPECIAL_OFS0  # noqa: F401
from artiq.coredevice.ad53xx import AD53XX_SPECIAL_OFS1, AD53XX_SPECIAL_READ  # noqa: F401
from artiq.coredevice.ad53xx import AD53XX_SPECIAL_AB0, AD53XX_SPECIAL_AB1, AD53XX_SPECIAL_AB2  # noqa: F401
from artiq.coredevice.ad53xx import AD53XX_SPECIAL_AB3, AD53XX_SPECIAL_AB  # noqa: F401
from artiq.coredevice.ad53xx import AD53XX_READ_X1A, AD53XX_READ_X1B, AD53XX_READ_OFFSET, AD53XX_READ_GAIN  # noqa: F401
from artiq.coredevice.ad53xx import AD53XX_READ_CONTROL, AD53XX_READ_OFS0, AD53XX_READ_OFS1  # noqa: F401
from artiq.coredevice.ad53xx import AD53XX_READ_AB0, AD53XX_READ_AB1, AD53XX_READ_AB2, AD53XX_READ_AB3  # noqa: F401

from dax.sim.device import DaxSimDevice
from dax.sim.signal import get_signal_manager


def _mu_to_voltage(voltage_mu, offset_dacs=0x2000, vref=5.):
    return ((voltage_mu - (offset_dacs * 0x4)) / 0x10000) * (4. * vref)


class _DummyTTL:
    @portable
    def on(self):
        pass

    @portable
    def off(self):
        pass


class AD53xx(DaxSimDevice):

    def __init__(self, dmgr, vref=5., offset_dacs=8192, **kwargs):
        # Call super
        super(AD53xx, self).__init__(dmgr, **kwargs)

        # Register signals
        self._signal_manager = get_signal_manager()
        self._init = self._signal_manager.register(self, 'init', bool, size=1)
        self._dac = [self._signal_manager.register(self, f'v_out_{i}', float) for i in range(32)]
        self._offset = [self._signal_manager.register(self, f'v_offset_{i}', float) for i in range(32)]
        self._gain = [self._signal_manager.register(self, f'gain_mu_{i}', int) for i in range(32)]

        # Store attributes (from ARTIQ code)
        self.vref = vref
        self.offset_dacs = offset_dacs

        # Internal registers
        self._dac_reg = [0.0] * 32  # Float signals can only take float values
        self._offset_reg = [0.0] * 32  # Float signals can only take float values
        self._gain_reg = ['x'] * 32

    @kernel
    def init(self, blind=False):
        # Delays and calls from ARTIQ code
        self.write_offset_dacs_mu(self.offset_dacs)
        if not blind:
            delay(25 * us)
            delay(15 * us)
        self._signal_manager.event(self._init, 1)

    @kernel
    def read_reg(self, channel=0, op=AD53XX_READ_X1A):
        delay(270 * ns)  # t_21 min sync high in readback
        raise NotImplementedError

    @kernel
    def write_offset_dacs_mu(self, value):
        value &= 0x3fff
        self.offset_dacs = value

    @kernel
    def write_gain_mu(self, channel, gain=0xffff):
        # noinspection PyTypeChecker
        self._gain_reg[channel] = gain & 0xFFFF

    @kernel
    def write_offset_mu(self, channel, offset=0x8000):
        self.write_offset(channel, _mu_to_voltage(offset, self.offset_dacs, self.vref))

    @kernel
    def write_offset(self, channel, voltage):
        self._offset_reg[channel] = voltage

    @kernel
    def write_dac_mu(self, channel, value):
        self.write_dac(channel, _mu_to_voltage(value, self.offset_dacs, self.vref))

    @kernel
    def write_dac(self, channel, voltage):
        self._dac_reg[channel] = voltage

    @kernel
    def load(self):
        delay(10 * ns)  # t13 = 10ns ldac pulse width low
        for signals, values in [(self._dac, self._dac_reg),
                                (self._offset, self._offset_reg),
                                (self._gain, self._gain_reg)]:
            for s, v in zip(signals, values):
                self._signal_manager.event(s, v)

    # noinspection PyDefaultArgument
    @kernel
    def set_dac_mu(self, values, channels=list(range(40))):
        voltages = [_mu_to_voltage(v, self.offset_dacs, self.vref) for v in values]
        self.set_dac(voltages, channels)

    # noinspection PyDefaultArgument
    @kernel
    def set_dac(self, voltages, channels=list(range(40))):
        for i in range(len(voltages)):
            self.write_dac(channels[i], voltages[i])
        self.load()

    @kernel
    def calibrate(self, channel, vzs, vfs):
        offset_err = voltage_to_mu(vzs, self.offset_dacs, self.vref)
        gain_err = voltage_to_mu(vfs, self.offset_dacs, self.vref) - (offset_err + 0xffff)

        assert offset_err <= 0  # noqa: ATQ401
        assert gain_err >= 0  # noqa: ATQ401

        self.core.break_realtime()
        self.write_offset_mu(channel, 0x8000 - offset_err)
        self.write_gain_mu(channel, 0xffff - gain_err)
