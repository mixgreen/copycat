# mypy: disallow_untyped_defs = False
# mypy: disallow_incomplete_defs = False
# mypy: check_untyped_defs = False

from artiq.language.core import *
from artiq.language.units import *
from artiq.coredevice.ad53xx import voltage_to_mu, AD53XX_READ_X1A  # type: ignore

from dax.sim.device import DaxSimDevice
from dax.sim.signal import get_signal_manager


def _mu_to_voltage(voltage_mu, *, vref, offset_dacs=0x0):
    return ((voltage_mu - (offset_dacs * 0x4)) / 0x10000) * (4. * vref)


class AD53xx(DaxSimDevice):
    _NUM_CHANNELS = 32
    """Number of output channels."""

    def __init__(self, dmgr, vref=5., offset_dacs=8192, **kwargs):
        # Call super
        super(AD53xx, self).__init__(dmgr, **kwargs)

        # Register signals
        self._signal_manager = get_signal_manager()
        self._init = self._signal_manager.register(self, 'init', bool, size=1)
        self._dac = [self._signal_manager.register(self, f'v_out_{i}', float) for i in range(self._NUM_CHANNELS)]
        self._offset = [self._signal_manager.register(self, f'v_offset_{i}', float) for i in range(self._NUM_CHANNELS)]
        self._gain = [self._signal_manager.register(self, f'gain_{i}', float) for i in range(self._NUM_CHANNELS)]

        # Store attributes (from ARTIQ code)
        assert 2 * V <= vref <= 5 * V, 'Reference voltage out of range'
        self.vref = vref
        assert 0 <= offset_dacs < 2 ** 14, 'Offset DACs out of range'
        if vref == 5 * V:
            assert offset_dacs <= 8192
        self.offset_dacs = offset_dacs

        # Internal registers
        self._dac_reg_mu = [0] * self._NUM_CHANNELS  # Kept in machine units for JIT conversion
        self._offset_reg = [0.0] * self._NUM_CHANNELS  # Float signals can only take float values
        self._gain_reg = [0.0] * self._NUM_CHANNELS  # Float signals can only take float values

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
        raise NotImplementedError

    @kernel
    def write_offset_dacs_mu(self, value):
        self.offset_dacs = value & 0x3fff
        self._update_signals()  # Changing offset DACs takes effect immediately

    @kernel
    def write_gain_mu(self, channel, gain=0xffff):
        assert 0 <= channel < self._NUM_CHANNELS, 'Channel out of range'
        # noinspection PyTypeChecker
        self._gain_reg[channel] = ((gain & 0xFFFF) + 1) / 2 ** 16

    @kernel
    def write_offset_mu(self, channel, offset=0x8000):
        self.write_offset(channel, _mu_to_voltage((offset & 0xFFFF) - 2 ** 15, vref=self.vref))

    @kernel
    def write_offset(self, channel, voltage):
        assert 0 <= channel < self._NUM_CHANNELS, 'Channel out of range'
        assert -2 * self.vref <= voltage <= 2 * self.vref, 'Offset voltage out of range'
        self._offset_reg[channel] = float(voltage)

    @kernel
    def write_dac_mu(self, channel, value):
        assert 0 <= channel < self._NUM_CHANNELS, 'Channel out of range'
        self._dac_reg_mu[channel] = value & 0xFFFF

    @kernel
    def write_dac(self, channel, voltage):
        self.write_dac_mu(channel, self.voltage_to_mu(voltage))

    @kernel
    def load(self):
        delay(10 * ns)  # t13 = 10ns ldac pulse width low
        self._update_signals()

    def _update_signals(self):
        for i in range(self._NUM_CHANNELS):
            v_out = _mu_to_voltage(self._dac_reg_mu[i], vref=self.vref, offset_dacs=self.offset_dacs)
            self._signal_manager.event(self._dac[i], v_out)
            self._signal_manager.event(self._offset[i], self._offset_reg[i])
            self._signal_manager.event(self._gain[i], self._gain_reg[i])

    @kernel
    def set_dac_mu(self, values, channels=None):
        if channels is None:
            channels = list(range(40))  # Note: 40 is too large, but this is taken from the ARTIQ driver
        for i in range(len(values)):
            self.write_dac_mu(channels[i], values[i])
        self.load()

    @kernel
    def set_dac(self, voltages, channels=None):
        voltages_mu = [self.voltage_to_mu(v) for v in voltages]
        assert all(0 <= v_mu < 2 ** 16 for v_mu in voltages_mu), 'One or more voltages out of range'
        self.set_dac_mu(voltages_mu, channels)

    @kernel
    def calibrate(self, channel, vzs, vfs):
        offset_err = self.voltage_to_mu(vzs)
        gain_err = self.voltage_to_mu(vfs) - (offset_err + 0xffff)

        assert offset_err <= 0
        assert gain_err >= 0

        self.core.break_realtime()
        self.write_offset_mu(channel, 0x8000 - offset_err)
        self.write_gain_mu(channel, 0xffff - gain_err)

    @portable
    def voltage_to_mu(self, voltage):
        return voltage_to_mu(voltage, self.offset_dacs, self.vref)
