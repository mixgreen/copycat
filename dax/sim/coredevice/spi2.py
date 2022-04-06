"""Write-only generic SPI simulation driver for use with Dax.Sim."""
# mypy: disallow_untyped_defs = False
# mypy: disallow_incomplete_defs = False
# mypy: check_untyped_defs = False
# mypy: disallow_subclassing_any = False
import typing

from artiq.coredevice.spi2 import SPIMaster as _SPIMaster   # type: ignore[import]
from artiq.language.core import kernel, delay_mu
from artiq.language.types import TNone
from dax.sim.device import DaxSimDevice
from dax.sim.signal import get_signal_manager


class SPIMaster(DaxSimDevice, _SPIMaster):
    """Wraps calls to ARTIQ SPI devices."""

    def __init__(self, dmgr: typing.Any, channel: int, **kwargs) -> None:
        kwargs_without_key = kwargs.copy()
        kwargs_without_key.pop("_key")
        super().__init__(dmgr, **kwargs)
        signal_manager = get_signal_manager()
        self._config_length = signal_manager.register(self, "cfg_transfer_length", int)
        self._config_cs = signal_manager.register(self, "cfg_chip_select", int)
        self._config_clk_div = signal_manager.register(self, "cfg_clk_divider", int)
        self._config_flags = signal_manager.register(self, "cfg_flags", int)
        self._out_data = signal_manager.register(self, "mosi", int)
        self._init_spi_device(channel, **kwargs_without_key)
        self._config_set = False

    def _init_spi_device(self, channel: int, div: int = 0, length: int = 0) -> None:
        self.ref_period_mu = self.core.seconds_to_mu(self.core.coarse_ref_period)
        self.channel = channel
        self.update_xfer_duration_mu(div, length)

    @kernel
    def set_config(self, flags, length, freq, cs) -> TNone:
        self.set_config_mu(flags, length, self.frequency_to_div(freq), cs)

    @kernel
    def set_config_mu(self, flags, length, div, cs) -> TNone:
        if length > 32 or length < 1:
            raise ValueError("Invalid SPI transfer length")
        if div > 257 or div < 2:
            raise ValueError("Invalid SPI clock divider")
        self._config_length.push(length)
        self._config_clk_div.push(div)
        self._config_cs.push(cs)
        self._config_flags.push(flags)
        delay_mu(self.ref_period_mu)
        self._config_set = True

    @kernel
    def write(self, data) -> TNone:
        if not self._config_set:
            raise RuntimeError("Data written to SPI device without configuring first")
        self._out_data.push(data)
        delay_mu(self.xfer_duration_mu)
        self._out_data.push("X")

    @kernel
    def read(self) -> TNone:
        raise NotImplementedError("Reads not simulated")
