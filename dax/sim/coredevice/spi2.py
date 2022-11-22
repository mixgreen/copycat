"""Write-only generic SPI simulation driver for use with DAX.sim."""
# mypy: disallow_untyped_defs = False
# mypy: disallow_incomplete_defs = False
# mypy: check_untyped_defs = False
# mypy: disallow_subclassing_any = False

import typing

from artiq.coredevice.spi2 import SPIMaster as _SPIMaster  # type: ignore[import]
from artiq.language.core import kernel, rpc, host_only, delay_mu
from artiq.language.types import TNone, TInt32

from dax.sim.device import DaxSimDevice
from dax.sim.signal import get_signal_manager


class SPIMaster(DaxSimDevice, _SPIMaster):
    """Wraps calls to ARTIQ SPI devices."""

    _set_config_mu_subscribers: typing.List[typing.Callable[[int, int, int, int], typing.Any]]
    _write_subscribers: typing.List[typing.Callable[[int], typing.Any]]

    def __init__(self, dmgr, div=0, length=0, **kwargs) -> None:
        # Call super
        super().__init__(dmgr, **kwargs)

        # Subscribers
        self._set_config_mu_subscribers = []
        self._write_subscribers = []

        # Register signals
        signal_manager = get_signal_manager()
        self._config_length = signal_manager.register(self, "cfg_transfer_length", int)
        self._config_cs = signal_manager.register(self, "cfg_chip_select", int)
        self._config_clk_div = signal_manager.register(self, "cfg_clk_divider", int)
        self._config_flags = signal_manager.register(self, "cfg_flags", int)
        self._out_data = signal_manager.register(self, "mosi", int)

        # Store attributes and initialization (from ARTIQ code)
        self.ref_period_mu = self.core.seconds_to_mu(
            self.core.coarse_ref_period)
        assert self.ref_period_mu == self.core.ref_multiplier
        self.update_xfer_duration_mu(div, length)

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
        self.update_xfer_duration_mu(div, length)
        delay_mu(self.ref_period_mu)
        self._set_config_mu_notify(flags, length, div, cs)

    @host_only
    def set_config_mu_subscribe(self, fn: typing.Callable[[int, int, int, int], typing.Any]) -> None:
        """Subscribe to :func:`set_config` and :func:`set_config_mu` calls of this device.

        :param fn: Callback function for the notification
        """
        self._set_config_mu_subscribers.append(fn)

    @rpc
    def _set_config_mu_notify(self, flags: int, length: int, div: int, cs: int) -> TNone:
        for fn in self._set_config_mu_subscribers:
            fn(flags, length, div, cs)

    @kernel
    def write(self, data) -> TNone:
        self._out_data.push(data)
        delay_mu(self.xfer_duration_mu)
        self._out_data.push("X")
        self._write_notify(data)

    @host_only
    def write_subscribe(self, fn: typing.Callable[[int], typing.Any]) -> None:
        """Subscribe to :func:`write` calls of this device.

        :param fn: Callback function for the notification
        """
        self._write_subscribers.append(fn)

    @rpc
    def _write_notify(self, data: int) -> TNone:
        for fn in self._write_subscribers:
            fn(data)

    @kernel
    def read(self) -> TInt32:
        raise NotImplementedError("Reads not simulated")
