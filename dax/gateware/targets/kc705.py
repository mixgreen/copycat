#!/usr/bin/env python3

from artiq.gateware.targets.kc705 import *
from artiq.gateware.targets.kc705 import _StandaloneBase


class KC705_BARE(_StandaloneBase):
    """
    Bare KC705 board with only onboard hardware. Based on NIST_CLOCK class.
    """

    def __init__(self, **kwargs):
        _StandaloneBase.__init__(self, **kwargs)

        platform = self.platform

        rtio_channels = []

        # USER_SMA_GPIO_P
        phy = ttl_serdes_7series.InOut_8X(platform.request("user_sma_gpio_p_33"))
        self.submodules += phy
        rtio_channels.append(rtio.Channel.from_phy(phy, ififo_depth=512))

        # USER_SMA_GPIO_N
        phy = ttl_serdes_7series.InOut_8X(platform.request("user_sma_gpio_n_33"))
        self.submodules += phy
        rtio_channels.append(rtio.Channel.from_phy(phy, ififo_depth=512))

        for i in range(2, 8):
            # GPIO_LED_[2..7]
            phy = ttl_simple.Output(platform.request("user_led", i))
            self.submodules += phy
            rtio_channels.append(rtio.Channel.from_phy(phy))

        ams101_dac = self.platform.request("ams101_dac", 0)
        phy = ttl_simple.Output(ams101_dac.ldac)
        self.submodules += phy
        rtio_channels.append(rtio.Channel.from_phy(phy))

        phy = spi2.SPIMaster(ams101_dac)
        self.submodules += phy
        rtio_channels.append(rtio.Channel.from_phy(
            phy, ififo_depth=4))

        phy = spi2.SPIMaster(platform.request("sdcard_spi_33"))
        self.submodules += phy
        rtio_channels.append(rtio.Channel.from_phy(
            phy, ififo_depth=4))

        self.config["HAS_RTIO_LOG"] = None
        self.config["RTIO_LOG_CHANNEL"] = len(rtio_channels)
        rtio_channels.append(rtio.LogChannel())

        self.add_rtio(rtio_channels)


VARIANTS = {cls.__name__.lower(): cls for cls in [KC705_BARE, NIST_CLOCK, NIST_QC2, SMA_SPI]}


def main():
    parser = argparse.ArgumentParser(description="KC705 gateware and firmware builder")
    builder_args(parser)
    soc_kc705_args(parser)
    parser.set_defaults(output_dir="artiq_kc705")
    parser.add_argument("-V", "--variant", choices=VARIANTS, default="kc705_bare")
    args = parser.parse_args()

    variant = args.variant.lower()
    try:
        cls = VARIANTS[variant]
    except KeyError:
        raise SystemExit("Invalid variant (-V/--variant)")

    soc = cls(**soc_kc705_argdict(args))
    build_artiq_soc(soc, builder_argdict(args))


if __name__ == "__main__":
    main()
