#!/usr/bin/env python3
"""Build ARTIQ for EURIQA's hardware based on the Xilinx KC705 FPGA.

Uses hardware (DAC/ADC/GPIO/etc) built by Duke. Located in blue "pulser" box.
"""
import argparse
import itertools
import logging

from artiq.build_soc import build_artiq_soc
from artiq.gateware import rtio
from artiq.gateware.rtio.phy import spi2
from artiq.gateware.rtio.phy import ttl_serdes_7series
from artiq.gateware.rtio.phy import ttl_simple
from artiq.gateware.targets.kc705 import _StandaloneBase
from misoc.integration.builder import builder_argdict
from misoc.integration.builder import builder_args
from misoc.targets.kc705 import soc_kc705_argdict
from misoc.targets.kc705 import soc_kc705_args

from . import euriqa

_LOGGER = logging.getLogger(__name__)


class EURIQA(_StandaloneBase):
    """EURIQA pulser setup."""

    def __init__(self, **kwargs):
        """Declare hardware available on Euriqa's KC705 & Duke Breakout."""
        add_sandia_dac_spi = kwargs.pop("sandia_dac_spi", False)
        _StandaloneBase.__init__(self, **kwargs)
        unused_count = itertools.count()

        platform = self.platform
        platform.add_extension(euriqa.fmc_adapter_io)
        if add_sandia_dac_spi:
            # segment to prevent accidentally adding x100 DAC comm/pins
            platform.add_extension(euriqa.x100_dac_spi)

        rtio_channels = list()

        # USER_SMA_GPIO_P
        phy = ttl_serdes_7series.InOut_8X(platform.request("user_sma_gpio_p_33"))
        self.submodules += phy
        rtio_channels.append(rtio.Channel.from_phy(phy, ififo_depth=512))

        # USER_SMA_GPIO_N
        phy = ttl_serdes_7series.InOut_8X(platform.request("user_sma_gpio_n_33"))
        self.submodules += phy
        rtio_channels.append(rtio.Channel.from_phy(phy, ififo_depth=512))

        for i in range(2, 4):
            # GPIO_LED_[2..3]
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


        # Output GPIO/TTL Banks
        for bank, i in itertools.product(["out1", "out2", "out3",  "out4"], range(8)):
            # out1-0, out1-1, ..., out3-7
            if add_sandia_dac_spi and bank == "out2" and i == 7:
                # add unused dummy channel. to keep channel #s same.
                # Won't output to useful digital line
                phy = ttl_serdes_7series.Output_8X(
                    platform.request("unused", next(unused_count))
                )
            else:
                phy = ttl_serdes_7series.Output_8X(platform.request(bank, i))
            self.submodules += phy
            rtio_channels.append(rtio.Channel.from_phy(phy))
        _LOGGER.debug("Ending output GPIO chan: %i", len(rtio_channels))

        # Input GPIO/TTL Banks
        for bank, i in itertools.product(["in1", "in2", "in3"], range(8)):
            # in1-0, in1-1, ..., in5-7
            phy = ttl_serdes_7series.InOut_8X(platform.request(bank, i))
            self.submodules += phy
            rtio_channels.append(rtio.Channel.from_phy(phy, ififo_depth=512))
        _LOGGER.debug("Ending input GPIO chan: %i", len(rtio_channels))

        # Tri-state buffer to disable the TTL/GPIO outputs (out1, ...)
        phy = ttl_simple.Output(platform.request("oeb", 0))
        self.submodules += phy
        rtio_channels.append(rtio.Channel.from_phy(phy))
        _LOGGER.debug("OEB GPIO chan: %i", len(rtio_channels))

        # TODO: figure out usage/what this is
        for i in range(9):
            phy = ttl_serdes_7series.Output_8X(platform.request("sma", i))
            self.submodules += phy
            rtio_channels.append(rtio.Channel.from_phy(phy))
        _LOGGER.debug("Ending sma chan: %i", len(rtio_channels))

        # TODO: update name for io_update everywhere
        # Update triggers for DDS. Edge will trigger output settings update
        for i in range(8):
            phy = ttl_serdes_7series.Output_8X(
                platform.request("io_update", i))
            self.submodules += phy
            rtio_channels.append(rtio.Channel.from_phy(phy))
        _LOGGER.debug("Ending io_update GPIO chan: %i", len(rtio_channels))

        # Reset lines for the DDS boards.
        for i in range(4):
            phy = ttl_serdes_7series.Output_8X(platform.request("reset", i))
            self.submodules += phy
            rtio_channels.append(rtio.Channel.from_phy(phy))
        _LOGGER.debug("Ending DDS Reset GPIO chan: %i", len(rtio_channels))

        # SPI, CLR, RESET and LDAC interfaces to control the MEMS system
        for i in range(2)
            spi_bus = self.platform.request("spi", i)
            phy = spi2.SPIMaster(spi_bus)
            self.submodules += phy
            rtio_channels.append(
                rtio.Channel.from_phy(phy, ififo_depth=128)
            )
        _LOGGER.debug("Ending MEMS SPI channels: %i", len(rtio_channels))

        phy = ttl_simple.Output(platform.request("hv209_clr", 0))
        self.submodules += phy
        rtio_channels.append(rtio.Channel.from_phy(phy))
        _LOGGER.debug("MEMS switch HV209 clear channel: %i", len(rtio_channels))

        phy = ttl_simple.Output(platform.request("dac8734_reset", 0))
        self.submodules += phy
        rtio_channels.append(rtio.Channel.from_phy(phy))
        _LOGGER.debug("MEMS DAC8734 reset channel: %i", len(rtio_channels))

        phy = ttl_simple.Output(platform.request("ldac_mems", 0))
        self.submodules += phy
        rtio_channels.append(rtio.Channel.from_phy(phy))
        _LOGGER.debug("MEMS LDAC GPIO channel: %i", len(rtio_channels))

        # SPI interfaces to control the DDS board outputs
        for i in range(2, 6):
            spi_bus = self.platform.request("spi", i)
            phy = spi2.SPIMaster(spi_bus)
            self.submodules += phy
            rtio_channels.append(rtio.Channel.from_phy(phy, ififo_depth=128))
            odd_channel_sdio = platform.request("odd_channel_sdio", (i-2))
            self.comb += odd_channel_sdio.eq(spi_bus.mosi)
        _LOGGER.debug("Ending SPI chan: %i", len(rtio_channels))

        # SPI & Load DAC (LDAC) pins for Controlling 8x DAC (DAC 8568)
        spi_bus = self.platform.request("spi", 6)
        phy = spi2.SPIMaster(spi_bus)
        self.submodules += phy
        rtio_channels.append(
            rtio.Channel.from_phy(phy, ififo_depth=128)
        )
        _LOGGER.debug("DAC8568 SPI RTIO channel: %i", len(rtio_channels))
        phy = ttl_simple.Output(platform.request("ldac", 0))
        self.submodules += phy
        rtio_channels.append(rtio.Channel.from_phy(phy))
        _LOGGER.debug("DAC8568 LDAC GPIO channel: %i", len(rtio_channels))

        # SPI for Coredevice serial comm to Sandia DAC
        if add_sandia_dac_spi:
            print("Adding SPI for Sandia DAC comms")
            spi_bus = self.platform.request("spi", 7)
            phy = spi2.SPIMaster(spi_bus)
            self.submodules += phy
            rtio_channels.append(
                rtio.Channel.from_phy(phy, ififo_depth=128)
            )

        self.config["HAS_RTIO_LOG"] = None
        self.config["RTIO_LOG_CHANNEL"] = len(rtio_channels)
        _LOGGER.debug("RTIO log chan: %i", len(rtio_channels))
        rtio_channels.append(rtio.LogChannel())

        _LOGGER.debug("Euriqa KC705 RTIO channels: %s",
                      list(enumerate(rtio_channels)))
        self.add_rtio(rtio_channels)


VARIANTS = {cls.__name__.lower(): cls for cls in [EURIQA]}

def get_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser for kc705 gateware builder."""
    parser = argparse.ArgumentParser(
        description="KC705 gateware and firmware builder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    builder_args(parser)
    soc_kc705_args(parser)
    parser.set_defaults(output_dir="artiq4_kc705_euriqa")
    parser.add_argument(
        "-V",
        "--variant",
        choices=VARIANTS.keys(),
        default="euriqa",
        help="variant: %(choices)s (default: %(default)s)",
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        action="count",
        default=0,
        help="increase logging verbosity level (default=WARNING)",
    )
    parser.add_argument(
        "--sandia-dac-spi",
        action="store_true",
        help="Add SPI for real-time Sandia 100x DAC serial communication",
    )
    return parser


def main() -> None:
    """Build gateware for specified KC705 FPGA variant."""
    args = get_argument_parser().parse_args()
    logging.basicConfig(level=logging.WARNING - args.verbosity)

    variant = args.variant.lower()
    try:
        cls = VARIANTS[variant]
    except KeyError:
        raise SystemExit("Invalid variant (-V/--variant)")

    soc = cls(**soc_kc705_argdict(args), sandia_dac_spi=args.sandia_dac_spi)
    build_artiq_soc(soc, builder_argdict(args))
    # NOTE: if you get a XILINX license error,
    #   check you have the proper license in ~/.Xilinx/


if __name__ == "__main__":
    main()
