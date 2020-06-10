#!/usr/bin/env python3
# type: ignore

import itertools
import argparse

from migen import *
from misoc.targets.kc705 import soc_kc705_args, soc_kc705_argdict
from misoc.integration.builder import builder_args, builder_argdict

from artiq.gateware import rtio
from artiq.gateware.rtio.phy import ttl_simple, ttl_serdes_7series, spi2
from artiq.build_soc import build_artiq_soc
# noinspection PyProtectedMember
from artiq.gateware.targets.kc705 import _RTIOCRG, _StandaloneBase

import dax.gateware.euriqa as euriqa


# noinspection PyPep8Naming
class KC705_BARE(_StandaloneBase):
    """Bare KC705 board with only onboard hardware. Based on NIST_CLOCK and SMA_SPI class."""

    def __init__(self, sd_card_spi=False, **kwargs):
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
        rtio_channels.append(rtio.Channel.from_phy(phy, ififo_depth=4))

        if sd_card_spi:
            phy = spi2.SPIMaster(platform.request("sdcard_spi_33"))
            self.submodules += phy
            rtio_channels.append(rtio.Channel.from_phy(phy, ififo_depth=4))

        self.config["HAS_RTIO_LOG"] = None
        self.config["RTIO_LOG_CHANNEL"] = len(rtio_channels)
        rtio_channels.append(rtio.LogChannel())

        self.add_rtio(rtio_channels)

    def add_rtio(self, rtio_channels):
        self.submodules.rtio_crg = _RTIOCRG(self.platform, self.crg.cd_sys.clk, use_sma=False)
        self.csr_devices.append("rtio_crg")
        self.config["HAS_RTIO_CLOCK_SWITCH"] = None
        self.submodules.rtio_tsc = rtio.TSC("async", glbl_fine_ts_width=3)
        self.submodules.rtio_core = rtio.Core(self.rtio_tsc, rtio_channels)
        self.csr_devices.append("rtio_core")
        self.submodules.rtio = rtio.KernelInitiator(self.rtio_tsc)
        self.submodules.rtio_dma = ClockDomainsRenamer("sys_kernel")(
            rtio.DMA(self.get_native_sdram_if()))
        self.register_kernel_cpu_csrdevice("rtio")
        self.register_kernel_cpu_csrdevice("rtio_dma")
        self.submodules.cri_con = rtio.CRIInterconnectShared(
            [self.rtio.cri, self.rtio_dma.cri],
            [self.rtio_core.cri])
        self.register_kernel_cpu_csrdevice("cri_con")
        self.submodules.rtio_moninj = rtio.MonInj(rtio_channels)
        self.csr_devices.append("rtio_moninj")

        self.platform.add_period_constraint(self.rtio_crg.cd_rtio.clk, 8.)
        self.platform.add_false_path_constraints(
            self.crg.cd_sys.clk,
            self.rtio_crg.cd_rtio.clk)

        self.submodules.rtio_analyzer = rtio.Analyzer(self.rtio_tsc, self.rtio_core.cri,
                                                      self.get_native_sdram_if())
        self.csr_devices.append("rtio_analyzer")


class EURIQA(_StandaloneBase):
    """EURIQA setup (Red Chamber)."""

    def __init__(self, **kwargs):
        """Declare hardware available on EURIQA KC705 & Duke Breakout."""
        add_sandia_dac_spi = kwargs.pop("sandia_dac_spi", False)
        _StandaloneBase.__init__(self, **kwargs)
        unused_count = itertools.count()

        platform = self.platform
        platform.add_extension(euriqa.fmc_adapter_io)
        if add_sandia_dac_spi:
            # segment to prevent accidentally adding x100 DAC comm/pins
            platform.add_extension(euriqa.x100_dac_spi)

        rtio_channels = list()

        # Note: prints were used instead of a logger because the logger gets disabled for unknown reasons

        # USER_SMA_GPIO_P
        print("user_sma_gpio_p_33 at channel {:d}".format(len(rtio_channels)))
        phy = ttl_serdes_7series.InOut_8X(platform.request("user_sma_gpio_p_33"))
        self.submodules += phy
        rtio_channels.append(rtio.Channel.from_phy(phy, ififo_depth=512))

        # USER_SMA_GPIO_N
        print("user_sma_gpio_n_33 at channel {:d}".format(len(rtio_channels)))
        phy = ttl_serdes_7series.InOut_8X(platform.request("user_sma_gpio_n_33"))
        self.submodules += phy
        rtio_channels.append(rtio.Channel.from_phy(phy, ififo_depth=512))

        for i in range(2, 4):
            # GPIO_LED_[2..3]
            print("user_led-{:d} at channel {:d}".format(i, len(rtio_channels)))
            phy = ttl_simple.Output(platform.request("user_led", i))
            self.submodules += phy
            rtio_channels.append(rtio.Channel.from_phy(phy))

        # Output GPIO/TTL Banks
        for bank, i in itertools.product(["out1", "out2", "out3", "out4"], range(8)):
            # out1-0, out1-1, ..., out3-7
            print("{:s}-{:d} at channel {:d}".format(bank, i, len(rtio_channels)))
            if add_sandia_dac_spi and bank == "out2" and i == 7:
                # add unused dummy channel. to keep channel #s same.
                # Won't output to useful digital line
                phy = ttl_serdes_7series.Output_8X(platform.request("unused", next(unused_count)))
            else:
                phy = ttl_serdes_7series.Output_8X(platform.request(bank, i))
            self.submodules += phy
            rtio_channels.append(rtio.Channel.from_phy(phy))
        print("Ending output GPIO channels at {:d}".format(len(rtio_channels)))

        # Input GPIO/TTL Banks
        for bank, i in itertools.product(["in1", "in2", "in3"], range(8)):
            # in1-0, in1-1, ..., in5-7
            print("{:s}-{:d} at channel {:d}".format(bank, i, len(rtio_channels)))
            phy = ttl_serdes_7series.InOut_8X(platform.request(bank, i))
            self.submodules += phy
            rtio_channels.append(rtio.Channel.from_phy(phy, ififo_depth=512))
        print("Ending input GPIO channels at {:d}".format(len(rtio_channels)))

        # Tri-state buffer to disable the TTL/GPIO outputs (out1, ...)
        phy = ttl_simple.Output(platform.request("oeb", 0))
        self.submodules += phy
        rtio_channels.append(rtio.Channel.from_phy(phy))
        print("OEB GPIO channels at {:d}".format(len(rtio_channels)))

        # TODO: figure out usage/what this is
        for i in range(9):
            print("{:s}-{:d} at channel {:d}".format('sma', i, len(rtio_channels)))
            phy = ttl_serdes_7series.Output_8X(platform.request("sma", i))
            self.submodules += phy
            rtio_channels.append(rtio.Channel.from_phy(phy))
        print("Ending SMA channels at {:d}".format(len(rtio_channels)))

        # TODO: update name for io_update everywhere
        # Update triggers for DDS. Edge will trigger output settings update
        for i in range(8):
            print("{:s}-{:d} at channel {:d}".format('dds-io_update', i, len(rtio_channels)))
            phy = ttl_serdes_7series.Output_8X(platform.request("io_update", i))
            self.submodules += phy
            rtio_channels.append(rtio.Channel.from_phy(phy))
        print("Ending DDS io_update GPIO channels at {:d}".format(len(rtio_channels)))

        # Reset lines for the DDS boards.
        for i in range(4):
            print("{:s}-{:d} at channel {:d}".format('dds-reset', i, len(rtio_channels)))
            phy = ttl_serdes_7series.Output_8X(platform.request("reset", i))
            self.submodules += phy
            rtio_channels.append(rtio.Channel.from_phy(phy))
        print("Ending DDS reset GPIO channels at {:d}".format(len(rtio_channels)))

        # SPI, CLR, RESET and LDAC interfaces to control the MEMS system
        for i in range(2):
            print("{:s}-{:d} at channel {:d}".format('MEMS-spi', i, len(rtio_channels)))
            spi_bus = self.platform.request("spi", i)
            phy = spi2.SPIMaster(spi_bus)
            self.submodules += phy
            rtio_channels.append(rtio.Channel.from_phy(phy, ififo_depth=128))
        print("Ending MEMS SPI channels at {:d}".format(len(rtio_channels)))

        print("MEMS switch HV209 clear channel at {:d}".format(len(rtio_channels)))
        phy = ttl_simple.Output(platform.request("hv209_clr", 0))
        self.submodules += phy
        rtio_channels.append(rtio.Channel.from_phy(phy))

        print("MEMS DAC8734 reset channel at {:d}".format(len(rtio_channels)))
        phy = ttl_simple.Output(platform.request("dac8734_reset", 0))
        self.submodules += phy
        rtio_channels.append(rtio.Channel.from_phy(phy))

        print("MEMS LDAC GPIO channel at {:d}".format(len(rtio_channels)))
        phy = ttl_simple.Output(platform.request("ldac_mems", 0))
        self.submodules += phy
        rtio_channels.append(rtio.Channel.from_phy(phy))

        # SPI interfaces to control the DDS board outputs
        for i in range(2, 6):
            print("{:s}-{:d} at channel {:d}".format('dds-spi', i, len(rtio_channels)))
            spi_bus = self.platform.request("spi", i)
            phy = spi2.SPIMaster(spi_bus)
            self.submodules += phy
            rtio_channels.append(rtio.Channel.from_phy(phy, ififo_depth=128))
            odd_channel_sdio = platform.request("odd_channel_sdio", (i - 2))
            self.comb += odd_channel_sdio.eq(spi_bus.mosi)
        print("Ending DDS SPI channels at {:d}".format(len(rtio_channels)))

        # SPI & Load DAC (LDAC) pins for Controlling 8x DAC (DAC 8568)
        print("DAC8568 SPI RTIO channel at {:d}".format(len(rtio_channels)))
        spi_bus = self.platform.request("spi", 6)
        phy = spi2.SPIMaster(spi_bus)
        self.submodules += phy
        rtio_channels.append(rtio.Channel.from_phy(phy, ififo_depth=128))

        print("DAC8568 LDAC GPIO channel at {:d}".format(len(rtio_channels)))
        phy = ttl_simple.Output(platform.request("ldac", 0))
        self.submodules += phy
        rtio_channels.append(rtio.Channel.from_phy(phy))

        # SPI for core device serial comm to Sandia DAC
        if add_sandia_dac_spi:
            print("{:s} at channel {:d}".format('Sandia DAC-spi', len(rtio_channels)))
            spi_bus = self.platform.request("spi", 7)
            phy = spi2.SPIMaster(spi_bus)
            self.submodules += phy
            rtio_channels.append(rtio.Channel.from_phy(phy, ififo_depth=128))

        self.config["HAS_RTIO_LOG"] = None
        self.config["RTIO_LOG_CHANNEL"] = len(rtio_channels)
        print("RTIO log channel at {:d}".format(len(rtio_channels)))
        rtio_channels.append(rtio.LogChannel())

        print("Euriqa KC705 RTIO channels:\n{:s}".format(
            '\n'.join('{{"channel": {:#x}}}, {}'.format(i, c) for i, c in enumerate(rtio_channels))))
        self.add_rtio(rtio_channels)

    def add_rtio(self, rtio_channels):
        self.submodules.rtio_crg = _RTIOCRG(self.platform, self.crg.cd_sys.clk, use_sma=False)
        self.csr_devices.append("rtio_crg")
        self.config["HAS_RTIO_CLOCK_SWITCH"] = None
        self.submodules.rtio_tsc = rtio.TSC("async", glbl_fine_ts_width=3)
        self.submodules.rtio_core = rtio.Core(self.rtio_tsc, rtio_channels)
        self.csr_devices.append("rtio_core")
        self.submodules.rtio = rtio.KernelInitiator(self.rtio_tsc)
        self.submodules.rtio_dma = ClockDomainsRenamer("sys_kernel")(
            rtio.DMA(self.get_native_sdram_if()))
        self.register_kernel_cpu_csrdevice("rtio")
        self.register_kernel_cpu_csrdevice("rtio_dma")
        self.submodules.cri_con = rtio.CRIInterconnectShared(
            [self.rtio.cri, self.rtio_dma.cri],
            [self.rtio_core.cri])
        self.register_kernel_cpu_csrdevice("cri_con")
        self.submodules.rtio_moninj = rtio.MonInj(rtio_channels)
        self.csr_devices.append("rtio_moninj")

        self.platform.add_period_constraint(self.rtio_crg.cd_rtio.clk, 8.)
        self.platform.add_false_path_constraints(
            self.crg.cd_sys.clk,
            self.rtio_crg.cd_rtio.clk)

        self.submodules.rtio_analyzer = rtio.Analyzer(self.rtio_tsc, self.rtio_core.cri,
                                                      self.get_native_sdram_if())
        self.csr_devices.append("rtio_analyzer")


# Update the available variants
VARIANTS = {cls.__name__.lower(): cls for cls in [KC705_BARE, EURIQA]}


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="KC705 gateware and firmware builder")
    builder_args(parser)
    soc_kc705_args(parser)
    parser.set_defaults(output_dir="kc705")
    parser.add_argument("variant", choices=VARIANTS,
                        help="The variant to build")
    parser.add_argument("--sandia-dac-spi", action="store_true",
                        help="Add SPI for real-time Sandia 100x DAC serial communication (EURIQA only)")
    args = parser.parse_args()

    # Obtain variant class
    cls = VARIANTS[args.variant.lower()]

    # Prepare kwargs
    kwargs = soc_kc705_argdict(args)
    if cls is EURIQA:
        # Flags only for EURIQA gateware
        kwargs.update(sandia_dac_spi=args.sandia_dac_spi)

    # Build
    soc = cls(**kwargs)
    build_artiq_soc(soc, builder_argdict(args))


if __name__ == "__main__":
    main()
