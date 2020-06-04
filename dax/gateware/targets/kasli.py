#!/usr/bin/env python3
# type: ignore

"""Build ARTIQ for EURIQA's hardware based on the ARTIQ Kasli."""

import argparse
import json

from artiq.build_soc import build_artiq_soc
from artiq.gateware import rtio
from artiq.gateware.eem import _eem_pin
from artiq.gateware.rtio.phy import edge_counter
from artiq.gateware.rtio.phy import ttl_serdes_7series
from artiq.gateware.rtio.phy import ttl_simple
from artiq.gateware.rtio.phy.spi2 import SPIMaster
from artiq.gateware.targets.kasli import StandaloneBase
from artiq.gateware.targets.kasli_generic import add_peripherals
from migen.build.generic_platform import IOStandard
from migen.build.generic_platform import Pins
from migen.build.generic_platform import Subsignal
from misoc.integration.builder import builder_argdict
from misoc.integration.builder import builder_args
from misoc.targets.kasli import soc_kasli_argdict
from misoc.targets.kasli import soc_kasli_args


class EURIQA(StandaloneBase):
    """EURIQA Kasli setup."""

    def __init__(self, description, hw_rev=None, **kwargs):  # noqa: C901
        """Build like regular Kasli, just replace some DIOs with sandia dac spi."""
        add_sandia_dac_spi = kwargs.pop("sandia_dac_spi", False)
        if hw_rev is None:
            hw_rev = description["hw_rev"]
        self.class_name_override = description["variant"]
        StandaloneBase.__init__(self, hw_rev=hw_rev, **kwargs)

        self.config["SI5324_AS_SYNTHESIZER"] = None
        self.config["RTIO_FREQUENCY"] = "{:.1f}".format(
            description.get("rtio_frequency", 125e6) / 1e6)
        if "ext_ref_frequency" in description:
            self.config["SI5324_EXT_REF"] = None
            self.config["EXT_REF_FREQUENCY"] = "{:.1f}".format(
                description["ext_ref_frequency"] / 1e6)
        if hw_rev == "v1.0":
            # EEM clock fan-out from Si5324, not MMCX
            self.comb += self.platform.request("clk_sel").eq(1)

        has_grabber = any(peripheral["type"] == "grabber"
                          for peripheral in description["peripherals"])
        if has_grabber:
            self.grabber_csr_group = []

        self.rtio_channels = []
        if add_sandia_dac_spi:
            # remove last bank of DIOs from description so that we can do it manually
            try:
                last_bank = [peripheral for peripheral in description["peripherals"]
                             if peripheral["type"] == "dio"].pop()
            except IndexError:
                raise ValueError("No DIOs present in description."
                                 "Can't add sandia dac spi")
            description["peripherals"].remove(last_bank)

        add_peripherals(self, description["peripherals"])

        if add_sandia_dac_spi:
            eem = last_bank["ports"][0]
            ttl_classes = {
                "input": ttl_serdes_7series.InOut_8X,
                "output": ttl_serdes_7series.Output_8X
            }
            if len(last_bank["ports"]) != 1:
                raise ValueError("wrong number of ports")
            if last_bank.get("edge_counter", False):
                edge_counter_cls = edge_counter.SimpleEdgeCounter
            else:
                edge_counter_cls = None

            self.platform.add_extension([
                ("dio{}".format(eem), i,
                 Subsignal("p", Pins(_eem_pin(eem, i, "p"))),
                 Subsignal("n", Pins(_eem_pin(eem, i, "n"))),
                 IOStandard("LVDS_25"))
                for i in range(5)
            ])
            print("{} (EEM{}) starting at RTIO channel {}".format(
                "DIO", eem, len(self.rtio_channels)))

            phys = []
            # ports 0-4 unchanged
            for i in range(5):
                pads = self.platform.request("dio{}".format(eem), i)
                phy = ttl_classes[last_bank["bank_direction_low"]](pads.p, pads.n)
                phys.append(phy)
                self.submodules += phy
                self.rtio_channels.append(rtio.Channel.from_phy(phy))

            if edge_counter_cls is not None:
                for phy in phys:
                    state = getattr(phy, "input_state", None)
                    if state is not None:
                        counter = edge_counter_cls(state)
                        self.submodules += counter
                        self.rtio_channels.append(rtio.Channel.from_phy(counter))

            # add dac spi
            self.platform.add_extension([
                (
                    "sandia_dac_spi_p", 0,
                    Subsignal("clk", Pins(_eem_pin(eem, 5, "p"))),
                    Subsignal("mosi", Pins(_eem_pin(eem, 6, "p"))),
                    Subsignal("cs_n", Pins(_eem_pin(eem, 7, "p"))),
                    IOStandard("LVDS_25")
                ),
                (
                    "sandia_dac_spi_n", 0,
                    Subsignal("clk", Pins(_eem_pin(eem, 5, "n"))),
                    Subsignal("mosi", Pins(_eem_pin(eem, 6, "n"))),
                    Subsignal("cs_n", Pins(_eem_pin(eem, 7, "n"))),
                    IOStandard("LVDS_25")
                )
            ])
            print("Sandia DAC SPI starting at RTIO channel {}".format(
                len(self.rtio_channels)))

            phy = SPIMaster(self.platform.request("sandia_dac_spi_p"),
                            self.platform.request("sandia_dac_spi_n"))
            self.submodules += phy
            self.rtio_channels.append(rtio.Channel.from_phy(phy, ififo_depth=128))

        for i in (1, 2):
            print("SFP LED at RTIO channel {}".format(len(self.rtio_channels)))
            sfp_ctl = self.platform.request("sfp_ctl", i)
            phy = ttl_simple.Output(sfp_ctl.led)
            self.submodules += phy
            self.rtio_channels.append(rtio.Channel.from_phy(phy))

        self.config["HAS_RTIO_LOG"] = None
        self.config["RTIO_LOG_CHANNEL"] = len(self.rtio_channels)
        self.rtio_channels.append(rtio.LogChannel())

        self.add_rtio(self.rtio_channels)
        if has_grabber:
            self.config["HAS_GRABBER"] = None
            self.add_csr_group("grabber", self.grabber_csr_group)
            for grabber in self.grabber_csr_group:
                self.platform.add_false_path_constraints(
                    self.rtio_crg.cd_rtio.clk,
                    getattr(self, grabber).deserializer.cd_cl.clk)


def main():
    """Build gateware for specified Kasli variant."""
    parser = argparse.ArgumentParser(
        description="ARTIQ device binary builder for generic Kasli systems")
    builder_args(parser)
    soc_kasli_args(parser)
    parser.set_defaults(output_dir="artiq_kasli")
    parser.add_argument("description", metavar="DESCRIPTION",
                        help="JSON system description file")
    parser.add_argument(
        "--sandia-dac-spi",
        action="store_true",
        help="Add SPI for real-time Sandia 100x DAC serial communication",
    )
    args = parser.parse_args()

    with open(args.description, "r") as f:
        description = json.load(f)

    if description["target"] != "kasli":
        raise ValueError("Description is for a different target")

    if description["base"] != "standalone":
        raise ValueError("Invalid base")

    soc = EURIQA(description, **soc_kasli_argdict(args),
                 sandia_dac_spi=args.sandia_dac_spi)
    args.variant = description["variant"]
    build_artiq_soc(soc, builder_argdict(args))


if __name__ == "__main__":
    main()
