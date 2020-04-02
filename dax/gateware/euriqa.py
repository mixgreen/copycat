"""Extension card for the EURIQA project.

Describes the "FMC" adapter that plugs into the KC705 and interfaces with the
Sandia/Duke Pulser hardware: DDS's, ADC, DAC, GPIO (TTL), etc.

Description of the pins & interfaces available on the FMC connector.
"""
from migen.build.generic_platform import IOStandard
from migen.build.generic_platform import Pins
from migen.build.generic_platform import Subsignal

fmc_adapter_io = [
    # Input Bank 1
    ("in1", 0, Pins("HPC:HA13_N"), IOStandard("LVCMOS33")),
    ("in1", 1, Pins("HPC:HA02_N"), IOStandard("LVCMOS33")),
    ("in1", 2, Pins("HPC:HA14_P"), IOStandard("LVCMOS33")),
    ("in1", 3, Pins("HPC:HA02_P"), IOStandard("LVCMOS33")),
    ("in1", 4, Pins("HPC:HA14_N"), IOStandard("LVCMOS33")),
    ("in1", 5, Pins("HPC:HA01_CC_N"), IOStandard("LVCMOS33")),
    ("in1", 6, Pins("HPC:LA23_P"), IOStandard("LVCMOS33")),
    ("in1", 7, Pins("HPC:LA28_N"), IOStandard("LVCMOS33")),
    # Input Bank 2
    ("in2", 0, Pins("HPC:HA11_N"), IOStandard("LVCMOS33")),
    ("in2", 1, Pins("HPC:HA19_P"), IOStandard("LVCMOS33")),
    ("in2", 2, Pins("HPC:HA12_P"), IOStandard("LVCMOS33")),
    ("in2", 3, Pins("HPC:HA04_P"), IOStandard("LVCMOS33")),
    ("in2", 4, Pins("HPC:HA12_N"), IOStandard("LVCMOS33")),
    ("in2", 5, Pins("HPC:HA03_N"), IOStandard("LVCMOS33")),
    ("in2", 6, Pins("HPC:HA13_P"), IOStandard("LVCMOS33")),
    ("in2", 7, Pins("HPC:HA03_P"), IOStandard("LVCMOS33")),
    # Input Bank 3
    ("in3", 0, Pins("HPC:HA04_N"), IOStandard("LVCMOS33")),
    ("in3", 1, Pins("HPC:HA05_P"), IOStandard("LVCMOS33")),
    ("in3", 2, Pins("HPC:HA05_N"), IOStandard("LVCMOS33")),
    ("in3", 3, Pins("HPC:HA06_P"), IOStandard("LVCMOS33")),
    ("in3", 5, Pins("HPC:HA06_N"), IOStandard("LVCMOS33")),
    ("in3", 4, Pins("HPC:HA07_P"), IOStandard("LVCMOS33")),
    ("in3", 6, Pins("HPC:HA07_N"), IOStandard("LVCMOS33")),
    ("in3", 7, Pins("HPC:HA08_P"), IOStandard("LVCMOS33")),
    # Output Bank 1
    ("out1", 0, Pins("HPC:LA03_N"), IOStandard("LVCMOS33")),
    ("out1", 1, Pins("HPC:LA03_P"), IOStandard("LVCMOS33")),
    ("out1", 2, Pins("HPC:LA02_N"), IOStandard("LVCMOS33")),
    ("out1", 3, Pins("HPC:LA02_P"), IOStandard("LVCMOS33")),
    ("out1", 4, Pins("HPC:LA01_CC_N"), IOStandard("LVCMOS33")),
    ("out1", 5, Pins("HPC:LA01_CC_P"), IOStandard("LVCMOS33")),
    ("out1", 6, Pins("HPC:LA00_CC_N"), IOStandard("LVCMOS33")),
    ("out1", 7, Pins("HPC:LA00_CC_P"), IOStandard("LVCMOS33")),
    # Output Bank 2
    ("out2", 0, Pins("HPC:LA07_N"), IOStandard("LVCMOS33")),
    ("out2", 1, Pins("HPC:LA07_P"), IOStandard("LVCMOS33")),
    ("out2", 3, Pins("HPC:LA06_N"), IOStandard("LVCMOS33")),
    ("out2", 2, Pins("HPC:LA06_P"), IOStandard("LVCMOS33")),
    ("out2", 4, Pins("HPC:LA05_N"), IOStandard("LVCMOS33")),
    ("out2", 5, Pins("HPC:LA05_P"), IOStandard("LVCMOS33")),
    ("out2", 6, Pins("HPC:LA04_N"), IOStandard("LVCMOS33")),
    ("out2", 7, Pins("HPC:LA04_P"), IOStandard("LVCMOS33")),
    # Output Bank 3
    ("out3", 0, Pins("HPC:LA32_N"), IOStandard("LVCMOS33")),
    ("out3", 1, Pins("HPC:LA09_N"), IOStandard("LVCMOS33")),
    ("out3", 2, Pins("HPC:LA32_P"), IOStandard("LVCMOS33")),
    ("out3", 3, Pins("HPC:LA09_P"), IOStandard("LVCMOS33")),
    ("out3", 4, Pins("HPC:LA28_P"), IOStandard("LVCMOS33")),
    ("out3", 5, Pins("HPC:LA08_N"), IOStandard("LVCMOS33")),
    ("out3", 6, Pins("HPC:LA30_P"), IOStandard("LVCMOS33")),
    ("out3", 7, Pins("HPC:HA08_P"), IOStandard("LVCMOS33")),

    # Output Bank 4
    ("out4", 0, Pins("HPC:LA19_P"), IOStandard("LVCMOS33")),
    ("out4", 1, Pins("HPC:LA11_N"), IOStandard("LVCMOS33")),
    ("out4", 2, Pins("HPC:LA19_N"), IOStandard("LVCMOS33")),
    ("out4", 3, Pins("HPC:LA11_P"), IOStandard("LVCMOS33")),
    ("out4", 4, Pins("HPC:LA33_N"), IOStandard("LVCMOS33")),
    ("out4", 5, Pins("HPC:LA10_N"), IOStandard("LVCMOS33")),
    ("out4", 6, Pins("HPC:LA33_P"), IOStandard("LVCMOS33")),
    ("out4", 7, Pins("HPC:LA10_P"), IOStandard("LVCMOS33")),
    # SMA output/input??
    ("sma", 0, Pins("HPC:HA09_P"), IOStandard("LVCMOS33")),
    ("sma", 1, Pins("HPC:LA24_P"), IOStandard("LVCMOS33")),
    ("sma", 2, Pins("HPC:LA24_N"), IOStandard("LVCMOS33")),
    ("sma", 3, Pins("HPC:HA19_N"), IOStandard("LVCMOS33")),
    ("sma", 4, Pins("HPC:HA10_N"), IOStandard("LVCMOS33")),
    ("sma", 5, Pins("HPC:LA20_N"), IOStandard("LVCMOS33")),
    ("sma", 6, Pins("HPC:HA11_P"), IOStandard("LVCMOS33")),
    ("sma", 7, Pins("HPC:HA09_N"), IOStandard("LVCMOS33")),
    ("sma", 8, Pins("HPC:LA20_P"), IOStandard("LVCMOS33")),
    # OEB = Output enable buffer
    ("oeb", 0, Pins("HPC: HA08_N"), IOStandard("LVCMOS33")),
    # trigger for updating output of the DDS
    ("io_update", 0, Pins("HPC:HA00_CC_N"), IOStandard("LVCMOS33")),
    ("io_update", 1, Pins("HPC:HA18_P"), IOStandard("LVCMOS33")),
    ("io_update", 2, Pins("HPC:LA22_P"), IOStandard("LVCMOS33")),
    ("io_update", 3, Pins("HPC:LA26_P"), IOStandard("LVCMOS33")),
    ("io_update", 4, Pins("HPC:LA16_P"), IOStandard("LVCMOS33")),
    ("io_update", 5, Pins("HPC:LA14_N"), IOStandard("LVCMOS33")),
    ("io_update", 6, Pins("HPC:LA18_CC_P"), IOStandard("LVCMOS33")),
    ("io_update", 7, Pins("HPC:LA12_N"), IOStandard("LVCMOS33")),
    # Resets for the DDS control boards
    ("reset", 0, Pins("HPC:HA01_CC_P"), IOStandard("LVCMOS33")),
    ("reset", 1, Pins("HPC:LA22_N"), IOStandard("LVCMOS33")),
    ("reset", 2, Pins("HPC:LA16_N"), IOStandard("LVCMOS33")),
    ("reset", 3, Pins("HPC:LA18_CC_N"), IOStandard("LVCMOS33")),
    # data channel for odd DDS channels on the DDS board,
    # Because the data channels are separated
    # the even channels are in the mosi Subsignal of the SPI busses
    ("odd_channel_sdio", 0, Pins("HPC:HA18_N"), IOStandard("LVCMOS33")),
    ("odd_channel_sdio", 1, Pins("HPC:LA26_N"), IOStandard("LVCMOS33")),
    ("odd_channel_sdio", 2, Pins("HPC:LA14_P"), IOStandard("LVCMOS33")),
    ("odd_channel_sdio", 3, Pins("HPC:LA12_P"), IOStandard("LVCMOS33")),
    # SPI outputs for MEMS
    (
        "spi", # used for MEMS system: HV209
        0,
        Subsignal("clk", Pins("LPC:LA20_P")),
        Subsignal("cs_n", Pins("")),
        Subsignal("mosi", Pins("LPC:LA21_P")),
        IOStandard("LVCMOS33"),
    ),
    (
        "spi", # used for MEMS system: DAC8734
        1,
        Subsignal("clk", Pins("LPC:LA23_P")),
        Subsignal("cs_n", Pins("LPC:LA23_N")),
        Subsignal("mosi", Pins("LPC:LA22_N")),
        IOStandard("LVCMOS33"),
    ),
    ("hv209_clr", 0, Pins("LPC:LA20_N"), IOStandard("LVCMOS33")),
    ("dac8734_reset", 0, Pins("LPC:LA21_N"), IOStandard("LVCMOS33")),
    ("ldac_mems", 0, Pins("LPC:LA22_P"), IOStandard("LVCMOS33")),
    # SPI channels for communicating to DDS boards
    (
        "spi",
        2,
        Subsignal("clk", Pins("HPC:HA16_N")),
        Subsignal("cs_n", Pins("HPC:HA17_CC_P HPC:HA00_CC_P")),
        Subsignal("mosi", Pins("HPC:HA17_CC_N")),
        IOStandard("LVCMOS33"),
    ),
    (
        "spi",
        3,
        Subsignal("clk", Pins("HPC:LA21_P")),
        Subsignal("cs_n", Pins("HPC:LA25_P HPC:LA21_N")),
        Subsignal("mosi", Pins("HPC:LA25_N")),
        IOStandard("LVCMOS33"),
    ),
    (
        "spi",
        4,
        Subsignal("clk", Pins("HPC:LA27_P")),
        Subsignal("cs_n", Pins("HPC:LA23_N HPC:LA27_N")),
        Subsignal("mosi", Pins("HPC:LA15_P")),
        IOStandard("LVCMOS33"),
    ),
    (
        "spi",
        5,
        Subsignal("clk", Pins("HPC:LA17_CC_P")),
        Subsignal("cs_n", Pins("HPC:LA13_N HPC:LA17_CC_N")),
        Subsignal("mosi", Pins("HPC:LA13_P")),
        IOStandard("LVCMOS33"),
    ),
    # DAC8568 Control pins: SPI & Load DAC TTL/GPIO trigger (LDAC)
    (
        "spi",
        6,
        Subsignal("clk", Pins("HPC: HA10_P")),
        Subsignal("cs_n", Pins("HPC: HA23_P")),
        Subsignal("mosi", Pins("HPC: HA23_N")),
        IOStandard("LVCMOS33"),
    ),
    ("ldac", 0, Pins("HPC: HA21_P"), IOStandard("LVCMOS33")),
]

x100_dac_spi = [
    # SPI to hack UART-like communication to Sandia DAC. Takes over out2-7.
    # SHOULD NOT ALWAYS BE USED. Only if using real-time comm to 100x DAC.
    # Only relevant pin is "miso", overwrites the SD card pins.
    # COULD SCREW UP YOUR SD CARD. CAREFUL
    (
        "spi",
        7,
        Subsignal("clk", Pins("AB22")),  # Unassigned # AB22 = SD Card MOSI
        Subsignal("mosi", Pins("HPC:LA04_P")),
        Subsignal("cs_n", Pins("AC21")),    # AC21 = SD Card CS_n
        IOStandard("LVCMOS33"),
    ),
    # unused acts as dummy pin, not routed to relevant location.
    ("unused", 0, Pins("AC20"), IOStandard("LVCMOS33"))     # AC20 = SD Card MISO
]
