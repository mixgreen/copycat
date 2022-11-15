"""Test :mod:`dax.sim.coredevice.spi2`."""
import artiq.language.environment as artiq_env
import artiq.coredevice.spi2 as spi  # type: ignore[import]

import dax.sim.test_case
import dax.sim.coredevice.spi2


class _Environment(artiq_env.HasEnvironment):
    def build(self):
        self.core = self.get_device("core")
        self.dut = self.get_device("dut")


class SPIPeekTestCase(dax.sim.test_case.PeekTestCase):
    _DEVICE_DB = {
        "core": {
            "type": "local",
            "module": "artiq.coredevice.core",
            "class": "Core",
            "arguments": {"host": None, "ref_period": 1e-9},
        },
        "dut": {
            "type": "local",
            "module": "artiq.coredevice.spi2",
            "class": "SPIMaster",
            "arguments": {"channel": 0},
        },
    }
    _SPI_CONFIG = spi.SPI_END | spi.SPI_CS_POLARITY | spi.SPI_CLK_POLARITY
    _CONFIG_SIGNALS = (
        "cfg_chip_select",
        "cfg_transfer_length",
        "cfg_clk_divider",
        "cfg_flags",
    )

    def setUp(self) -> None:
        # self.rng = random.Random(self.SEED)
        self.env = self.construct_env(_Environment, device_db=self._DEVICE_DB)

    def test_set_config(self):
        for sig in self._CONFIG_SIGNALS:
            self.expect(self.env.dut, sig, "x")
        self.env.dut.set_config(self._SPI_CONFIG, 32, 25e6, 1)
        self.expect(self.env.dut, "cfg_chip_select", 1)
        self.expect(self.env.dut, "cfg_clk_divider", 5)
        self.expect(self.env.dut, "cfg_transfer_length", 32)
        self.expect(self.env.dut, "cfg_flags", self._SPI_CONFIG)
        self.expect(self.env.dut, "mosi", "x")

    def test_set_config_equal(self):
        """Check that :meth:`set_config` and :meth:`set_config_mu` are equivalent."""
        for sig in self._CONFIG_SIGNALS:
            self.expect(self.env.dut, sig, "x")
        self.env.dut.set_config(self._SPI_CONFIG, 32, 25e6, 1)
        self.expect(self.env.dut, "cfg_chip_select", 1)
        self.expect(self.env.dut, "cfg_clk_divider", 5)
        self.expect(self.env.dut, "cfg_transfer_length", 32)
        self.expect(self.env.dut, "cfg_flags", self._SPI_CONFIG)
        self.expect(self.env.dut, "mosi", "x")
        self.env.dut.set_config_mu(self._SPI_CONFIG, 32, 5, 1)
        self.expect(self.env.dut, "cfg_chip_select", 1)
        self.expect(self.env.dut, "cfg_clk_divider", 5)
        self.expect(self.env.dut, "cfg_transfer_length", 32)
        self.expect(self.env.dut, "cfg_flags", self._SPI_CONFIG)
        self.expect(self.env.dut, "mosi", "x")

    def test_set_config_subscribe(self):
        callback_args = []

        def callback(*args):
            callback_args.append(args)

        self.env.dut.set_config_mu_subscribe(callback)

        ref = [(1, 2, 3, 4), (5, 6, 7, 8)]
        for args in ref:
            self.env.dut.set_config_mu(*args)
        self.assertListEqual(ref, callback_args)

    def test_write(self):
        self.expect(self.env.dut, "mosi", "x")

        self.env.dut.set_config(self._SPI_CONFIG, 32, 25e6, 1)
        self.env.dut.write(10)
        self.assertEqual(self.env.dut._out_data.pull(offset=-2), 10)
        self.expect(self.env.dut, "mosi", "x")

    def test_write_subscribe(self):
        callback_data = []

        def callback(data):
            callback_data.append(data)

        self.env.dut.write_subscribe(callback)

        ref = [1, 2, 3, 4]
        for data in ref:
            self.env.dut.write(data)
        self.assertListEqual(ref, callback_data)

    def test_read(self):
        with self.assertRaises(NotImplementedError):
            self.env.dut.read()
