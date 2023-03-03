from artiq.experiment import *

import dax.sim.test_case
import dax.sim.coredevice.suservo
import test.sim.coredevice._compile_testcase as compile_testcase


_DEVICE_DB = {
    'core': {
        'type': 'local',
        'module': 'artiq.coredevice.core',
        'class': 'Core',
        'arguments': {'host': None, 'ref_period': 1e-9}
    }
}

_DEVICE_DB["suservo0_ch0"] = {
    "type": "local",
    "module": "artiq.coredevice.suservo",
    "class": "Channel",
    "arguments": {"channel": 0x000028, "servo_device": "suservo0"},
}

_DEVICE_DB["suservo0_ch1"] = {
    "type": "local",
    "module": "artiq.coredevice.suservo",
    "class": "Channel",
    "arguments": {"channel": 0x000029, "servo_device": "suservo0"},
}

_DEVICE_DB["suservo0_ch2"] = {
    "type": "local",
    "module": "artiq.coredevice.suservo",
    "class": "Channel",
    "arguments": {"channel": 0x00002A, "servo_device": "suservo0"},
}

_DEVICE_DB["suservo0_ch3"] = {
    "type": "local",
    "module": "artiq.coredevice.suservo",
    "class": "Channel",
    "arguments": {"channel": 0x00002B, "servo_device": "suservo0"},
}

_DEVICE_DB["suservo0_ch4"] = {
    "type": "local",
    "module": "artiq.coredevice.suservo",
    "class": "Channel",
    "arguments": {"channel": 0x00002C, "servo_device": "suservo0"},
}

_DEVICE_DB["suservo0_ch5"] = {
    "type": "local",
    "module": "artiq.coredevice.suservo",
    "class": "Channel",
    "arguments": {"channel": 0x00002D, "servo_device": "suservo0"},
}

_DEVICE_DB["suservo0_ch6"] = {
    "type": "local",
    "module": "artiq.coredevice.suservo",
    "class": "Channel",
    "arguments": {"channel": 0x00002E, "servo_device": "suservo0"},
}

_DEVICE_DB["suservo0_ch7"] = {
    "type": "local",
    "module": "artiq.coredevice.suservo",
    "class": "Channel",
    "arguments": {"channel": 0x00002F, "servo_device": "suservo0"},
}

_DEVICE_DB["suservo0"] = {
    "type": "local",
    "module": "artiq.coredevice.suservo",
    "class": "SUServo",
    "arguments": {
            "channel": 0x000030,
            "pgia_device": "spi_sampler0_pgia",
            "cpld0_device": "urukul0_cpld",
            "cpld1_device": "urukul1_cpld",
            "dds0_device": "urukul0_dds",
            "dds1_device": "urukul1_dds",
    },
}

_DEVICE_DB["spi_sampler0_pgia"] = {
    "type": "local",
    "module": "artiq.coredevice.spi2",
    "class": "SPIMaster",
    "arguments": {"channel": 0x000031},
}

_DEVICE_DB["spi_urukul0"] = {
    "type": "local",
    "module": "artiq.coredevice.spi2",
    "class": "SPIMaster",
    "arguments": {"channel": 0x000018}
}

_DEVICE_DB["ttl_urukul0_io_update"] = {
    "type": "local",
    "module": "artiq.coredevice.ttl",
    "class": "TTLOut",
    "arguments": {"channel": 0x000019}
}

_DEVICE_DB["ttl_urukul0_sw0"] = {
    "type": "local",
    "module": "artiq.coredevice.ttl",
    "class": "TTLOut",
    "arguments": {"channel": 0x00001a}
}

_DEVICE_DB["ttl_urukul0_sw1"] = {
    "type": "local",
    "module": "artiq.coredevice.ttl",
    "class": "TTLOut",
    "arguments": {"channel": 0x00001b}
}

_DEVICE_DB["ttl_urukul0_sw2"] = {
    "type": "local",
    "module": "artiq.coredevice.ttl",
    "class": "TTLOut",
    "arguments": {"channel": 0x00001c}
}

_DEVICE_DB["ttl_urukul0_sw3"] = {
    "type": "local",
    "module": "artiq.coredevice.ttl",
    "class": "TTLOut",
    "arguments": {"channel": 0x00001d}
}

_DEVICE_DB["urukul0_cpld"] = {
    "type": "local",
    "module": "artiq.coredevice.urukul",
    "class": "CPLD",
    "arguments": {"spi_device": "spi_urukul0", "refclk": 125000000.0, "clk_sel": 2},
}

_DEVICE_DB["urukul0_dds"] = {
    "type": "local",
    "module": "artiq.coredevice.ad9910",
    "class": "AD9910",
    "arguments": {"pll_n": 32, "chip_select": 3, "cpld_device": "urukul0_cpld"},
}

_DEVICE_DB["spi_urukul1"] = {
    "type": "local",
    "module": "artiq.coredevice.spi2",
    "class": "SPIMaster",
    "arguments": {"channel": 0x000033},
}

_DEVICE_DB["urukul1_cpld"] = {
    "type": "local",
    "module": "artiq.coredevice.urukul",
    "class": "CPLD",
    "arguments": {"spi_device": "spi_urukul1", "refclk": 125000000.0, "clk_sel": 2},
}

_DEVICE_DB["urukul1_dds"] = {
    "type": "local",
    "module": "artiq.coredevice.ad9910",
    "class": "AD9910",
    "arguments": {
        "pll_n": 32,
        "chip_select": 5,
        "cpld_device": "urukul0_cpld",
        "sw_device": "ttl_urukul0_sw1"
    }
}


class _Environment(HasEnvironment):
    def build(self, *, dut):
        self.core = self.get_device('core')
        self.dut = self.get_device(dut)


class SuservoCompileTestCase(compile_testcase.CoredeviceCompileTestCase):
    DEVICE_CLASS = dax.sim.coredevice.suservo.SUServo
    DEVICE_KWARGS = {
        "channel": 0x000030,
        "pgia_device": "spi_sampler0_pgia",
        "cpld0_device": "urukul0_cpld",
        "cpld1_device": "urukul1_cpld",
        "dds0_device": "urukul0_dds",
        "dds1_device": "urukul1_dds",
    }
    FN_ARGS = {
        'set_config': (0),
        'set_pgia_mu': (0, 0.0),
    }
    FN_EXCLUDE = {'write', 'read', 'get_adc_mu', 'get_adc'}
    DEVICE_DB = _DEVICE_DB


class SuservoTestCase(dax.sim.test_case.PeekTestCase):

    def setUp(self) -> None:
        self.env = self.construct_env(_Environment,
                                      device_db=_DEVICE_DB,
                                      build_kwargs=dict(dut='suservo0')
                                      )

    def test_init(self):
        self.expect(self.env.dut, 'init', 'x')
        self.env.dut.init()
        self.expect(self.env.dut, 'init', 1)

    def test_config(self):
        self.assertEqual(self.env.dut.get_status(), 0)
        self.env.dut.set_config(enable=1)
        self.assertEqual(self.env.dut.get_status(), 1)

    def test_set_cpld_att(self):
        self.env.dut.cpld0.set_att(0, 10.0)

    def test_set_pgia(self):
        # TODO: update with sampler sim driver
        self.env.dut.set_pgia_mu(1, 1)


class SuservoChannelTestCase(dax.sim.test_case.PeekTestCase):

    def setUp(self) -> None:
        self.env = self.construct_env(_Environment,
                                      device_db=_DEVICE_DB,
                                      build_kwargs=dict(dut='suservo0_ch0')
                                      )

    def test_init(self):
        self.expect(self.env.dut, 'init', 'x')
        self.env.dut.init()
        self.expect(self.env.dut, 'init', 1)

    def test_set(self):
        self.env.dut.set(en_out=1, en_iir=1, profile=0)
        self.assertEqual(self.env.dut.get_profile_mu(), 0)

    def test_dds(self):
        profile = 0
        offset = -0.1
        frequency = 200 * MHz
        phase = 0.0

        self.env.dut.set_dds(
            profile=profile,
            offset=offset,
            frequency=frequency,
            phase=phase,
        )
        self.env.dut.set_dds_offset(0, -0.2)

    def test_iir(self):
        self.env.dut.set_iir(
            profile=0,
            adc=0,
            kp=-0.1,
            ki=-300.0 / s,
            g=0.0,
            delay=0.0,
        )

    def test_y(self):
        y = 0.0
        self.env.dut.set_y(profile=0, y=y)
        self.assertEqual(self.env.dut.get_y(), y)
