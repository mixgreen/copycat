import random

from artiq.experiment import *

import dax.sim.test_case
import dax.sim.coredevice.adf5356
from dax.sim.device import ARTIQ_MAJOR_VERSION

import test.sim.coredevice._compile_testcase as compile_testcase

_DEVICE_DB = {
    'core': {
        'type': 'local',
        'module': 'artiq.coredevice.core',
        'class': 'Core',
        'arguments': {'host': None, 'ref_period': 1e-9}
    },
    'spi_mirny0': {
        'type': 'local',
        'module': 'artiq.coredevice.spi2',
        'class': 'SPIMaster',
    },
    'ttl_mirny0_sw0': {
        'type': 'local',
        'module': 'artiq.coredevice.ttl',
        'class': 'TTLOut',
        'arguments': {},
    },
    'mirny0_ch0': {
        "type": "local",
        "module": "artiq.coredevice.adf5356",
        "class": "ADF5356",
        "arguments": {
            "channel": 0,
            "sw_device": "ttl_mirny0_sw0",
            "cpld_device": "mirny0_cpld",
        }
    },
    "mirny0_cpld": {
        "type": "local",
        "module": "artiq.coredevice.mirny",
        "class": "Mirny",
        "arguments": {
            "spi_device": "spi_mirny0",
            "refclk": 100000000.0,
            "clk_sel": 0
        },
    },
    "dut": "mirny0_ch0",
}


class _Environment(HasEnvironment):
    def build(self):
        self.core = self.get_device('core')
        self.dut = self.get_device('dut')


class Adf5356TestCase(dax.sim.test_case.PeekTestCase):
    SEED = None

    def setUp(self) -> None:
        self.rng = random.Random(self.SEED)
        self.env = self.construct_env(_Environment, device_db=_DEVICE_DB)

    def test_init(self):
        self.expect(self.env.dut, 'init', 'x')
        self.env.dut.init()
        self.expect(self.env.dut, 'init', 1)

    if ARTIQ_MAJOR_VERSION >= 7:
        def test_set_att(self):
            signal = f'att_{self.env.dut.channel}'
            self.expect(self.env.dut.cpld, signal, 'x')
            self.env.dut.set_att(0.0)
            self.expect(self.env.dut.cpld, signal, 0.0)

    def test_set_att_mu(self):
        signal = f'att_{self.env.dut.channel}'
        self.expect(self.env.dut.cpld, signal, 'x')
        self.env.dut.set_att_mu(255)
        self.expect(self.env.dut.cpld, signal, 0.0)

    def test_set_output_power(self):
        self.expect(self.env.dut, 'power', 'x')
        for power in range(4):
            self.env.dut.set_output_power_mu(power)
            self.expect(self.env.dut, 'power', power)

    def test_enable_output(self):
        self.expect(self.env.dut, 'enable', 'x')
        self.env.dut.enable_output()
        self.expect(self.env.dut, 'enable', True)
        self.env.dut.disable_output()
        self.expect(self.env.dut, 'enable', False)

    def test_set_frequency(self):
        for freq in [60e6, 1e9, 5e9]:
            self.env.dut.set_frequency(freq)
            self.expect(self.env.dut, 'freq', freq)

    def test_info(self):
        self.assertIsInstance(self.env.dut.info(), dict)


class CompileTestCase(compile_testcase.CoredeviceCompileTestCase):
    DEVICE_CLASS = dax.sim.coredevice.adf5356.ADF5356
    DEVICE_KWARGS = {
        "channel": 0,
        "sw_device": "ttl_mirny0_sw0",
        "cpld_device": "mirny0_cpld",
    }
    FN_ARGS = {
        'set_att_mu': (0,),
        'set_att': (0.0,),
        'set_output_power_mu': (0,),
        'set_frequency': (1e9,),
    }
    FN_EXCLUDE = {'write', 'read_muxout', '_init_registers', '_compute_pfd_frequency'}
    DEVICE_DB = _DEVICE_DB
