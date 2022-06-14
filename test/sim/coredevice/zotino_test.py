import typing

import dax.sim.coredevice.zotino

import test.sim.coredevice.ad53xx_test

_DEVICE_DB = {
    'core': {
        'type': 'local',
        'module': 'artiq.coredevice.core',
        'class': 'Core',
        'arguments': {'host': None, 'ref_period': 1e-9}
    },
    'spi': {
        'type': 'local',
        'module': 'artiq.coredevice.spi2',
        'class': 'SPIMaster',
    },
    "dut": {
        "type": "local",
        "module": "artiq.coredevice.zotino",
        "class": "Zotino",
        "arguments": {"spi_device": "spi"}
    },
}


class ZotinoPeekTestCase(test.sim.coredevice.ad53xx_test.AD53xxPeekTestCase):
    DDB = _DEVICE_DB

    def test_leds(self):
        self.expect(self.env.dut, 'led', 'x')  # Equivalent to SignalNotSet

        ref = '000000010000000'
        for i in range(8):
            self.env.dut.set_leds(0x1 << i)
            value = ref[i:8 + i]
            assert value[-1 - i] == '1'
            self.expect(self.env.dut, 'led', value)
            self.env.dut.set_leds(0x0)
            self.expect(self.env.dut, 'led', '00000000')


class CompileTestCase(test.sim.coredevice.ad53xx_test.CompileTestCase):
    DEVICE_CLASS: typing.ClassVar[typing.Type] = dax.sim.coredevice.zotino.Zotino
    DEVICE_KWARGS = {
        'spi_device': 'spi',
    }
    FN_KWARGS = {
        'set_leds': {'leds': 0},
    }
    FN_KWARGS.update(test.sim.coredevice.ad53xx_test.CompileTestCase.FN_KWARGS)
    DEVICE_DB = _DEVICE_DB
