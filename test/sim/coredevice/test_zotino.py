import test.sim.coredevice.test_ad53xx

_DEVICE_DB = {
    'core': {
        'type': 'local',
        'module': 'artiq.coredevice.core',
        'class': 'Core',
        'arguments': {'host': '1.2.3.4', 'ref_period': 1e-9}
    },
    "dut": {
        "type": "local",
        "module": "artiq.coredevice.zotino",
        "class": "Zotino"
    },
}


class ZotinoPeekTestCase(test.sim.coredevice.test_ad53xx.AD53xxPeekTestCase):
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
