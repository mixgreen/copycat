import artiq.coredevice.ttl

from dax.base import *


class MemsMirrorModule(DaxModule):
    """Module to control a MEMS mirrors."""

    def build(self, mems_trig, mems_sw, mems_dac):
        # Devices for the MEMS mirror board
        self.setattr_device(mems_trig, 'mems_trig', (artiq.coredevice.ttl.TTLOut, artiq.coredevice.ttl.TTLInOut))
        self.setattr_device(mems_sw, 'mems_sw')
        self.setattr_device(mems_dac, 'mems_dac')

    def load_module(self):
        pass

    @kernel
    def init_module(self):
        # Set direction of trigger signal
        self.mems_trig.output()

    @kernel
    def config_module(self):
        # Guarantee trigger is off
        self.mems_trig.off()
