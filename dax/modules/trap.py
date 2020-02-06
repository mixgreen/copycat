import artiq.coredevice.ttl

from dax.base import *


class TrapModule(DaxModule):
    """Module for a trap controlled by a Sandia DAC board."""

    def build(self, sdac_trig, sdac_config, sdac_data):
        # Devices for Sandia DAC board
        self.setattr_device(sdac_trig, 'sdac_trig', (artiq.coredevice.ttl.TTLOut, artiq.coredevice.ttl.TTLInOut))
        self.setattr_device(sdac_config, 'sdac_config')
        self.setattr_device(sdac_data, 'sdac_data')

    def load_module(self):
        pass

    @kernel
    def init_module(self):
        # Set direction of trigger signal
        self.sdac_trig.output()

    @kernel
    def config_module(self):
        # Guarantee trigger is off
        self.sdac_trig.off()
