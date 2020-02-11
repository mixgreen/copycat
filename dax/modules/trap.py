import artiq.coredevice.ttl

from dax.base import *
from dax.modules.interfaces.trap_if import *

# Alias for TTLOut and TTLInOut type as a tuple
_TTL_OUT_TYPE = (artiq.coredevice.ttl.TTLOut, artiq.coredevice.ttl.TTLInOut)


class TrapModule(DaxModule, TrapInterface):
    """Module for a trap controlled by a Sandia DAC board."""

    def build(self, oven_sw, cool_pump_sw, ion_repump_sw, sdac_trig, sdac_config, sdac_data):
        # Switch device for oven
        self.setattr_device(oven_sw, 'oven_sw', _TTL_OUT_TYPE)

        # Switch device for cool and pump laser
        self.setattr_device(cool_pump_sw, 'cool_pump_sw', _TTL_OUT_TYPE)
        # Switch device for ion and repump laser
        self.setattr_device(ion_repump_sw, 'ion_repump_sw', _TTL_OUT_TYPE)

        # Devices for Sandia DAC board
        self.setattr_device(sdac_trig, 'sdac_trig', _TTL_OUT_TYPE)
        self.setattr_device(sdac_config, 'sdac_config')
        self.setattr_device(sdac_data, 'sdac_data')

    def load(self):
        pass

    @kernel
    def init(self):
        # Break realtime to get some slack
        self.core.break_realtime()

        # Configure oven switch as output
        self.oven_sw.output()

        # Configure optical switches as outputs
        self.cool_pump_sw.output()
        self.ion_repump_sw.output()

        # Set direction of trigger signal
        self.sdac_trig.output()

    @kernel
    def config(self):
        # Break realtime to get some slack
        self.core.break_realtime()

        # Oven off
        self.oven_off()

        # Ion off by default
        self.ion_on()
        # Repump on by default
        self.repump_on()

        # Guarantee Sandia DAC trigger is off
        self.sdac_trig.off()

    @kernel
    def set_oven_o(self, state):
        self.oven_sw.set_o(state)

    @kernel
    def oven_on(self):
        self.set_oven_o(True)

    @kernel
    def oven_off(self):
        self.set_oven_o(False)

    @kernel
    def set_cool_o(self, state):
        self.cool_pump_sw.set_o(state)

    @kernel
    def cool_on(self):
        self.set_cool_o(True)

    @kernel
    def cool_off(self):
        self.set_cool_o(False)

    @kernel
    def cool(self, duration):
        """Cool for a specified duration."""
        self.cool_on()
        delay(duration)
        self.cool_off()

    @kernel
    def cool_mu(self, duration):
        """Cool for a specified duration in machine units."""
        self.cool_on()
        delay_mu(duration)
        self.cool_off()

    @kernel
    def set_pump_o(self, state):
        # Cool and pump share the same switch
        self.set_cool_o(state)

    @kernel
    def pump_on(self):
        self.set_pump_o(True)

    @kernel
    def pump_off(self):
        self.set_pump_o(False)

    @kernel
    def set_ion_o(self, state):
        self.ion_repump_sw.set_o(state)

    @kernel
    def ion_on(self):
        self.set_ion_o(True)

    @kernel
    def ion_off(self):
        self.set_ion_o(False)

    @kernel
    def set_repump_o(self, state):
        # Ion and repump share the same switch
        self.set_ion_o(state)

    @kernel
    def repump_on(self):
        self.set_repump_o(True)

    @kernel
    def repump_off(self):
        self.set_repump_o(False)
