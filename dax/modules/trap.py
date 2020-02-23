import numpy as np

import artiq.coredevice.ttl

from dax.base import *
from dax.modules.interfaces.trap_if import *

# Alias for TTLOut and TTLInOut type as a tuple
_TTL_OUT_TYPE = (artiq.coredevice.ttl.TTLOut, artiq.coredevice.ttl.TTLInOut)


class TrapModule(DaxModule, TrapInterface):
    """Module for a trap controlled by a Sandia DAC board."""

    COOL_DURATION_KEY = 'cool_duration'
    PUMP_DURATION_KEY = 'pump_duration'
    LOADED_IONS_KEY = 'loaded_ions'

    def build(self, oven_sw, cool_pump_sw, ion_sw, repump_sw, sdac_trig, sdac_config, sdac_data):
        # Switch device for oven
        self.setattr_device(oven_sw, 'oven_sw', _TTL_OUT_TYPE)

        # Switch device for cool and pump laser
        self.setattr_device(cool_pump_sw, 'cool_pump_sw', _TTL_OUT_TYPE)
        # Switch device for ion and repump laser
        self.setattr_device(ion_sw, 'ion_sw', _TTL_OUT_TYPE)
        # Switch device for ion and repump laser
        self.setattr_device(repump_sw, 'repump_sw', _TTL_OUT_TYPE)

        # Devices for Sandia DAC board
        self.setattr_device(sdac_trig, 'sdac_trig', _TTL_OUT_TYPE)
        self.setattr_device(sdac_config, 'sdac_config')
        self.setattr_device(sdac_data, 'sdac_data')

    def load(self):
        # Default cool duration
        self.setattr_dataset_sys(self.COOL_DURATION_KEY, 5 * us)
        # Default pump duration
        self.setattr_dataset_sys(self.PUMP_DURATION_KEY, 5 * us)
        # Number of loaded ions
        self.setattr_dataset_sys(self.LOADED_IONS_KEY, np.int32(0))

    def init(self):
        pass

    @kernel
    def config(self):
        # Break realtime to get some slack
        self.core.break_realtime()

        # Oven off by default
        self.oven_off()

        # Repump on by default
        self.repump_on()
        # Ion off by default
        self.ion_off()
        # Cool off by default
        self.cool_off()

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
        self.cool_mu(self.core.seconds_to_mu(duration))

    @kernel
    def cool_mu(self, duration):
        self.cool_on()
        delay_mu(duration)
        self.cool_off()

    @kernel
    def cool_default(self):
        self.cool(self.cool_duration)

    @kernel
    def set_pump_o(self, state):
        self.set_cool_o(state)

    @kernel
    def pump_on(self):
        self.set_pump_o(True)

    @kernel
    def pump_off(self):
        self.set_pump_o(False)

    @kernel
    def pump(self, duration):
        self.pump_mu(self.core.seconds_to_mu(duration))

    @kernel
    def pump_mu(self, duration):
        self.pump_on()
        delay_mu(duration)
        self.pump_off()

    @kernel
    def pump_default(self):
        self.pump(self.pump_duration)

    @kernel
    def set_ion_o(self, state):
        self.ion_sw.set_o(state)

    @kernel
    def ion_on(self):
        self.set_ion_o(True)

    @kernel
    def ion_off(self):
        self.set_ion_o(False)

    @kernel
    def set_repump_o(self, state):
        self.repump_sw.set_o(state)

    @kernel
    def repump_on(self):
        self.set_repump_o(True)

    @kernel
    def repump_off(self):
        self.set_repump_o(False)

    @portable
    def num_loaded_ions(self):
        return self.loaded_ions

    @host_only
    def set_loaded_ions(self, loaded_ions):
        # Set a new number of loaded ions
        self.loaded_ions = np.int32(loaded_ions)
        self.set_dataset_sys(self.LOADED_IONS_KEY, self.loaded_ions)

    @portable
    def get_targets(self):
        return [np.int32(i) for i in range(self.loaded_ions)]
