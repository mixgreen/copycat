import artiq.coredevice.ttl

from dax.base import *


class ChamberModule(DaxModule):
    """Module representing a chamber with standard actuators."""

    COOL_AOM_FREQ_KEY = 'cool_aom_freq'
    COOL_AOM_PHASE_KEY = 'cool_aom_phase'
    COOL_AOM_ATT_KEY = 'cool_aom_att'

    PUMP_AOM_FREQ_KEY = 'pump_aom_freq'
    PUMP_AOM_PHASE_KEY = 'pump_aom_phase'
    PUMP_AOM_ATT_KEY = 'pump_aom_att'

    ION_AOM_FREQ_KEY = 'ion_aom_freq'
    ION_AOM_PHASE_KEY = 'ion_aom_phase'
    ION_AOM_ATT_KEY = 'ion_aom_att'

    REPUMP_AOM_FREQ_KEY = 'repump_aom_freq'
    REPUMP_AOM_PHASE_KEY = 'repump_aom_phase'
    REPUMP_AOM_ATT_KEY = 'repump_aom_att'

    def build(self, oven_sw, cool_aom, pump_aom, ion_aom, repump_aom):
        # Switch device for oven
        self.setattr_device(oven_sw, 'oven_sw', (artiq.coredevice.ttl.TTLOut, artiq.coredevice.ttl.TTLInOut))

        # Various AOMs
        self.setattr_device(cool_aom, 'cool_aom')
        self.setattr_device(pump_aom, 'pump_aom')
        self.setattr_device(ion_aom, 'ion_aom')
        self.setattr_device(repump_aom, 'repump_aom')

    def load_module(self):
        # For all AOMs: frequency, phase, and attenuation
        self.setattr_dataset_sys(self.COOL_AOM_FREQ_KEY, 100 * MHz)
        self.setattr_dataset_sys(self.COOL_AOM_PHASE_KEY, 0.0)
        self.setattr_dataset_sys(self.COOL_AOM_ATT_KEY, 0.0 * dB)

        self.setattr_dataset_sys(self.PUMP_AOM_FREQ_KEY, 100 * MHz)
        self.setattr_dataset_sys(self.PUMP_AOM_PHASE_KEY, 0.0)
        self.setattr_dataset_sys(self.PUMP_AOM_ATT_KEY, 0.0 * dB)

        self.setattr_dataset_sys(self.ION_AOM_FREQ_KEY, 100 * MHz)
        self.setattr_dataset_sys(self.ION_AOM_PHASE_KEY, 0.0)
        self.setattr_dataset_sys(self.ION_AOM_ATT_KEY, 0.0 * dB)

        self.setattr_dataset_sys(self.REPUMP_AOM_FREQ_KEY, 100 * MHz)
        self.setattr_dataset_sys(self.REPUMP_AOM_PHASE_KEY, 0.0)
        self.setattr_dataset_sys(self.REPUMP_AOM_ATT_KEY, 0.0 * dB)

    @kernel
    def init_module(self):
        # Break realtime to get some slack
        self.core.break_realtime()

        # Configure oven switch as output
        self.oven_sw.output()

        # Initialize all AOMs
        self.cool_aom.init()
        self.pump_aom.init()
        self.ion_aom.init()
        self.repump_aom.init()

    @kernel
    def config_module(self):
        # Break realtime to get some slack
        self.core.break_realtime()

        # Oven off
        self.oven_off()

        # Configure repump AOM and switch on by default
        self.repump_aom.set(self.repump_aom_freq, phase=self.repump_aom_phase)
        self.repump_aom.set_att(self.repump_aom_att)
        self.repump_aom.cfg_sw(True)

        # Switch other AOMs off
        self.cool_off()
        self.pump_off()
        self.ion_off()

        # Configure other AOMs
        self.cool_aom.set(self.cool_aom_freq, phase=self.cool_aom_phase)
        self.cool_aom.set_att(self.cool_aom_att)
        self.pump_aom.set(self.pump_aom_freq, phase=self.pump_aom_phase)
        self.pump_aom.set_att(self.pump_aom_att)
        self.ion_aom.set(self.ion_aom_freq, phase=self.ion_aom_phase)
        self.ion_aom.set_att(self.ion_aom_att)

    @kernel
    def set_oven_sw(self, state):
        self.oven_sw.set_o(state)

    @kernel
    def oven_on(self):
        self.set_oven_sw(True)

    @kernel
    def oven_off(self):
        self.set_oven_sw(False)

    @kernel
    def set_cool_sw(self, state):
        self.cool_aom.cfg_sw(state)

    @kernel
    def cool_on(self):
        self.set_cool_sw(True)

    @kernel
    def cool_off(self):
        self.set_cool_sw(False)

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
    def set_pump_sw(self, state):
        self.pump_aom.cfg_sw(state)

    @kernel
    def pump_on(self):
        self.set_pump_sw(True)

    @kernel
    def pump_off(self):
        self.set_pump_sw(False)

    @kernel
    def set_ion_sw(self, state):
        self.ion_aom.cfg_sw(state)

    @kernel
    def ion_on(self):
        self.set_ion_sw(True)

    @kernel
    def ion_off(self):
        self.set_ion_sw(False)
