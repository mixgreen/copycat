import artiq.coredevice.ttl

from dax.experiment import *
from dax.modules.interfaces.global_beam_if import *


class GlobalBeamModule(DaxModule, GlobalBeamInterface):
    """Module for global beam control."""

    AOM_NAMES = 'brcz'
    AOM_FREQ_KEY = '{name:s}_aom_freq'
    AOM_PHASE_KEY = '{name:s}_aom_phase'
    AOM_ATT_KEY = '{name:s}_aom_att'

    # Configuration of switch (which configuration refers to which state)
    SW_BRC = False
    SW_Z = not SW_BRC

    def build(self, b_aom, r_aom, c_aom, z_aom, sw):
        # Global beam AOMs
        for k, n in zip((b_aom, r_aom, c_aom, z_aom), self.AOM_NAMES):
            self.setattr_device(k, '{name:s}_aom'.format(name=n))

        # Switch between BRC and Z
        self.setattr_device(sw, 'sw', artiq.coredevice.ttl.TTLOut)

        # Make switch configurations kernel invariant
        self.update_kernel_invariants('SW_BRC', 'SW_Z')

    def init(self):
        for n in self.AOM_NAMES:
            # Set AOM frequency, phase, and attenuation
            self.setattr_dataset_sys(self.AOM_FREQ_KEY.format(name=n), 100 * MHz)
            self.setattr_dataset_sys(self.AOM_PHASE_KEY.format(name=n), 0.0)
            self.setattr_dataset_sys(self.AOM_ATT_KEY.format(name=n), 0.0 * dB)

        # Initialize the devices
        self._init()

    @kernel
    def _init(self):
        # Reset core
        self.core.reset()

        # Make state of switch unambiguous
        self.set_sw(self.SW_BRC)

        # All AOM switches off
        self.off()

        # Set default configuration for all AOMs
        self.b_aom.set(self.b_aom_freq, phase=self.b_aom_phase)
        self.b_aom.set_att(self.b_aom_att)
        self.r_aom.set(self.r_aom_freq, phase=self.r_aom_phase)
        self.r_aom.set_att(self.r_aom_att)
        self.c_aom.set(self.c_aom_freq, phase=self.c_aom_phase)
        self.c_aom.set_att(self.c_aom_att)
        self.z_aom.set(self.z_aom_freq, phase=self.z_aom_phase)
        self.z_aom.set_att(self.z_aom_att)

        # Guarantee all events are submitted
        self.core.wait_until_mu(now_mu())

    def post_init(self):
        pass

    @kernel
    def set_sw(self, state):
        self.sw.set_o(state)

    @kernel
    def sw_brc(self):
        self.set_sw(self.SW_BRC)

    @kernel
    def sw_z(self):
        self.set_sw(self.SW_Z)

    @kernel
    def set_o(self, state):
        # Set switches of all AOMs (independent of self.sw state)
        self.b_aom.cfg_sw(state)
        self.r_aom.cfg_sw(state)
        self.c_aom.cfg_sw(state)
        self.z_aom.cfg_sw(state)

    @kernel
    def on(self):
        self.set_o(True)

    @kernel
    def off(self):
        self.set_o(False)

    @kernel
    def pulse(self, duration):
        self.on()
        delay(duration)
        self.off()

    @kernel
    def pulse_mu(self, duration):
        self.on()
        delay_mu(duration)
        self.off()
