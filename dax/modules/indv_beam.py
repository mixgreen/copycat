import artiq.coredevice.ttl

from dax.base import *
from dax.base.indv_beam_if import *


class _MemsMirrorModule(DaxModule):
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
        # Break realtime to get some slack
        self.core.break_realtime()

        # Set direction of trigger signal
        self.mems_trig.output()

    @kernel
    def config_module(self):
        # Break realtime to get some slack
        self.core.break_realtime()

        # Guarantee trigger is off
        self.mems_trig.off()


class IndvBeamMemsModule(DaxModule, IndvBeamInterface):
    """Module for individual beam path controlled with MEMS mirrors."""

    DPASS_AOM_0_FREQ_KEY = 'dpass_aom_0_freq'
    DPASS_AOM_0_PHASE_KEY = 'dpass_aom_0_phase'
    DPASS_AOM_0_ATT_KEY = 'dpass_aom_0_att'

    DPASS_AOM_1_FREQ_KEY = 'dpass_aom_1_freq'
    DPASS_AOM_1_PHASE_KEY = 'dpass_aom_1_phase'
    DPASS_AOM_1_ATT_KEY = 'dpass_aom_1_att'

    INDV_AOM_0_FREQ_KEY = 'indv_aom_0_freq'
    INDV_AOM_0_PHASE_KEY = 'indv_aom_0_phase'
    INDV_AOM_0_ATT_KEY = 'indv_aom_0_att'

    INDV_AOM_1_FREQ_KEY = 'indv_aom_1_freq'
    INDV_AOM_1_PHASE_KEY = 'indv_aom_1_phase'
    INDV_AOM_1_ATT_KEY = 'indv_aom_1_att'

    INDV_AOM_RESP_TIME_KEY = 'indv_aom_resp_time'
    INDV_AOM_RESP_COMP_KEY = 'indv_aom_resp_comp'

    PID_ENABLE_KEY = 'pid_enable'

    def build(self, dpass_aom_0, dpass_aom_1, indv_aom_0, indv_aom_1, pid_sw_0, pid_sw_1, **kwargs):
        # Double pass AOMs
        self.setattr_device(dpass_aom_0, 'dpass_aom_0')
        self.setattr_device(dpass_aom_1, 'dpass_aom_1')

        # Individual AOMs
        self.setattr_device(indv_aom_0, 'indv_aom_0')
        self.setattr_device(indv_aom_1, 'indv_aom_1')

        # PID switches
        self.setattr_device(pid_sw_0, 'pid_sw_0')
        self.setattr_device(pid_sw_1, 'pid_sw_1')

        # MEMS mirror module
        self.mems_mirror = dax.modules.mems_mirror.MemsMirrorModule(self, 'mems_mirror', **kwargs)

    def load_module(self):
        # For all AOMs: frequency, phase, and attenuation
        self.setattr_dataset_sys(self.DPASS_AOM_0_FREQ_KEY, 100 * MHz)
        self.setattr_dataset_sys(self.DPASS_AOM_0_PHASE_KEY, 0.0)
        self.setattr_dataset_sys(self.DPASS_AOM_0_ATT_KEY, 0.0 * dB)

        self.setattr_dataset_sys(self.DPASS_AOM_1_FREQ_KEY, 100 * MHz)
        self.setattr_dataset_sys(self.DPASS_AOM_1_PHASE_KEY, 0.0)
        self.setattr_dataset_sys(self.DPASS_AOM_1_ATT_KEY, 0.0 * dB)

        self.setattr_dataset_sys(self.INDV_AOM_0_FREQ_KEY, 100 * MHz)
        self.setattr_dataset_sys(self.INDV_AOM_0_PHASE_KEY, 0.0)
        self.setattr_dataset_sys(self.INDV_AOM_0_ATT_KEY, 0.0 * dB)

        self.setattr_dataset_sys(self.INDV_AOM_1_FREQ_KEY, 100 * MHz)
        self.setattr_dataset_sys(self.INDV_AOM_1_PHASE_KEY, 0.0)
        self.setattr_dataset_sys(self.INDV_AOM_1_ATT_KEY, 0.0 * dB)

        # Individual AOM response time and compensation flag
        self.setattr_dataset_sys(self.INDV_AOM_RESP_TIME_KEY, 10 * us)
        self.setattr_dataset_sys(self.INDV_AOM_RESP_COMP_KEY, True)

        # PID enable flag
        self.setattr_dataset_sys(self.PID_ENABLE_KEY, True)

    @kernel
    def init_module(self):
        # Break realtime to get some slack
        self.core.break_realtime()

        # Configure PIDs as output
        self.pid_sw_0.output()
        self.pid_sw_1.output()

        # Initialize all AOMs
        self.dpass_aom_0.init()
        self.dpass_aom_1.init()
        self.indv_aom_0.init()
        self.indv_aom_1.init()

    @kernel
    def config_module(self):
        # Break realtime to get some slack
        self.core.break_realtime()

        # Configure DPASS AOMs and switch on by default
        self.dpass_aom_0.set(self.dpass_aom_0_freq, phase=self.dpass_aom_0_phase)
        self.dpass_aom_0.set_att(self.dpass_aom_0_att)
        self.dpass_aom_0.cfg_sw(True)
        self.dpass_aom_1.set(self.dpass_aom_1_freq, phase=self.dpass_aom_1_phase)
        self.dpass_aom_1.set_att(self.dpass_aom_1_att)
        self.dpass_aom_1.cfg_sw(True)

        # Indv AOMs and PID switches are initially off
        self.indv_aom_0.cfg_sw(False)
        self.indv_aom_1.cfg_sw(False)
        self.pid_sw_0.set_o(False)
        self.pid_sw_0.set_o(False)

        # Set default configuration for indv AOMs
        self.indv_aom_0.set(self.indv_aom_0_freq, phase=self.indv_aom_0_phase)
        self.indv_aom_0.set_att(self.indv_aom_0_att)
        self.indv_aom_1.set(self.indv_aom_1_freq, phase=self.indv_aom_1_phase)
        self.indv_aom_1.set_att(self.indv_aom_1_att)

    @kernel
    def set_o(self, state):
        if state:
            self.on()
        else:
            self.off()

    @kernel
    def on(self):
        if self.indv_aom_resp_comp:
            # Move cursor to compensate for response time
            delay(-self.indv_aom_resp_time)

        # Set switches of indv AOMs
        self.indv_aom_0.cfg_sw(True)
        self.indv_aom_1.cfg_sw(True)

        if self.indv_aom_resp_comp:
            # Move cursor to compensate for response time
            delay(self.indv_aom_resp_time)

        if self.pid_enable:
            # Set PIDs
            self.pid_sw_0.set_o(True)
            self.pid_sw_0.set_o(True)

    @kernel
    def off(self):
        if self.pid_enable:
            # Set PIDs
            self.pid_sw_0.set_o(False)
            self.pid_sw_0.set_o(False)

        if self.indv_aom_resp_comp:
            # Move cursor to compensate for response time
            delay(self.indv_aom_resp_time)

        # Set switches of indv AOMs
        self.indv_aom_0.cfg_sw(False)
        self.indv_aom_1.cfg_sw(False)

        if self.indv_aom_resp_comp:
            # Move cursor to compensate for response time
            delay(-self.indv_aom_resp_time)

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
