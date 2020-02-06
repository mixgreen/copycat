import artiq.coredevice.ttl

from dax.base import *
import dax.modules.mems_mirror


class IndvBeamMemsModule(DaxModule):
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

    INDV_AOM_RESP_TIME = 'indv_aom_response_time'

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

        # Individual AOM response time
        self.setattr_dataset_sys(self.INDV_AOM_RESP_TIME, 10 * us)

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
        self.off()

        # Set default configuration for indv AOMs
        self.indv_aom_0.set(self.indv_aom_0_freq, phase=self.indv_aom_0_phase)
        self.indv_aom_0.set_att(self.indv_aom_0_att)
        self.indv_aom_1.set(self.indv_aom_1_freq, phase=self.indv_aom_1_phase)
        self.indv_aom_1.set_att(self.indv_aom_1_att)

    @kernel
    def set_o(self, state):
        # Set switches of indv AOMs
        self.indv_aom_0.cfg_sw(state)
        self.indv_aom_1.cfg_sw(state)

        # Set PIDs
        self.pid_sw_0.set_o(state & self.pid_enable)
        self.pid_sw_0.set_o(state & self.pid_enable)

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
