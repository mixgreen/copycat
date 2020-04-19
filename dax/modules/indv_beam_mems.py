import numpy as np

import artiq.coredevice.ttl

from dax.experiment import *


class _MemsMirrorModule(DaxModule):
    """Module to control a set of MEMS mirrors connected to one board."""

    def build(self, mems_trig, mems_sw, mems_dac):
        # Devices for the MEMS mirror board
        self.setattr_device(mems_trig, 'mems_trig', artiq.coredevice.ttl.TTLOut)
        self.setattr_device(mems_sw, 'mems_sw')
        self.setattr_device(mems_dac, 'mems_dac')

    def init(self):
        # TODO, should store MEMS settings ions (targets)
        pass

    def post_init(self):
        pass


class IndvBeamMemsModule(DaxModule):
    """Module for individual beam path controlled with MEMS mirrors."""

    DPASS_AOM_DEVICE_KEY = '{name:s}_{beam:d}_{signal:d}'
    DPASS_AOM_FREQ_KEY = 'dpass_aom_freq'
    DPASS_AOM_PHASE_KEY = 'dpass_aom_phase'
    DPASS_AOM_ATT_KEY = 'dpass_aom_att'

    INDV_AOM_DEVICE_KEY = '{name:s}_{beam:d}'
    INDV_AOM_FREQ_KEY = 'indv_aom_freq'
    INDV_AOM_PHASE_KEY = 'indv_aom_phase'
    INDV_AOM_ATT_KEY = 'indv_aom_att'

    PID_DEVICE_KEY = '{name:s}_{beam:d}'
    PID_ENABLE_KEY = 'pid_enable'

    INDV_AOM_RESP_TIME_KEY = 'indv_aom_resp_time'
    INDV_AOM_RESP_COMP_KEY = 'indv_aom_resp_comp'

    class _BeamConfig:
        """Class that holds attributes for a beam."""

        # Indicator for no beam
        NO_BEAM = np.int32(-1)

        kernel_invariants = {'NO_BEAM'}

        def __init__(self):
            self.target = self.NO_BEAM  # Initially it is unknown what the target is for a beam
            self.state = False  # Initialization should make sure all beams are initially off

        def is_target(self, target):
            return target == self.target

        def is_available(self):
            return not self.state

    def build(self, dpass_aom, indv_aom, pid_sw, num_beams, num_dpass_signals, **kwargs):
        assert isinstance(dpass_aom, str) and dpass_aom
        assert isinstance(indv_aom, str) and indv_aom
        assert isinstance(pid_sw, str) and pid_sw
        assert isinstance(num_beams, int) and num_beams > 0
        assert isinstance(num_dpass_signals, int) and num_dpass_signals > 0

        # Store attributes
        self.num_beams = np.int32(num_beams)
        self.num_dpass_signals = np.int32(num_dpass_signals)

        # Current beam configurations (should be managed by the module only)
        self._beam_configurations = [self._BeamConfig() for _ in range(num_beams)]

        # Double pass AOMs (self.dpass_aom[beam][signal])
        self.dpass_aom = [[self.get_device(self.DPASS_AOM_DEVICE_KEY.format(name=dpass_aom, beam=b, signal=s))
                           for s in range(num_dpass_signals)] for b in range(num_beams)]
        # Individual AOMs (self.indv_aom[beam])
        self.indv_aom = [self.get_device(self.INDV_AOM_DEVICE_KEY.format(name=indv_aom, beam=b))
                         for b in range(num_beams)]
        # PID switches (self.pid_sw[beam])
        self.pid_sw = [self.get_device(self.PID_DEVICE_KEY.format(name=pid_sw, beam=b)) for b in range(num_beams)]
        # MEMS mirror module
        self.mems_mirror = dax.modules.mems_mirror.MemsMirrorModule(self, 'mems_mirror', **kwargs)

        # Update kernel invariants
        self.update_kernel_invariants('num_beams', 'num_dpass_signals', 'dpass_aom', 'indv_aom', 'pid_sw')

    def init(self):
        # Double-pass AOMs
        self.setattr_dataset_sys(self.DPASS_AOM_FREQ_KEY, [[100 * MHz] * self.num_dpass_signals] * self.num_beams)
        self.setattr_dataset_sys(self.DPASS_AOM_PHASE_KEY, [[0.0] * self.num_dpass_signals] * self.num_beams)
        self.setattr_dataset_sys(self.DPASS_AOM_ATT_KEY, [[0.0 * dB] * self.num_dpass_signals] * self.num_beams)

        # Individual AOMs
        self.setattr_dataset_sys(self.INDV_AOM_FREQ_KEY, [100 * MHz] * self.num_beams)
        self.setattr_dataset_sys(self.INDV_AOM_PHASE_KEY, [0.0] * self.num_beams)
        self.setattr_dataset_sys(self.INDV_AOM_ATT_KEY, [0.0 * dB] * self.num_beams)

        # PIDs
        self.setattr_dataset_sys(self.PID_ENABLE_KEY, [True] * self.num_beams)

        # Individual AOMs response time and compensation flag
        self.setattr_dataset_sys(self.INDV_AOM_RESP_TIME_KEY, 10 * us)
        self.setattr_dataset_sys(self.INDV_AOM_RESP_COMP_KEY, True)

        # Initialize devices
        self._init()

    @kernel
    def _init(self):
        # Reset core
        self.core.reset()

        for b in range(self.num_beams):  # For all beams
            # Configure PID switch as output
            self.pid_sw[b].set_o(False)  # PID by default off

            # Set default configuration for individual AOMs
            self.indv_aom[b].set(self.indv_aom_freq[b], phase=self.indv_aom_phase[b])
            self.indv_aom[b].set_att(self.indv_aom_att[b])
            self.indv_aom[b].cfg_sw(False)  # INDV by default off

            for s in range(self.num_dpass_signals):  # For all signals
                # Configure double-pass AOMs
                self.dpass_aom[b][s].set(self.dpass_aom_freq[b][s], phase=self.dpass_aom_phase[b][s])
                self.dpass_aom[b][s].set_att(self.dpass_aom_att[b][s])
                self.dpass_aom[b][s].cfg_sw(True)  # DPASS by default on

        # Guarantee all events are submitted
        self.core.wait_until_mu(now_mu())

    def post_init(self):
        pass

    @portable
    def _get_beam(self, target, state):
        """Provide a target and future state, beam index will be returned.

        When switching to the on state:
         - Returns NO_BEAM if a beam was already on and pointed at the target
         - Returns a beam index when a beam is available but not targeted
         - Raises when no beam was pointed at the target and no beams are available

        When switching to off state:
         - Returns NO_BEAM if no beam is on and pointed at the target
         - Returns a beam index when a beam is on and pointing at the target
        """

        if state:  # We switch a beam on, which means we need to find a beam
            # Keep track of available beams in case we do not find a beam matching the target
            available_beam = self._BeamConfig.NO_BEAM

            for b in range(self.num_beams):
                # Search for a beam matching the target
                c = self._beam_configurations[b]
                if c.is_target(target):
                    # We found a beam matching the target
                    if c.state == state:
                        # Beam already in position and in the correct state, nothing needs to be done
                        return self._BeamConfig.NO_BEAM
                    else:
                        # We found a beam matching the target with a different state, update state and return beam index
                        c.state = state
                        return np.int32(b)
                elif c.is_available():
                    # We found an available beam, store it
                    available_beam = np.int32(b)

            # We exhausted the search which means that no beam is actively pointed at our target
            if available_beam != self._BeamConfig.NO_BEAM:
                # Save configuration
                self._beam_configurations[available_beam].target = np.int32(target)
                self._beam_configurations[available_beam].state = state
                # Return the index of the available beam
                return available_beam
            else:
                # No beam was available
                raise RuntimeError('Requested more targets than available beams')

        else:  # We switch a beam off, which means we do not have to find a beam
            for b in range(self.num_beams):
                # Search for a beam matching the target
                c = self._beam_configurations[b]
                if c.is_target(target):
                    # We found a beam matching the target
                    if c.state == state:
                        # Beam already in position and in the correct state, nothing needs to be done
                        return self._BeamConfig.NO_BEAM
                    else:
                        # We found a beam matching the target with a different state, update state and return beam index
                        c.state = state
                        return np.int32(b)

            # We exhausted the search which means that no beam is actively pointed at our target
            return self._BeamConfig.NO_BEAM

    @kernel
    def set_targets_o(self, state, targets):
        if state:
            self.on_targets(targets)
        else:
            self.off_targets(targets)

    @kernel
    def on_targets(self, targets):
        # Get the beams corresponding to the targets (or NO_BEAM if no action is required)
        beams = [self._get_beam(t, True) for t in targets]

        # Set AOM first to potentially minimize the number of timeline alterations
        if self.indv_aom_resp_comp:
            # Move cursor to compensate for response time
            delay(-self.indv_aom_resp_time)

        for b in beams:
            if b != self._BeamConfig.NO_BEAM:
                # Set switch of indv AOM
                self.indv_aom[b].cfg_sw(True)
                # TODO, configure MEMS to point to correct target

        if self.indv_aom_resp_comp:
            # Move cursor to undo compensation for response time
            delay(self.indv_aom_resp_time)

        if self.pid_enable:
            for b in beams:
                if b != self._BeamConfig.NO_BEAM:
                    # Set PID
                    self.pid_sw[b].set_o(True)

    @kernel
    def off_targets(self, targets):
        # Get the beams corresponding to the targets (or NO_BEAM if no action is required)
        beams = [self._get_beam(t, False) for t in targets]

        # Set PID first to potentially minimize the number of timeline alterations
        if self.pid_enable:
            for b in beams:
                if b != self._BeamConfig.NO_BEAM:
                    # Set PID
                    self.pid_sw[b].set_o(False)

        if self.indv_aom_resp_comp:
            # Move cursor to compensate for response time
            delay(self.indv_aom_resp_time)

        for b in beams:
            if b != self._BeamConfig.NO_BEAM:
                # Set switch of indv AOM
                self.indv_aom[b].cfg_sw(False)
                # TODO, configure MEMS to point to correct target

        if self.indv_aom_resp_comp:
            # Move cursor to undo compensation for response time
            delay(-self.indv_aom_resp_time)

    @kernel
    def pulse_targets(self, duration, targets):
        self.on_targets(targets)
        delay(duration)
        self.off_targets(targets)

    @kernel
    def pulse_targets_mu(self, duration, targets):
        self.on_targets(targets)
        delay_mu(duration)
        self.off_targets(targets)
