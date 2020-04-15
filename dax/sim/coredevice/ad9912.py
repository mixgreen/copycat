from dax.sim.coredevice import *


class AD9912(DaxSimDevice):

    def __init__(self, dmgr, sw_device=None, **kwargs):
        # Call super
        super(AD9912, self).__init__(dmgr, **kwargs)

        # Register variables
        self._signal_manager = get_signal_manager()
        self._init = self._signal_manager.register(self.key, 'init', bool, size=1)
        self._sw = self._signal_manager.register(self.key, 'sw', bool, size=1)
        self._freq = self._signal_manager.register(self.key, 'freq', float)
        self._phase = self._signal_manager.register(self.key, 'phase', float)
        self._att = self._signal_manager.register(self.key, 'att', float)

        if sw_device:
            # Add switch device
            self.sw = dmgr.get(sw_device)

    @kernel
    def init(self):
        self._signal_manager.event(self._init, 1)

    @kernel
    def set(self, frequency, phase=0.0):
        self._signal_manager.event(self._freq, frequency)
        self._signal_manager.event(self._phase, phase)

    @kernel
    def cfg_sw(self, state):
        self._signal_manager.event(self._sw, state)

    @kernel
    def set_att(self, att):
        self._signal_manager.event(self._att, att)
