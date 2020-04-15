from dax.sim.coredevice import *


class AD9914(DaxSimDevice):

    def __init__(self, dmgr, **kwargs):
        # Call super
        super(AD9914, self).__init__(dmgr, **kwargs)

        # Register variables
        self._signal_manager = get_signal_manager()
        self._init = self._signal_manager.register(self.key, 'init', bool, size=1)
        self._freq = self._signal_manager.register(self.key, 'freq', float)
        self._phase = self._signal_manager.register(self.key, 'phase', float)
        self._amp = self._signal_manager.register(self.key, 'amp', float)

    @kernel
    def init(self):
        self._signal_manager.event(self._init, 1)

    @kernel
    def set(self, frequency, phase=0.0, phase_mode=-1, amplitude=1.0):
        self._signal_manager.event(self._freq, frequency)
        self._signal_manager.event(self._phase, phase)
        self._signal_manager.event(self._amp, amplitude)
