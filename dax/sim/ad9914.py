from artiq.language.core import *
from artiq.language.types import *
from artiq.language.units import *

import dax.sim.time as time

# Taken from ad9914.py
_PHASE_MODE_DEFAULT = -1


class AD9914:

    def __init__(self, dmgr, name):
        self.core = dmgr.get("core")
        self.name = name

        # Register variables
        self._init = time.manager.register(self.name, 'init', 'reg', size=1)
        self._freq = time.manager.register(self.name, 'freq', 'real')
        self._phase = time.manager.register(self.name, 'phase', 'real')
        self._amp = time.manager.register(self.name, 'amp', 'real')

    @kernel
    def init(self):
        time.manager.event(self._init, 1)

    @kernel
    def set(self, frequency, phase=0.0, phase_mode=_PHASE_MODE_DEFAULT, amplitude=1.0):
        time.manager.event(self._freq, frequency)
        time.manager.event(self._phase, phase)
        time.manager.event(self._amp, amplitude)
