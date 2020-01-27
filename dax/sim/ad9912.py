from artiq.language.core import *
from artiq.language.types import *
from artiq.language.units import *

import dax.sim.time as time


class AD9912:

    def __init__(self, dmgr, name):
        self.core = dmgr.get("core")
        self.name = name

        # Register variables
        self._init = time.manager.register(self.name, 'init', 'reg', size=1)
        self._sw = time.manager.register(self.name, 'sw', 'reg', size=1)
        self._freq = time.manager.register(self.name, 'freq', 'real')
        self._phase = time.manager.register(self.name, 'phase', 'real')
        self._att = time.manager.register(self.name, 'att', 'real')

    @kernel
    def init(self):
        time.manager.event(self._init, 1)

    @kernel
    def set(self, frequency, phase=0.0):
        time.manager.event(self._freq, frequency)
        time.manager.event(self._phase, phase)

    @kernel
    def cfg_sw(self, state):
        time.manager.event(self._sw, state)

    @kernel
    def set_att(self, att):
        time.manager.event(self._att, att)
