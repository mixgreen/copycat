# mypy: disallow_untyped_defs = False
# mypy: disallow_incomplete_defs = False
# mypy: check_untyped_defs = False

from artiq.language.core import kernel

from dax.sim.coredevice.ad53xx import AD53xx


class Zotino(AD53xx):

    def __init__(self, dmgr, **kwargs):
        # Call super
        super(Zotino, self).__init__(dmgr, **kwargs)

        # Register additional signals
        self._led = self._signal_manager.register(self, f'led', bool, size=8)

    @kernel
    def set_leds(self, leds):
        self._signal_manager.event(self._led, f'{leds & 0xFF:08b}')
