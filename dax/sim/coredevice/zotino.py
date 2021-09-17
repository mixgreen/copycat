# mypy: disallow_untyped_defs = False
# mypy: disallow_incomplete_defs = False
# mypy: check_untyped_defs = False

from artiq.language.core import kernel

from dax.sim.coredevice.ad53xx import AD53xx


# noinspection PyAbstractClass
class Zotino(AD53xx):

    def __init__(self, dmgr, **kwargs):
        # Call super
        super(Zotino, self).__init__(dmgr, **kwargs)

        # Register additional signals
        self._led = self._signal_manager.register(self, 'led', bool, size=8)

    def _set_leds(self, leds):
        self._signal_manager.push(self._led, f'{leds & 0xFF:08b}')

    @kernel
    def set_leds(self, leds):
        self._set_leds(leds)
