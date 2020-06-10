import numpy as np

import artiq.coredevice.ttl  # type: ignore

from dax.experiment import *

__all__ = ['LedModule']


class LedModule(DaxModule):
    """Module to control user LED's."""

    def build(self, *leds: str, init: bool = False) -> None:  # type: ignore
        # Check arguments
        if 1 > len(leds) > 8:
            raise TypeError("Number of LED's must be in the range [1..8]")
        assert all(isinstance(led, str) for led in leds), 'Provided LED keys must be of type str'
        assert isinstance(init, bool), 'Initialization flag must be of type bool'

        # Store attributes
        self._init_flag = init
        self.logger.debug('Init flag: {}'.format(self._init_flag))

        # LED array
        self.led = [self.get_device(led, artiq.coredevice.ttl.TTLOut) for led in leds]
        self.logger.debug("Number of LED's: {:d}".format(len(self.led)))

        # Store kernel invariants
        self.update_kernel_invariants('_init_flag', 'led')

    def init(self) -> None:
        if self._init_flag:
            # Initialize the LED's if the init flag is set
            self._init()

    @kernel
    def _init(self):  # type: () -> None
        # Reset the core
        self.core.reset()

        # Turn all LED's off
        self.off_all()

        # Wait until event is submitted
        self.core.wait_until_mu(now_mu())

    def post_init(self) -> None:
        pass

    """Module functionality"""

    @kernel
    def set_o(self, o, index: TInt32 = np.int32(0)):
        self.led[index].set_o(o)

    @kernel
    def on(self, index: TInt32 = np.int32(0)):
        self.led[index].on()

    @kernel
    def off(self, index: TInt32 = np.int32(0)):
        self.led[index].off()

    @kernel
    def pulse(self, duration: TFloat, index: TInt32 = np.int32(0)):
        self.led[index].pulse(duration)

    @kernel
    def pulse_mu(self, duration: TInt64, index=np.int32(0)):
        self.led[index].pulse_mu(duration)

    @kernel
    def on_all(self):  # type: () -> None
        """Switch all LED's on."""
        for led in self.led:
            # Number of LED's does not exceed 8, hence they can all be set in parallel
            led.on()

    @kernel
    def off_all(self):  # type: () -> None
        """Switch all LED's off."""
        for led in self.led:
            # Number of LED's does not exceed 8, hence they can all be set in parallel
            led.off()

    @kernel
    def set_code(self, code: TInt32):
        """Visualize the lower bits of the code using the LED's."""
        for led in self.led:
            # Set LED (explicit casting required)
            led.set_o(bool(code & np.int32(0x1)))
            # Shift code
            code >>= np.int32(0x1)
