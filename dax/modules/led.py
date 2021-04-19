import numpy as np

import artiq.coredevice.ttl  # type: ignore

from dax.experiment import *

__all__ = ['LedModule']


class LedModule(DaxModule):
    """Module to control user LED's."""

    _init_kernel: bool

    def build(self, *leds: str, init_kernel: bool = False) -> None:  # type: ignore
        """Build the LED module.

        :param leds: Keys of the LED devices to use in order from least to most significant
        :param init_kernel: Run initialization kernel during default module initialization
        """
        # Check arguments
        if not 1 <= len(leds) <= 8:
            raise TypeError("Number of LED's must be in the range [1..8]")
        assert all(isinstance(led, str) for led in leds), 'Provided LED keys must be of type str'
        assert isinstance(init_kernel, bool), 'Init kernel flag must be of type bool'

        # Store attributes
        self._init_kernel = init_kernel
        self.logger.debug(f'Init kernel: {self._init_kernel}')

        # LED array
        self.led = [self.get_device(led, artiq.coredevice.ttl.TTLOut) for led in leds]
        self.update_kernel_invariants('led')
        self.logger.debug(f"Number of LED's: {len(self.led)}")

    def init(self, *, force: bool = False) -> None:
        """Initialize this module.

        :param force: Force full initialization
        """
        if self._init_kernel or force:
            # Initialize the LED's if the init flag is set
            self.logger.debug('Running initialization kernel')
            self.init_kernel()

    @kernel
    def init_kernel(self):  # type: () -> None
        """Kernel function to initialize this module.

        This function is called automatically during initialization unless the user configured otherwise.
        In that case, this function has to be called manually.
        """
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
    def set_o(self, o: TBool, index: TInt32 = 0):
        self.led[index].set_o(o)

    @kernel
    def on(self, index: TInt32 = 0):
        self.led[index].on()

    @kernel
    def off(self, index: TInt32 = 0):
        self.led[index].off()

    @kernel
    def pulse(self, duration: TFloat, index: TInt32 = 0):
        self.led[index].pulse(duration)

    @kernel
    def pulse_mu(self, duration: TInt64, index: TInt32 = 0):
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
