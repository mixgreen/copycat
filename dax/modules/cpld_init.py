import artiq.coredevice.urukul  # type: ignore
import artiq.coredevice.ad9910  # type: ignore
import artiq.coredevice.ad9912  # type: ignore
import artiq.coredevice.suservo  # type: ignore

from dax.experiment import *
import dax.util.units

__all__ = ['CpldInitModule']


class CpldInitModule(DaxModule):
    """Module to automatically initialize CPLD devices.

    This module searches for CPLD devices and calls :func:`get_att_mu` such that
    the attenuation settings are loaded to the device driver.
    """

    DEVICE_TYPES = (artiq.coredevice.ad9910.AD9910,
                    artiq.coredevice.ad9912.AD9912,
                    artiq.coredevice.suservo.SUServo,)
    """Devices types that use Urukul CPLD."""

    def build(self, *,  # type: ignore
              interval: float = 5 * us, check_registered_devices: bool = True, init_kernel: bool = True) -> None:
        """Build the CPLD initialization module.

        :param interval: Interval/delay between initialization of multiple CPLD devices
        :param check_registered_devices: Enable verification if devices were already registered by an other module
        :param init_kernel: Run initialization kernel during default module initialization
        """
        assert isinstance(interval, float), 'Interval must be a time which has type float'
        assert isinstance(check_registered_devices, bool), 'Check registered devices flag must be of type bool'
        assert isinstance(init_kernel, bool), 'Init kernel flag must be of type bool'

        # Store attributes
        self._interval: float = interval
        self.update_kernel_invariants('_interval')
        self.logger.debug(f'Interval set to {dax.util.units.time_to_str(self._interval)}')
        self._init_kernel: bool = init_kernel
        self.logger.debug(f'Init kernel: {self._init_kernel}')

        if check_registered_devices:
            # Check if no devices have been requested yet
            self.logger.debug('Checking if devices that use CPLD have already been registered')
            devices = self.registry.search_devices(self.DEVICE_TYPES)
            if devices:
                # Warn the user that devices using CPLD already have been registered
                self.logger.warning(f'The following devices that use CPLD have already been registered '
                                    f'before this module was created: {", ".join(devices)}')

        # List of CPLD device keys
        self.keys = [k for k, v in self.registry.device_db.items()
                     if isinstance(v, dict) and v.get('class') == 'CPLD']

        # CPLD array
        self.cpld = [self.get_device(key, artiq.coredevice.urukul.CPLD) for key in self.keys]
        self.update_kernel_invariants('cpld')
        self.logger.debug(f'Number of CPLD devices: {len(self.cpld)}')

        if not self.cpld:
            # Disable CPLD initialization kernel if there are no devices
            self.init_kernel = self._nop  # type: ignore[assignment]
            self.logger.debug('Initialization kernel disabled due to the lack of CPLD devices')

    def init(self, *, force: bool = False) -> None:
        """Initialize this module.

        :param force: Force full initialization
        """
        if (self._init_kernel or force) and self.cpld:
            # Initialize CPLD devices
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

        for c in self.cpld:
            # Load attenuation values to device driver
            c.get_att_mu()
            delay(self._interval)

        # Wait until event is submitted
        self.core.wait_until_mu(now_mu())

    @kernel
    def _nop(self):  # type: () -> None
        """Empty function."""
        pass

    def post_init(self) -> None:
        pass
