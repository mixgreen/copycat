import artiq.coredevice.urukul  # type: ignore
import artiq.coredevice.ad9910  # type: ignore
import artiq.coredevice.ad9912  # type: ignore
import artiq.coredevice.suservo  # type: ignore

from dax.experiment import *
import dax.util.units

__all__ = ['CpldInitModule']


class CpldInitModule(DaxModule):
    """Module to automatically initialize CPLD devices.

    This module searches for CPLD devices and calls `get_att_mu()` such that
    the attenuation settings are loaded to the device driver.
    """

    DEVICE_TYPES = (artiq.coredevice.ad9910.AD9910,
                    artiq.coredevice.ad9912.AD9912,
                    artiq.coredevice.suservo.SUServo,)
    """Devices types that use Urukul CPLD."""

    def build(self, interval: float = 5 * us, check_registered_devices: bool = True) -> None:  # type: ignore
        """Build the CPLD initialization module.

        :param interval: Interval/delay between initialization of multiple CPLD devices
        :param check_registered_devices: Enable verification if devices were already registered by an other module
        """
        assert isinstance(interval, float), 'Interval must be a time which has type float'
        assert isinstance(check_registered_devices, bool), 'Check registered devices flag must be of type bool'

        # Store interval
        self._interval: float = interval
        self.logger.debug(f'Interval set to {dax.util.units.time_to_str(self._interval):s}')

        if check_registered_devices:
            # Check if no devices have been requested yet
            self.logger.debug('Checking if devices that use CPLD have already been registered')
            devices = self.registry.search_devices(self.DEVICE_TYPES)
            if devices:
                # Warn the user that devices using CPLD already have been registered
                self.logger.warning(f'The following devices that use CPLD have already been registered '
                                    f'before this module was created: {", ".join(devices):s}')

        # List of CPLD device keys
        cpld_device_keys = [k for k, v in self.get_device_db().items()
                            if isinstance(v, dict) and v.get('class') == 'CPLD']

        # CPLD array
        self._cpld = [self.get_device(key, artiq.coredevice.urukul.CPLD) for key in cpld_device_keys]
        self.logger.debug(f'Number of CPLD devices: {len(self._cpld):d}')

        # Store kernel invariants
        self.update_kernel_invariants('_interval', '_cpld')

    def init(self) -> None:
        if self._cpld:
            # Initialize CPLD devices
            self._init()

    @kernel
    def _init(self):  # type: () -> None
        # Reset the core
        self.core.reset()

        for c in self._cpld:
            # Load attenuation values to device driver
            c.get_att_mu()
            delay(self._interval)

        # Wait until event is submitted
        self.core.wait_until_mu(now_mu())

    def post_init(self) -> None:
        pass
