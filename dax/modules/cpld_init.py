import artiq.coredevice.urukul
import artiq.coredevice.ad9910
import artiq.coredevice.ad9912
import artiq.coredevice.suservo

from dax.experiment import *
import dax.util.units


class CpldInitModule(DaxModule):
    """Module to automatically initialize CPLD devices.

    This module searches for CPLD devices and calls `get_att_mu()` such that
    the attenuation settings are loaded to the device driver.
    """

    # Devices types that use Urukul CPLD
    DEVICE_TYPES = (artiq.coredevice.ad9910.AD9910,
                    artiq.coredevice.ad9912.AD9912,
                    artiq.coredevice.suservo.SUServo)

    def build(self, interval=5 * us, check_registered_devices=True):
        assert isinstance(interval, float), 'Interval must be a time which has type float'
        assert isinstance(check_registered_devices, bool), 'Check registered devices flag must be of type bool'

        # Store interval
        # TODO: default interval time is not optimized
        self._interval = interval
        self.logger.debug('Interval set to {:s}'.format(dax.util.units.time_to_str(self._interval)))

        if check_registered_devices:
            # Check if no devices have been requested yet
            self.logger.debug('Checking if devices that use CPLD have already been registered')
            devices = self.registry.search_devices(self.DEVICE_TYPES)
            if devices:
                # Warn the user that devices using CPLD already have been registered
                self.logger.warning('The following devices that use CPLD have already been registered '
                                    'before this module was created: {:s}'.format(', '.join(devices)))

        # List of CPLD device keys
        cpld_device_keys = [k for k, v in self.get_device_db().items()
                            if isinstance(v, dict) and v.get('class') == 'CPLD']

        # CPLD array
        self._cpld = [self.get_device(key, artiq.coredevice.urukul.CPLD) for key in cpld_device_keys]
        self.logger.debug('Number of CPLD devices: {:d}'.format(len(self._cpld)))

        # Store kernel invariants
        self.update_kernel_invariants('_interval', '_cpld')

    def init(self):
        if self._cpld:
            # Initialize CPLD devices
            self._init()

    @kernel
    def _init(self):
        # Reset the core
        self.core.reset()

        for c in self._cpld:
            # Load attenuation values to device driver
            c.get_att_mu()
            delay(self._interval)

        # Wait until event is submitted
        self.core.wait_until_mu(now_mu())

    def post_init(self):
        pass
