import numpy as np
import itertools
import typing
import time

from dax.experiment import *
from dax.util.units import freq_to_str, time_to_str

import artiq.coredevice.ad9910

__all__ = ['TrapRfModule']


class TrapRfModule(DaxModule):
    """A trap RF module using an AD9910.

    This module controls an AD9910 used for trap RF. It can safely ramp the power of the DDS to not damage the trap.

    Notes when considering using this module:

    - It might not be desired to use a DDS controlled by ARTIQ. If the core device needs to be power cycled, the RF
      signal will be interrupted.
    - Any code outside this DAX module might control the DDS in an unsafe manner. This includes the startup kernel and
      the idle kernel.
    - Code outside this DAX module could modify system datasets or (class) attributes in an unsafe manner.
    """

    _RESONANCE_FREQ_KEY = 'resonance_freq'
    """Key for the resonance frequency, used to calculate the allowed frequency range."""
    _RESONANCE_DIFF_KEY = 'resonance_diff'
    """Key for the allowed frequency difference from the resonance frequency (both positive and negative)."""
    _RAMP_SLOPE_KEY = 'ramp_slope'
    """Key for the slope of the amplitude ramp."""
    _RAMP_COMPRESSION_KEY = 'ramp_compression'
    """Key for the ramp compression flag."""
    _MAX_AMP_KEY = 'max_amp'
    """Key for the maximum amplitude in dB."""

    _ENABLED_KEY = 'enabled'
    """Key to store the enabled flag."""
    _LAST_AMP_KEY = 'last_amp'
    """Key to store the last amplitude."""
    _LAST_FREQ_KEY = 'last_freq'
    """Key to store the last frequency."""
    _CACHE_LAST_ASF_KEY = 'last_asf'
    """Core device cache key to store the last ASF."""

    _MAX_AMP: float = 0.0 * dB
    """Constant for maximum amplitude."""
    _MIN_AMP: float = -91.0 * dB
    """Constant for minimum amplitude in the dynamic range of the device."""
    _ATT_MU: int = 0xff
    """Constant attenuation value in machine units (att is fixed such that we can do conversion between dB and dBm)."""
    _RAMP_STEP_PERIOD: float = 1.0 * ms
    """Ramp step period (i.e. delay between two ramp steps)."""

    _default_resonance_freq: float
    _default_resonance_diff: float
    _default_ramp_slope: float
    _ramp_cutoff: float
    _trap_rf: artiq.coredevice.ad9910.AD9910
    _system: DaxSystem
    _resonance_freq: float
    _resonance_diff: float
    _ramp_slope: float
    _ramp_compression: bool
    _max_amp: float

    def build(self, *, key: str,
              default_resonance_freq: float,
              default_resonance_diff: float,
              default_ramp_slope: float = 1.0 * dB,
              ramp_cutoff: float = -30.0 * dB) -> None:
        """Build the trap RF module.

        :param key: The key of the DDS device
        :param default_resonance_freq: Default resonance frequency, used to calculate the allowed frequency range
        :param default_resonance_diff: Default allowed resonance frequency difference (both positive and negative)
        :param default_ramp_slope: Default ramp slope in dBm/second
        :param ramp_cutoff: Ramp cutoff: below this value we can switch amplitude freely
        """
        assert isinstance(default_resonance_freq, float)
        assert default_resonance_freq > 0 * Hz
        assert isinstance(default_resonance_diff, float)
        assert default_resonance_diff > 0 * Hz
        assert isinstance(default_ramp_slope, float)
        assert default_ramp_slope > 0 * Hz
        assert isinstance(ramp_cutoff, float)
        assert ramp_cutoff <= 0.0 * dB, 'Ramp cutoff can not be greater than 0 dB'

        # Check class variables
        assert self._MAX_AMP <= 0.0 * dB, 'Maximum amplitude can not be more than 0 dB'
        assert self._MIN_AMP < self._MAX_AMP, 'Minimum amplitude must be less than the maximum amplitude'
        assert 0x00 <= self._ATT_MU <= 0xff, 'Invalid attenuation value'
        assert self._RAMP_STEP_PERIOD >= 1.0 * ms, 'Ramp step period can not be too small (it provides slack)'
        self.update_kernel_invariants('_CACHE_LAST_ASF_KEY', '_ATT_MU', '_RAMP_STEP_PERIOD')

        # Store attributes
        self._default_resonance_freq = default_resonance_freq
        self._default_resonance_diff = default_resonance_diff
        self._default_ramp_slope = default_ramp_slope
        self._ramp_cutoff = ramp_cutoff

        # Trap RF device
        self._trap_rf = self.get_device(key, artiq.coredevice.ad9910.AD9910)
        self.update_kernel_invariants('_trap_rf')

    @host_only
    def init(self) -> None:
        """Initialize this module."""
        # Get a reference to the system
        self._system = self.registry.find_module(DaxSystem)

        # System datasets
        self._resonance_freq = self.get_dataset_sys(self._RESONANCE_FREQ_KEY, self._default_resonance_freq)
        self._resonance_diff = self.get_dataset_sys(self._RESONANCE_DIFF_KEY, self._default_resonance_diff)
        self._ramp_slope = self.get_dataset_sys(self._RAMP_SLOPE_KEY, self._default_ramp_slope)
        self._ramp_compression = self.get_dataset_sys(self._RAMP_COMPRESSION_KEY, False)
        self._max_amp = self.get_dataset_sys(self._MAX_AMP_KEY, self._MAX_AMP)

    @host_only
    def post_init(self) -> None:
        pass

    """Module functionality"""

    @host_only
    def ramp(self, amp: float, freq: typing.Optional[float] = None, enabled: bool = True) -> None:
        """Set the trap RF frequency and slowly ramp the amplitude to the given value.

        Note that the amplitude is given in **dbm**.

        :param amp: Target amplitude in dBm (ignored if the enabled flag is `False`)
        :param freq: New frequency (last frequency if not given)
        :param enabled: Enable trap RF
        :raises KeyError: Raised if no frequency was given and no previous value was available
        """
        assert isinstance(amp, float), 'Amplitude must be of type float'
        assert isinstance(freq, float) or freq is None, 'Frequency must be of type float or None'
        assert isinstance(enabled, bool), 'Enabled flag must be a bool'

        if freq is None:
            # Default to the last stored frequency
            freq = self.get_freq()
        assert isinstance(freq, float)

        # Verify if the frequency is in the allowed range
        freq_min: float = self._resonance_freq - self._resonance_diff
        freq_max: float = self._resonance_freq + self._resonance_diff
        assert freq_min <= freq_max
        if not freq_min <= freq <= freq_max:
            raise ValueError(f'Frequency {freq_to_str(freq)} is outside of the allowed trap RF frequency range')

        if enabled:
            # Convert dBm input amplitude to dB
            amp = self.dbm_to_db(amp)
        else:
            # Set the minimum amp when disabling the trap RF
            amp = self._MIN_AMP

        # Verify if the amplitude is reasonable
        if not amp <= self._MAX_AMP:
            raise ValueError(f'Amplitude {amp} dB is outside of the allowed trap RF amplitude range')
        if amp > self._max_amp:
            raise ValueError(f'Amplitude {amp} dB is above the trap protection limit')
        if amp < self._MIN_AMP:
            self.logger.warning(f'Amplitude {amp} dB is lower than the dynamic range of the amplitude')
        self.logger.debug(f'Ramp trap RF, target amplitude: {amp} dB')

        # Get last ASF from cache
        last_asf = self.core_cache.get(self._CACHE_LAST_ASF_KEY)

        # Check if the system was just booted
        boot: bool = len(last_asf) == 0
        self.logger.debug(f'Ramp trap RF, system boot detected: {boot}')

        # Obtain last amplitude (assume lowest amplitude if none was available)
        last_amp: float = self.get_dataset_sys(self._LAST_AMP_KEY, self._MIN_AMP)
        assert isinstance(last_amp, float), 'Expected type float for last amplitude'
        assert last_amp <= self._MAX_AMP, 'Amplitude can not be greater than the maximum valid amplitude'
        self.logger.debug(f'Ramp trap RF, last amplitude: {last_amp:f} dB')

        # Decide initial amplitude
        init_amp: float = self._MIN_AMP if boot else last_amp
        assert init_amp <= self._MAX_AMP, 'Amplitude can not be greater than the maximum valid amplitude'
        self.logger.debug(f'Ramp trap RF, initial amplitude: {init_amp:f} dB')

        # Decide the number of steps we need to get from the initial to the target amp
        num_steps: int = int(abs(amp - init_amp) / self._ramp_slope / self._RAMP_STEP_PERIOD)
        num_steps = max(num_steps, 2)  # Submit at least two steps
        assert num_steps >= 2, 'Number of steps must be at least 2'

        # Create the steps in dB
        amp_list = np.linspace(init_amp, amp, num=num_steps, endpoint=True, dtype=float)

        # Apply the cutoff (ramp not required below cutoff)
        amp_filter = amp_list >= self._ramp_cutoff
        amp_filter[-1] = True  # The last value (target amp) should always remain
        amp_list = amp_list[amp_filter]
        assert len(amp_list) > 0, 'Length of the amp list can not be zero'
        assert amp_list[-1] == amp, 'The last value in the amp list was not the target amplitude'
        assert max(amp_list) <= self._MAX_AMP, 'Amplitude can not be greater than the maximum valid amplitude'
        self.logger.debug(f'Ramp trap RF, generating ramp from {amp_list[0]:f} to {amp_list[-1]:f} dB')

        # Transform values to DDS amplitude scale
        amp_scale_list = 10 ** (amp_list / 20)
        assert max(amp_scale_list) <= 1.0, 'Amplitude scale can not be greater than 1.0'
        assert min(amp_scale_list) >= 0.0, 'Amplitude scale can not be less than 0.0'

        # Transform amplitudes to machine units (ASF)
        asf_list = np.vectorize(self._trap_rf.amplitude_to_asf)(amp_scale_list)
        if self._ramp_compression:
            # Compress the list of values for efficiency, potentially speeds up ramping by removing values
            asf_list = np.asarray([k for k, _ in itertools.groupby(asf_list)], dtype=np.int32)
            assert len(asf_list) > 0, 'Length of the asf list can not be zero'
        if init_amp >= self._ramp_cutoff and asf_list[0] != last_asf:
            raise RuntimeError('First ASF of ramp does not match last cached value, '
                               'use safety ramp down to return to a consistent state')

        # Call the kernel function
        self.logger.debug(f'Ramp trap RF, created ramp with {len(asf_list)} step(s)')
        self.logger.info(f'Ramping trap RF amplitude ({time_to_str(len(asf_list) * self._RAMP_STEP_PERIOD)})...')
        self._ramp(self._trap_rf.frequency_to_ftw(freq), asf_list, boot, enabled)
        self.logger.info('Trap RF amplitude ramping done')

        # Store latest values
        self.set_dataset_sys(self._ENABLED_KEY, enabled)
        self.set_dataset_sys(self._LAST_AMP_KEY, amp)
        self.set_dataset_sys(self._LAST_FREQ_KEY, freq)

    @kernel
    def _ramp(self, ftw: TInt32, asf_list: TArray(TInt32), boot: TBool, enabled: TBool):
        # Reset core
        self.core.reset()

        # Make sure CPLD attenuation was loaded
        self._trap_rf.cpld.get_att_mu()
        self.core.break_realtime()

        if boot:
            # Set amplitude
            delay(1 * ms)
            self._trap_rf.set_mu(ftw, asf=0)
            # Set attenuation to desired value
            self._trap_rf.set_att_mu(self._ATT_MU)
            # Store minimum amplitude in cache
            self.core_cache.put(self._CACHE_LAST_ASF_KEY, [0])

        # Extract cached value
        last_asf = self.core_cache.get(self._CACHE_LAST_ASF_KEY)
        if len(last_asf) == 0:
            raise IndexError('Trap RF last ASF cache line was empty for unknown reason, aborting for safety')

        # Guarantee slack after cache operation
        self.core.break_realtime()

        if enabled:
            # Switching on BEFORE ramping
            self._trap_rf.sw.on()

        for asf in asf_list:
            # Set amplitude (in steps) and update value in cache
            delay(self._RAMP_STEP_PERIOD)
            self._trap_rf.set_mu(ftw, asf=asf)
            last_asf[0] = asf

        if not enabled:
            # Switching off AFTER ramping
            delay(self._RAMP_STEP_PERIOD)
            self._trap_rf.sw.off()

        # Guarantee events are submitted
        self.core.wait_until_mu(now_mu())

    @host_only
    def re_ramp(self, pause: float = 0.0) -> None:
        """Slowly ramp the trap RF amplitude down before ramping back up again to the last amplitude.

        :param pause: Pause time between ramp down and ramp up in seconds
        :raises KeyError: Raised if no previous value for frequency or amplitude is available
        """
        assert isinstance(pause, float) and pause >= 0.0, 'Pause must be of type float and greater or equal to zero'

        # Store last amplitude in dBm
        amp = self.get_amp()
        # Ramp down to minimum amplitude
        self.ramp(self.db_to_dbm(self._MIN_AMP), enabled=False)

        if pause > 0.0:
            # Pause for the given duration
            self.logger.debug(f'Pausing for {pause} seconds')
            time.sleep(pause)

        # Ramp back up to the last amplitude and frequency
        self.ramp(amp)

    @host_only
    def safety_ramp_down(self):
        """Ramp down trap RF based on last cached amplitude, **should only be used in case of problems**.

        This function has limited checks and configuration to allow ramping down in a wide set of scenarios.
        """

        # Get last ASF from cache
        last_asf = self.core_cache.get(self._CACHE_LAST_ASF_KEY)

        # Check if the system was just booted
        boot: bool = len(last_asf) == 0
        self.logger.debug(f'Safety ramp down trap RF, system boot detected: {boot}')

        if boot:
            # If the device was just booted, we can not perform a safety ramp down
            self.logger.warning('Can not perform safety ramp down, no cached amplitude found')
            return

        try:
            # Default to the last stored frequency
            freq: float = self.get_freq()
            self.logger.info(f'Safety ramp down trap RF, using last frequency: {freq_to_str(freq)}')
        except KeyError:
            # Fall back on the resonance frequency
            freq = self._resonance_freq
            self.logger.info(f'Safety ramp down trap RF, fallback on resonance frequency: {freq_to_str(freq)}')
        assert isinstance(freq, float)

        # Load constants
        enabled: bool = False
        amp: float = self._MIN_AMP
        amp_scale: float = 10 ** (amp / 20)
        asf: int = self._trap_rf.amplitude_to_asf(amp_scale)
        assert asf == 0

        # Decide the number of steps we need
        num_steps: int = (last_asf[0] - asf) + 1
        num_steps = max(num_steps, 2)  # We can not have less than two steps
        assert num_steps >= 2, 'Number of steps must be at least 2'

        # Create ASF list
        asf_list = np.linspace(last_asf[0], asf, num=num_steps, endpoint=True, dtype=np.int32)
        assert min(asf_list) == 0
        assert max(asf_list) < 2 ** 14
        assert len(asf_list) <= 2 ** 14

        # Call the kernel function
        self.logger.debug(f'Safety ramp down trap RF, created ramp with {len(asf_list)} step(s)')
        self.logger.info(f'Safety ramping trap RF amplitude ({time_to_str(len(asf_list) * self._RAMP_STEP_PERIOD)})...')
        self._ramp(self._trap_rf.frequency_to_ftw(freq), asf_list, boot, enabled)
        self.logger.info('Trap RF amplitude safety ramping done')

        # Store latest values
        self.set_dataset_sys(self._ENABLED_KEY, enabled)
        self.set_dataset_sys(self._LAST_AMP_KEY, amp)
        self.set_dataset_sys(self._LAST_FREQ_KEY, freq)

    @host_only
    def dbm_to_db(self, amplitude: float) -> float:
        """Convert trap RF amplitude in dBm to the device native amplitude in dB.

        Note: this conversion is EXPLICITLY tied to the attenuation and the specific output device!

        :param amplitude: The amplitude in dBm
        """
        assert isinstance(amplitude, float), 'Amplitude must be of type float'
        # Convert value
        amplitude -= 10.0
        # Basic value checks which do not rely on system datasets
        assert amplitude <= self._MAX_AMP, 'Amplitude can not be greater than the maximum valid amplitude'
        # Return value
        return amplitude

    @host_only
    def db_to_dbm(self, amplitude: float) -> float:
        """Convert the device native amplitude in dB to trap RF amplitude in dBm.

        Note: this conversion is EXPLICITLY tied to the attenuation and the specific output device!

        :param amplitude: The amplitude in dB
        """
        assert isinstance(amplitude, float), 'Amplitude must be of type float'
        # Basic value checks which do not rely on system datasets
        assert amplitude <= self._MAX_AMP, 'Amplitude can not be greater than the maximum valid amplitude'
        # Convert value
        amplitude += 10.0
        # Return value
        return amplitude

    """Helper functions"""

    @host_only
    def get_amp(self) -> float:
        """Get the last registered trap RF amplitude in dBm.

        The key for the last trap RF amplitude is private for safety reasons.
        This function can be used instead, even at build() time.

        Note: This value represents the last registered value and there is no guarantee that
        this is the actual current value at the device level.

        :return: Last trap RF amplitude in dBm
        :raises KeyError: Raised if no previous value was available
        """
        return self.db_to_dbm(self.get_dataset_sys(self._LAST_AMP_KEY))

    @host_only
    def get_freq(self) -> float:
        """Get the last registered trap RF frequency.

        The key for the last trap RF frequency is private for safety reasons.
        This function can be used instead, even at build() time.

        Note: This value represents the last registered value and there is no guarantee that
        this is the actual current value at the device level.

        :return: Last trap RF frequency
        :raises KeyError: Raised if no previous value was available
        """
        return self.get_dataset_sys(self._LAST_FREQ_KEY)

    @host_only
    def is_enabled(self) -> bool:
        """Return the trap RF enabled flag.

        The key for the trap RF enabled flag is private for safety reasons.
        This function checks if the device was just rebooted and if the trap RF enabled flag was set.
        This function does not check if the trap RF amplitude meets some minimum value.

        Note: This function is safe to use at any time, even after a reboot.

        :return: True if trap RF is enabled
        :raises KeyError: Raised if no previous value was available, which means the state is ambiguous
        """
        if not self._system.dax_sim_enabled:
            # Check if the system was just booted
            last_asf = self.core_cache.get(self._CACHE_LAST_ASF_KEY)
            if len(last_asf) == 0:
                # Device was just booted, trap RF is off
                return False

        # Return the enabled flag stored as a system dataset
        # Can raise a KeyError if the key was not set before, which means the state is ambiguous
        return self.get_dataset_sys(self._ENABLED_KEY)

    @host_only
    def update_resonance_freq(self, freq: float) -> None:
        """Update resonance frequency.

        :param freq: Resonance frequency in Hz
        """
        assert isinstance(freq, float)
        assert freq > 0.0 * Hz
        self.set_dataset_sys(self._RESONANCE_FREQ_KEY, freq)

    @host_only
    def update_resonance_diff(self, diff: float) -> None:
        """Update resonance diff.

        :param diff: Resonance frequency diff in Hz
        """
        assert isinstance(diff, float)
        assert diff > 0.0 * Hz
        self.set_dataset_sys(self._RESONANCE_DIFF_KEY, diff)

    @host_only
    def update_max_amp(self, amp: float) -> None:
        """Update maximum amplitude.

        :param amp: Maximum trap RF power for device protection in dBm
        """
        assert isinstance(amp, float)
        amp = self.dbm_to_db(amp)
        assert self._MIN_AMP <= amp <= self._MAX_AMP
        self.set_dataset_sys(self._MAX_AMP_KEY, amp)

    @host_only
    def update_ramp_slope(self, slope: float) -> None:
        """Update ramp slope.

        :param slope: Ramp slope in dB
        """
        assert isinstance(slope, float)
        assert slope > 0.0 * dB
        self.set_dataset_sys(self._RAMP_SLOPE_KEY, slope)

    @host_only
    def update_ramp_compression(self, compression: bool) -> None:
        """Update ramp compression.

        Ramp compression, enable for more efficient ramps by removing duplicate ramp values (can increase ramp speed).

        :param compression: Flag to enable or disable ramp compression
        """
        assert isinstance(compression, bool)
        self.set_dataset_sys(self._RAMP_COMPRESSION_KEY, compression)
