import numpy as np

from dax.experiment import *
from dax.modules.interfaces.global_beam_if import *
from dax.modules.interfaces.indv_beam_if import *
from dax.modules.interfaces.detection_if import *
from dax.modules.interfaces.trap_if import *


class SamplerService(DaxService):
    """Service that provides generic sampling.

    It is assumed that ions are loaded.
    It is assumed that individual beam infrastructure is aligned.
    """

    SERVICE_NAME = 'sampler'

    PAUSE_TIME_KEY = 'pause_time'

    def build(self):
        # Obtain required modules
        self.gbeam = self.registry.find_module(GlobalBeamInterface)
        self.ibeam = self.registry.find_module(IndvBeamInterface)
        self.detect = self.registry.find_module(DetectionInterface)
        self.trap = self.registry.find_module(TrapInterface)

        # Attribute to store result dataset key
        self.result_dataset_key = ''  # This value could be mutated inside a kernel

    def init(self):
        # Pause time between samples (required to obtain slack)
        self.setattr_dataset_sys(self.PAUSE_TIME_KEY, 5 * us)

    def post_init(self):
        pass

    @portable
    def prepare_sampler(self, key='y'):
        """Prepare the dataset to write results to, can be called more than once."""

        # Prepare an empty list to append to
        self.set_dataset(key, [])
        # Store key
        self.result_dataset_key = key
        # Debug message
        self.logger.debug('Prepared key "{:s}" as result dataset'.format(key))

    @portable
    def finalize_sampler(self, x_data, key='x'):
        """Store x-values in dataset along with one or multiple sets of results."""

        # Store x-axis data
        self.set_dataset(key, x_data)
        # Reset key
        self.result_dataset_key = ''
        # Debug message
        self.logger.debug('Stored x-values with key "{:s}" and finalized sampling'.format(key))

    @kernel
    def sample_active(self, duration, num_samples=1):
        self.sample_active_mu(self.core.seconds_to_mu(duration), num_samples=np.int32(num_samples))

    @kernel
    def sample_active_mu(self, duration, num_samples=1):
        self.sample_targets_mu(duration, targets=self.trap.get_targets(), num_samples=np.int32(num_samples))

    @kernel
    def sample_targets(self, duration, targets, num_samples=1):
        self.sample_targets_mu(self.core.seconds_to_mu(duration), targets=targets, num_samples=np.int32(num_samples))

    @kernel
    def sample_targets_mu(self, duration, targets, num_samples=1):
        assert duration > 0
        assert len(targets) > 0
        assert num_samples > 0

        # Guarantee types
        num_samples = np.int32(num_samples)

        if self.result_dataset_key == '':
            raise RuntimeError('Result dataset was not prepared before using the sampler')

        # Prepare storage for counts
        num_targets = np.int32(len(targets))
        counts = [np.int32(0) for _ in range(num_targets)]

        # Iterate over samples
        for _ in range(num_samples):
            # Pump
            self.trap.pump_default()

            # Turn beams on
            self.ibeam.on_targets(targets)  # Indv beam first for better timeline sequence
            self.gbeam.on()

            # Detect while beams are on
            t = self.detect.detect_targets_mu(duration, targets)  # Moves cursor

            # Turn beams off
            self.gbeam.off()
            self.ibeam.off_targets(targets)  # Indv beam last for better timeline sequence

            # Create some slack by adding a pause time
            delay(self.pause_time)

            # Obtain count results
            r = self.detect.count_targets(t, targets)

            # Sum counts per target
            for i in range(num_targets):
                counts[i] += r[i]

        # Calculate mean counts over all samples
        counts_mean = [float(c / num_targets) for c in counts]

        # Append the array of mean counts to the result dataset
        self.append_to_dataset(self.result_dataset_key, counts_mean)
