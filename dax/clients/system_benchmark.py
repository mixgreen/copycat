import logging
import time

import artiq.master.worker_db

from dax.experiment import *
import dax.util.units

__all__ = ['SystemBenchmarkDaxInit']


@dax_client_factory
class SystemBenchmarkDaxInit(DaxClient, EnvExperiment):
    """DAX system initialization benchmark."""

    def build(self):
        # Arguments
        number_kwargs = {'scale': 1, 'step': 1, 'ndecimals': 0}
        self.setattr_argument('num_samples', NumberValue(5, min=1, **number_kwargs))

    def prepare(self):
        # Get the system (which is actually self after using the client factory)
        self.system = self.registry.find_module(DaxSystem)  # Not possible in build() because the system is `self`

    def run(self):
        # Store input values in dataset
        self.set_dataset('num_samples', self.num_samples)
        # Prepare result dataset
        result_key = 'dax_init_time'
        self.set_dataset(result_key, [])

        # Total time recorded
        total = 0.0

        # Suppress dataset warnings because of multiple initializations
        artiq.master.worker_db.logger.setLevel(logging.WARNING + 1)

        for _ in range(self.num_samples):
            # Create a new experiment class which is an instance of the type of self
            exp = type(self)(self)

            # Record start time
            start = time.perf_counter()
            # Run DAX system initialization
            exp.dax_init()
            # Record time
            stop = time.perf_counter()

            # Add difference to total time
            diff = stop - start
            total += diff

            # Store intermediate result
            self.append_to_dataset(result_key, diff)

        # Restore logging level
        artiq.master.worker_db.logger.setLevel(logging.NOTSET)

        # Store recorded average initialization time
        init_time = total / self.num_samples
        self.system.set_dataset_sys(self.system.DAX_INIT_TIME_KEY, init_time)

    def analyze(self):
        # Report result
        init_time = dax.util.units.time_to_str(self.system.get_dataset_sys(self.system.DAX_INIT_TIME_KEY))
        self.logger.info('Average execution time of dax_init() is {:s}'.format(init_time))
