import logging
import time

import artiq.master.worker_db

from dax.experiment import *
import dax.util.units


@dax_client_factory
class SystemBenchmarkDaxInit(DaxClient, EnvExperiment):
    """DAX system initialization benchmark."""

    def build(self):
        # Arguments
        number_kwargs = {'scale': 1, 'step': 1, 'ndecimals': 0}
        self.setattr_argument('num_samples', NumberValue(5, min=1, **number_kwargs))

    def run(self):
        # Suppress dataset warnings because of multiple initializations
        artiq.master.worker_db.logger.setLevel(logging.WARNING + 1)

        # Total time recorded
        total = 0.0

        for _ in range(self.num_samples):
            # Record start time
            start = time.perf_counter()
            # Run DAX system initialization
            self.dax_init()  # After using the client factory, this object will be a DaxSystem
            # Record time
            stop = time.perf_counter()

            # Add difference to total time
            total += stop - start

        # Store recorded average initialization time
        self.init_time = total / self.num_samples

        # Restore logging level
        artiq.master.worker_db.logger.setLevel(logging.NOTSET)

    def analyze(self):
        # Report result
        sys_id = self.SYS_ID
        init_time = dax.util.units.time_to_str(self.init_time)
        self.logger.info('Average execution time of dax_init() for system "{:s}" is {:s}'.format(sys_id, init_time))
