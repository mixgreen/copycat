import unittest
import unittest.mock

from artiq.experiment import EnvExperiment, Experiment, HasEnvironment
from artiq.master.worker_impl import Scheduler  # type: ignore

import dax.util.artiq
import dax.util.experiments


class _SubmitBarrierExperiment(EnvExperiment):
    def __init__(self, *args, **kwargs):
        # Call super
        super(_SubmitBarrierExperiment, self).__init__(*args, **kwargs)
        # The mock scheduler
        self.scheduler = unittest.mock.Mock(spec=Scheduler)

    def get_device(self, key):
        if key == 'scheduler':
            # Return the mock scheduler
            return self.scheduler
        else:
            return super(_SubmitBarrierExperiment, self).get_device(key)

    def run(self):
        dax.util.experiments.Barrier.submit(self)


class ExperimentsTestCase(unittest.TestCase):
    def test_build(self):
        experiments = [
            dax.util.experiments.Barrier,
        ]

        for exp in experiments:
            with self.subTest(experiment_cls=exp.__name__):
                self.assertTrue(issubclass(exp, Experiment), 'Experiment class is not a subclass of ARTIQ Experiment')
                self.assertTrue(issubclass(exp, HasEnvironment),
                                'Experiment class is not a subclass of ARTIQ HasEnvironment')
                # Build the experiment
                exp(dax.util.artiq.get_managers())

    def test_run_barrier(self):
        # Create experiment
        exp = dax.util.experiments.Barrier(dax.util.artiq.get_managers())
        # Replace scheduler with mock scheduler
        exp._scheduler = unittest.mock.NonCallableMock(**{'check_pause.return_value': True})

        # Run experiment
        exp.prepare()
        exp.run()
        exp.analyze()

        # Check calls
        self.assertListEqual(exp._scheduler.method_calls, [unittest.mock.call.check_pause()])

    def test_submit_barrier(self):
        # Create experiment
        exp = _SubmitBarrierExperiment(dax.util.artiq.get_managers())
        exp.run()

        # Verify if scheduler was called correctly
        self.assertEqual(exp.scheduler.submit.call_count, 1, 'Scheduler was not called')
