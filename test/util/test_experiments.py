import unittest
import unittest.mock
import numpy as np

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
            (dax.util.experiments.Barrier, {}),
            (dax.util.experiments.SetDataset, {'Key': 'key', 'Value': '3'}),
        ]

        for exp, arguments in experiments:
            with self.subTest(experiment_cls=exp.__name__, arguments=arguments):
                self.assertTrue(issubclass(exp, Experiment), 'Experiment class is not a subclass of ARTIQ Experiment')
                self.assertTrue(issubclass(exp, HasEnvironment),
                                'Experiment class is not a subclass of ARTIQ HasEnvironment')

                with dax.util.artiq.get_managers(arguments=arguments) as managers:
                    # Build the experiment
                    exp(managers)

    def test_run_barrier(self):
        with dax.util.artiq.get_managers() as managers:
            # Create experiment
            exp = dax.util.experiments.Barrier(managers)
            # Replace scheduler with mock scheduler
            exp._scheduler = unittest.mock.NonCallableMock(**{'check_pause.return_value': True})

            # Run experiment
            exp.prepare()
            exp.run()
            exp.analyze()

            # Check calls
            self.assertListEqual(exp._scheduler.method_calls, [unittest.mock.call.check_pause()])

    def test_submit_barrier(self):
        with dax.util.artiq.get_managers() as managers:
            # Create experiment
            exp = _SubmitBarrierExperiment(managers)
            exp.run()

            # Verify if scheduler was called correctly
            self.assertEqual(exp.scheduler.submit.call_count, 1, 'Scheduler was not called')

    def test_run_set_dataset(self):
        arguments = [
            {'Key': 'key', 'Value': '3'},
            {'Key': 'key', 'Value': 'np.int32(4)'},
            {'Key': 'key', 'Value': 'np.int64(5)'},
            {'Key': 'key', 'Value': 'float(5)'},
            {'Key': 'key', 'Value': '3.4'},
            {'Key': 'key', 'Value': '"value"'},
            {'Key': 'key', 'Value': '[1, 2, 3, 4]'},
            {'Key': 'key', 'Value': 'list(range(7))'},
            {'Key': 'key', 'Value': 'np.asarray([3, 4], dtype=np.int32)'},
            {'Key': 'key', 'Value': '3 * ms'},
            {'Key': 'key', 'Value': '3 * s'},
            {'Key': 'key', 'Value': '3 * Hz'},
            {'Key': 'key', 'Value': '3 * dB'},
            {'Key': 'key', 'Value': '3 * V'},
            {'Key': 'key', 'Value': '3 * A'},
            {'Key': 'key', 'Value': '3 * W'},
        ]

        for args in arguments:
            with self.subTest(arguments=args):
                with dax.util.artiq.get_managers(arguments=args) as managers:
                    # Create experiment
                    exp = dax.util.experiments.SetDataset(managers)

                    # Run experiment
                    exp.prepare()
                    exp.run()
                    exp.analyze()

                    # Verify value
                    g = {'np': np}
                    g.update(dax.util.experiments.SetDataset._UNITS)
                    ref_value = eval(args['Value'], g, {})
                    value = exp.get_dataset(args['Key'])
                    if isinstance(value, np.ndarray):
                        self.assertTrue(np.array_equal(value, ref_value),
                                        'Obtained dataset does not match written dataset (type: ndarray)')
                    else:
                        self.assertEqual(value, ref_value, 'Obtained dataset does not match written dataset')
