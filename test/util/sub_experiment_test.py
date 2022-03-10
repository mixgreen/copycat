import unittest
import pathlib

from artiq.experiment import *

from dax.util.output import temp_dir
from dax.util.artiq import get_managers
from dax.util.sub_experiment import SubExperiment


class _TestEnv(HasEnvironment):
    pass


class _TestExperiment(EnvExperiment):
    def build(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

        self.prepare_count = 0
        self.run_count = 0
        self.analyze_count = 0

    def prepare(self):
        self.prepare_count += 1

    def run(self):
        self.run_count += 1

    def analyze(self):
        self.analyze_count += 1


class _TestExperimentWithArguments(_TestExperiment):
    def build(self, *args, **kwargs):
        super(_TestExperimentWithArguments, self).build(*args, **kwargs)

        self.foo = self.get_argument('foo', NumberValue())
        self.bar = self.get_argument('bar', StringValue())


class SubExperimentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.managers = get_managers()
        self.env = _TestEnv(self.managers)

    def test_sub_experiment(self, *, class_=_TestExperiment, arguments=None, iterations=1):
        build_args = (1, 'b', 3.0)
        build_kwargs = {
            'build': 20,
            'kwargs': 'c',
        }

        with temp_dir() as path:
            temp_path = pathlib.Path(path)
            self.assertEqual(len(list(temp_path.glob('**/*.h5'))), 0)

            for i in range(iterations):
                sub_experiment = SubExperiment(self.env, self.managers)
                result = sub_experiment.run(class_, 'foo',
                                            arguments=arguments, build_args=build_args, build_kwargs=build_kwargs)
                self.assertIsInstance(result, class_)

                if isinstance(arguments, dict):
                    for k, v in arguments.items():
                        self.assertEqual(getattr(result, k), v)
                self.assertTupleEqual(result.args, build_args)
                self.assertDictEqual(result.kwargs, build_kwargs)

                for count in [result.prepare_count, result.run_count, result.analyze_count]:
                    self.assertEqual(count, 1)

                self.assertEqual(len(list(temp_path.glob('**/*.h5'))), i + 1)

    def test_sub_experiment_with_arguments(self):
        arguments = {
            'foo': 10,
            'bar': 'bar',
        }
        self.test_sub_experiment(class_=_TestExperimentWithArguments, arguments=arguments)

    def test_sub_experiment_with_iterations(self):
        self.test_sub_experiment(iterations=3)
