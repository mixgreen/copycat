import typing
import logging
import time
import inspect

from artiq.experiment import EnvExperiment, HasEnvironment

__all__ = ['Barrier']


class Barrier(EnvExperiment):
    """Barrier

    This experiment functions as a barrier that prevents other experiments from running.

    When importing this class directly into the global namespace, it will be recognized by ARTIQ as an experiment.
    Alternatively, users can also call the :func:`submit` class method at runtime to schedule this experiment
    using the scheduler of the current environment.
    """

    PRIORITY: int = 100
    """Scheduling priority of the barrier."""
    CLOCK_PERIOD: float = 0.5
    """Internal clock period to check for pause conditions."""

    def build(self) -> None:  # type: ignore
        assert isinstance(self.PRIORITY, int), 'Priority must be of type int'
        assert isinstance(self.CLOCK_PERIOD, float), 'Clock period must be of type float'
        assert self.CLOCK_PERIOD > 0.0, 'Clock period must be greater than 0.0'

        # Set priority
        self.set_default_scheduling(priority=self.PRIORITY)
        # Get the scheduler
        self._scheduler = self.get_device('scheduler')

    def run(self) -> None:
        # Sleep until pause condition occurs
        while not self._scheduler.check_pause():
            time.sleep(self.CLOCK_PERIOD)

    @classmethod
    def submit(cls, environment: HasEnvironment, *, pipeline: typing.Optional[str] = None) -> None:
        """Submit this barrier experiment to the ARTIQ scheduler.

        :param environment: An object which inherits from ARTIQ HasEnvironment, required to get the ARTIQ scheduler
        :param pipeline: The pipeline to submit to, default to the current pipeline
        """
        assert isinstance(environment, HasEnvironment), 'The given environment must be of type HasEnvironment'
        assert isinstance(pipeline, str) or pipeline is None, 'The given pipeline must be None or of type str'

        # Construct expid
        expid: typing.Dict[str, typing.Any] = {
            'file': inspect.getabsfile(cls),
            'class_name': cls.__name__,
            'arguments': {},
            'log_level': logging.WARNING,
        }

        # Submit this class to the scheduler
        environment.get_device('scheduler').submit(pipeline_name=pipeline, expid=expid, priority=cls.PRIORITY)
