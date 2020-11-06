import typing
import logging
import time
import inspect
import numpy as np

import artiq.experiment

__all__ = ['Barrier', 'SetDataset']


class Barrier(artiq.experiment.EnvExperiment):
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
    @artiq.experiment.host_only
    def submit(cls, environment: artiq.experiment.HasEnvironment, *, pipeline: typing.Optional[str] = None) -> None:
        """Submit this barrier experiment to the ARTIQ scheduler.

        Note that submitting the barrier experiment does not pause the current experiment.

        :param environment: An object which inherits from ARTIQ HasEnvironment, required to get the ARTIQ scheduler
        :param pipeline: The pipeline to submit to, default to the current pipeline
        """
        assert isinstance(environment, artiq.experiment.HasEnvironment), \
            'The given environment must be of type HasEnvironment'
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


class SetDataset(artiq.experiment.EnvExperiment):
    """Set dataset

    This experiment is a utility to set/write arbitrary datasets.
    When importing this class directly into the global namespace, it will be recognized by ARTIQ as an experiment.
    """

    _UNITS: typing.Dict[str, float] = {unit: getattr(artiq.experiment, unit)
                                       for unit in ['ps', 'ns', 'us', 'ms', 's',
                                                    'mHz', 'Hz', 'kHz', 'MHz', 'GHz',
                                                    'dB',
                                                    'uV', 'mV', 'V', 'kV',
                                                    'uA', 'mA', 'A',
                                                    'uW', 'mW', 'W']}
    """Dict with all units."""

    def build(self) -> None:  # type: ignore
        # Dataset key
        self.key: str = self.get_argument('Key', artiq.experiment.StringValue(),
                                          tooltip='The key of the dataset')
        # Dataset value
        self.value: str = self.get_argument('Value', artiq.experiment.StringValue(),
                                            tooltip='The value to store, which is directly interpreted using `eval()`\n'
                                                    'Globals include `np` (Numpy) and the ARTIQ units')

        # Persist flag
        self.persist: bool = self.get_argument('Persist', artiq.experiment.BooleanValue(False),
                                               tooltip='The master should store the data on-disk')
        # Overwrite flag
        self.overwrite: bool = self.get_argument('Overwrite', artiq.experiment.BooleanValue(False),
                                                 tooltip='Allow overwriting of existing values')

    def run(self) -> None:
        if not self.overwrite:
            try:
                # Try to obtain dataset
                self.get_dataset(self.key, archive=False)
            except KeyError:
                # Key does not exist, we are not overwriting
                pass
            else:
                # Key does exist, we are overwriting
                raise RuntimeError(f'Key "{self.key}" already exists and overwrite is disabled')

        # Set up globals
        g: typing.Dict[str, typing.Any] = {'np': np}
        g.update(self._UNITS)
        # Evaluate value
        value: typing.Any = eval(self.value, g, {})
        # Set dataset
        self.set_dataset(self.key, value,
                         broadcast=True, persist=self.persist, archive=False)
