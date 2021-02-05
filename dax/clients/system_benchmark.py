# mypy: no_warn_unused_ignores

import logging
import typing
import time
import cProfile
import pstats

import artiq.master.worker_db

from dax.experiment import *
import dax.util.units
import dax.util.output

__all__ = ['SystemBenchmarkDaxInit', 'SystemBenchmarkDaxInitProfile', 'SystemBenchmarkBuildProfile']


@dax_client_factory
class SystemBenchmarkDaxInit(DaxClient, Experiment):
    """DAX system initialization benchmark."""

    DAX_INIT: bool = False
    """Disable DAX init."""

    def build(self) -> None:  # type: ignore
        # Arguments
        self.num_samples = self.get_argument('num_samples', NumberValue(5, min=1, step=1, ndecimals=0))

    def prepare(self) -> None:
        # Get the system
        self.system = self.registry.find_module(DaxSystem)

    def run(self) -> None:
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
            # Create a new experiment class which is an instance of the system type
            system = type(self.system)(self)

            # Record start time
            start = time.perf_counter()
            # Run DAX system initialization
            system.dax_init()
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

    def analyze(self) -> None:
        # Report result
        init_time = dax.util.units.time_to_str(self.system.get_dataset_sys(self.system.DAX_INIT_TIME_KEY))
        self.logger.info(f'Average execution time of dax_init() is {init_time}')


@dax_client_factory
class SystemBenchmarkDaxInitProfile(DaxClient, Experiment):
    """DAX system initialization profiler."""

    DAX_INIT: bool = False
    """Disable DAX init."""

    SORT_KEYS = [k.value for k in pstats.SortKey]  # type: ignore[attr-defined]
    """Profile stats sort keys."""

    def build(self) -> None:  # type: ignore
        # Arguments
        self.sort_stats = self.get_argument('Sort stats',
                                            EnumerationValue(self.SORT_KEYS, 'cumulative'),
                                            tooltip='Sort the txt output based on the given key')
        self.strip_dirs = self.get_argument('Strip dirs',
                                            BooleanValue(False),
                                            tooltip='Remove leading path information from file names in txt output')

    def prepare(self) -> None:
        # Get the system
        self.system = self.registry.find_module(DaxSystem)
        # Create the profile object
        self.profile = cProfile.Profile()

    def run(self) -> None:
        # Profile just the dax_init() call
        self.profile.enable()
        try:
            self.system.dax_init()
        finally:
            self.profile.disable()

    def analyze(self) -> None:
        # Dump raw stats
        file_name_generator = dax.util.output.get_file_name_generator(self.get_device('scheduler'))
        self.profile.dump_stats(file_name_generator('cprofile', 'stats'))

        with open(file_name_generator('cprofile', 'txt'), 'w') as file:
            # Create stats object
            stats = pstats.Stats(self.profile, stream=file)
            # Modify stats based on arguments
            if self.strip_dirs:
                stats.strip_dirs()
            stats.sort_stats(self.sort_stats)
            # Write txt stats
            stats.print_stats()


class _SystemBenchmarkBuildProfile(EnvExperiment):
    """Build profile experiment class template."""

    SORT_KEYS = [k.value for k in pstats.SortKey]  # type: ignore[attr-defined]
    """Profile stats sort keys."""

    SYSTEM_TYPE: typing.Type[DaxSystem]
    """System type to profile."""
    SYSTEM_ARGS: typing.Sequence[typing.Any]
    """Positional arguments for building the system."""
    SYSTEM_KWARGS: typing.Dict[str, typing.Any]
    """Keyword arguments for building the system."""

    def build(self) -> None:  # type: ignore
        # Arguments
        self.sort_stats = self.get_argument('Sort stats',
                                            EnumerationValue(self.SORT_KEYS, 'cumulative'),
                                            tooltip='Sort the txt output based on the given key')
        self.strip_dirs = self.get_argument('Strip dirs',
                                            BooleanValue(False),
                                            tooltip='Remove leading path information from file names in txt output')

    def prepare(self) -> None:
        # Call super
        super(_SystemBenchmarkBuildProfile, self).prepare()

        # Create the profile object
        self.profile = cProfile.Profile()

        # Profile call to build()
        self.profile.enable()
        try:
            self.SYSTEM_TYPE(self, *self.SYSTEM_ARGS, **self.SYSTEM_KWARGS)
        finally:
            self.profile.disable()

    def run(self) -> None:
        # Profiling is done in the prepare() function as we do not need to have write access to datasets
        pass

    def analyze(self) -> None:
        # Dump raw stats
        file_name_generator = dax.util.output.get_file_name_generator(self.get_device('scheduler'))
        self.profile.dump_stats(file_name_generator('cprofile', 'stats'))

        with open(file_name_generator('cprofile', 'txt'), 'w') as file:
            # Create stats object
            stats = pstats.Stats(self.profile, stream=file)
            # Modify stats based on arguments
            if self.strip_dirs:
                stats.strip_dirs()
            stats.sort_stats(self.sort_stats)
            # Write txt stats
            stats.print_stats()


# noinspection PyPep8Naming
def SystemBenchmarkBuildProfile(
        system_type: typing.Type[DaxSystem],
        *system_args: typing.Any, **system_kwargs: typing.Any) -> typing.Type[_SystemBenchmarkBuildProfile]:
    """Create a new build profile experiment class.

    This factory function will create a new build profile class for a given system type.

    Note that this is a custom factory function that matches the signature
    of the :func:`dax_client_factory` decorator output.

    :param system_type: The system type to build
    :param system_args: Positional arguments forwarded to the systems :func:`build` function
    :param system_kwargs: Keyword arguments forwarded to the systems :func:`build` function
    :return: A build profile class for the given system type
    :raises TypeError: Raised if the provided `system_type` parameter is not a subclass of `DaxSystem`
    """

    # Check the system type
    assert isinstance(system_type, type), 'System type must be a type'
    if not issubclass(system_type, DaxSystem):
        raise TypeError('System type must be a subclass of DaxSystem')

    class BuildProfileExperiment(_SystemBenchmarkBuildProfile):
        SYSTEM_TYPE = system_type
        SYSTEM_ARGS = system_args
        SYSTEM_KWARGS = system_kwargs

    # The factory function returns the newly constructed class
    return BuildProfileExperiment
