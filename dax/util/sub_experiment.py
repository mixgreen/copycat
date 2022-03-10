import typing
import collections.abc

import artiq.language.environment

import dax.util.artiq
import dax.util.output

__all__ = ['SubExperiment']


class SubExperiment:
    """Helper class to run sub-experiments with cloned managers.

    See also :func:`dax.util.artiq.clone_managers`.
    """

    _EXP_T = typing.TypeVar('_EXP_T', bound=artiq.language.environment.HasEnvironment)  # Experiment type variable

    _scheduler: typing.Any
    _file_name_generator: dax.util.output.FileNameGenerator
    _managers: typing.Any

    def __init__(self, parent: artiq.language.environment.HasEnvironment, managers: typing.Any):
        """Initialize a sub-experiment helper class.

        :param parent: The parent environment, normally the environment that spawns the sub-experiments
        :param managers: The tuple with ARTIQ manager objects (must be captured in the constructor of the parent)
        """
        if not isinstance(parent, artiq.language.environment.HasEnvironment):
            raise TypeError('The parent environment must be an ARTIQ environment object')

        # Obtain the scheduler from the parent
        self._scheduler = parent.get_device('scheduler')
        # Create a file name generator
        self._file_name_generator = dax.util.output.FileNameGenerator(self._scheduler)
        # Store the managers
        self._managers = managers

    def run(self, experiment_class: typing.Type[_EXP_T], name: str, *,
            arguments: typing.Optional[typing.Dict[str, typing.Any]] = None,
            metadata: typing.Optional[typing.Dict[str, typing.Any]] = None,
            build_args: typing.Sequence[typing.Any] = (),
            build_kwargs: typing.Optional[typing.Dict[str, typing.Any]] = None) -> _EXP_T:
        """Run a sub-experiment.

        Builds, prepares, runs, and analyzes a sub-experiment.
        At the end of the experiment, a HDF5 file with results will be written in the result directory.

        :param experiment_class: The class of the sub-experiment to run, must be an ARTIQ experiment
        :param name: Name for this sub-experiment to identify its output (will be made unique if necessary)
        :param arguments: Arguments for the sub-experiment
        :param metadata: Metadata stored in the output HDF5 file
        :param build_args: Positional arguments passed to the build function of the sub-experiment
        :param build_kwargs: Keyword arguments passed to the build function of the sub-experiment
        :return: The instance of the sub-experiment
        """

        # Set default values
        if arguments is None:
            arguments = {}
        if metadata is None:
            metadata = {}
        if build_kwargs is None:
            build_kwargs = {}

        assert issubclass(experiment_class, artiq.language.environment.HasEnvironment), \
            'Experiment class must be a subclass of ARTIQ HasEnvironment'
        assert issubclass(experiment_class, artiq.language.environment.Experiment), \
            'Experiment class must be a subclass of ARTIQ experiment'
        assert isinstance(name, str), 'Name must be of type str'
        assert isinstance(build_args, collections.abc.Sequence), 'Build arguments must be a sequence'
        for d in [arguments, metadata, build_kwargs]:
            assert isinstance(d, dict), 'Arguments, metadata, and build_kwargs must be of type dict'
            assert all(isinstance(k, str) for k in d), \
                'All keys in arguments, metadata, and build_kwargs must be of type str'

        with dax.util.artiq.clone_managers(self._managers, arguments=arguments) as managers:
            # Create the sub-experiment class
            # noinspection PyArgumentList
            exp = experiment_class(managers, *build_args, **build_kwargs)

            # Run all phases of the experiment
            exp.prepare()
            try:
                exp.run()
                exp.analyze()
            finally:
                # Store HDF5 file
                meta = {'rid': self._scheduler.rid}
                meta.update(metadata)
                managers.write_hdf5(self._file_name_generator(name, 'h5'), metadata=meta)

            # Return the sub-experiment class
            # noinspection PyTypeChecker
            return exp
