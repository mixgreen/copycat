from __future__ import annotations  # Postponed evaluation of annotations

import typing
import logging
import collections
import time
import enum
import networkx as nx
import graphviz

import artiq.experiment

import dax.base.system
import dax.util.output

__all__ = ['JobAction', 'Job', 'Policy', 'DaxScheduler']


def _str_to_time(string: str) -> float:
    """Convert a string to a time in seconds.

    This function converts simple time strings with a number and a character to a time in seconds.
    The available units are:

    - `'s'` for seconds
    - `'m'` for minutes
    - `'h'` for hours
    - `'d'` for days
    - `'w'` for weeks

    Examples of valid strings are `'10 s'`, `'-2m'`, `'8h'`, and `'2.5 w'`.
    An empty string will return zero.

    :param string: The string
    :return: Time in seconds as a float
    :raises ValueError: Raised if the number or the unit is invalid
    """
    assert isinstance(string, str), 'Input must be of type str'

    # Dict with available units
    units: typing.Dict[str, float] = {
        's': 1.0,
        'm': 60.0,
        'h': 3600.0,
        'd': 86400.0,
        'w': 604800.0,
    }

    if string:
        try:
            # Get the value and the unit
            value = float(string[:-1])
            unit = units[string[-1]]
            # Return the product
            return value * unit
        except KeyError:
            raise ValueError(f'No valid time unit was found at the end of string "{string}"')
        except ValueError:
            raise ValueError(f'No valid number was found at the start of string "{string}"')
    else:
        # Return zero in case the string is empty
        return 0.0


class JobAction(enum.Enum):
    """Job action enumeration."""

    RUN = enum.auto()
    """Run this job."""
    PASS = enum.auto()
    """Pass this job."""

    def submittable(self) -> bool:
        """Check if a job should be submitted or not.

        :return: True if the job is submittable
        """
        return self is JobAction.RUN

    def __str__(self) -> str:
        """String representation of this job action.

        :return: The name of the action
        """
        return self.name


class Job(dax.base.system.DaxHasKey):
    """Job class to define a job for the scheduler.

    Users only have to override class attributes to create a job definition.
    The following main attributes can be overridden:

    - :attr:`FILE`: The file name containing the experiment
    - :attr:`CLASS_NAME`: The class name of the experiment
    - :attr:`ARGUMENTS`: A dictionary with experiment arguments (scan objects can be used directly as arguments)
    - :attr:`INTERVAL`: The job submit interval
    - :attr:`DEPENDENCIES`: A collection of job classes on which this job depends
    """

    FILE: typing.Optional[str] = None
    """File containing the experiment, relative from the `repository` directory."""
    CLASS_NAME: typing.Optional[str] = None
    """Class name of the experiment."""
    ARGUMENTS: typing.Dict[str, typing.Any] = {}
    """The experiment arguments."""

    INTERVAL: typing.Optional[str] = None
    """Interval to run this job, defaults to no interval."""
    DEPENDENCIES: typing.Collection[typing.Type[Job]] = set()
    """Collection of dependencies of this job."""

    LOG_LEVEL: typing.Union[int, str] = logging.WARNING
    """The log level for the experiment."""
    PIPELINE: typing.Optional[str] = None
    """The pipeline to submit this job to, defaults to the pipeline assigned by the scheduler."""
    PRIORITY: int = 0
    """Job priority relative to the base job priority of the scheduler."""
    FLUSH: bool = False
    """The flush flag when submitting a job."""

    _RID_LIST_KEY: str = 'rid_list'
    """Key to store every submitted RID."""
    _LAST_SUBMIT_KEY: str = 'last_submit'
    """Key to store the last submit timestamp."""

    def __init__(self, managers_or_parent: DaxScheduler,
                 *args: typing.Any, **kwargs: typing.Any):
        """Initialize a job object.

        :param managers_or_parent: The manager or parent, must be a :class:`DaxScheduler`
        :param args: Positional arguments passed to the superclass
        :param kwargs: Keyword arguments passed to the superclass
        """

        # Check file, class, and arguments
        assert isinstance(self.FILE, str) or self.FILE is None, 'The file attribute must be of type str or None'
        assert isinstance(self.CLASS_NAME, str) or self.CLASS_NAME is None, \
            'The class name attribute must be of type str or None'
        assert isinstance(self.ARGUMENTS, dict), 'The arguments must be of type dict'
        assert all(isinstance(k, str) for k in self.ARGUMENTS), 'All argument keys must be of type str'
        # Check interval and dependencies
        assert isinstance(self.INTERVAL, str) or self.INTERVAL is None, 'Interval must be of type str or None'
        assert isinstance(self.DEPENDENCIES, collections.abc.Collection), 'The dependencies must be a collection'
        assert all(issubclass(d, Job) for d in self.DEPENDENCIES), 'All dependencies must be subclasses of Job'
        # Check log level, pipeline, priority, and flush
        assert isinstance(self.LOG_LEVEL, (int, str)), 'Log level must be of type int or str'
        assert self.PIPELINE is None or isinstance(self.PIPELINE, str), 'Pipeline must be of type str or None'
        assert isinstance(self.PRIORITY, int), 'Priority must be of type int'
        assert -99 <= self.PRIORITY <= 99, 'Priority must be in the domain [-99, 99]'
        assert isinstance(self.FLUSH, bool), 'Flush must be of type bool'

        # Check parent
        if not isinstance(managers_or_parent, DaxScheduler):
            raise TypeError(f'Parent of job {self.get_name()} is not a DAX scheduler')

        # Take key attributes from parent
        self._take_parent_key_attributes(managers_or_parent)

        # Call super
        super(Job, self).__init__(managers_or_parent, *args,
                                  name=self.get_name(), system_key=managers_or_parent.get_system_key(self.get_name()),
                                  **kwargs)

    def build(self) -> None:  # type: ignore
        # Obtain the scheduler
        self._scheduler = self.get_device('scheduler')

        # Construct an expid for this job
        if not self.is_meta():
            self._expid: typing.Dict[str, typing.Any] = {
                'file': self.FILE,
                'class_name': self.CLASS_NAME,
                'arguments': self._process_arguments(),
                'log_level': self.LOG_LEVEL,
                'repo_rev': None,  # Requests current revision
            }
            self.logger.debug(f'expid: {self._expid}')
        else:
            self._expid = {}
            self.logger.debug('This job is a meta-job')

        # Convert the interval
        if self.is_timed():
            # Convert the interval string
            self._interval: float = _str_to_time(typing.cast(str, self.INTERVAL))
            # Check the value
            if self._interval > 0.0:
                self.logger.info(f'Interval set to {self._interval:.0f} second(s)')
            else:
                msg: str = 'The job interval must be greater than zero'
                self.logger.error(msg)
                raise ValueError(msg)
        else:
            # No interval was set
            self._interval = 0.0
            self.logger.info('No interval set')

    def init(self, *, reset: bool) -> None:
        """Initialize the job, should be called once just before the scheduler starts.

        :param reset: Reset the previous state of this job
        """

        # Initialize last RID to a non-existing value
        self._last_rid: int = -1

        # Initialize RID list
        self._rid_list_key: str = self.get_system_key(self._RID_LIST_KEY)
        self.set_dataset(self._rid_list_key, [])  # Archive only

        if self.is_timed():
            if reset:
                # Reset state and set the current time instead
                self._next_submit: float = time.time()
                self.logger.debug('Initialized job by resetting next submit time')
            else:
                # Try to obtain the last submit time
                last_submit = self.get_dataset_sys(self._LAST_SUBMIT_KEY, 0.0, data_store=False)
                assert isinstance(last_submit, float), 'Unexpected type returned from dataset'
                # Add the interval to the last submit time
                self._next_submit = last_submit + self._interval
                self.logger.debug('Initialized job by obtaining the last submit time')
        else:
            # This job is untimed, next submit will never happen
            self._next_submit = float('inf')
            self.logger.debug('Initialized untimed job')

    def visit(self, *, wave: float) -> JobAction:
        """Visit this job.

        :param wave: Wave identifier
        :return: The job action for this job
        """
        assert isinstance(wave, float), 'Wave must be of type float'

        if self._next_submit <= wave:
            # Interval expired, run this job
            return JobAction.RUN
        else:
            # Interval did not expire, do not run
            return JobAction.PASS

    def submit(self, *, wave: float, pipeline: str, priority: int) -> None:
        """Submit this job.

        :param wave: Wave identifier
        :param pipeline: The default pipeline to submit to
        :param priority: The baseline priority of the experiment
        """
        assert isinstance(wave, float), 'Wave must be of type float'
        assert isinstance(pipeline, str) and pipeline, 'Pipeline name must be of type string and can not be empty'
        assert isinstance(priority, int), 'Priority must be of type int'

        # Store current wave timestamp
        self.set_dataset_sys(self._LAST_SUBMIT_KEY, wave, data_store=False)

        if not self.is_meta():
            if self._last_rid not in self._scheduler.get_status():
                # Submit experiment of this job
                self._last_rid = self._scheduler.submit(
                    pipeline_name=pipeline if self.PIPELINE is None else self.PIPELINE,
                    expid=self._expid,
                    priority=priority + self.PRIORITY,
                    flush=self.FLUSH
                )
                self.logger.info(f'Submitted job with RID {self._last_rid}')

                # Archive the RID
                self.append_to_dataset(self._rid_list_key, self._last_rid)
                self.data_store.append(self._rid_list_key, self._last_rid)
            else:
                # Previous job was still running
                self.logger.warning(f'Skipping job, previous job with RID {self._last_rid} is still running')

        # Reschedule job
        self.schedule(wave=wave)

    def schedule(self, *, wave: float) -> None:
        """Schedule this job.

        :param wave: Wave identifier
        """
        assert isinstance(wave, float), 'Wave must be of type float'

        if self.is_timed():
            if self.visit(wave=wave).submittable():
                # Update next submit time using the interval
                self._next_submit += self._interval
            else:
                # Involuntary submit, reset phase
                self._next_submit = wave + self._interval

            if self._next_submit <= time.time():
                # Prevent setting next submit time in the past
                self._next_submit = time.time() + self._interval
                self.logger.warning('Next job was rescheduled because the interval time expired')

    def cancel(self) -> None:
        """Cancel this job."""
        if self._last_rid >= 0:
            # Cancel the last RID (returns if the RID is not running)
            self.logger.debug(f'Cancelling job (RID {self._last_rid})')
            self._scheduler.request_termination(self._last_rid)

    def _process_arguments(self) -> typing.Dict[str, typing.Any]:
        """Process and return the arguments of this job.

        :return: The processed arguments, ready to be used in the expid
        """

        def process(argument: typing.Any) -> typing.Any:
            if isinstance(argument, artiq.experiment.ScanObject):
                return argument.describe()  # type: ignore[attr-defined]
            else:
                return argument

        return {key: process(arg) for key, arg in self.ARGUMENTS.items()}

    def is_meta(self) -> bool:
        """Check if this job is a meta-job (i.e. no experiment is associated with it).

        :return: True if this job is a meta-job
        """
        if self.FILE is None and self.CLASS_NAME is not None or self.FILE is not None and self.CLASS_NAME is None:
            raise ValueError('The FILE and CLASS_NAME attributes should both be None or not None')

        return self.FILE is None and self.CLASS_NAME is None

    def is_timed(self) -> bool:
        """Check if this job is timed.

        :return: True if this job has an interval
        """
        return self.INTERVAL is not None

    @classmethod
    def get_name(cls) -> str:
        """Get the name of this job.

        :return: The name of this job as a string
        """
        return cls.__name__


class Policy(enum.Enum):
    """Policy enumeration for the scheduler.

    The policy enumeration includes definitions for the policies using a mapping table.
    """

    if typing.TYPE_CHECKING:
        # Only add type when type checking is enabled to not conflict with iterations over the Policy enum
        __P_T = typing.Dict[typing.Tuple[JobAction, JobAction], JobAction]  # Policy enum type

    LAZY: __P_T = {
        (JobAction.PASS, JobAction.PASS): JobAction.PASS,
        (JobAction.PASS, JobAction.RUN): JobAction.RUN,
        (JobAction.RUN, JobAction.PASS): JobAction.PASS,
        (JobAction.RUN, JobAction.RUN): JobAction.RUN,
    }
    """Lazy scheduling policy, only submit jobs that expired."""

    GREEDY: __P_T = {
        (JobAction.PASS, JobAction.PASS): JobAction.PASS,
        (JobAction.PASS, JobAction.RUN): JobAction.RUN,
        (JobAction.RUN, JobAction.PASS): JobAction.RUN,
        (JobAction.RUN, JobAction.RUN): JobAction.RUN,
    }
    """Greedy scheduling policy, submit jobs that expired or depend on an expired job."""

    def action(self, previous: JobAction, current: JobAction) -> JobAction:
        """Apply the policy on two job actions.

        :param previous: The job action of the predecessor (previous node)
        :param current: The current job action
        :return: The new job action based on this policy
        """
        assert isinstance(previous, JobAction)
        assert isinstance(current, JobAction)
        policy: Policy.__P_T = self.value
        return policy[previous, current]

    @staticmethod
    def from_str(string_: str) -> Policy:
        """Convert a string into its corresponding policy enumeration.

        :param string_: The name of the policy as a string
        :return: The policy enumeration object
        :raises KeyError: Raised if the policy name does not exist (case sensitive)
        """
        return {str(p): p for p in Policy}[string_]

    def __str__(self) -> str:
        """Return the name of this policy.

        :return: The name of this policy as a string
        """
        return self.name


class DaxScheduler(dax.base.system.DaxHasKey):
    """DAX scheduler class to inherit from.

    Users only have to override class attributes to create a scheduling definition.
    **The scheduler subclass must also inherit from :class:`Experiment` or :class:`EnvExperiment`
    to make the scheduler available as an ARTIQ experiment.**

    The following attributes must be overridden:

    - :attr:`NAME`: The name of this scheduler
    - :attr:`JOBS`: A collection of job classes that form the job set for this scheduler

    Other optional attributes that can be overridden are:

    - :attr:`SYSTEM`: A DAX system type to enable additional logging of data
    - :attr:`DEFAULT_PIPELINE`: The default pipeline to submit jobs to, the scheduler can not run in the same pipeline
    - :attr:`DEFAULT_JOB_PRIORITY`: The baseline priority for jobs submitted by this scheduler
    """

    NAME: str
    """Scheduler name, used as top key."""
    JOBS: typing.Collection[typing.Type[Job]]
    """The collection of job classes."""

    SYSTEM: typing.Optional[typing.Type[dax.base.system.DaxSystem]] = None
    """Optional DAX system type, enables extra features if provided."""

    DEFAULT_PIPELINE: str = 'main'
    """Default pipeline to submit jobs to."""
    DEFAULT_JOB_PRIORITY: int = 0
    """Default baseline priority to submit jobs."""

    _GRAPHVIZ_FORMAT: str = 'pdf'
    """Format specification for the graphviz renderer."""

    def __init__(self, managers_or_parent: typing.Any,
                 *args: typing.Any, **kwargs: typing.Any):
        """Initialize the scheduler object.

        :param managers_or_parent: The manager or parent
        :param args: Positional arguments passed to the superclass
        :param kwargs: Keyword arguments passed to the superclass
        """

        # Check name
        assert hasattr(self, 'NAME'), 'No name was provided'
        # Check jobs
        assert hasattr(self, 'JOBS'), 'No job list was provided'
        assert isinstance(self.JOBS, collections.abc.Collection), 'The jobs attribute must be a collection'
        assert all(issubclass(job, Job) for job in self.JOBS), 'All jobs must be subclasses of Job'
        # Check system
        assert self.SYSTEM is None or issubclass(self.SYSTEM, dax.base.system.DaxSystem)
        # Check pipeline
        assert isinstance(self.DEFAULT_PIPELINE, str) and self.DEFAULT_PIPELINE, 'Default pipeline must be of type str'
        # Check default job priority
        assert isinstance(self.DEFAULT_JOB_PRIORITY, int), 'Default job priority must be of type int'
        assert -99 <= self.DEFAULT_JOB_PRIORITY <= 99, 'Default job priority must be in the domain [-99, 99]'

        # Call super
        super(DaxScheduler, self).__init__(managers_or_parent, *args,
                                           name=self.NAME, system_key=self.NAME, **kwargs)

    def build(self, *args: typing.Any, **kwargs: typing.Any) -> None:
        # Set default scheduling options for the scheduler itself
        self.set_default_scheduling(pipeline_name=self.NAME)

        # Scheduling arguments
        self._policy_arg = self.get_argument('Policy',
                                             artiq.experiment.EnumerationValue(sorted(str(p) for p in Policy)),
                                             tooltip='Scheduling policy',
                                             group='Scheduler')
        self._wave_interval = self.get_argument('Wave interval',
                                                artiq.experiment.NumberValue(60, 's', min=1, step=1, ndecimals=0),
                                                tooltip='Interval to visit jobs',
                                                group='Scheduler')
        self._clock_period = self.get_argument('Clock period',
                                               artiq.experiment.NumberValue(0.5, 's', min=0.1),
                                               tooltip='Internal scheduler clock period',
                                               group='Scheduler')

        # Job arguments
        self._pipeline = self.get_argument('Pipeline',
                                           artiq.experiment.StringValue(self.DEFAULT_PIPELINE),
                                           tooltip='Default pipeline to submit jobs to',
                                           group='Jobs')
        self._job_priority = self.get_argument('Priority',
                                               artiq.experiment.NumberValue(self.DEFAULT_JOB_PRIORITY,
                                                                            min=-99, max=99, step=1, ndecimals=0),
                                               tooltip='Baseline job priority',
                                               group='Jobs')
        self._reset_jobs = self.get_argument('Reset',
                                             artiq.experiment.BooleanValue(False),
                                             tooltip='Reset job timestamps at startup, making them all expired',
                                             group='Jobs')
        self._terminate_jobs = self.get_argument('Terminate at exit',
                                                 artiq.experiment.BooleanValue(False),
                                                 tooltip='Terminate running jobs at exit',
                                                 group='Jobs')

        # Dependency graph arguments
        self._reduce_graph = self.get_argument('Reduce',
                                               artiq.experiment.BooleanValue(True),
                                               tooltip='Use transitive reduction of the job dependency graph',
                                               group='Dependency graph')
        self._view_graph = self.get_argument('View',
                                             artiq.experiment.BooleanValue(False),
                                             tooltip='View the job dependency graph at startup',
                                             group='Dependency graph')

        # The ARTIQ scheduler
        self._scheduler = self.get_device('scheduler')

        # Instantiate the data store
        if self.SYSTEM is not None and self.SYSTEM.DAX_INFLUX_DB_KEY is not None:
            # Create an Influx DB data store
            self.__data_store = dax.base.system.DaxDataStoreInfluxDb.get_instance(self, self.SYSTEM)
        else:
            # No data store configured
            self.__data_store: dax.base.system.DaxDataStore = dax.base.system.DaxDataStore()

        # Call super and forward arguments, for compatibility with other libraries
        # noinspection PyArgumentList
        super(DaxScheduler, self).build(*args, **kwargs)

    @property
    def data_store(self) -> dax.base.system.DaxDataStore:
        """Get the data store.

        :return: The data store object
        """
        return self.__data_store

    def prepare(self) -> None:
        # Check pipeline
        if self._scheduler.pipeline_name == self._pipeline:
            raise ValueError(f'The scheduler can not run in pipeline "{self._pipeline}"')

        # Check arguments
        if self._wave_interval < 1.0:
            raise ValueError('The chosen wave interval is too small')
        if self._clock_period < 0.1:
            raise ValueError('The chosen clock period is too small')
        if self._wave_interval < 2 * self._clock_period:
            raise ValueError('The wave interval is too small compared to the clock period')
        if -99 > self._job_priority > 99:
            raise ValueError('Job priority must be in the domain [-99, 99]')

        # Obtain the scheduling policy
        self._policy: Policy = Policy.from_str(self._policy_arg)

        # Create the job objects
        jobs: typing.Dict[typing.Type[Job], Job] = {job: job(self) for job in self.JOBS}
        self.logger.debug(f'Created {len(jobs)} job(s)')
        if len(jobs) < len(self.JOBS):
            self.logger.warning('Duplicate jobs were dropped from the job set')
        if len({job.get_name() for job in jobs}) < len(jobs):
            msg = 'Job name conflict, two jobs are not allowed to have the same class name'
            self.logger.error(msg)
            raise ValueError(msg)

        # Create the job dependency graph
        self._job_graph: nx.DiGraph[Job] = nx.DiGraph()
        self._job_graph.add_nodes_from(jobs.values())
        try:
            self._job_graph.add_edges_from(((job, jobs[d]) for job in jobs.values() for d in job.DEPENDENCIES))
        except KeyError as e:
            raise KeyError(f'Dependency "{e.args[0].get_name()}" is not part of the given job set') from None

        # Check graph
        if not nx.algorithms.is_directed_acyclic_graph(self._job_graph):
            raise RuntimeError('Dependency graph is not a directed acyclic graph')
        if self._policy is Policy.LAZY and any(not j.is_timed() for j in self._job_graph):
            self.logger.warning('Found one or more unreachable jobs (untimed jobs in a lazy scheduling policy')

        if self._reduce_graph:
            # Get the transitive reduction of the job dependency graph
            self._job_graph = nx.algorithms.transitive_reduction(self._job_graph)

        # Plot graph
        plot = graphviz.Digraph(name=self.NAME, directory=str(dax.util.output.get_base_path(self._scheduler)))
        for job in self._job_graph:
            plot.node(job.get_name())
        plot.edges(((j.get_name(), k.get_name()) for j, k in self._job_graph.edges))
        plot.render(view=self._view_graph, format=self._GRAPHVIZ_FORMAT)

        # Find root jobs
        # noinspection PyTypeChecker
        self._root_jobs: typing.Set[Job] = {job for job, degree in self._job_graph.in_degree if degree == 0}
        self.logger.debug(f'Found {len(self._root_jobs)} root job(s)')

        # Terminate other instances of this scheduler
        self._terminate_running_instances()

    def run(self) -> None:
        """Entry point for the scheduler."""

        # Initialize all jobs
        self.logger.debug(f'Initializing {len(self._job_graph)} job(s)')
        for job in self._job_graph:
            job.init(reset=self._reset_jobs)

        # Time for the next wave
        next_wave: float = time.time()

        try:
            while True:
                while next_wave > time.time():
                    if self._scheduler.check_pause():
                        # Pause
                        self.logger.debug('Pausing scheduler')
                        self._scheduler.pause()
                    else:
                        # Sleep for a clock period
                        time.sleep(self._clock_period)

                # Update next wave time
                next_wave += self._wave_interval
                # Start the wave
                self.wave()
                # Prevent setting next wave time in the past
                if next_wave <= time.time():
                    next_wave = time.time() + self._wave_interval
                    self.logger.warning('Next wave was rescheduled because the interval time expired')

        except artiq.experiment.TerminationRequested:
            # Scheduler terminated
            self.logger.info('Scheduler was terminated')

        finally:
            if self._terminate_jobs:
                # Cancel all jobs
                self.logger.debug(f'Cancelling {len(self._job_graph)} job(s)')
                for job in self._job_graph:
                    job.cancel()

    def wave(self) -> None:
        """Run a wave over the job set."""

        # Generate the unique wave timestamp
        wave: float = time.time()
        self.logger.debug(f'Starting wave {wave:.0f}')

        def recurse(job: Job, action: JobAction, submitted: typing.Set[Job]) -> typing.Set[Job]:
            """Recurse over the dependencies of a job.

            :param job: The current job to process
            :param action: The action provided by the previous job
            :param submitted: The set of jobs submitted in this wave
            :return: The updated set of submitted jobs
            """

            # Visit the current job
            current_action = job.visit(wave=wave)
            # Get the new action based on the policy
            new_action = self._policy.action(action, current_action)
            # Recurse over successors
            for successor in self._job_graph.successors(job):
                submitted = recurse(successor, new_action, submitted)

            if new_action.submittable() and job not in submitted:
                # Submit this job
                job.submit(wave=wave,
                           pipeline=self._pipeline,
                           priority=self._job_priority)
                submitted.add(job)

            # Return set with submitted jobs
            return submitted

        # Set of submitted jobs in this wave
        submitted_jobs: typing.Set[Job] = set()

        for root_job in self._root_jobs:
            # Recurse over all root jobs
            submitted_jobs = recurse(root_job, JobAction.PASS, submitted_jobs)

        if submitted_jobs:
            # Log submitted jobs
            self.logger.debug(f'Submitted jobs: {", ".join(sorted(j.get_name() for j in submitted_jobs))}')

    def _terminate_running_instances(self) -> None:
        """Terminate running instances of this scheduler."""

        # Obtain the schedule with the expid objects
        schedule = {rid: status['expid'] for rid, status in self._scheduler.get_status().items()}
        # Filter schedule to find other instances of this scheduler
        other_instances = [rid for rid, expid in schedule.items()
                           if expid['file'] == self._scheduler.expid['file']
                           and expid['class_name'] == self._scheduler.expid['class_name']
                           and rid != self._scheduler.rid]

        # Request termination of other instances
        for rid in other_instances:
            self.logger.info(f'Terminating other scheduler instance with RID {rid}')
            self._scheduler.request_termination(rid)

        if other_instances:
            # Wait until all other instances disappeared from the schedule
            self.logger.info(f'Waiting for {len(other_instances)} other instance(s) to terminate')
            # The timeout counter
            timeout = 20

            while any(rid in self._scheduler.get_status() for rid in other_instances):
                if timeout <= 0:
                    # Timeout elapsed
                    raise RuntimeError('Timeout while waiting for other instances to terminate')
                else:
                    # Update timeout counter
                    timeout -= 1

                # Sleep
                time.sleep(0.5)

            # Other instances were terminated
            self.logger.info('All other instances were terminated successfully')
