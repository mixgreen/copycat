from __future__ import annotations  # Postponed evaluation of annotations

import typing
import logging
import collections.abc
import time
import enum
import networkx as nx
import graphviz
import asyncio
import json
import hashlib
import dataclasses

import artiq.experiment
import artiq.master.worker_db
import sipyco.pc_rpc  # type: ignore

import dax.base.system
import dax.util.output

__all__ = ['JobAction', 'Job', 'Policy', 'DaxScheduler', 'dax_scheduler_client']


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


def _hash_dict(d: typing.Dict[str, typing.Any]) -> str:
    """Return a stable hash of a dictionary.

    :param d: The dictionary to hash
    :return: Hash of the dictionary
    :raises TypeError: Raised if a key or value type can not be hashed
    """

    assert isinstance(d, dict), 'Input must be a dict'
    assert all(isinstance(k, str) for k in d), 'All keys must be of type str'

    # Convert the dict to a sorted json string, which should be unique
    json_ = json.dumps(d, sort_keys=True)
    # Take a hash of the unique string
    arguments_hash = hashlib.md5(json_.encode())
    # Return the digest
    return arguments_hash.hexdigest()


class JobAction(enum.Enum):
    """Job action enumeration."""

    PASS = enum.auto()
    """Pass this job."""
    RUN = enum.auto()
    """Run this job."""
    FORCE = enum.auto()
    """Force this job."""

    def submittable(self) -> bool:
        """Check if a job should be submitted or not.

        :return: True if the job is submittable
        """
        return self in {JobAction.RUN, JobAction.FORCE}

    @classmethod
    def from_str(cls, string_: str) -> JobAction:
        """Convert a string into its corresponding job action enumeration.

        :param string_: The name of the job action as a string (case insensitive)
        :return: The job action enumeration object
        :raises KeyError: Raised if the job action name does not exist
        """
        return {str(p).lower(): p for p in cls}[string_.lower()]

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

    Optionally, users can override the :func:`build_job` method to add configurable arguments.
    """

    FILE: typing.Optional[str] = None
    """File containing the experiment, relative from the `repository` directory."""
    CLASS_NAME: typing.Optional[str] = None
    """Class name of the experiment."""
    ARGUMENTS: typing.Dict[str, typing.Any] = {}
    """The experiment arguments."""

    INTERVAL: typing.Optional[str] = None
    """Interval to run this job, defaults to no interval."""
    DEPENDENCIES: typing.Collection[typing.Type[Job]] = []
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
    _ARGUMENTS_HASH_KEY: str = 'arguments_hash'
    """Key to store the last arguments hash."""

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
            raise TypeError(f'Parent of job "{self.get_name()}" is not a DAX scheduler')

        # Take key attributes from parent
        self._take_parent_key_attributes(managers_or_parent)

        # Call super
        super(Job, self).__init__(managers_or_parent, *args,
                                  name=self.get_name(), system_key=managers_or_parent.get_system_key(self.get_name()),
                                  **kwargs)

    def build(self) -> None:  # type: ignore
        """Build the job object.

        To add configurable arguments to this job, override the :func:`build_job` method.
        """

        # Obtain the scheduler
        self._scheduler = self.get_device('scheduler')
        # Copy the class arguments attribute (prevents class attribute from being mutated)
        self.ARGUMENTS = self.ARGUMENTS.copy()
        # Build this job
        self.build_job()
        # Process the arguments
        self._arguments = self._process_arguments()

        # Construct an expid for this job
        if not self.is_meta():
            self._expid: typing.Dict[str, typing.Any] = {
                'file': self.FILE,
                'class_name': self.CLASS_NAME,
                'arguments': self._arguments,
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
                raise ValueError(f'The interval of job "{self.get_name()}" must be greater than zero')
        else:
            # No interval was set
            self._interval = 0.0
            self.logger.info('No interval set')

    def build_job(self) -> None:
        """Build this job.

        Override this function to add configurable arguments to this job.
        Please note that **argument keys must be unique over all jobs in the job set and the scheduler**.
        It is up to the programmer to ensure that there are no duplicate argument keys.

        Configurable arguments should be added directly to the :attr:`ARGUMENTS` attribute.
        For example::

            self.ARGUMENTS['foo'] = self.get_argument('foo', BooleanValue())
            self.ARGUMENTS['bar'] = self.get_argument('bar', Scannable(RangeScan(10 * us, 200 * us, 10)))
        """
        pass

    def init(self, *, reset: bool) -> None:
        """Initialize the job, should be called once just before the scheduler starts.

        :param reset: Reset the previous state of this job
        """

        # Initialize last RID to a non-existing value
        self._last_rid: int = -1

        # Initialize RID list
        self._rid_list_key: str = self.get_system_key(self._RID_LIST_KEY)
        self.set_dataset(self._rid_list_key, [])  # Archive only

        # Calculate the argument hash
        arguments_hash: str = _hash_dict(self._arguments)
        try:
            # Check if the argument hash changed
            arguments_changed: bool = arguments_hash != self.get_dataset_sys(self._ARGUMENTS_HASH_KEY)
        except KeyError:
            # No previous arguments hash was available, so we can consider it changed
            arguments_changed = True

        if arguments_changed:
            # Store the new arguments hash
            self.logger.debug(f'Job arguments changed: {arguments_hash}')
            self.set_dataset_sys(self._ARGUMENTS_HASH_KEY, arguments_hash, data_store=False)

        if self.is_timed():
            self.logger.debug('Initializing timed job')
            if reset:
                # Reset the job by resetting the next submit time
                self._next_submit: float = time.time()
                self.logger.debug('Job reset requested')
            elif arguments_changed:
                # Arguments changed, reset the next submit time
                self._next_submit = time.time()
                self.logger.debug('Job was reset due to changed arguments')
            else:
                # Try to obtain the last submit time
                last_submit = self.get_dataset_sys(self._LAST_SUBMIT_KEY, 0.0, data_store=False)
                assert isinstance(last_submit, float), 'Unexpected type returned from dataset'
                # Add the interval to the last submit time
                self._next_submit = last_submit + self._interval
                self.logger.debug('Initialized job by loading the last submit time')
        else:
            # This job is untimed, next submit will not happen
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
        attributes = (self.FILE, self.CLASS_NAME)
        meta = all(a is None for a in attributes)
        if meta or all(a is not None for a in attributes):
            return meta
        else:
            raise ValueError(f'The FILE and CLASS_NAME attributes of job "{self.get_name()}" '
                             f'should both be None or not None')

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
        (JobAction.PASS, JobAction.FORCE): JobAction.RUN,
        (JobAction.RUN, JobAction.PASS): JobAction.PASS,
        (JobAction.RUN, JobAction.RUN): JobAction.RUN,
        (JobAction.RUN, JobAction.FORCE): JobAction.RUN,
        (JobAction.FORCE, JobAction.PASS): JobAction.RUN,
        (JobAction.FORCE, JobAction.RUN): JobAction.RUN,
        (JobAction.FORCE, JobAction.FORCE): JobAction.RUN,
    }
    """Lazy scheduling policy, only submit jobs that expired."""

    GREEDY: __P_T = {
        (JobAction.PASS, JobAction.PASS): JobAction.PASS,
        (JobAction.PASS, JobAction.RUN): JobAction.RUN,
        (JobAction.PASS, JobAction.FORCE): JobAction.RUN,
        (JobAction.RUN, JobAction.PASS): JobAction.RUN,
        (JobAction.RUN, JobAction.RUN): JobAction.RUN,
        (JobAction.RUN, JobAction.FORCE): JobAction.RUN,
        (JobAction.FORCE, JobAction.PASS): JobAction.RUN,
        (JobAction.FORCE, JobAction.RUN): JobAction.RUN,
        (JobAction.FORCE, JobAction.FORCE): JobAction.RUN,
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

    @classmethod
    def from_str(cls, string_: str) -> Policy:
        """Convert a string into its corresponding policy enumeration.

        :param string_: The name of the policy as a string (case insensitive)
        :return: The policy enumeration object
        :raises KeyError: Raised if the policy name does not exist
        """
        return {str(p).lower(): p for p in cls}[string_.lower()]

    def __str__(self) -> str:
        """Return the name of this policy.

        :return: The name of this policy as a string
        """
        return self.name


@dataclasses.dataclass(frozen=True)
class _Request:
    """Data class for scheduler controller requests."""
    jobs: typing.Collection[str]
    action: str
    policy: typing.Optional[str]
    reverse: typing.Optional[bool]


if typing.TYPE_CHECKING:
    __Q_T = asyncio.Queue[_Request]  # Type variable for the async request queue


class _SchedulerController:
    """Scheduler controller class, which exposes an external interface to the DAX scheduler."""

    def __init__(self, queue: __Q_T):
        # Store a reference to the queue
        self._queue: __Q_T = queue

    def submit(self, *jobs: str,
               action: str = str(JobAction.FORCE),
               policy: typing.Optional[str] = None,
               reverse: typing.Optional[bool] = None) -> None:
        """Submit a request to the scheduler.

        :param jobs: A sequence of job names as strings (case sensitive)
        :param action: The root job action as a string (defaults to :attr:`JobAction.FORCE`)
        :param policy: The scheduling policy as a string (defaults to the schedulers policy)
        :param reverse: Reverse the job dependencies (defaults to the schedulers reverse flag)
        """

        # Put this request in the queue without checking any of the inputs
        self._queue.put_nowait(_Request(jobs, action, policy, reverse))


class DaxScheduler(dax.base.system.DaxHasKey):
    """DAX scheduler class to inherit from.

    Users only have to override class attributes to create a scheduling definition.
    **The scheduler subclass must also inherit from :class:`Experiment` or :class:`EnvExperiment`
    to make the scheduler available as an ARTIQ experiment.**

    The following attributes must be overridden:

    - :attr:`NAME`: The name of this scheduler
    - :attr:`JOBS`: A collection of job classes that form the job set for this scheduler

    Other optional attributes that can be overridden are:

    - :attr:`ROOT_JOBS`: A collection of job classes that are the root jobs, defaults to all entry nodes
    - :attr:`SYSTEM`: A DAX system type to enable additional logging of data
    - :attr:`CONTROLLER`: The scheduler controller name as defined in the device DB
    - :attr:`DEFAULT_SCHEDULING_POLICY`: The default scheduling policy
    - :attr:`DEFAULT_REVERSE_DEPENDENCIES`: The default value for the reverse job dependencies flag
    - :attr:`DEFAULT_JOB_PIPELINE`: The default pipeline to submit jobs to
    - :attr:`DEFAULT_JOB_PRIORITY`: The baseline priority for jobs submitted by this scheduler
    - :attr:`DEFAULT_RESET_JOBS`: The default value for the reset jobs flag
    """

    NAME: str
    """Scheduler name, used as top key."""
    JOBS: typing.Collection[typing.Type[Job]]
    """The collection of job classes."""

    ROOT_JOBS: typing.Collection[typing.Type[Job]] = []
    """The collection of root jobs, all entry nodes if not provided."""
    SYSTEM: typing.Optional[typing.Type[dax.base.system.DaxSystem]] = None
    """Optional DAX system type, enables Influx DB logging if provided."""
    CONTROLLER: typing.Optional[str] = None
    """Optional scheduler controller name, as defined in the device DB."""

    DEFAULT_SCHEDULING_POLICY: Policy = Policy.LAZY
    """Default scheduling policy."""
    DEFAULT_REVERSE_DEPENDENCIES: bool = False
    """Default value for the reverse job dependencies flag."""
    DEFAULT_JOB_PIPELINE: str = 'main'
    """Default pipeline to submit jobs to."""
    DEFAULT_JOB_PRIORITY: int = 0
    """Default baseline priority to submit jobs."""
    DEFAULT_RESET_JOBS: bool = False
    """Default value for the reset jobs flag."""

    _GRAPHVIZ_FORMAT: str = 'pdf'
    """Format specification for the graphviz renderer."""

    def __init__(self, managers_or_parent: typing.Any,
                 *args: typing.Any, **kwargs: typing.Any):
        """Initialize the scheduler object.

        :param managers_or_parent: The manager or parent
        :param args: Positional arguments passed to the superclass
        :param kwargs: Keyword arguments passed to the superclass
        """

        # Check attributes
        self.check_attributes()
        # Call super
        super(DaxScheduler, self).__init__(managers_or_parent, *args,
                                           name=self.NAME, system_key=self.NAME, **kwargs)

    def build(self, *args: typing.Any, **kwargs: typing.Any) -> None:
        """Build the scheduler.

        :param args: Positional arguments forwarded to the super class
        :param kwargs: Keyword arguments forwarded to the super class
        """

        # Set default scheduling options for the scheduler itself
        self.set_default_scheduling(pipeline_name=self.NAME, priority=99)

        # The ARTIQ scheduler
        self._scheduler = self.get_device('scheduler')

        # Instantiate the data store
        if self.SYSTEM is not None and self.SYSTEM.DAX_INFLUX_DB_KEY is not None:
            # Create an Influx DB data store
            self.__data_store = dax.base.system.DaxDataStoreInfluxDb.get_instance(self, self.SYSTEM)
        else:
            # No data store configured
            self.__data_store: dax.base.system.DaxDataStore = dax.base.system.DaxDataStore()

        # Create the job objects
        self._jobs: typing.Dict[typing.Type[Job], Job] = {job: job(self) for job in self.JOBS}
        # Store a map from job names to jobs
        self._job_name_map: typing.Dict[str, Job] = {job.get_name(): job for job in self._jobs.values()}

        # Scheduling arguments
        default_scheduling_policy: str = str(self.DEFAULT_SCHEDULING_POLICY)
        self._policy_arg: str = self.get_argument('Scheduling policy',
                                                  artiq.experiment.EnumerationValue(sorted(str(p) for p in Policy),
                                                                                    default=default_scheduling_policy),
                                                  tooltip='Scheduling policy',
                                                  group='Scheduler')
        self._reverse: bool = self.get_argument('Reverse dependencies',
                                                artiq.experiment.BooleanValue(self.DEFAULT_REVERSE_DEPENDENCIES),
                                                tooltip='Reverse the job dependencies when visiting jobs',
                                                group='Scheduler')
        self._wave_interval: int = self.get_argument('Wave interval',
                                                     artiq.experiment.NumberValue(60, 's', min=1, step=1, ndecimals=0),
                                                     tooltip='Interval to visit jobs',
                                                     group='Scheduler')
        self._clock_period: float = self.get_argument('Clock period',
                                                      artiq.experiment.NumberValue(0.5, 's', min=0.1),
                                                      tooltip='Internal scheduler clock period',
                                                      group='Scheduler')
        if self.CONTROLLER is None:
            self._enable_controller: bool = False
        else:
            self._enable_controller = self.get_argument('Enable controller',
                                                        artiq.experiment.BooleanValue(True),
                                                        tooltip='Enable the scheduler controller',
                                                        group='Scheduler')

        # Job arguments
        self._job_pipeline: str = self.get_argument('Job pipeline',
                                                    artiq.experiment.StringValue(self.DEFAULT_JOB_PIPELINE),
                                                    tooltip='Default pipeline to submit jobs to',
                                                    group='Jobs')
        self._job_priority: int = self.get_argument('Job priority',
                                                    artiq.experiment.NumberValue(self.DEFAULT_JOB_PRIORITY,
                                                                                 min=-99, max=99, step=1, ndecimals=0),
                                                    tooltip='Baseline job priority',
                                                    group='Jobs')
        self._reset_jobs: bool = self.get_argument('Reset jobs',
                                                   artiq.experiment.BooleanValue(self.DEFAULT_RESET_JOBS),
                                                   tooltip='Reset job timestamps at startup, marking them all expired',
                                                   group='Jobs')
        self._terminate_jobs: bool = self.get_argument('Terminate jobs at exit',
                                                       artiq.experiment.BooleanValue(False),
                                                       tooltip='Terminate running jobs at exit',
                                                       group='Jobs')

        # Graph arguments
        self._reduce_graph: bool = self.get_argument('Reduce graph',
                                                     artiq.experiment.BooleanValue(True),
                                                     tooltip='Use transitive reduction of the job dependency graph',
                                                     group='Dependency graph')
        self._view_graph: bool = self.get_argument('View graph',
                                                   artiq.experiment.BooleanValue(False),
                                                   tooltip='View the job dependency graph at startup',
                                                   group='Dependency graph')

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
        if self._scheduler.pipeline_name == self._job_pipeline:
            raise ValueError(f'The scheduler can not run in pipeline "{self._job_pipeline}"')

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
        # Process the job set
        self._process_job_set()

        # Plot the job dependency graph
        self._plot_dependency_graph()
        # Terminate other instances of this scheduler
        self._terminate_running_instances()

    def run(self) -> None:
        """Entry point for the scheduler."""
        asyncio.run(self._async_run())

    async def _async_run(self) -> None:
        """Async entry point for the scheduler."""

        # Create a request queue
        queue: __Q_T = asyncio.Queue()

        if self._enable_controller:
            # Run the scheduler and the controller
            await self._run_scheduler_and_controller(queue)
        else:
            # Only run the scheduler
            await self._run_scheduler(queue)

    async def _run_scheduler(self, queue: __Q_T) -> None:
        """Coroutine for running the scheduler"""

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
                    elif not queue.empty():
                        # Handle requests
                        self.logger.debug('Handling external request')
                        self._handle_external_request(queue)
                    else:
                        # Sleep for a clock period
                        await asyncio.sleep(self._clock_period)

                # Generate the unique wave timestamp
                wave: float = time.time()
                # Start the wave
                self.logger.debug(f'Starting wave {wave:.0f}')
                self.wave(wave=wave,
                          root_jobs=self._root_jobs,
                          root_action=JobAction.PASS,
                          policy=self._policy,
                          reverse=self._reverse)
                # Update next wave time
                next_wave += self._wave_interval
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

    async def _run_scheduler_and_controller(self, queue: __Q_T) -> None:
        """Coroutine for running the scheduler and the controller."""

        # Create the controller and the server objects
        controller = _SchedulerController(queue)
        server = sipyco.pc_rpc.Server({'DaxSchedulerController': controller},
                                      description=f'DaxScheduler controller: {self.get_identifier()}')

        # Start the server task
        host, port = self._get_controller_details()
        self.logger.debug(f'Starting the scheduler controller: bind address {host}, port {port}')
        task = asyncio.create_task(server.start(host, port))
        await task  # Allow the server to start

        try:
            # Run the scheduler
            await self._run_scheduler(queue)
        finally:
            # Stop the server
            self.logger.debug('Stopping the scheduler controller')
            await server.stop()

            if not queue.empty():
                # Warn if not all schedule requests were handled
                self.logger.warning(f'The scheduler controller cancelled {queue.qsize()} request(s)')

    def wave(self, *, wave: float,
             root_jobs: typing.Collection[Job],
             root_action: JobAction,
             policy: Policy,
             reverse: bool) -> None:
        """Run a wave over the job set.

        :param wave: The wave identifier
        :param root_jobs: A collection of root jobs
        :param root_action: The root job action
        :param policy: The policy for this wave
        :param reverse: Reverse the job dependencies
        """
        assert isinstance(wave, float), 'Wave must be of type float'
        assert isinstance(root_jobs, collections.abc.Collection), 'Root jobs must be a collection'
        assert all(isinstance(j, Job) for j in root_jobs), 'All root jobs must be of type Job'
        assert isinstance(root_action, JobAction), 'Root action must be of type JobAction'
        assert isinstance(policy, Policy), 'Policy must be of type Policy'
        assert isinstance(reverse, bool), 'Reverse flag must be of type bool'

        # Select the correct graph for this wave
        graph: nx.DiGraph[Job] = self._job_graph_reversed if reverse else self._job_graph
        # Set of submitted jobs in this wave
        submitted: typing.Set[Job] = set()

        def recurse(job: Job, action: JobAction) -> None:
            """Recurse over the dependencies of a job.

            :param job: The current job to process
            :param action: The action provided by the previous job
            """

            # Visit the current job
            current_action = job.visit(wave=wave)
            # Get the new action based on the policy
            new_action = policy.action(action, current_action)
            # Recurse over successors
            for successor in graph.successors(job):
                recurse(successor, new_action)

            if new_action.submittable() and job not in submitted:
                # Submit this job
                job.submit(wave=wave,
                           pipeline=self._job_pipeline,
                           priority=self._job_priority)
                submitted.add(job)

        for root_job in root_jobs:
            # Recurse over all root jobs using the root action
            recurse(root_job, root_action)

        if submitted:
            # Log submitted jobs
            self.logger.debug(f'Submitted jobs: {", ".join(sorted(j.get_name() for j in submitted))}')

    def _handle_external_request(self, queue: __Q_T) -> None:
        """Handle a single external request from the queue."""

        # Get one element from the queue (there must be an element)
        request: _Request = queue.get_nowait()

        if not isinstance(request.jobs, collections.abc.Collection):
            self.logger.error('Dropping invalid request, jobs parameter is not a collection')
            return

        try:
            # Convert the input parameters
            root_jobs: typing.Collection[Job] = {self._job_name_map[job] for job in request.jobs}
            root_action: JobAction = JobAction.from_str(request.action)
            policy: Policy = self._policy if request.policy is None else Policy.from_str(request.policy)
            reverse: bool = self._reverse if request.reverse is None else request.reverse
        except KeyError:
            # Log the error
            self.logger.exception(f'Dropping invalid request: {request}')
        else:
            # Generate the unique wave timestamp
            wave: float = time.time()
            # Submit a wave for the external request
            self.logger.debug(f'Starting externally triggered wave {wave:.0f}')
            self.wave(wave=wave,
                      root_jobs=root_jobs,
                      root_action=root_action,
                      policy=policy,
                      reverse=reverse)

    def _process_job_set(self) -> None:
        """Process the job set of this scheduler."""

        # Check job set integrity
        self.logger.debug(f'Created {len(self._jobs)} job(s)')
        if len(self._jobs) < len(self.JOBS):
            self.logger.warning('Duplicate jobs were dropped from the job set')
        if len(self._job_name_map) < len(self._jobs):
            msg = 'Job name conflict, two jobs are not allowed to have the same class name'
            self.logger.error(msg)
            raise ValueError(msg)

        # Log the job set
        self.logger.debug(f'Job set: {", ".join(sorted(self._job_name_map))}')

        # Create the job dependency graph
        self._job_graph: nx.DiGraph[Job] = nx.DiGraph()
        self._job_graph.add_nodes_from(self._jobs.values())
        try:
            self._job_graph.add_edges_from(((job, self._jobs[d])
                                            for job in self._jobs.values() for d in job.DEPENDENCIES))
        except KeyError as e:
            raise KeyError(f'Dependency "{e.args[0].get_name()}" is not in the job set') from None

        # Check graph
        if not nx.algorithms.is_directed_acyclic_graph(self._job_graph):
            raise RuntimeError('Dependency graph is not a directed acyclic graph')
        if self._policy is Policy.LAZY and self.CONTROLLER is None and any(not j.is_timed() for j in self._job_graph):
            self.logger.warning('Found unreachable jobs (untimed jobs in a lazy scheduling policy)')

        if self._reduce_graph:
            # Get the transitive reduction of the job dependency graph
            self._job_graph = nx.algorithms.transitive_reduction(self._job_graph)

        # Store reversed graph
        self._job_graph_reversed: nx.DiGraph[Job] = self._job_graph.reverse(copy=False)

        if self.ROOT_JOBS:
            try:
                # Get the root jobs
                self._root_jobs: typing.Set[Job] = {self._jobs[job] for job in self.ROOT_JOBS}
            except KeyError as e:
                raise KeyError(f'Root job "{e.args[0].get_name()}" is not in the job set') from None
        else:
            # Find entry nodes to use as root jobs
            graph: nx.DiGraph[Job] = self._job_graph_reversed if self._reverse else self._job_graph
            # noinspection PyTypeChecker
            self._root_jobs = {job for job, degree in graph.in_degree if degree == 0}

        # Log the root jobs
        self.logger.debug(f'Root jobs: {", ".join(sorted(j.get_name() for j in self._root_jobs))}')

    def _plot_dependency_graph(self) -> None:
        """Plot the job dependency graph."""

        # Create a directed graph object
        plot = graphviz.Digraph(name=self.NAME, directory=str(dax.util.output.get_base_path(self._scheduler)))
        for job in self._job_graph:
            plot.node(job.get_name())
        plot.edges(((j.get_name(), k.get_name()) for j, k in self._job_graph.edges))
        # Render the graph
        plot.render(view=self._view_graph, format=self._GRAPHVIZ_FORMAT)

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

    def _get_controller_details(self) -> typing.Tuple[str, int]:
        """Get the scheduler controller details from the device DB and verify the details."""

        assert isinstance(self.CONTROLLER, str), 'Controller attribute must be of types str'

        # Obtain the device DB
        device_db = self.get_device_db()

        try:
            # Try to get the controller entry
            controller = device_db[self.CONTROLLER]
        except KeyError:
            raise KeyError(f'Could not find controller "{self.CONTROLLER}" in the device DB') from None
        try:
            # Try to get the host and the port
            host: str = controller['host']
            port: int = controller['port']
        except KeyError as e:
            raise KeyError(f'Could not obtain host and port information for controller "{self.CONTROLLER}"') from e

        # Check obtained values
        if not isinstance(host, str):
            raise TypeError(f'Host information for controller "{self.CONTROLLER}" must be of type str')
        if not isinstance(port, int):
            raise TypeError(f'Port information for controller "{self.CONTROLLER}" must be of type str')

        # Check that best effort is disabled
        if controller.get('best_effort'):
            raise ValueError(f'Controller "{self.CONTROLLER}" should have best effort disabled')

        # Verify if no controller is already running
        try:
            self.get_device(self.CONTROLLER)
        except artiq.master.worker_db.DeviceError:
            pass  # No server is running
        else:
            raise RuntimeError(f'Controller "{self.CONTROLLER}" is already running')

        # Return the controller details of interest
        return host, port

    @classmethod
    def check_attributes(cls) -> None:
        """Check if all attributes of this class are correctly overridden.

        :raises AssertionError: Raised if attributes are not correctly overridden
        """

        # Check name
        assert hasattr(cls, 'NAME'), 'No name was provided'
        # Check jobs
        assert hasattr(cls, 'JOBS'), 'No job list was provided'
        assert isinstance(cls.JOBS, collections.abc.Collection), 'The jobs attribute must be a collection'
        assert all(issubclass(job, Job) for job in cls.JOBS), 'All jobs must be subclasses of Job'
        # Check root jobs, system, and controller
        assert isinstance(cls.ROOT_JOBS, collections.abc.Collection), 'The root jobs attribute must be a collection'
        assert all(issubclass(job, Job) for job in cls.ROOT_JOBS), 'All root jobs must be subclasses of Job'
        assert cls.SYSTEM is None or issubclass(cls.SYSTEM, dax.base.system.DaxSystem), \
            'The provided system must be a subclass of DaxSystem or None'
        assert cls.CONTROLLER is None or isinstance(cls.CONTROLLER, str), 'Controller must be of type str or None'
        assert cls.CONTROLLER != 'scheduler', 'Controller can not be "scheduler" (aliases with the ARTIQ scheduler)'
        # Check default policy, reverse, pipeline, job priority, and reset jobs flag
        assert isinstance(cls.DEFAULT_SCHEDULING_POLICY, Policy), 'Default policy must be of type Policy'
        assert isinstance(cls.DEFAULT_REVERSE_DEPENDENCIES, bool), \
            'Default reverse dependencies flag must be of type bool'
        assert isinstance(cls.DEFAULT_JOB_PIPELINE, str) and cls.DEFAULT_JOB_PIPELINE, \
            'Default job pipeline must be of type str'
        assert isinstance(cls.DEFAULT_JOB_PRIORITY, int), 'Default job priority must be of type int'
        assert -99 <= cls.DEFAULT_JOB_PRIORITY <= 99, 'Default job priority must be in the domain [-99, 99]'
        assert isinstance(cls.DEFAULT_RESET_JOBS, bool), 'Default reset jobs flag must be of type bool'
        # Check graphviz format
        assert isinstance(cls._GRAPHVIZ_FORMAT, str), 'Graphviz format must be of type str'


class _DaxSchedulerClient(dax.base.system.DaxBase, artiq.experiment.Experiment):
    """A client experiment class for a scheduler."""

    SCHEDULER_NAME: str
    """The name of the scheduler."""
    JOB_NAMES: typing.List[str]
    """A List with the names of the jobs."""
    CONTROLLER_KEY: str
    """Key of the scheduler controller."""

    _JOB_ACTION_NAMES: typing.List[str] = sorted(str(a) for a in JobAction)
    """A list with job action names."""
    _SCHEDULER_POLICY: str = '<Scheduler policy>'
    """The policy option to use the schedulers policy."""
    _POLICY_NAMES: typing.List[str] = [_SCHEDULER_POLICY] + sorted(str(p) for p in Policy)
    """A list with policy names."""
    _SCHEDULER_REVERSE: str = '<Scheduler reverse>'
    """The reverse job dependencies option to use the schedulers reverse flag."""
    _REVERSE_DICT: typing.Dict[str, typing.Optional[bool]] = {_SCHEDULER_REVERSE: None,
                                                              'False': False, 'True': True}
    """A dict with reverse job dependencies names and values."""

    def build(self) -> None:  # type: ignore
        # Set default scheduling options for the client
        self.set_default_scheduling(pipeline_name=f'_{self.SCHEDULER_NAME}')

        # Arguments
        self._job: str = self.get_argument('Job',
                                           artiq.experiment.EnumerationValue(self.JOB_NAMES),
                                           tooltip='Job to submit')
        self._action: str = self.get_argument('Action',
                                              artiq.experiment.EnumerationValue(self._JOB_ACTION_NAMES,
                                                                                default=str(JobAction.FORCE)),
                                              tooltip='Initial job action')
        self._policy: str = self.get_argument('Policy',
                                              artiq.experiment.EnumerationValue(self._POLICY_NAMES,
                                                                                default=self._SCHEDULER_POLICY),
                                              tooltip='Scheduling policy')
        self._reverse: str = self.get_argument('Reverse',
                                               artiq.experiment.EnumerationValue(sorted(self._REVERSE_DICT),
                                                                                 default=self._SCHEDULER_REVERSE),
                                               tooltip='Reverse the job dependencies when visiting jobs')

        # Get the DAX scheduler controller
        self._dax_scheduler: _SchedulerController = self.get_device(self.CONTROLLER_KEY)
        # Get the ARTIQ scheduler
        self._scheduler = self.get_device('scheduler')

    def prepare(self) -> None:
        # Check pipeline (loosely checked)
        if self._scheduler.pipeline_name == self.SCHEDULER_NAME:
            self.logger.warning('The scheduler client should not be submitted to the same pipeline as the scheduler')

    def run(self) -> None:
        # Submit the request
        self.logger.info(f'Submitting request: job={self._job}, action={self._action}, '
                         f'policy={self._policy}, reverse={self._reverse}')
        self._dax_scheduler.submit(self._job,
                                   action=self._action,
                                   policy=None if self._policy == self._SCHEDULER_POLICY else self._policy,
                                   reverse=self._REVERSE_DICT[self._reverse])

    def get_identifier(self) -> str:
        """Return the identifier of this scheduler client."""
        return f'[{self.SCHEDULER_NAME}]({self.__class__.__name__})'


def dax_scheduler_client(scheduler_class: typing.Type[DaxScheduler]) -> typing.Type[_DaxSchedulerClient]:
    """Decorator to generate a client experiment class from a :class:`DaxScheduler` class.

    The client experiment can be used to manually trigger jobs.

    :param scheduler_class: The scheduler class
    :return: An instantiated client experiment class
    """

    assert issubclass(scheduler_class, DaxScheduler), 'The scheduler class must be a subclass of DaxScheduler'
    scheduler_class.check_attributes()
    assert isinstance(scheduler_class.CONTROLLER, str), 'The scheduler class must have a controller key'

    class WrapperClass(_DaxSchedulerClient):
        """A wrapped/instantiated client experiment class for a scheduler."""

        SCHEDULER_NAME = scheduler_class.NAME
        JOB_NAMES = sorted(j.get_name() for j in scheduler_class.JOBS)
        if isinstance(scheduler_class.CONTROLLER, str):
            CONTROLLER_KEY = scheduler_class.CONTROLLER
        else:
            raise TypeError('The scheduler class must have a valid controller key to generate a client')

    # Return the wrapped client class
    return WrapperClass
