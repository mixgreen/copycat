from __future__ import annotations  # Postponed evaluation of annotations

import abc
import typing
import logging
import collections.abc
import time
import enum
import networkx as nx
import asyncio
import json
import hashlib
import dataclasses
import os
import os.path
import inspect
import pathlib

import artiq.experiment
import artiq.master.worker_db
import artiq.tools
import sipyco.pc_rpc
from sipyco import pyon

import dax.base.system
import dax.base.exceptions
import dax.util.output
import dax.util.artiq
import dax.util.experiments

__all__ = ['NodeAction', 'Policy', 'Job', 'Trigger',
           'CalibrationJob', 'create_calibration',
           'DaxScheduler', 'dax_scheduler_client']

_unit_to_seconds: typing.Dict[str, float] = {
    's': 1.0,
    'm': 60.0,
    'h': 3600.0,
    'd': 86400.0,
    'w': 604800.0,
}
"""Dict to map a string time unit to seconds."""


def _str_to_time(string: str) -> float:
    """Convert a string to a time in seconds.

    This function converts simple time strings with a number and a character to a time in seconds.
    The available units are:

    - ``'s'`` for seconds
    - ``'m'`` for minutes
    - ``'h'`` for hours
    - ``'d'`` for days
    - ``'w'`` for weeks

    Examples of valid strings are ``'10 s'``, ``'-2m'``, ``'8h'``, and ``'2.5 w'``.
    An empty string will return zero.

    :param string: The string
    :return: Time in seconds as a float
    :raises ValueError: Raised if the number or the unit is invalid
    """
    assert isinstance(string, str), 'Input must be of type str'

    # Format string
    string = string.strip()

    if string:
        try:
            # Get the value and the unit
            value = float(string[:-1])
            unit = _unit_to_seconds[string[-1]]
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


class NodeAction(enum.Enum):
    """Node action enumeration."""

    PASS = enum.auto()
    """Pass this node."""
    RUN = enum.auto()
    """Run this node."""
    FORCE = enum.auto()
    """Force this node."""

    def submittable(self) -> bool:
        """Check if a node should be submitted or not.

        :return: True if the node is submittable
        """
        return self in {NodeAction.RUN, NodeAction.FORCE}

    @classmethod
    def from_str(cls, string_: str) -> NodeAction:
        """Convert a string into its corresponding node action enumeration.

        :param string_: The name of the node action as a string (case insensitive)
        :return: The node action enumeration object
        :raises KeyError: Raised if the node action name does not exist
        """
        return {str(p).lower(): p for p in cls}[string_.lower()]

    def __str__(self) -> str:
        """String representation of this node action.

        :return: The name of the node action
        """
        return self.name


class Policy(enum.Enum):
    """Policy enumeration for the scheduler.

    The policy enumeration includes definitions for the policies using a mapping table.
    """

    if typing.TYPE_CHECKING:
        # Only add type when type checking is enabled to not conflict with iterations over the Policy enum
        __P_T = typing.Dict[typing.Tuple[NodeAction, NodeAction], NodeAction]  # Policy enum type

    LAZY: __P_T = {
        (NodeAction.PASS, NodeAction.PASS): NodeAction.PASS,
        (NodeAction.PASS, NodeAction.RUN): NodeAction.RUN,
        (NodeAction.PASS, NodeAction.FORCE): NodeAction.RUN,
        (NodeAction.RUN, NodeAction.PASS): NodeAction.PASS,
        (NodeAction.RUN, NodeAction.RUN): NodeAction.RUN,
        (NodeAction.RUN, NodeAction.FORCE): NodeAction.RUN,
        (NodeAction.FORCE, NodeAction.PASS): NodeAction.RUN,
        (NodeAction.FORCE, NodeAction.RUN): NodeAction.RUN,
        (NodeAction.FORCE, NodeAction.FORCE): NodeAction.RUN,
    }
    """Lazy scheduling policy."""

    GREEDY: __P_T = {
        (NodeAction.PASS, NodeAction.PASS): NodeAction.PASS,
        (NodeAction.PASS, NodeAction.RUN): NodeAction.RUN,
        (NodeAction.PASS, NodeAction.FORCE): NodeAction.RUN,
        (NodeAction.RUN, NodeAction.PASS): NodeAction.RUN,
        (NodeAction.RUN, NodeAction.RUN): NodeAction.RUN,
        (NodeAction.RUN, NodeAction.FORCE): NodeAction.RUN,
        (NodeAction.FORCE, NodeAction.PASS): NodeAction.RUN,
        (NodeAction.FORCE, NodeAction.RUN): NodeAction.RUN,
        (NodeAction.FORCE, NodeAction.FORCE): NodeAction.RUN,
    }
    """Greedy scheduling policy."""

    def action(self, previous: NodeAction, current: NodeAction) -> NodeAction:
        """Apply the policy on two node actions.

        :param previous: The node action of the predecessor (previous node)
        :param current: The current node action
        :return: The new node action based on this policy
        """
        assert isinstance(previous, NodeAction)
        assert isinstance(current, NodeAction)
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
    nodes: typing.Collection[str]
    action: str
    policy: typing.Optional[str]
    reverse: typing.Optional[bool]
    priority: typing.Optional[int]
    depth: int
    start_depth: int


# Workaround required for Python<3.9
if typing.TYPE_CHECKING:
    _RQ_T = asyncio.Queue[_Request]  # Type for the request queue
else:
    _RQ_T = asyncio.Queue


class Node(dax.base.system.DaxHasKey, abc.ABC):
    """Abstract node class for the scheduler."""

    INTERVAL: typing.ClassVar[typing.Optional[str]] = None
    """Interval to run this node, defaults to no interval."""
    DEPENDENCIES: typing.ClassVar[typing.Collection[typing.Type[Node]]] = []
    """Collection of node dependencies."""

    _LAST_SUBMIT_KEY: typing.ClassVar[str] = 'last_submit'
    """Key to store the last submit timestamp."""

    _dependencies: typing.FrozenSet[typing.Type[Node]]
    _interval: float
    _next_submit: float

    def __init__(self, managers_or_parent: DaxScheduler,
                 *args: typing.Any, **kwargs: typing.Any):
        """Initialize the node object.

        :param managers_or_parent: The manager or parent, must be of type :class:`DaxScheduler`
        :param args: Positional arguments passed to the superclass
        :param kwargs: Keyword arguments passed to the superclass
        """

        # Check interval and dependencies
        assert isinstance(self.INTERVAL, str) or self.INTERVAL is None, 'Interval must be of type str or None'
        assert isinstance(self.DEPENDENCIES, collections.abc.Collection), 'The dependencies must be a collection'
        assert all(issubclass(node, Node) for node in self.DEPENDENCIES), 'All dependencies must be subclasses of Node'

        # Check parent
        if not isinstance(managers_or_parent, DaxScheduler):
            raise TypeError(f'Parent of node "{self.get_name()}" is not a DAX scheduler')

        # Take key attributes from parent
        self._take_parent_key_attributes(managers_or_parent)

        # Call super
        super(Node, self).__init__(managers_or_parent, *args,
                                   name=self.get_name(), system_key=managers_or_parent.get_system_key(self.get_name()),
                                   **kwargs)

    def build(self) -> None:  # type: ignore
        """Build the node object."""

        # Process dependencies
        self._dependencies = frozenset(self.DEPENDENCIES)
        if len(self._dependencies) < len(self.DEPENDENCIES):
            self.logger.warning('Duplicate dependencies were dropped')

        # Obtain the scheduler
        self._scheduler = self.get_device('scheduler')

        # Convert the interval
        if self.is_timed():
            # Convert the interval string
            self._interval = _str_to_time(typing.cast(str, self.INTERVAL))
            # Check the value
            if self._interval > 0.0:
                self.logger.info(f'Interval set to {self._interval:.0f} second(s)')
            else:
                raise ValueError(f'The interval of node "{self.get_name()}" must be greater than zero')
        else:
            # No interval was set
            self._interval = 0.0
            self.logger.info('No interval set')

    def init(self, *, reset: bool, **_: typing.Any) -> None:
        """Initialize the node, called once before the scheduler starts.

        :param reset: Reset the state of this node
        """
        assert isinstance(reset, bool), 'The reset flag must be of type bool'

        if self.is_timed():
            self.logger.debug('Initializing timed node')
            if reset:
                # Reset the node by resetting the next submit time
                self._next_submit = time.time()
                self.logger.debug('Node reset requested')
            else:
                # Try to obtain the last submit time
                last_submit = self.get_dataset_sys(self._LAST_SUBMIT_KEY, 0.0, data_store=False)
                assert isinstance(last_submit, float), 'Unexpected type returned from dataset'
                # Add the interval to the last submit time
                self._next_submit = last_submit + self._interval
                self.logger.debug('Loaded last submit timestamp')
        else:
            # This node is untimed, next submit will not happen
            self._next_submit = float('inf')
            self.logger.debug('Initialized untimed node')

    def visit(self, *, wave: float) -> NodeAction:
        """Visit this node.

        :param wave: Wave identifier
        :return: The action for this node
        """
        assert isinstance(wave, float), 'Wave must be of type float'

        if self._next_submit <= wave:
            # Interval expired
            return NodeAction.RUN
        else:
            # Interval did not expire
            return NodeAction.PASS

    def submit(self, *, wave: float, priority: int) -> None:
        """Submit this node.

        :param wave: Wave identifier
        :param priority: Submit priority
        """
        assert isinstance(wave, float), 'Wave must be of type float'
        assert isinstance(priority, int), 'Priority must be of type int'

        # Store current wave timestamp
        self.set_dataset_sys(self._LAST_SUBMIT_KEY, wave, data_store=False)

        if not self.is_meta():
            # Submit
            self._submit(wave=wave, priority=priority)

        # Reschedule node
        self.schedule(wave=wave)

    @abc.abstractmethod
    def _submit(self, *, wave: float, priority: int) -> None:
        pass

    def schedule(self, *, wave: float) -> None:
        """Schedule this node.

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
                self.logger.warning('Next submit was rescheduled because the interval time expired')

    @abc.abstractmethod
    def cancel(self) -> None:
        """Cancel the submit action of this node."""
        pass

    def is_timed(self) -> bool:
        """Check if this node is timed.

        :return: True if this node has an interval
        """
        return self.INTERVAL is not None

    @abc.abstractmethod
    def is_meta(self) -> bool:
        """Check if this node is a meta-node (i.e. a node without submittable action).

        :return: True if this node is a meta-node
        """
        pass

    def get_dependencies(self) -> typing.FrozenSet[typing.Type[Node]]:
        """Get the dependencies of this node.

        :return: A set of node on which this node depends
        """
        return self._dependencies

    @classmethod
    def get_name(cls) -> str:
        """Get the name of this node.

        :return: The name of this node as a string
        """
        return cls.__name__


class BaseJob(Node, abc.ABC):
    """Base class for jobs."""

    LOG_LEVEL: typing.ClassVar[int] = logging.WARNING
    """The log level for the experiment."""
    PIPELINE: typing.ClassVar[typing.Optional[str]] = None
    """The pipeline to submit this job to, defaults to the pipeline assigned by the scheduler."""
    PRIORITY: typing.ClassVar[int] = 0
    """Job priority relative to the base job priority of the scheduler."""
    FLUSH: typing.ClassVar[bool] = False
    """The flush flag when submitting a job."""
    REPOSITORY: typing.ClassVar[bool] = True
    """True if the given file(s) is (are) in the experiment repository."""

    _RID_LIST_KEY: typing.ClassVar[str] = 'rid_list'
    """Key to store every submitted RID."""
    _ARGUMENTS_HASH_KEY: typing.ClassVar[str] = 'arguments_hash'
    """Key to store the last arguments hash."""

    _pipeline: str
    _last_rid: int
    _rid_list_key: str

    def build(self) -> None:  # type: ignore
        # Check log level, pipeline, priority, flush, and repository
        assert isinstance(self.LOG_LEVEL, int), 'Log level must be of type int'
        assert self.PIPELINE is None or isinstance(self.PIPELINE, str), 'Pipeline must be of type str or None'
        assert isinstance(self.PRIORITY, int), 'Priority must be of type int'
        assert isinstance(self.FLUSH, bool), 'Flush must be of type bool'
        assert isinstance(self.REPOSITORY, bool), 'Repository flag must be of type bool'

        # Type attributes
        self._arguments: typing.Dict[str, typing.Any]
        self._expid: typing.Dict[str, typing.Any]

        # Call super
        super(BaseJob, self).build()

    def init(self, *, reset: bool, **kwargs: typing.Any) -> None:
        # Check if attributes are populated
        assert hasattr(self, '_arguments')
        assert hasattr(self, '_expid')

        # Check parameters
        assert isinstance(reset, bool)
        # Extract arguments
        self._pipeline = kwargs['job_pipeline'] if self.PIPELINE is None else self.PIPELINE
        assert isinstance(self._pipeline, str) and self._pipeline

        # Initialize last RID to a non-existing value
        self._last_rid = -1

        # Initialize RID list
        self._rid_list_key = self.get_system_key(self._RID_LIST_KEY)
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
            self.logger.debug(f'Forcing reset because job arguments changed: {arguments_hash}')
            self.set_dataset_sys(self._ARGUMENTS_HASH_KEY, arguments_hash, data_store=False)
            # Force a reset
            reset = True

        # Call super
        super(BaseJob, self).init(reset=reset, **kwargs)

    def _submit(self, *, wave: float, priority: int) -> None:
        status = self._scheduler.get_status()
        if self._last_rid not in status or status[self._last_rid]['status'] in {'run_done', 'analyzing', 'deleting'}:
            # Submit experiment of this job
            self._last_rid = self._scheduler.submit(
                pipeline_name=self._pipeline,
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

    def cancel(self) -> None:
        if self._last_rid >= 0:
            # Cancel the last RID (returns if the RID is not running)
            self.logger.debug(f'Cancelling job (RID {self._last_rid})')
            self._scheduler.request_termination(self._last_rid)


class Job(BaseJob):
    """Job class to define a job for the scheduler.

    Users only have to override class attributes to create a job definition.
    The following main attributes can be overridden:

    - :attr:`FILE`: The file name containing the experiment
    - :attr:`CLASS_NAME`: The class name of the experiment
    - :attr:`ARGUMENTS`: A dictionary with experiment arguments (scan objects can be used directly as arguments)
    - :attr:`Node.INTERVAL`: The submit interval
    - :attr:`Node.DEPENDENCIES`: A collection of node classes on which this job depends

    Optionally, users can override the :func:`build_job` method to add configurable arguments.
    """

    FILE: typing.ClassVar[typing.Optional[str]] = None
    """File containing the experiment (by default relative from the `repository` directory,
    see :attr:`BaseJob.REPOSITORY`)."""
    CLASS_NAME: typing.ClassVar[typing.Optional[str]] = None
    """Class name of the experiment."""
    ARGUMENTS: typing.Dict[str, typing.Any] = {}
    """The experiment arguments."""

    _arguments: typing.Dict[str, typing.Any]
    _expid: typing.Dict[str, typing.Any]

    def build(self) -> None:  # type: ignore
        """Build the job object.

        To add configurable arguments to this job, override the :func:`build_job` method instead.
        """
        # Check file, repository flag, class, and arguments
        assert isinstance(self.FILE, str) or self.FILE is None, 'The file attribute must be of type str or None'
        assert isinstance(self.CLASS_NAME, str) or self.CLASS_NAME is None, \
            'The class name attribute must be of type str or None'
        assert isinstance(self.ARGUMENTS, dict), 'The arguments must be of type dict'
        assert all(isinstance(k, str) for k in self.ARGUMENTS), 'All argument keys must be of type str'

        # Call super
        super(Job, self).build()

        # Copy the class arguments attribute (prevents class attribute from being mutated)
        self.ARGUMENTS = self.ARGUMENTS.copy()
        # Build this job
        self.build_job()
        # Process the arguments
        self._arguments = dax.util.artiq.process_arguments(self.ARGUMENTS)

        if self.FILE is not None and not self.REPOSITORY:
            # Expand file
            expanded_file: str = os.path.expanduser(self.FILE)
            if not os.path.isabs(expanded_file):
                raise ValueError('The given file must be an absolute path')
            file: typing.Optional[str] = expanded_file
        else:
            file = self.FILE

        # Construct an expid for this job
        if not self.is_meta():
            self._expid = {
                'file': file,
                'class_name': self.CLASS_NAME,
                'arguments': self._arguments,
                'log_level': self.LOG_LEVEL,
            }
            if self.REPOSITORY:
                # Request current revision and interpret file location relative from the repository
                self._expid['repo_rev'] = None
            self.logger.debug(f'expid: {self._expid}')
        else:
            self._expid = {}
            self.logger.debug('This job is a meta-job')

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

    def is_meta(self) -> bool:
        # Decide if this job is a meta-job
        attributes = (self.FILE, self.CLASS_NAME)
        meta = all(a is None for a in attributes)

        if meta or all(a is not None for a in attributes):
            return meta
        else:
            # Attributes not consistent, raise an exception
            raise ValueError(f'The FILE and CLASS_NAME attributes of job "{self.get_name()}" '
                             f'should both be None or not None')


class Trigger(Node):
    """Trigger class to define a trigger for the scheduler.

    Users only have to override class attributes to create a trigger definition.
    The following main attributes can be overridden:

    - :attr:`NODES`: A collection of nodes to trigger
    - :attr:`ACTION`: The root node action of this trigger (defaults to :attr:`NodeAction.FORCE`)
    - :attr:`POLICY`: The scheduling policy of this trigger (defaults to the schedulers policy)
    - :attr:`REVERSE`: The reverse wave flag of this trigger (defaults to the schedulers reverse wave flag)
    - :attr:`PRIORITY`: The job priority of this trigger (defaults to the schedulers job priority)
    - :attr:`DEPTH`: Maximum recursion depth (:const:`-1` for infinite recursion depth, which is the default)
    - :attr:`START_DEPTH`: Depth to start visiting nodes (:const:`0` to start at the root nodes, which is the default)
    - :attr:`Node.INTERVAL`: The trigger interval
    - :attr:`Node.DEPENDENCIES`: A collection of node classes on which this trigger depends
    """

    NODES: typing.ClassVar[typing.Collection[typing.Type[Node]]] = []
    """Collection of nodes to trigger."""
    ACTION: typing.ClassVar[NodeAction] = NodeAction.FORCE
    """The root node action of this trigger."""
    POLICY: typing.ClassVar[typing.Optional[Policy]] = None
    """The scheduling policy of this trigger."""
    REVERSE: typing.ClassVar[typing.Optional[bool]] = None
    """The reverse wave flag for this trigger."""
    PRIORITY: typing.ClassVar[typing.Optional[int]] = None
    """The job priority of this trigger."""
    DEPTH: typing.ClassVar[int] = -1
    """Maximum recursion depth (:const:`-1` for infinite recursion depth)."""
    START_DEPTH: typing.ClassVar[int] = 0
    """Depth to start visiting nodes (:const:`0` to start at the root nodes)."""

    _nodes: typing.Set[str]
    _request_queue: _RQ_T

    def build(self) -> None:  # type: ignore
        # Check nodes, action, policy, and reverse flag
        assert isinstance(self.NODES, collections.abc.Collection), 'The nodes must be a collection'
        assert all(issubclass(node, Node) for node in self.NODES), 'All nodes must be subclasses of Node'
        assert isinstance(self.ACTION, NodeAction), 'Action must be of type NodeAction'
        assert isinstance(self.POLICY, Policy) or self.POLICY is None, 'Policy must be of type Policy or None'
        assert isinstance(self.REVERSE, bool) or self.REVERSE is None, 'Reverse must be of type bool or None'
        assert isinstance(self.PRIORITY, int) or self.PRIORITY is None, 'Priority must be of type int or None'
        assert isinstance(self.DEPTH, int), 'Depth must be of type int'
        assert isinstance(self.START_DEPTH, int), 'Start depth must be of type int'

        # Assemble the collection of node names
        self._nodes = {node.get_name() for node in self.NODES}
        if len(self._nodes) < len(self.NODES):
            self.logger.warning('Duplicate nodes were dropped')

        # Call super
        super(Trigger, self).build()

    def init(self, *, reset: bool, **kwargs: typing.Any) -> None:
        # Extract attributes
        self._request_queue = kwargs['request_queue']
        assert isinstance(self._request_queue, asyncio.Queue)

        # Call super
        super(Trigger, self).init(reset=reset, **kwargs)

    def _submit(self, *, wave: float, priority: int) -> None:
        # Add the request to the request queue
        self._request_queue.put_nowait(_Request(nodes=self._nodes,
                                                action=str(self.ACTION),
                                                policy=str(self.POLICY) if self.POLICY is not None else None,
                                                reverse=self.REVERSE,
                                                priority=self.PRIORITY,
                                                depth=self.DEPTH,
                                                start_depth=self.START_DEPTH))
        self.logger.info('Submitted trigger')

    def cancel(self) -> None:
        # Triggers can not be cancelled
        pass

    def is_meta(self) -> bool:
        # This trigger is a meta-trigger if there are no nodes to trigger
        return len(self.NODES) == 0


class CalibrationJob(BaseJob):
    """Meta-class to create a job for the scheduler that wraps two experiments.

    The two experiments are intended to be ``check_data`` and ``calibrate`` experiments
    as detailed in https://arxiv.org/abs/1803.03226.

    In order to 'communicate' the results of each experiment, certain exceptions must be raised, corresponding to the
    possible results of the experiments as detailed in the above paper.

    For ``check_data``:

    - :class:`dax.base.exceptions.OutOfSpecError`: raise in the case that the data is out of spec and the parameter
      should be re-calibrated.
    - :class:`dax.base.exceptions.BadDataError`: raise in the case of bad data (i.e. if you suspect that there is some
      condition that would prevent the calibration from being resolvable).

    For ``calibrate``:

    - :class:`dax.base.exceptions.FailedCalibrationError`: raise in the case that the calibration has failed without
      resolution. Ideally, this should happen very rarely, as most conditions that would cause a calibration to fail
      should be resolved by ``diagnose``, which is triggered by raising a BadDataError in the ``check_data`` experiment.

    Users only have to override class attributes to create a job definition.
    The following main attributes **must** be overridden:

    - :attr:`CHECK_FILE`: The file containing the check experiment
    - :attr:`CALIBRATION_FILE`: The file containing the calibration experiment
    - :attr:`CHECK_CLASS_NAME`: The class name of the check experiment
    - :attr:`CALIBRATION_CLASS_NAME`: The class name of the calibration experiment

    The following attributes can optionally be overridden:

    - :attr:`CHECK_ARGUMENTS`: A dictionary with experiment arguments for the `check_data` experiment
    - :attr:`CALIBRATION_ARGUMENTS`: A dictionary with experiment arguments for the `calibrate` experiment
    - :attr:`CALIBRATION_TIMEOUT`: The timeout period for the `check_state` phase (as specified by the optimus
      algorithm). If set to :const:`None` (default), the calibration will always go straight to the `check_data` phase.
    - :attr:`GREEDY_FAILURE`: Whether or not to fail 'greedily'. In other words, if an experiment fails due to an
      uncaught exception (not a calibration exception), whether to schedule the :class:`dax.util.experiments.Barrier`
      experiment to prevent other experiments from executing. Defaults to :const:`True`. If :const:`False`, the
      experiment in question will fail and the rest of the scheduled experiments will be unaffected.
    - :attr:`Node.INTERVAL`: The submit interval
    - :attr:`Node.DEPENDENCIES`: A collection of node classes on which this job depends

    Optionally, users can override the :func:`build_job` method to add configurable arguments.

    **Notes regarding intended use**:

    - In order to exactly implement the 'Optimus' algorithm, :class:`CalibrationJob`\\ s should not specify an
      :attr:`Node.INTERVAL` (using :attr:`CALIBRATION_TIMEOUT` instead). The root node(s) of the calibration graph
      should instead be submitted by a :class:`Trigger` with a :attr:`Policy.GREEDY` :attr:`Trigger.POLICY`, with
      whatever :attr:`Node.INTERVAL` is desired.
    - :class:`Trigger`\\ s can also be added to submit a subgraph of the calibration graph.
    - :class:`Job`\\ s **can** be present in the calibration graph. Due to the :attr:`Policy.GREEDY` :class:`Trigger`,
      they will be submitted along with the :class:`CalibrationJob`\\ s. However, in the case of a ``diagnose`` wave,
      those :class:`Job`\\ s will not be submitted as they are not taken into account by the 'Optimus' algorithm.
    """

    CHECK_FILE: typing.ClassVar[str]
    """File containing the check experiment."""
    CALIBRATION_FILE: typing.ClassVar[str]
    """File containing the calibration experiment."""
    CHECK_CLASS_NAME: typing.ClassVar[str]
    """Class name of check experiment."""
    CALIBRATION_CLASS_NAME: typing.ClassVar[str]
    """Class name of calibration experiment."""
    CHECK_ARGUMENTS: typing.Dict[str, typing.Any] = {}
    """The experiment arguments."""
    CALIBRATION_ARGUMENTS: typing.Dict[str, typing.Any] = {}
    """The experiment arguments."""
    CALIBRATION_TIMEOUT: typing.ClassVar[typing.Optional[str]] = None
    """Calibration timeout period. This is should be used instead of :attr:`Node.INTERVAL`."""
    GREEDY_FAILURE: typing.ClassVar[bool] = True
    """Whether or not to fail 'greedily'. In other words, if an experiment fails due to an uncaught exception (not a
     calibration exception), whether to schedule the :class:`dax.util.experiments.Barrier` experiment to prevent
     other experiments from executing. Defaults to :const:`True`. If :const:`False`, the experiment in question will
     fail and the rest of the scheduled experiments will be unaffected."""

    # Public so that meta exp can access them
    LAST_CAL_KEY: typing.ClassVar[str] = 'last_calibration'
    """Key to store the last time the calibration was successfully run."""
    LAST_CHECK_KEY: typing.ClassVar[str] = 'last_check'
    """Key to store the last time `check_data` passed."""
    DIAGNOSE_FLAG_KEY: typing.ClassVar[str] = 'diagnose'
    """Key to store a flag that tells experiment to run diagnose instead of maintain (i.e. skip ``check_state``)."""

    _META_EXP_FILE: typing.ClassVar[str]
    """Path to the file that the meta experiment is inserted into. Should only be written to via the decorator calling
    ``build_meta_exp()``."""

    _expid: typing.Dict[str, typing.Any]

    @classmethod
    def _meta_exp_name(cls) -> str:
        return f'_{cls.__name__}MetaExp'

    @classmethod  # noqa: C901
    def build_meta_exp(cls, file: str) -> typing.Tuple[typing.Type[artiq.experiment.Experiment], str]:  # noqa: C901
        """Build the meta-experiment class."""

        # set file attribute to use when submitting the meta experiment
        cls._META_EXP_FILE = file

        class MetaExp(dax.base.system.DaxBase, artiq.experiment.Experiment):  # pragma: no cover
            """The generated meta-experiment class."""

            _controller_key: str
            _my_dataset_keys: typing.Dict[str, str]
            _dep_dataset_keys: typing.Dict[str, typing.Dict[str, str]]
            _timeout: float
            _check_exp: typing.Any
            _check_exception: typing.Optional[Exception]
            _check_analyze: bool
            _cal_analyze: bool
            _cal_cls: typing.Any
            _calibration_exp: typing.Any

            def __init__(self, managers_or_parent: typing.Any, *args: typing.Any, **kwargs: typing.Any):
                _, _, argument_mgr, _ = managers_or_parent
                self._check_args = argument_mgr.unprocessed_arguments.pop('check')
                self._check_managers = dax.util.artiq.clone_managers(
                    managers_or_parent,
                    arguments=self._check_args
                )
                self._cal_args = argument_mgr.unprocessed_arguments.pop('calibration')
                self._cal_managers = dax.util.artiq.clone_managers(
                    managers_or_parent,
                    arguments=self._cal_args
                )
                self._controller_key = argument_mgr.unprocessed_arguments.pop('controller_key')
                self._my_dataset_keys = argument_mgr.unprocessed_arguments.pop('my_dataset_keys')
                self._dep_dataset_keys = argument_mgr.unprocessed_arguments.pop('dep_dataset_keys')
                self._timeout = argument_mgr.unprocessed_arguments.pop('timeout')
                check_file = argument_mgr.unprocessed_arguments.pop('check_file')
                cal_file = argument_mgr.unprocessed_arguments.pop('calibration_file')
                check_mod = artiq.tools.file_import(check_file, prefix='')
                cal_mod = artiq.tools.file_import(cal_file, prefix='')
                check_cls = artiq.tools.get_experiment(check_mod, cls.CHECK_CLASS_NAME)
                self._cal_cls = artiq.tools.get_experiment(cal_mod, cls.CALIBRATION_CLASS_NAME)
                # need the HasEnvironment constructor as well as the usual Experiment methods
                assert issubclass(check_cls, artiq.experiment.HasEnvironment)
                assert issubclass(self._cal_cls, artiq.experiment.HasEnvironment)
                # mypy/python doesn't support "intersection" style typing (yet), so just have to use typing.Any
                try:
                    self._check_exp = check_cls(self._check_managers)
                except Exception as e:
                    self._check_exception = e
                else:
                    self._check_exception = None
                # flags for analyze phase
                self._check_analyze = False
                self._cal_analyze = False
                # call super at the end (instead of beginning) so that we can do the above arg pre-processing
                # before the sub-experiments are built
                super(MetaExp, self).__init__(managers_or_parent, *args, **kwargs)

            def get_identifier(self) -> str:
                return cls._meta_exp_name()

            def _check_state(self) -> bool:
                # 1st check: has this calibration timed out
                last_cal = self.get_dataset(self._my_dataset_keys[cls.LAST_CAL_KEY], 0.0, archive=False)
                last_check = self.get_dataset(self._my_dataset_keys[cls.LAST_CHECK_KEY], 0.0, archive=False)
                assert isinstance(last_cal, float), 'Unexpected return type from dataset'
                assert isinstance(last_check, float), 'Unexpected return type from dataset'
                last_cal_or_check: float = max(last_cal, last_check)
                if time.time() > last_cal_or_check + self._timeout:
                    self.logger.info('check_state failed: timeout')
                    return False

                # 2nd check: have any dependencies been re-calibrated since the last time this cal passed check_data
                # or was re-calibrated
                last_cals: typing.Dict[str, float] = {
                    name: typing.cast(float, self.get_dataset(key_dict[cls.LAST_CAL_KEY], 0.0, archive=False))
                    for name, key_dict in self._dep_dataset_keys.items()
                }
                assert all(isinstance(t, float) for t in last_cals.values()), 'Unexpected return type from dataset'
                re_calibrated = {name for name, cal_time in last_cals.items()
                                 if cal_time > last_cal_or_check}
                if re_calibrated:
                    self.logger.info(f'check_state failed: one or more dependencies re-calibrated: {re_calibrated}')
                    return False
                else:
                    self.logger.info('check_state passed')
                    return True

            def build(self, *args: typing.Any, **kwargs: typing.Any) -> None:
                self._scheduler = self.get_device('scheduler')
                self._dax_scheduler = self.get_device(self._controller_key)
                # construct system keys for dependency datasets
                self._dep_dataset_keys = {
                    name: {
                        key_name: self._dax_scheduler.get_foreign_key(name, key_value)
                        for key_name, key_value in key_dict.items()
                    } for name, key_dict in self._dep_dataset_keys.items()
                }

            def prepare(self) -> None:
                # might be unnecessary to prepare check_exp, but in the case that it does run this will save some time
                # vs. putting this call in run()
                if self._check_exception is None:
                    try:
                        self._check_exp.prepare()
                    except Exception as e:
                        self._check_exception = e

            def run(self) -> None:
                # noinspection PyBroadException
                try:
                    # check_state (only if diagnose flag is False and timeout is not None)
                    if not self.get_dataset(self._my_dataset_keys[cls.DIAGNOSE_FLAG_KEY], False, archive=False) \
                            and self._timeout is not None and self._check_state():
                        return

                    # check_data
                    try:
                        if self._check_exception is not None:
                            raise self._check_exception
                        self._check_analyze = True
                        self._check_exp.run()
                    except dax.base.exceptions.BadDataError:
                        self.logger.info('Bad data, triggering diagnose wave')
                        for name, key_dict in self._dep_dataset_keys.items():
                            self.set_dataset(key_dict[cls.DIAGNOSE_FLAG_KEY], True,
                                             broadcast=True, persist=True, archive=False)
                        self._dax_scheduler.submit(cls.__name__, policy=str(Policy.GREEDY), depth=1, start_depth=1,
                                                   priority=self._scheduler.priority + 1)
                        dax.util.artiq.pause_strict_priority(self._scheduler)
                        self.logger.info('Diagnose finished, continuing to calibration')
                    except dax.base.exceptions.OutOfSpecError:
                        self.logger.info('Out of spec, continuing to calibration')
                    else:
                        self.logger.info('Check data passed, returning')
                        self.set_dataset(self._my_dataset_keys[cls.LAST_CHECK_KEY], time.time(),
                                         broadcast=True, persist=True, archive=False)
                        return

                    # calibrate
                    try:
                        self._calibration_exp = self._cal_cls(self._cal_managers)
                        self._calibration_exp.prepare()
                        self._calibration_exp.run()
                    except dax.base.exceptions.FailedCalibrationError as fce:
                        self.logger.exception(fce)
                        self._submit_barrier()
                    else:
                        self.logger.info('Calibration succeeded')
                        self._cal_analyze = True
                        self.set_dataset(self._my_dataset_keys[cls.LAST_CAL_KEY], time.time(),
                                         broadcast=True, persist=True, archive=False)
                except Exception:
                    if cls.GREEDY_FAILURE:
                        self.logger.exception('Uncaught exception')
                        self._submit_barrier()
                    else:
                        raise
                finally:
                    self.set_dataset(self._my_dataset_keys[cls.DIAGNOSE_FLAG_KEY], False,
                                     broadcast=True, persist=True, archive=False)

            def analyze(self) -> None:
                # make sure file names are unique in the case that the check class and cal class are the same
                file_name_gen = dax.util.output.FileNameGenerator(self._scheduler)
                check_name = file_name_gen(cls.CHECK_CLASS_NAME, 'h5')
                cal_name = file_name_gen(cls.CALIBRATION_CLASS_NAME, 'h5')
                if self._check_analyze:
                    try:
                        self._check_exp.analyze()
                    finally:
                        check_meta = {
                            'rid': self._scheduler.rid,
                            'arguments': pyon.encode(self._check_args)
                        }
                        self._check_managers.write_hdf5(check_name, metadata=check_meta)
                if self._cal_analyze:
                    try:
                        self._calibration_exp.analyze()
                    finally:
                        cal_meta = {
                            'rid': self._scheduler.rid,
                            'arguments': pyon.encode(self._cal_args)
                        }
                        self._cal_managers.write_hdf5(cal_name, metadata=cal_meta)

            def _submit_barrier(self) -> None:
                # Ensure the priority of the barrier is higher than the priority of the current experiment
                priority = self._scheduler.priority + dax.util.experiments.Barrier.PRIORITY
                dax.util.experiments.Barrier.submit(self, pipeline=self._scheduler.pipeline_name, priority=priority)
                dax.util.artiq.pause_strict_priority(self._scheduler)

        # Return the meta experiment class and its name
        return MetaExp, cls._meta_exp_name()

    def build(self) -> None:  # type: ignore
        """Build the job object.

        To add configurable arguments to this job, override the :func:`build_job` method instead.
        """

        # Check classes and arguments
        assert self.hasattr('_META_EXP_FILE'), 'The @create_calibration decorator must be used'
        assert isinstance(self.CHECK_FILE, str), 'The check file must by provided'
        assert isinstance(self.CALIBRATION_FILE, str), 'The calibration file must by provided'
        assert isinstance(self.CHECK_CLASS_NAME, str), 'The check class name must be provided'
        assert isinstance(self.CALIBRATION_CLASS_NAME, str), 'The calibration class name must be provided'
        assert isinstance(self.CHECK_ARGUMENTS, dict), 'The check arguments must be of type dict'
        assert isinstance(self.CALIBRATION_ARGUMENTS, dict), 'The calibration arguments must be of type dict'
        assert all(isinstance(k, str) for k in self.CHECK_ARGUMENTS), 'All argument keys must be of type str'
        assert all(isinstance(k, str) for k in self.CALIBRATION_ARGUMENTS), 'All argument keys must be of type str'
        assert isinstance(self.CALIBRATION_TIMEOUT, str) or self.CALIBRATION_TIMEOUT is None, \
            'The calibration timeout attribute must be of type str or None'
        assert isinstance(self.GREEDY_FAILURE, bool), 'The greedy failure attribute must be of type bool'
        if self.INTERVAL is not None:
            self.logger.warning(
                f'Non-None INTERVAL "{self.INTERVAL}" could result in unexpected behavior of calibration jobs.')
        # Call super
        super(CalibrationJob, self).build()

        # Copy the class arguments attribute (prevents class attribute from being mutated)
        self.CHECK_ARGUMENTS = self.CHECK_ARGUMENTS.copy()
        self.CALIBRATION_ARGUMENTS = self.CALIBRATION_ARGUMENTS.copy()
        # Build this job
        self.build_job()

    def build_job(self) -> None:
        """Build this job.

        Override this function to add configurable arguments to this job.
        Please note that **argument keys must be unique over all jobs in the job set and the scheduler**.
        It is up to the programmer to ensure that there are no duplicate argument keys.

        Configurable arguments should be added directly to the :attr:`CHECK_ARGUMENTS`
        or :attr:`CALIBRATION_ARGUMENTS` attributes.
        For example::

            self.CHECK_ARGUMENTS['foo'] = self.get_argument('foo', BooleanValue())
            self.CALIBRATION_ARGUMENTS['bar'] = self.get_argument('bar', Scannable(RangeScan(10 * us, 200 * us, 10)))
        """
        pass

    def init(self, *, reset: bool, **kwargs: typing.Any) -> None:
        # Process the arguments for check/calibration experiments
        check_arguments = dax.util.artiq.process_arguments(self.CHECK_ARGUMENTS)
        calibration_arguments = dax.util.artiq.process_arguments(self.CALIBRATION_ARGUMENTS)
        # construct 'arguments' for meta exp
        my_dataset_keys = {
            self.LAST_CAL_KEY: self.get_system_key(self.LAST_CAL_KEY),
            self.LAST_CHECK_KEY: self.get_system_key(self.LAST_CHECK_KEY),
            self.DIAGNOSE_FLAG_KEY: self.get_system_key(self.DIAGNOSE_FLAG_KEY)
        }
        # reset timestamps if `reset` is True
        if reset:
            self.set_dataset(my_dataset_keys[self.LAST_CAL_KEY], 0.0, broadcast=True, persist=True, archive=False)
            self.set_dataset(my_dataset_keys[self.LAST_CHECK_KEY], 0.0, broadcast=True, persist=True, archive=False)
        # type checking on dependencies because regular Jobs won't have these datasets
        dep_dataset_keys = {
            dep.get_name(): {
                self.LAST_CAL_KEY: dep.LAST_CAL_KEY,
                self.LAST_CHECK_KEY: dep.LAST_CHECK_KEY,
                self.DIAGNOSE_FLAG_KEY: dep.DIAGNOSE_FLAG_KEY
            } for dep in self.DEPENDENCIES if issubclass(dep, CalibrationJob)
        }
        # convert timeout to float
        timeout: typing.Optional[float] = _str_to_time(
            self.CALIBRATION_TIMEOUT) if self.CALIBRATION_TIMEOUT is not None else None

        # process file paths
        if not self.REPOSITORY:
            # Expand files
            check_file = os.path.expanduser(self.CHECK_FILE)
            assert os.path.isabs(check_file), 'The given check file must be an absolute path'
            cal_file = os.path.expanduser(self.CALIBRATION_FILE)
            assert os.path.isabs(cal_file), 'The given check file must be an absolute path'
        else:
            # check that job definition is in-repo
            assert 'repo_rev' in self._scheduler.expid, 'Job definition must be inside the repository.'
            # find repository path - only works if directory name is actually 'repository'
            cwd = pathlib.PurePath(os.getcwd())
            # in ARTIQ, cwd is some subdirectory of 'results', which resides at the same level as 'repository'
            # so, trim the path all the way up to 'results', and then append 'repository'
            # todo: this is a bit of a hack because the repository directory isn't currently exposed through ARTIQ.
            #  if/when that changes, this should be updated
            repo_path = pathlib.PurePath(*cwd.parts[:cwd.parts.index('results')], 'repository')
            assert os.path.exists(repo_path), f'Path {repo_path} does not exist'
            check_file = str(repo_path.joinpath(self.CHECK_FILE))
            cal_file = str(repo_path.joinpath(self.CALIBRATION_FILE))

        # actual arguments to pass to meta exp
        self._arguments = {
            'check': check_arguments,
            'calibration': calibration_arguments,
            'check_file': check_file,
            'calibration_file': cal_file,
            'controller_key': kwargs['controller_key'],
            'my_dataset_keys': my_dataset_keys,
            'dep_dataset_keys': dep_dataset_keys,
            'timeout': timeout
        }

        # Construct an expid for this job
        self._expid = {
            'file': self._META_EXP_FILE,
            'class_name': self._meta_exp_name(),
            'arguments': self._arguments,
            'log_level': self.LOG_LEVEL,
            'repo_rev': None,  # assumes that the scheduler instance is in the user repo
        }
        self.logger.debug(f'expid: {self._expid}')

        # Call super
        super(CalibrationJob, self).init(reset=reset, **kwargs)

    def is_meta(self) -> bool:
        """Returns `False`. Calibration jobs can not be meta-jobs."""
        return False


__CJ_T = typing.TypeVar('__CJ_T', bound=CalibrationJob)  # Calibration job type var


def create_calibration(cls: typing.Type[__CJ_T]) -> typing.Type[__CJ_T]:
    """Mandatory decorator for calibration jobs to ensure that the meta experiment is built at import time.

    :param cls: A subclass of :class:`CalibrationJob`
    :return: The unmodified calibration job class
    """
    if not issubclass(cls, CalibrationJob):
        raise TypeError(f'Class {cls.__name__} is not a subclass of dax.base.scheduler.CalibrationJob.')

    # Obtain the global namespace of the caller
    gn = inspect.stack()[1].frame.f_globals
    file = gn['__file__']
    # Build the meta experiment
    meta_exp, name = cls.build_meta_exp(file)

    # Insert meta experiment into the global namespace of the caller
    if name in gn:
        raise LookupError(f'Name "{name}" already exists')
    gn[name] = meta_exp

    # Return the unmodified calibration job class
    return cls


class SchedulerController:
    """Scheduler controller class, which exposes an external interface to a running DAX scheduler."""

    _scheduler: DaxScheduler
    _request_queue: _RQ_T

    def __init__(self, scheduler: DaxScheduler, request_queue: _RQ_T):
        """Create a new scheduler controller.

        :param scheduler: The scheduler that spawned this controller
        :param request_queue: The request queue
        """
        assert isinstance(scheduler, DaxScheduler), 'Scheduler must be of type DaxScheduler'

        # Store a reference to the scheduler and the request queue
        self._scheduler = scheduler
        self._request_queue = request_queue

    async def submit(self, *nodes: str,
                     action: str = str(NodeAction.FORCE),
                     policy: typing.Optional[str] = None,
                     reverse: typing.Optional[bool] = None,
                     priority: typing.Optional[int] = None,
                     depth: int = -1,
                     start_depth: int = 0,
                     block: bool = True) -> None:
        """Submit a request to the scheduler.

        :param nodes: A sequence of node names as strings (case sensitive)
        :param action: The root node action as a string (defaults to :attr:`NodeAction.FORCE`)
        :param policy: The scheduling policy as a string (defaults to the schedulers policy)
        :param reverse: The reverse wave flag (defaults to the schedulers reverse wave flag)
        :param priority: The job priority of this trigger (defaults to the schedulers job priority)
        :param depth: Maximum recursion depth (:const:`-1` for infinite recursion depth, which is the default)
        :param start_depth: Depth to start visiting nodes (:const:`0` to start at the root nodes, which is the default)
        :param block: Block until the request was handled
        """

        # Put this request in the queue (inputs are not checked in this async function)
        self._request_queue.put_nowait(_Request(nodes=nodes,
                                                action=action,
                                                policy=policy,
                                                reverse=reverse,
                                                priority=priority,
                                                depth=depth,
                                                start_depth=start_depth))

        if block:
            # Wait until all requests are handled
            await self._request_queue.join()

    def get_foreign_key(self, node: str, *keys: str) -> str:
        """Obtain a system key of a foreign node.

        :param node: The node name as a string (case sensitive)
        :param keys: The keys to append to the system key of the foreign node
        :return: The foreign system key as a string
        :raises KeyError: Raised if the node is not in the scheduling graph
        :raises ValueError: Raised if any key has an invalid format
        """
        assert isinstance(node, str), 'Node must be of type str'
        assert all(isinstance(key, str) for key in keys), 'Keys must be of type str'

        if node in self._scheduler:
            # Return the foreign system key
            return self._scheduler.get_system_key(node, *keys)
        else:
            # Node is not in the scheduling graph
            raise KeyError(f'Node "{node}" is not in the scheduling graph')


class DaxScheduler(dax.base.system.DaxHasKey, abc.ABC):
    """DAX scheduler class to inherit.

    Users only have to override class attributes to create a scheduling definition.
    The scheduler subclass must also inherit the ARTIQ :class:`Experiment` or
    :class:`EnvExperiment` class to make the scheduler available as an ARTIQ experiment.

    The following attributes must be overridden:

    - :attr:`NAME`: The name of this scheduler
    - :attr:`NODES`: A collection of node classes for this scheduler

    Other optional attributes that can be overridden are:

    - :attr:`ROOT_NODES`: A collection of node classes that are the root nodes, defaults to all entry nodes
    - :attr:`SYSTEM`: A DAX system type to enable additional logging of data
    - :attr:`CONTROLLER`: The scheduler controller name as defined in the device DB
    - :attr:`TERMINATE_TIMEOUT`: The timeout for terminating other instances in seconds
    - :attr:`DEFAULT_SCHEDULING_POLICY`: The default scheduling policy
    - :attr:`DEFAULT_WAVE_INTERVAL`: The default wave interval in seconds
    - :attr:`DEFAULT_CLOCK_PERIOD`: The default clock period in seconds
    - :attr:`DEFAULT_REVERSE_WAVE`: The default value for the reverse wave flag
    - :attr:`DEFAULT_RESET_NODES`: The default value for the reset nodes flag
    - :attr:`DEFAULT_CANCEL_NODES`: The default value for the cancel nodes at exit flag
    - :attr:`DEFAULT_JOB_PIPELINE`: The default pipeline to submit jobs to
    - :attr:`DEFAULT_JOB_PRIORITY`: The baseline priority for jobs submitted by this scheduler
    """

    NAME: typing.ClassVar[str]
    """Scheduler name, used as top key."""
    NODES: typing.ClassVar[typing.Collection[typing.Type[Node]]]
    """The collection of node classes."""

    ROOT_NODES: typing.ClassVar[typing.Collection[typing.Type[Node]]] = []
    """The collection of root nodes, all entry nodes if not provided."""
    SYSTEM: typing.ClassVar[typing.Optional[typing.Type[dax.base.system.DaxSystem]]] = None
    """Optional DAX system type, enables Influx DB logging if provided."""
    CONTROLLER: typing.ClassVar[typing.Optional[str]] = None
    """Optional scheduler controller name, as defined in the device DB."""
    TERMINATE_TIMEOUT: typing.ClassVar[float] = 10.0
    """Timeout for terminating other instances in seconds."""

    DEFAULT_WAVE_INTERVAL: typing.ClassVar[float] = 60.0
    """Default wave interval in seconds."""
    DEFAULT_CLOCK_PERIOD: typing.ClassVar[float] = 0.5
    """Default clock period in seconds."""
    DEFAULT_SCHEDULING_POLICY: typing.ClassVar[Policy] = Policy.LAZY
    """Default scheduling policy."""
    DEFAULT_REVERSE_WAVE: typing.ClassVar[bool] = False
    """Default value for the reverse wave flag."""
    DEFAULT_RESET_NODES: typing.ClassVar[bool] = False
    """Default value for the reset nodes flag."""
    DEFAULT_CANCEL_NODES: typing.ClassVar[bool] = False
    """Default value for the cancel nodes at exit flag."""
    DEFAULT_JOB_PIPELINE: typing.ClassVar[str] = 'main'
    """Default pipeline to submit jobs to."""
    DEFAULT_JOB_PRIORITY: typing.ClassVar[int] = 0
    """Default baseline priority to submit jobs."""

    __data_store: dax.base.system.DaxDataStore
    _nodes: typing.Dict[typing.Type[Node], Node]
    _node_name_map: typing.Dict[str, Node]
    _policy_arg: str
    _reverse: bool
    _wave_interval: float
    _clock_period: float
    _enable_controller: bool
    _reset_nodes: bool
    _cancel_nodes: bool
    _job_pipeline: str
    _job_priority: int
    _reduce_graph: bool
    _view_graph: bool
    _policy: Policy
    _graph: nx.DiGraph[Node]
    _graph_reversed: nx.DiGraph[Node]
    _root_nodes: typing.Set[Node]

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
        self.set_default_scheduling(pipeline_name=self.NAME, priority=99, flush=False)

        # The ARTIQ scheduler
        self._scheduler = self.get_device('scheduler')

        # Instantiate the data store
        if self.SYSTEM is not None and self.SYSTEM.DAX_INFLUX_DB_KEY is not None:
            # Create an Influx DB data store
            self.__data_store = dax.base.system.DaxDataStoreInfluxDb.get_instance(self, self.SYSTEM)
        else:
            # No data store configured
            self.__data_store = dax.base.system.DaxDataStore()

        # Create the node objects
        self._nodes = {node: node(self) for node in set(self.NODES)}
        self.logger.debug(f'Created {len(self._nodes)} node(s)')
        if len(self._nodes) < len(self.NODES):
            self.logger.warning('Duplicate nodes were dropped')
        # Store a map from node names to nodes
        self._node_name_map = {node.get_name(): node for node in self._nodes.values()}

        # Scheduling arguments
        default_scheduling_policy: str = str(self.DEFAULT_SCHEDULING_POLICY)
        self._policy_arg = self.get_argument('Scheduling policy',
                                             artiq.experiment.EnumerationValue(sorted(str(p) for p in Policy),
                                                                               default=default_scheduling_policy),
                                             tooltip='Scheduling policy',
                                             group='Scheduler')
        self._reverse = self.get_argument('Reverse wave',
                                          artiq.experiment.BooleanValue(self.DEFAULT_REVERSE_WAVE),
                                          tooltip='Reverse the wave direction when traversing the graph',
                                          group='Scheduler')
        self._wave_interval = self.get_argument('Wave interval',
                                                artiq.experiment.NumberValue(self.DEFAULT_WAVE_INTERVAL, 's',
                                                                             min=1.0, step=1.0),
                                                tooltip='Interval to visit nodes',
                                                group='Scheduler')
        self._clock_period = self.get_argument('Clock period',
                                               artiq.experiment.NumberValue(self.DEFAULT_CLOCK_PERIOD, 's',
                                                                            min=0.1, step=0.1),
                                               tooltip='Internal scheduler clock period',
                                               group='Scheduler')
        if self.CONTROLLER is None:
            self._enable_controller = False
        else:
            self._enable_controller = self.get_argument('Enable controller',
                                                        artiq.experiment.BooleanValue(True),
                                                        tooltip='Enable the scheduler controller',
                                                        group='Scheduler')

        # Node arguments
        self._reset_nodes = self.get_argument('Reset nodes',
                                              artiq.experiment.BooleanValue(self.DEFAULT_RESET_NODES),
                                              tooltip='Reset the node states at scheduler startup',
                                              group='Nodes')
        self._cancel_nodes = self.get_argument('Cancel nodes at exit',
                                               artiq.experiment.BooleanValue(self.DEFAULT_CANCEL_NODES),
                                               tooltip='Cancel the submit action of nodes during scheduler exit',
                                               group='Nodes')
        self._job_pipeline = self.get_argument('Job pipeline',
                                               artiq.experiment.StringValue(self.DEFAULT_JOB_PIPELINE),
                                               tooltip='Default pipeline to submit jobs to',
                                               group='Nodes')
        self._job_priority = self.get_argument('Job priority',
                                               artiq.experiment.NumberValue(self.DEFAULT_JOB_PRIORITY,
                                                                            step=1, ndecimals=0),
                                               tooltip='Baseline job priority',
                                               group='Nodes')

        # Graph arguments
        self._reduce_graph = self.get_argument('Reduce graph',
                                               artiq.experiment.BooleanValue(True),
                                               tooltip='Use transitive reduction of the dependency graph',
                                               group='Dependency graph')
        self._view_graph = self.get_argument('View graph',
                                             artiq.experiment.BooleanValue(False),
                                             tooltip='View the dependency graph at startup',
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

        # Obtain the scheduling policy
        self._policy = Policy.from_str(self._policy_arg)
        # Process the graph
        self._process_graph()

        # Render the dependency graph
        self.logger.debug('Rendering graph')
        self._render_graph()
        # Terminate other instances of this scheduler
        self.logger.info('Terminating other scheduler instances...')
        terminated_instances = dax.util.artiq.terminate_running_instances(self._scheduler,
                                                                          timeout=self.TERMINATE_TIMEOUT)
        if terminated_instances:
            self.logger.info(f'Terminated instances with RID: {terminated_instances}')
        else:
            self.logger.info('No other scheduler instances found')

    def run(self) -> None:
        """Entry point for the scheduler."""
        asyncio.run(self._async_run())

    async def _async_run(self) -> None:
        """Async entry point for the scheduler."""

        # Create a request queue
        request_queue: _RQ_T = asyncio.Queue()

        if self._enable_controller:
            # Run the scheduler and the controller
            await self._run_scheduler_and_controller(request_queue=request_queue)
        else:
            # Only run the scheduler
            await self._run_scheduler(request_queue=request_queue)

    async def _run_scheduler_and_controller(self, *, request_queue: _RQ_T) -> None:
        """Coroutine for running the scheduler and the controller."""

        # Create the controller and the server objects
        controller = SchedulerController(self, request_queue)
        server = sipyco.pc_rpc.Server({'DaxSchedulerController': controller},
                                      description=f'DaxScheduler controller: {self.get_identifier()}')

        # Start the server task
        host, port = self._get_controller_details()
        self.logger.debug(f'Starting the scheduler controller on [{host}]:{port}')
        await asyncio.create_task(server.start(host, port))

        try:
            # Run the scheduler
            await self._run_scheduler(request_queue=request_queue)
        finally:
            # Stop the server
            self.logger.debug('Stopping the scheduler controller')
            await server.stop()

    async def _run_scheduler(self, *, request_queue: _RQ_T) -> None:
        """Coroutine for running the scheduler"""

        # Initialize all nodes
        self.logger.debug(f'Initializing {len(self._graph)} node(s)')
        for node in self._graph:
            node.init(reset=self._reset_nodes,
                      job_pipeline=self._job_pipeline,
                      request_queue=request_queue,
                      controller_key=self.CONTROLLER)

        # Start the request handler task
        self.logger.debug('Starting the request handler')
        request_handler = asyncio.create_task(self._run_request_handler(request_queue=request_queue))

        try:
            # Time for the next wave
            next_wave: float = time.time()

            while True:
                while next_wave > time.time():
                    if self._scheduler.check_pause():
                        # Pause
                        self.logger.debug('Pausing scheduler')
                        self._scheduler.pause()
                    else:
                        # Sleep for a clock period
                        await asyncio.sleep(self._clock_period)

                # Generate the unique wave timestamp
                wave: float = time.time()
                # Start the wave
                self.logger.debug(f'Starting timed wave {wave:.0f}')
                self.wave(wave=wave,
                          root_nodes=self._root_nodes,
                          root_action=NodeAction.PASS,
                          policy=self._policy,
                          reverse=self._reverse,
                          priority=self._job_priority)
                # Update next wave time
                next_wave += self._wave_interval
                # Prevent setting next wave time in the past
                if next_wave <= time.time():
                    next_wave = time.time() + self._wave_interval
                    self.logger.warning('Next wave was rescheduled because the interval time expired')

        except artiq.experiment.TerminationRequested:
            # Scheduler terminated
            self.logger.info(f'Scheduler with RID {self._scheduler.rid} was terminated')

        finally:
            # Cancel the request handler task
            self.logger.debug('Cancelling the request handler')
            request_handler.cancel()
            try:
                await request_handler
            except asyncio.CancelledError:
                pass  # Ignore expected error

            if not request_queue.empty():
                # Warn if not all scheduler requests were handled
                self.logger.warning(f'The scheduler dropped {request_queue.qsize()} unhandled request(s)')

            if self._cancel_nodes:
                # Cancel nodes
                self.logger.debug(f'Cancelling {len(self._graph)} node(s)')
                for node in self._graph:
                    node.cancel()

    async def _run_request_handler(self, *, request_queue: _RQ_T) -> None:
        """Coroutine for the request handler."""

        while True:
            # Wait for a request
            request: _Request = await request_queue.get()
            # Handle the request
            self._handle_request(request=request)
            # Mark the request done
            request_queue.task_done()

    def _handle_request(self, *, request: _Request) -> None:
        """Handle a single request."""

        if not isinstance(request.nodes, collections.abc.Collection):
            self.logger.error('Dropping invalid request, nodes parameter is not a collection')
            return

        try:
            # Convert the input parameters
            root_nodes: typing.Collection[Node] = {self._node_name_map[node] for node in request.nodes}
            root_action: NodeAction = NodeAction.from_str(request.action)
            policy: Policy = self._policy if request.policy is None else Policy.from_str(request.policy)
            reverse: bool = self._reverse if request.reverse is None else request.reverse
            priority: int = self._job_priority if request.priority is None else request.priority
            depth: int = request.depth
            start_depth: int = request.start_depth
            # Verify unchecked fields
            assert isinstance(reverse, bool), 'Reverse flag must be of type bool'
            assert isinstance(priority, int), 'Priority must be of type int'
            assert isinstance(depth, int), 'Depth must be of type int'
            assert isinstance(start_depth, int), 'Start depth must be of type int'
        except (KeyError, AssertionError):
            # Log the error
            self.logger.exception(f'Dropping invalid request: {request}')
        else:
            # Generate the unique wave timestamp
            wave: float = time.time()
            # Submit a wave for the request
            self.logger.debug(f'Starting triggered wave {wave:.0f}')
            self.wave(wave=wave,
                      root_nodes=root_nodes,
                      root_action=root_action,
                      policy=policy,
                      reverse=reverse,
                      priority=priority,
                      depth=depth,
                      start_depth=start_depth)

    def wave(self, *, wave: float,
             root_nodes: typing.Collection[Node],
             root_action: NodeAction,
             policy: Policy,
             reverse: bool,
             priority: int,
             depth: int = -1,
             start_depth: int = 0) -> None:
        """Run a wave over the graph.

        :param wave: The wave identifier
        :param root_nodes: A collection of root nodes
        :param root_action: The root node action
        :param policy: The policy for this wave
        :param reverse: The reverse wave flag
        :param priority: Submit priority
        :param depth: Maximum recursion depth (:const:`-1` for infinite recursion depth)
        :param start_depth: Depth to start visiting nodes (:const:`0` to start at the root nodes)
        """
        assert isinstance(wave, float), 'Wave must be of type float'
        assert isinstance(root_nodes, collections.abc.Collection), 'Root nodes must be a collection'
        assert all(isinstance(j, Node) for j in root_nodes), 'All root nodes must be of type Node'
        assert isinstance(root_action, NodeAction), 'Root action must be of type NodeAction'
        assert isinstance(policy, Policy), 'Policy must be of type Policy'
        assert isinstance(reverse, bool), 'Reverse flag must be of type bool'
        assert isinstance(priority, int), 'Priority must be of type int'
        assert isinstance(depth, int), 'Depth must be of type int'
        assert isinstance(start_depth, int), 'Start depth must be of type int'

        # Select the correct graph for this wave
        graph: nx.DiGraph[Node] = self._graph_reversed if reverse else self._graph
        # Set of submitted nodes in this wave
        submitted: typing.Set[Node] = set()

        def submit(node: Node) -> None:
            """Submit a node.

            :param node: The node to submit
            """

            if node not in submitted:
                # Submit this node
                node.submit(wave=wave, priority=priority)
                submitted.add(node)

        def recurse(node: Node, action: NodeAction, current_depth: int, current_start_depth: int) -> None:
            """Process nodes recursively.

            :param node: The current node to process
            :param action: The action provided by the previous node
            :param current_depth: Current remaining recursion depth
            :param current_start_depth: Current remaining start depth
            """

            if current_start_depth <= 0:
                # Visit the current node and get the new action based on the policy
                new_action: NodeAction = policy.action(action, node.visit(wave=wave))
                # See if the node is submittable
                submittable: bool = new_action.submittable()
            else:
                # Pass the provided action
                new_action = action
                # This node is not submittable
                submittable = False

            if submittable and reverse:
                # Submit this node before recursion
                submit(node)

            if current_depth != 0:
                # Recurse over successors
                for successor in graph.successors(node):
                    recurse(successor, new_action, current_depth - 1, current_start_depth - 1)

            if submittable and not reverse:
                # Submit this node after recursion
                submit(node)

        for root_node in root_nodes:
            # Recurse over all root nodes using the root action
            recurse(root_node, root_action, depth, start_depth)

        if submitted:
            # Log submitted nodes
            self.logger.debug(f'Submitted node(s): {", ".join(sorted(node.get_name() for node in submitted))}')

    def _process_graph(self) -> None:
        """Process the graph of this scheduler."""

        # Check node set integrity
        base_classes: typing.Sequence[typing.Type[Node]] = [Job, Trigger]
        for c in base_classes:
            if self._nodes.pop(c, None) is not None:
                self.logger.warning(f'Removed base class "{c.get_name()}" from nodes')
        if len(self._node_name_map) < len(self._nodes):
            msg = 'Node name conflict, two nodes are not allowed to have the same class name'
            self.logger.error(msg)
            raise ValueError(msg)

        # Log the node set
        self.logger.debug(f'Nodes: {", ".join(sorted(self._node_name_map))}')

        # Create the dependency graph
        self._graph = nx.DiGraph()
        self._graph.add_nodes_from(self._nodes.values())
        try:
            self._graph.add_edges_from(((node, self._nodes[d])
                                        for node in self._nodes.values() for d in node.get_dependencies()))
        except KeyError as e:
            raise KeyError(f'Dependency "{e.args[0].get_name()}" is not in the node set') from None

        # Check trigger nodes
        for trigger in (node for node in self._graph if isinstance(node, Trigger)):
            if any(n not in self._nodes for n in trigger.NODES):
                raise KeyError(f'Not all nodes of trigger "{trigger.get_name()}" are in the node set')

        # Check graph
        if not nx.algorithms.is_directed_acyclic_graph(self._graph):
            raise RuntimeError('The dependency graph is not a directed acyclic graph')
        if self._policy is Policy.LAZY and self.CONTROLLER is None and any(not node.is_timed() for node in self._graph):
            self.logger.warning('Found unreachable nodes (untimed nodes in a lazy scheduling policy)')

        if self._reduce_graph:
            # Get the transitive reduction of the dependency graph
            self._graph = nx.algorithms.transitive_reduction(self._graph)

        # Store reversed graph
        self._graph_reversed = self._graph.reverse(copy=False)

        if self.ROOT_NODES:
            try:
                # Get the root nodes
                self._root_nodes = {self._nodes[node] for node in self.ROOT_NODES}
            except KeyError as e:
                raise KeyError(f'Root node "{e.args[0].get_name()}" is not in the node set') from None
            else:
                if len(self._root_nodes) < len(self.ROOT_NODES):
                    self.logger.warning('Duplicate root nodes were dropped')
        else:
            # Find entry nodes to use as root nodes
            graph: nx.DiGraph[Node] = self._graph_reversed if self._reverse else self._graph
            # noinspection PyTypeChecker
            self._root_nodes = {node for node, degree in graph.in_degree if degree == 0}

        # Log the root nodes
        self.logger.debug(f'Root nodes: {", ".join(sorted(node.get_name() for node in self._root_nodes))}')

    def _render_graph(self) -> None:
        """Render the dependency graph."""

        # Lazy import
        import graphviz

        # Create a directed graph object
        plot = graphviz.Digraph(name=self.NAME, directory=str(dax.util.output.get_base_path(self._scheduler)))
        # Add all nodes
        for node in self._graph:
            plot.node(node.get_name(),
                      style='dashed' if node.is_meta() else None,
                      label=f'{node.get_name()}\n({node.INTERVAL})' if node.is_timed() else None,
                      shape='box' if isinstance(node, Trigger) else None)
        # Add all dependencies
        plot.edges(((m.get_name(), n.get_name()) for m, n in self._graph.edges))
        # Add trigger edges
        for t, n in ((t, n) for t in self._graph if isinstance(t, Trigger) for n in t.NODES):
            plot.edge(t.get_name(), n.get_name(), style='dashed')
        # Render the graph
        plot.render(view=self._view_graph)

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

    def __contains__(self, item: typing.Union[str, typing.Type[Node]]) -> bool:
        """True if the given node (name or class) is in the graph of this scheduler.

        :param item: The node to test for membership (node name or node class)
        :return: True if the given node is in the graph of this scheduler
        """
        return item in self._node_name_map or item in self._nodes

    @classmethod
    def check_attributes(cls) -> None:
        """Check if all attributes of this class are correctly overridden.

        :raises AssertionError: Raised if attributes are not correctly overridden
        """

        # Check name
        assert hasattr(cls, 'NAME'), 'No name was provided'
        # Check nodes
        assert hasattr(cls, 'NODES'), 'No nodes were provided'
        assert isinstance(cls.NODES, collections.abc.Collection), 'The nodes attribute must be a collection'
        assert all(issubclass(node, Node) for node in cls.NODES), 'All nodes must be subclasses of Node'
        # Check root nodes, system, controller, and terminate timeout
        assert isinstance(cls.ROOT_NODES, collections.abc.Collection), 'The root nodes attribute must be a collection'
        assert all(issubclass(node, Node) for node in cls.ROOT_NODES), 'All root nodes must be subclasses of Node'
        assert cls.SYSTEM is None or issubclass(cls.SYSTEM, dax.base.system.DaxSystem), \
            'The provided system must be a subclass of DaxSystem or None'
        assert cls.CONTROLLER is None or isinstance(cls.CONTROLLER, str), 'Controller must be of type str or None'
        assert cls.CONTROLLER != 'scheduler', 'Controller can not be "scheduler" (aliases with the ARTIQ scheduler)'
        assert isinstance(cls.TERMINATE_TIMEOUT, float), 'Terminate timeout must be of type float'
        assert cls.TERMINATE_TIMEOUT >= 0.0, 'Terminate timeout must be greater or equal to zero'
        # Check default wave interval, clock period, policy, reverse, reset nodes flag, pipeline, and job priority
        assert isinstance(cls.DEFAULT_WAVE_INTERVAL, float), 'Default wave interval must be of type float'
        assert isinstance(cls.DEFAULT_CLOCK_PERIOD, float), 'Default clock period must be of type float'
        assert isinstance(cls.DEFAULT_SCHEDULING_POLICY, Policy), 'Default policy must be of type Policy'
        assert isinstance(cls.DEFAULT_REVERSE_WAVE, bool), 'Default reverse wave flag must be of type bool'
        assert isinstance(cls.DEFAULT_RESET_NODES, bool), 'Default reset nodes flag must be of type bool'
        assert isinstance(cls.DEFAULT_CANCEL_NODES, bool), 'Default cancel nodes at startup flag must be of type bool'
        assert isinstance(cls.DEFAULT_JOB_PIPELINE, str) and cls.DEFAULT_JOB_PIPELINE, \
            'Default job pipeline must be of type str'
        assert isinstance(cls.DEFAULT_JOB_PRIORITY, int), 'Default job priority must be of type int'


class _DaxSchedulerClient(dax.base.system.DaxBase, artiq.experiment.Experiment):
    """A client experiment class for a scheduler."""

    SCHEDULER_NAME: typing.ClassVar[str]
    """The name of the scheduler."""
    NODE_NAMES: typing.ClassVar[typing.List[str]]
    """A List with the names of the nodes."""
    CONTROLLER_KEY: typing.ClassVar[str]
    """Key of the scheduler controller."""

    _NODE_ACTION_NAMES: typing.ClassVar[typing.List[str]] = sorted(str(a) for a in NodeAction)
    """A list with node action names."""
    _SCHEDULER_SETTING: typing.ClassVar[str] = '<Scheduler setting>'
    """The option for using the scheduler setting."""
    _POLICY_NAMES: typing.ClassVar[typing.List[str]] = [_SCHEDULER_SETTING] + sorted(str(p) for p in Policy)
    """A list with policy names."""
    _REVERSE_DICT: typing.ClassVar[typing.Dict[str, typing.Optional[bool]]] = {
        _SCHEDULER_SETTING: None, 'False': False, 'True': True}
    """A dict with reverse wave flag names and values."""

    _node: str
    _action: str
    _policy: str
    _reverse: str
    _priority: int
    _depth: int
    _start_depth: int
    _block: bool
    _dax_scheduler: SchedulerController

    def build(self) -> None:  # type: ignore
        # Set default scheduling options for the client
        self.set_default_scheduling(pipeline_name=f'_{self.SCHEDULER_NAME}')

        # Arguments
        self._node = self.get_argument('Node',
                                       artiq.experiment.EnumerationValue(self.NODE_NAMES),
                                       tooltip='Node to submit')
        self._action = self.get_argument('Action',
                                         artiq.experiment.EnumerationValue(self._NODE_ACTION_NAMES,
                                                                           default=str(NodeAction.FORCE)),
                                         tooltip='Initial node action')
        self._policy = self.get_argument('Policy',
                                         artiq.experiment.EnumerationValue(self._POLICY_NAMES,
                                                                           default=self._SCHEDULER_SETTING),
                                         tooltip='Scheduling policy')
        self._reverse = self.get_argument('Reverse',
                                          artiq.experiment.EnumerationValue(sorted(self._REVERSE_DICT),
                                                                            default=self._SCHEDULER_SETTING),
                                          tooltip='Reverse the wave direction when traversing the graph')
        self._priority = self.get_argument('Priority',
                                           artiq.experiment.NumberValue(0, step=1, ndecimals=0),
                                           tooltip='Baseline job priority')
        self._depth = self.get_argument('Depth',
                                        artiq.experiment.NumberValue(-1, min=-1, step=1, ndecimals=0),
                                        tooltip='Maximum recursion depth (`-1` for infinite recursion depth)')
        self._start_depth = self.get_argument('Start depth',
                                              artiq.experiment.NumberValue(0, min=0, step=1, ndecimals=0),
                                              tooltip='Depth to start visiting nodes (`0` to start at the root nodes)')
        self._block = self.get_argument('Block',
                                        artiq.experiment.BooleanValue(True),
                                        tooltip='Block until the request was handled')

        # Get the DAX scheduler controller
        self._dax_scheduler = self.get_device(self.CONTROLLER_KEY)
        # Get the ARTIQ scheduler
        self._scheduler = self.get_device('scheduler')

    def prepare(self) -> None:
        # Check pipeline (loosely checked)
        if self._scheduler.pipeline_name == self.SCHEDULER_NAME:
            self.logger.warning('The scheduler client should not be submitted to the same pipeline as the scheduler')

    def run(self) -> None:
        # Submit the request
        self.logger.info(f'Submitting request: node={self._node}, action={self._action}, '
                         f'policy={self._policy}, reverse={self._reverse}')
        self._dax_scheduler.submit(self._node,
                                   action=self._action,
                                   policy=None if self._policy == self._SCHEDULER_SETTING else self._policy,
                                   reverse=self._REVERSE_DICT[self._reverse],
                                   priority=self._priority,
                                   depth=self._depth,
                                   start_depth=self._start_depth,
                                   block=self._block)

    def get_identifier(self) -> str:
        """Return the identifier of this scheduler client."""
        return f'[{self.SCHEDULER_NAME}]({self.__class__.__name__})'


def dax_scheduler_client(scheduler_class: typing.Type[DaxScheduler]) -> typing.Type[_DaxSchedulerClient]:
    """Decorator to generate a client experiment class from a :class:`DaxScheduler` class.

    The client experiment can be used to manually trigger nodes.

    :param scheduler_class: The scheduler class
    :return: An instantiated client experiment class
    """

    assert issubclass(scheduler_class, DaxScheduler), 'The scheduler class must be a subclass of DaxScheduler'
    scheduler_class.check_attributes()

    class WrapperClass(_DaxSchedulerClient):
        """A wrapped/instantiated client experiment class for a scheduler."""

        SCHEDULER_NAME = scheduler_class.NAME
        NODE_NAMES = sorted({node.get_name() for node in scheduler_class.NODES})
        if isinstance(scheduler_class.CONTROLLER, str):
            CONTROLLER_KEY = scheduler_class.CONTROLLER
        else:
            raise TypeError('The scheduler class must have a valid controller key to generate a client')

    # Return the wrapped client class
    return WrapperClass
