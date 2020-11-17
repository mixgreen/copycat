from __future__ import annotations  # Postponed evaluation of annotations

import abc
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
import os

import artiq.experiment
import artiq.master.worker_db
import sipyco.pc_rpc  # type: ignore

import dax.base.system
import dax.util.output

__all__ = ['NodeAction', 'Policy', 'Job', 'Trigger', 'DaxScheduler', 'dax_scheduler_client']


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


if typing.TYPE_CHECKING:
    __RQ_T = asyncio.Queue[_Request]  # Type variable for the request queue
else:
    __RQ_T = asyncio.Queue


class Node(dax.base.system.DaxHasKey, abc.ABC):
    """Abstract node class for the scheduler."""

    INTERVAL: typing.Optional[str] = None
    """Interval to run this node, defaults to no interval."""
    DEPENDENCIES: typing.Collection[typing.Type[Node]] = []
    """Collection of node dependencies."""

    _LAST_SUBMIT_KEY: str = 'last_submit'
    """Key to store the last submit timestamp."""

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
        self._dependencies: typing.FrozenSet[typing.Type[Node]] = frozenset(self.DEPENDENCIES)
        if len(self._dependencies) < len(self.DEPENDENCIES):
            self.logger.warning('Duplicate dependencies were dropped')

        # Obtain the scheduler
        self._scheduler = self.get_device('scheduler')

        # Convert the interval
        if self.is_timed():
            # Convert the interval string
            self._interval: float = _str_to_time(typing.cast(str, self.INTERVAL))
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
                self._next_submit: float = time.time()
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


class Job(Node):
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

    FILE: typing.Optional[str] = None
    """File containing the experiment (by default relative from the `repository` directory, see :attr:`REPOSITORY`)."""
    REPOSITORY: bool = True
    """True if the given file is in the experiment repository."""
    CLASS_NAME: typing.Optional[str] = None
    """Class name of the experiment."""
    ARGUMENTS: typing.Dict[str, typing.Any] = {}
    """The experiment arguments."""

    LOG_LEVEL: int = logging.WARNING
    """The log level for the experiment."""
    PIPELINE: typing.Optional[str] = None
    """The pipeline to submit this job to, defaults to the pipeline assigned by the scheduler."""
    PRIORITY: int = 0
    """Job priority relative to the base job priority of the scheduler."""
    FLUSH: bool = False
    """The flush flag when submitting a job."""

    _RID_LIST_KEY: str = 'rid_list'
    """Key to store every submitted RID."""
    _ARGUMENTS_HASH_KEY: str = 'arguments_hash'
    """Key to store the last arguments hash."""

    def build(self) -> None:  # type: ignore
        """Build the job object.

        To add configurable arguments to this job, override the :func:`build_job` method instead.
        """

        # Check file, repository flag, class, and arguments
        assert isinstance(self.FILE, str) or self.FILE is None, 'The file attribute must be of type str or None'
        assert isinstance(self.REPOSITORY, bool), 'Repository flag must be of type bool'
        assert isinstance(self.CLASS_NAME, str) or self.CLASS_NAME is None, \
            'The class name attribute must be of type str or None'
        assert isinstance(self.ARGUMENTS, dict), 'The arguments must be of type dict'
        assert all(isinstance(k, str) for k in self.ARGUMENTS), 'All argument keys must be of type str'
        # Check log level, pipeline, priority, and flush
        assert isinstance(self.LOG_LEVEL, int), 'Log level must be of type int'
        assert self.PIPELINE is None or isinstance(self.PIPELINE, str), 'Pipeline must be of type str or None'
        assert isinstance(self.PRIORITY, int), 'Priority must be of type int'
        assert isinstance(self.FLUSH, bool), 'Flush must be of type bool'

        # Call super
        super(Job, self).build()

        # Copy the class arguments attribute (prevents class attribute from being mutated)
        self.ARGUMENTS = self.ARGUMENTS.copy()
        # Build this job
        self.build_job()
        # Process the arguments
        self._arguments = self._process_arguments()

        if self.FILE is not None and not self.REPOSITORY:
            # Expand file
            self.FILE = os.path.expanduser(self.FILE)
            assert os.path.isabs(self.FILE), 'The given file must be an absolute path'

        # Construct an expid for this job
        if not self.is_meta():
            self._expid: typing.Dict[str, typing.Any] = {
                'file': self.FILE,
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

    def init(self, *, reset: bool, **kwargs: typing.Any) -> None:
        assert isinstance(reset, bool)

        # Extract attributes
        self._pipeline: str = kwargs['job_pipeline'] if self.PIPELINE is None else self.PIPELINE
        assert isinstance(self._pipeline, str) and self._pipeline

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
            self.logger.debug(f'Forcing reset because job arguments changed: {arguments_hash}')
            self.set_dataset_sys(self._ARGUMENTS_HASH_KEY, arguments_hash, data_store=False)
            # Force a reset
            reset = True

        # Call super
        super(Job, self).init(reset=reset, **kwargs)

    def _submit(self, *, wave: float, priority: int) -> None:
        if self._last_rid not in self._scheduler.get_status():
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


class Trigger(Node):
    """Trigger class to define a trigger for the scheduler.

    Users only have to override class attributes to create a trigger definition.
    The following main attributes can be overridden:

    - :attr:`NODES`: A collection of nodes to trigger
    - :attr:`ACTION`: The root node action of this trigger (defaults to :attr:`NodeAction.FORCE`)
    - :attr:`POLICY`: The scheduling policy of this trigger (defaults to the schedulers policy)
    - :attr:`REVERSE`: The reverse wave flag of this trigger (defaults to the schedulers reverse wave flag)
    - :attr:`PRIORITY`: The job priority of this trigger (defaults to the schedulers job priority)
    - :attr:`DEPTH`: Maximum recursion depth (`-1` for infinite recursion depth, which is the default)
    - :attr:`Node.INTERVAL`: The trigger interval
    - :attr:`Node.DEPENDENCIES`: A collection of node classes on which this trigger depends
    """

    NODES: typing.Collection[typing.Type[Node]] = []
    """Collection of nodes to trigger."""
    ACTION: NodeAction = NodeAction.FORCE
    """The root node action of this trigger."""
    POLICY: typing.Optional[Policy] = None
    """The scheduling policy of this trigger."""
    REVERSE: typing.Optional[bool] = None
    """The reverse wave flag for this trigger."""
    PRIORITY: typing.Optional[int] = None
    """The job priority of this trigger."""
    DEPTH: int = -1
    """Maximum recursion depth (`-1` for infinite recursion depth)."""

    def build(self) -> None:  # type: ignore
        # Check nodes, action, policy, and reverse flag
        assert isinstance(self.NODES, collections.abc.Collection), 'The nodes must be a collection'
        assert all(issubclass(node, Node) for node in self.NODES), 'All nodes must be subclasses of Node'
        assert isinstance(self.ACTION, NodeAction), 'Action must be of type NodeAction'
        assert isinstance(self.POLICY, Policy) or self.POLICY is None, 'Policy must be of type Policy or None'
        assert isinstance(self.REVERSE, bool) or self.REVERSE is None, 'Reverse must be of type bool or None'
        assert isinstance(self.PRIORITY, int) or self.PRIORITY is None, 'Priority must be of type int or None'
        assert isinstance(self.DEPTH, int), 'Depth must be of type int'

        # Assemble the collection of node names
        self._nodes: typing.Set[str] = {node.get_name() for node in self.NODES}
        if len(self._nodes) < len(self.NODES):
            self.logger.warning('Duplicate nodes were dropped')

        # Call super
        super(Trigger, self).build()

    def init(self, *, reset: bool, **kwargs: typing.Any) -> None:
        # Extract attributes
        self._request_queue: __RQ_T = kwargs['request_queue']
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
                                                depth=self.DEPTH))
        self.logger.info('Submitted trigger')

    def cancel(self) -> None:
        # Triggers can not be cancelled
        pass

    def is_meta(self) -> bool:
        # This trigger is a meta-trigger if there are no nodes to trigger
        return len(self.NODES) == 0


class _SchedulerController:
    """Scheduler controller class, which exposes an external interface to the DAX scheduler."""

    def __init__(self, request_queue: __RQ_T):
        # Store a reference to the request queue
        self._request_queue: __RQ_T = request_queue

    async def submit(self, *nodes: str,
                     action: str = str(NodeAction.FORCE),
                     policy: typing.Optional[str] = None,
                     reverse: typing.Optional[bool] = None,
                     priority: typing.Optional[int] = None,
                     depth: int = -1,
                     block: bool = True) -> None:
        """Submit a request to the scheduler.

        :param nodes: A sequence of node names as strings (case sensitive)
        :param action: The root node action as a string (defaults to :attr:`NodeAction.FORCE`)
        :param policy: The scheduling policy as a string (defaults to the schedulers policy)
        :param reverse: The reverse wave flag (defaults to the schedulers reverse wave flag)
        :param priority: The job priority of this trigger (defaults to the schedulers job priority)
        :param depth: Maximum recursion depth (`-1` for infinite recursion depth, which is the default)
        :param block: Block until the request was handled
        """

        # Put this request in the queue without checking any of the inputs
        self._request_queue.put_nowait(_Request(nodes=nodes,
                                                action=action,
                                                policy=policy,
                                                reverse=reverse,
                                                priority=priority,
                                                depth=depth))

        if block:
            # Wait until all requests are handled
            await self._request_queue.join()


class DaxScheduler(dax.base.system.DaxHasKey, abc.ABC):
    """DAX scheduler class to inherit from.

    Users only have to override class attributes to create a scheduling definition.
    The scheduler subclass must also inherit from the ARTIQ :class:`Experiment` or
    :class:`EnvExperiment` class to make the scheduler available as an ARTIQ experiment.

    The following attributes must be overridden:

    - :attr:`NAME`: The name of this scheduler
    - :attr:`NODES`: A collection of node classes for this scheduler

    Other optional attributes that can be overridden are:

    - :attr:`ROOT_NODES`: A collection of node classes that are the root nodes, defaults to all entry nodes
    - :attr:`SYSTEM`: A DAX system type to enable additional logging of data
    - :attr:`CONTROLLER`: The scheduler controller name as defined in the device DB
    - :attr:`DEFAULT_SCHEDULING_POLICY`: The default scheduling policy
    - :attr:`DEFAULT_REVERSE_WAVE`: The default value for the reverse wave flag
    - :attr:`DEFAULT_RESET_NODES`: The default value for the reset nodes flag
    - :attr:`DEFAULT_CANCEL_NODES`: The default value for the cancel nodes at exit flag
    - :attr:`DEFAULT_JOB_PIPELINE`: The default pipeline to submit jobs to
    - :attr:`DEFAULT_JOB_PRIORITY`: The baseline priority for jobs submitted by this scheduler
    """

    NAME: str
    """Scheduler name, used as top key."""
    NODES: typing.Collection[typing.Type[Node]]
    """The collection of node classes."""

    ROOT_NODES: typing.Collection[typing.Type[Node]] = []
    """The collection of root nodes, all entry nodes if not provided."""
    SYSTEM: typing.Optional[typing.Type[dax.base.system.DaxSystem]] = None
    """Optional DAX system type, enables Influx DB logging if provided."""
    CONTROLLER: typing.Optional[str] = None
    """Optional scheduler controller name, as defined in the device DB."""

    DEFAULT_SCHEDULING_POLICY: Policy = Policy.LAZY
    """Default scheduling policy."""
    DEFAULT_REVERSE_WAVE: bool = False
    """Default value for the reverse wave flag."""
    DEFAULT_RESET_NODES: bool = False
    """Default value for the reset nodes flag."""
    DEFAULT_CANCEL_NODES: bool = False
    """Default value for the cancel nodes at exit flag."""
    DEFAULT_JOB_PIPELINE: str = 'main'
    """Default pipeline to submit jobs to."""
    DEFAULT_JOB_PRIORITY: int = 0
    """Default baseline priority to submit jobs."""

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
            self.__data_store: dax.base.system.DaxDataStore = dax.base.system.DaxDataStore()

        # Create the node objects
        self._nodes: typing.Dict[typing.Type[Node], Node] = {node: node(self) for node in set(self.NODES)}
        self.logger.debug(f'Created {len(self._nodes)} node(s)')
        if len(self._nodes) < len(self.NODES):
            self.logger.warning('Duplicate nodes were dropped')
        # Store a map from node names to nodes
        self._node_name_map: typing.Dict[str, Node] = {node.get_name(): node for node in self._nodes.values()}

        # Scheduling arguments
        default_scheduling_policy: str = str(self.DEFAULT_SCHEDULING_POLICY)
        self._policy_arg: str = self.get_argument('Scheduling policy',
                                                  artiq.experiment.EnumerationValue(sorted(str(p) for p in Policy),
                                                                                    default=default_scheduling_policy),
                                                  tooltip='Scheduling policy',
                                                  group='Scheduler')
        self._reverse: bool = self.get_argument('Reverse wave',
                                                artiq.experiment.BooleanValue(self.DEFAULT_REVERSE_WAVE),
                                                tooltip='Reverse the wave direction when traversing the graph',
                                                group='Scheduler')
        self._wave_interval: int = self.get_argument('Wave interval',
                                                     artiq.experiment.NumberValue(60, 's', min=1, step=1, ndecimals=0),
                                                     tooltip='Interval to visit nodes',
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

        # Node arguments
        self._reset_nodes: bool = self.get_argument('Reset nodes',
                                                    artiq.experiment.BooleanValue(self.DEFAULT_RESET_NODES),
                                                    tooltip='Reset the node states at scheduler startup',
                                                    group='Nodes')
        self._cancel_nodes: bool = self.get_argument('Cancel nodes at exit',
                                                     artiq.experiment.BooleanValue(self.DEFAULT_CANCEL_NODES),
                                                     tooltip='Cancel the submit action of nodes during scheduler exit',
                                                     group='Nodes')
        self._job_pipeline: str = self.get_argument('Job pipeline',
                                                    artiq.experiment.StringValue(self.DEFAULT_JOB_PIPELINE),
                                                    tooltip='Default pipeline to submit jobs to',
                                                    group='Nodes')
        self._job_priority: int = self.get_argument('Job priority',
                                                    artiq.experiment.NumberValue(self.DEFAULT_JOB_PRIORITY,
                                                                                 step=1, ndecimals=0),
                                                    tooltip='Baseline job priority',
                                                    group='Nodes')

        # Graph arguments
        self._reduce_graph: bool = self.get_argument('Reduce graph',
                                                     artiq.experiment.BooleanValue(True),
                                                     tooltip='Use transitive reduction of the dependency graph',
                                                     group='Dependency graph')
        self._view_graph: bool = self.get_argument('View graph',
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
        if self._wave_interval < 1:
            raise ValueError('The chosen wave interval is too small')
        if self._clock_period < 0.1:
            raise ValueError('The chosen clock period is too small')
        if self._wave_interval < 2 * self._clock_period:
            raise ValueError('The wave interval is too small compared to the clock period')

        # Obtain the scheduling policy
        self._policy: Policy = Policy.from_str(self._policy_arg)
        # Process the graph
        self._process_graph()

        # Plot the dependency graph
        self._plot_graph()
        # Terminate other instances of this scheduler
        self._terminate_running_instances()

    def run(self) -> None:
        """Entry point for the scheduler."""
        asyncio.run(self._async_run())

    async def _async_run(self) -> None:
        """Async entry point for the scheduler."""

        # Create a request queue
        request_queue: __RQ_T = asyncio.Queue()

        if self._enable_controller:
            # Run the scheduler and the controller
            await self._run_scheduler_and_controller(request_queue=request_queue)
        else:
            # Only run the scheduler
            await self._run_scheduler(request_queue=request_queue)

    async def _run_scheduler_and_controller(self, *, request_queue: __RQ_T) -> None:
        """Coroutine for running the scheduler and the controller."""

        # Create the controller and the server objects
        controller = _SchedulerController(request_queue)
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

    async def _run_scheduler(self, *, request_queue: __RQ_T) -> None:
        """Coroutine for running the scheduler"""

        # Initialize all nodes
        self.logger.debug(f'Initializing {len(self._graph)} node(s)')
        for node in self._graph:
            node.init(reset=self._reset_nodes,
                      job_pipeline=self._job_pipeline,
                      request_queue=request_queue)

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
            self.logger.info('Scheduler was terminated')

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

    async def _run_request_handler(self, *, request_queue: __RQ_T) -> None:
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
            # Verify unchecked fields
            assert isinstance(reverse, bool), 'Reverse flag must be of type bool'
            assert isinstance(priority, int), 'Priority must be of type int'
            assert isinstance(depth, int), 'Depth must be of type int'
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
                      depth=depth)

    def wave(self, *, wave: float,
             root_nodes: typing.Collection[Node],
             root_action: NodeAction,
             policy: Policy,
             reverse: bool,
             priority: int,
             depth: int = -1) -> None:
        """Run a wave over the graph.

        :param wave: The wave identifier
        :param root_nodes: A collection of root nodes
        :param root_action: The root node action
        :param policy: The policy for this wave
        :param reverse: The reverse wave flag
        :param priority: Submit priority
        :param depth: Maximum recursion depth (`-1` for infinite recursion depth)
        """
        assert isinstance(wave, float), 'Wave must be of type float'
        assert isinstance(root_nodes, collections.abc.Collection), 'Root nodes must be a collection'
        assert all(isinstance(j, Node) for j in root_nodes), 'All root nodes must be of type Node'
        assert isinstance(root_action, NodeAction), 'Root action must be of type NodeAction'
        assert isinstance(policy, Policy), 'Policy must be of type Policy'
        assert isinstance(reverse, bool), 'Reverse flag must be of type bool'
        assert isinstance(priority, int), 'Priority must be of type int'
        assert isinstance(depth, int), 'Depth must be of type int'

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

        def recurse(node: Node, action: NodeAction, current_depth: int) -> None:
            """Process nodes recursively.

            :param node: The current node to process
            :param action: The action provided by the previous node
            :param current_depth: Current remaining recursion depth
            """

            # Visit the current node
            current_action = node.visit(wave=wave)
            # Get the new action based on the policy
            new_action = policy.action(action, current_action)

            if reverse and new_action.submittable():
                # Submit this node before recursion
                submit(node)

            if current_depth != 0:
                # Recurse over successors
                for successor in graph.successors(node):
                    recurse(successor, new_action, current_depth - 1)

            if not reverse and new_action.submittable():
                # Submit this node after recursion
                submit(node)

        for root_node in root_nodes:
            # Recurse over all root nodes using the root action
            recurse(root_node, root_action, depth)

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
        self._graph: nx.DiGraph[Node] = nx.DiGraph()
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
        self._graph_reversed: nx.DiGraph[Node] = self._graph.reverse(copy=False)

        if self.ROOT_NODES:
            try:
                # Get the root nodes
                self._root_nodes: typing.Set[Node] = {self._nodes[node] for node in self.ROOT_NODES}
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

    def _plot_graph(self) -> None:
        """Plot the dependency graph."""

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
        # Check nodes
        assert hasattr(cls, 'NODES'), 'No nodes were provided'
        assert isinstance(cls.NODES, collections.abc.Collection), 'The nodes attribute must be a collection'
        assert all(issubclass(node, Node) for node in cls.NODES), 'All nodes must be subclasses of Node'
        # Check root nodes, system, and controller
        assert isinstance(cls.ROOT_NODES, collections.abc.Collection), 'The root nodes attribute must be a collection'
        assert all(issubclass(node, Node) for node in cls.ROOT_NODES), 'All root nodes must be subclasses of Node'
        assert cls.SYSTEM is None or issubclass(cls.SYSTEM, dax.base.system.DaxSystem), \
            'The provided system must be a subclass of DaxSystem or None'
        assert cls.CONTROLLER is None or isinstance(cls.CONTROLLER, str), 'Controller must be of type str or None'
        assert cls.CONTROLLER != 'scheduler', 'Controller can not be "scheduler" (aliases with the ARTIQ scheduler)'
        # Check default policy, reverse, reset nodes flag, pipeline, and job priority
        assert isinstance(cls.DEFAULT_SCHEDULING_POLICY, Policy), 'Default policy must be of type Policy'
        assert isinstance(cls.DEFAULT_REVERSE_WAVE, bool), 'Default reverse wave flag must be of type bool'
        assert isinstance(cls.DEFAULT_RESET_NODES, bool), 'Default reset nodes flag must be of type bool'
        assert isinstance(cls.DEFAULT_CANCEL_NODES, bool), 'Default cancel nodes at startup flag must be of type bool'
        assert isinstance(cls.DEFAULT_JOB_PIPELINE, str) and cls.DEFAULT_JOB_PIPELINE, \
            'Default job pipeline must be of type str'
        assert isinstance(cls.DEFAULT_JOB_PRIORITY, int), 'Default job priority must be of type int'


class _DaxSchedulerClient(dax.base.system.DaxBase, artiq.experiment.Experiment):
    """A client experiment class for a scheduler."""

    SCHEDULER_NAME: str
    """The name of the scheduler."""
    NODE_NAMES: typing.List[str]
    """A List with the names of the nodes."""
    CONTROLLER_KEY: str
    """Key of the scheduler controller."""

    _NODE_ACTION_NAMES: typing.List[str] = sorted(str(a) for a in NodeAction)
    """A list with node action names."""
    _SCHEDULER_SETTING: str = '<Scheduler setting>'
    """The option for using the scheduler setting."""
    _POLICY_NAMES: typing.List[str] = [_SCHEDULER_SETTING] + sorted(str(p) for p in Policy)
    """A list with policy names."""
    _REVERSE_DICT: typing.Dict[str, typing.Optional[bool]] = {_SCHEDULER_SETTING: None, 'False': False, 'True': True}
    """A dict with reverse wave flag names and values."""

    def build(self) -> None:  # type: ignore
        # Set default scheduling options for the client
        self.set_default_scheduling(pipeline_name=f'_{self.SCHEDULER_NAME}')

        # Arguments
        self._node: str = self.get_argument('Node',
                                            artiq.experiment.EnumerationValue(self.NODE_NAMES),
                                            tooltip='Node to submit')
        self._action: str = self.get_argument('Action',
                                              artiq.experiment.EnumerationValue(self._NODE_ACTION_NAMES,
                                                                                default=str(NodeAction.FORCE)),
                                              tooltip='Initial node action')
        self._policy: str = self.get_argument('Policy',
                                              artiq.experiment.EnumerationValue(self._POLICY_NAMES,
                                                                                default=self._SCHEDULER_SETTING),
                                              tooltip='Scheduling policy')
        self._reverse: str = self.get_argument('Reverse',
                                               artiq.experiment.EnumerationValue(sorted(self._REVERSE_DICT),
                                                                                 default=self._SCHEDULER_SETTING),
                                               tooltip='Reverse the wave direction when traversing the graph')
        self._priority: int = self.get_argument('Priority',
                                                artiq.experiment.NumberValue(0, step=1, ndecimals=0),
                                                tooltip='Baseline job priority')
        self._depth: int = self.get_argument('Depth',
                                             artiq.experiment.NumberValue(-1, min=-1, step=1, ndecimals=0),
                                             tooltip='Maximum recursion depth (`-1` for infinite recursion depth)')
        self._block: bool = self.get_argument('Block',
                                              artiq.experiment.BooleanValue(True),
                                              tooltip='Block until the request was handled')

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
        self.logger.info(f'Submitting request: node={self._node}, action={self._action}, '
                         f'policy={self._policy}, reverse={self._reverse}')
        self._dax_scheduler.submit(self._node,
                                   action=self._action,
                                   policy=None if self._policy == self._SCHEDULER_SETTING else self._policy,
                                   reverse=self._REVERSE_DICT[self._reverse],
                                   priority=self._priority,
                                   depth=self._depth,
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
