import unittest
import time
import logging
import typing
import collections
import contextlib
import socket
import abc
import asyncio
import sys
import os
import os.path
import subprocess
import numpy as np
import networkx as nx
import textwrap
import io
from unittest.mock import Mock, call

from artiq.language.scan import *
from artiq.experiment import TerminationRequested, NumberValue
import artiq.frontend.artiq_run  # type: ignore
from sipyco.sync_struct import Subscriber

from dax.base.scheduler import *
import dax.base.scheduler
from dax.util.artiq import get_managers, process_arguments
import dax.base.system
import dax.base.exceptions
from dax.util.output import temp_dir
import dax

from test.environment import CI_ENABLED

_NUM_WAVES = 5 if CI_ENABLED else 1


def _find_free_port() -> int:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return int(s.getsockname()[1])


_LOCALHOST: str = '127.0.0.1'  # IPv4 for CI
_ARTIQ_MASTER_NOTIFY_PORT: int = 3250

_REF_DICT_T = typing.Dict[typing.Optional[typing.Type[Job]], typing.Dict[str, int]]  # Type of a reference dict

_DEVICE_DB: typing.Dict[str, typing.Any] = {
    'core': {
        'type': 'local',
        'module': 'artiq.coredevice.core',
        'class': 'Core',
        'arguments': {'host': None, 'ref_period': 1e-9}
    },
    'core_cache': {
        'type': 'local',
        'module': 'artiq.coredevice.cache',
        'class': 'CoreCache'
    },
    'core_dma': {
        'type': 'local',
        'module': 'artiq.coredevice.dma',
        'class': 'CoreDMA'
    },
    # Reference to the scheduler controller
    'dax_scheduler': {
        'type': 'controller',
        'host': _LOCALHOST,
        'port': _find_free_port(),
    },
}


# noinspection PyProtectedMember
def _get_init_kwargs(scheduler: DaxScheduler, **kwargs):
    kwargs.setdefault('job_pipeline', scheduler._job_pipeline)
    kwargs.setdefault('job_priority', scheduler._job_priority)
    kwargs.setdefault('reset', scheduler._reset_nodes)
    return kwargs


class _DummyScheduler(artiq.frontend.artiq_run.DummyScheduler):
    """Used for quick termination of tests"""

    def __init__(self):
        super(_DummyScheduler, self).__init__()
        self._terminate = False

    def terminate_this_experiment(self):
        self._terminate = True

    def check_pause(self):
        return self._terminate

    def pause(self):
        if self.check_pause():
            raise TerminationRequested


class _Node(dax.base.scheduler.Node, abc.ABC):
    def __init__(self, *args, **kwargs):
        super(_Node, self).__init__(*args, **kwargs)
        self.counter = collections.Counter()

    def init(self, *, reset, **kwargs):
        self.counter['init'] += 1
        return super(_Node, self).init(reset=reset, **kwargs)

    def visit(self, *args, **kwargs):
        self.counter['visit'] += 1
        return super(_Node, self).visit(*args, **kwargs)

    def submit(self, *args, **kwargs):
        self.counter['submit'] += 1
        return super(_Node, self).submit(*args, **kwargs)

    def schedule(self, *args, **kwargs):
        self.counter['schedule'] += 1
        return super(_Node, self).schedule(*args, **kwargs)

    def cancel(self):
        self.counter['cancel'] += 1
        return super(_Node, self).cancel()


class _Job(Job, _Node):
    pass


class _Trigger(Trigger, _Node):
    pass


class _Job4(_Job):
    FILE = 'foo.py'
    CLASS_NAME = 'Bar'


class _Job2(_Job):
    DEPENDENCIES = {_Job4}


class _Job3(_Job):
    DEPENDENCIES = {_Job4}


class _Job1(_Job):
    DEPENDENCIES = {_Job2, _Job3}
    INTERVAL = '1h'
    FILE = 'foo.py'
    CLASS_NAME = 'Bar'


class _JobC(_Job):
    FILE = 'foo.py'
    CLASS_NAME = 'Bar'


class _JobB(_Job):
    DEPENDENCIES = {_JobC}
    INTERVAL = '1w'


class _JobA(_Job):
    DEPENDENCIES = {_JobB}
    FILE = 'foo.py'
    CLASS_NAME = 'Bar'


class _Scheduler(DaxScheduler):
    NAME = 'test_scheduler'
    NODES = []  # type: ignore[var-annotated]

    def __init__(self, *args, **kwargs):
        # Mock data store
        self._data_store = Mock(spec=dax.base.system.DaxDataStore)
        # Variable for testing purposes
        self.testing_handled_requests = 0
        # Call super
        super(_Scheduler, self).__init__(*args, **kwargs)
        # Replace scheduler with custom dummy scheduler
        self._scheduler = _DummyScheduler()

    @property
    def data_store(self):
        return self._data_store

    def wave(self, *,
             wave=None,
             root_nodes=None,
             root_action=None,
             policy=None,
             reverse=None,
             priority=None,
             **kwargs) -> None:
        # Use defaults for simplicity
        if wave is None:
            wave = time.time()
        if root_nodes is None:
            root_nodes = self._root_nodes
        if root_action is None:
            root_action = NodeAction.PASS
        if policy is None:
            policy = self._policy
        if reverse is None:
            reverse = self._reverse
        if priority is None:
            priority = self._job_priority

        super(_Scheduler, self).wave(wave=wave, root_nodes=root_nodes, root_action=root_action,
                                     policy=policy, reverse=reverse, priority=priority, **kwargs)

    async def _run_scheduler(self, *, request_queue) -> None:
        await self.controller_callback(request_queue)
        await super(_Scheduler, self)._run_scheduler(request_queue=request_queue)
        # Variable for testing purposes
        self.testing_request_queue_qsize = request_queue.qsize()

    async def controller_callback(self, request_queue) -> None:
        pass

    def _handle_request(self, *, request) -> None:
        self.testing_handled_requests += 1
        try:
            super(_Scheduler, self)._handle_request(request=request)
        except TerminationRequested:
            # Test ended, catch exception and mark this experiment as terminated using the dummy scheduler
            self._scheduler.terminate_this_experiment()

    def _render_graph(self) -> None:
        # Skip render graph, prevents output and call to the renderer
        pass


class SchedulerMiscTestCase(unittest.TestCase):
    def test_str_to_time(self):
        test_data = [
            ('', 0.0),
            ('0s', 0.0),
            ('0s  ', 0.0),
            ('  0s', 0.0),
            ('   0s   ', 0.0),
            ('  0  s  ', 0.0),
            ('3s', 3.0),
            ('3    s', 3.0),
            ('3 \t  \t s', 3.0),
            ('-3.3 s', -3.3),
            ('-inf s', float('-inf')),
            ('.3s', 0.3),
            ('33.s', 33.),
            ('46.5 m', 46.5 * 60),
            ('46.5 h', 46.5 * 60 * 60),
            ('46.5 d', 46.5 * 60 * 60 * 24),
            ('46.5 w', 46.5 * 60 * 60 * 24 * 7),
        ]

        for s, v in test_data:
            with self.subTest(string=s, value=v):
                self.assertEqual(dax.base.scheduler._str_to_time(s), v,
                                 'Converted time string does not equal expected float value')

    def test_job_action_str(self):
        for a in NodeAction:
            with self.subTest(job_action=a):
                self.assertEqual(str(a), a.name)

    def test_job_action_submittable(self):
        submittable = {NodeAction.RUN, NodeAction.FORCE}

        for a in NodeAction:
            with self.subTest(job_action=a):
                self.assertEqual(a in submittable, a.submittable())

    def test_policy_complete(self):
        for p in Policy:
            with self.subTest(policy=str(p)):
                self.assertEqual(len(p.value), len(NodeAction) ** 2, 'Policy not fully implemented')

    def test_policy_differences(self):
        for a0 in NodeAction:
            for a1 in NodeAction:
                lazy = Policy.LAZY.action(a0, a1)
                greedy = Policy.GREEDY.action(a0, a1)

                if a0 is NodeAction.RUN and a1 is NodeAction.PASS:
                    # Only for this one scenario the policies are different
                    self.assertEqual(lazy, NodeAction.PASS, 'Policy lazy returned an unexpected action')
                    self.assertEqual(greedy, NodeAction.RUN, 'Policy greedy returned an unexpected action')
                else:
                    # All other cases the policies return the same actions
                    self.assertEqual(lazy, greedy, 'Policies unexpectedly returned different actions')


class SchedulerClientTestCase(unittest.TestCase):

    def test_client_class_instantiation(self):
        class S(_Scheduler):
            CONTROLLER = 'dax_scheduler'

        self.assertTrue(issubclass(dax_scheduler_client(S), dax.base.scheduler._DaxSchedulerClient))

    def test_client_class_instantiation_bad(self):
        with self.assertRaises(TypeError, msg='Lack of scheduler controller did not raise'):
            dax_scheduler_client(_Scheduler)

        class S(_Scheduler):
            CONTROLLER = 'scheduler'

        with self.assertRaises(AssertionError, msg='Invalid scheduler controller name did not raise'):
            dax_scheduler_client(S)

    def test_client_instantiation(self):
        class S(_Scheduler):
            CONTROLLER = 'dax_scheduler'
            NODES = {_JobA, _JobB, _JobC}

        class Client(dax_scheduler_client(S)):
            pass

        device_db = _DEVICE_DB.copy()
        device_db['dax_scheduler'] = {
            'type': 'local',
            'module': 'dax.sim.coredevice.dummy',
            'class': 'Dummy',
            'arguments': {'_key': 'dax_scheduler'}  # Add key argument which is normally added by DAX.sim
        }

        with get_managers(device_db, Node=_JobB.get_name()) as managers:
            # noinspection PyArgumentList
            c = Client(managers)
            c.prepare()
            with self.assertRaises(AttributeError, msg='Dummy device did not raise expected error'):
                c.run()


class LazySchedulerTestCase(unittest.TestCase):
    POLICY = Policy.LAZY
    REVERSE_WAVE = False

    FAST_WAVE_ARGUMENTS = {
        'Wave interval': 1,
        'Clock period': 0.1,
    }

    def setUp(self) -> None:
        self.arguments: typing.Dict[str, typing.Any] = {'Scheduling policy': str(self.POLICY),
                                                        'Reverse wave': self.REVERSE_WAVE,
                                                        'Job pipeline': 'test_pipeline',
                                                        'View graph': False}
        self.managers = get_managers(arguments=self.arguments)

    def tearDown(self) -> None:
        # Close managers
        self.managers.close()

    def test_create_job(self):
        s = _Scheduler(self.managers)

        class J0(Job):
            pass

        class J1(Job):
            FILE = 'foo.py'
            CLASS_NAME = 'Bar'

        class J2(Job):
            DEPENDENCIES = [J1]

        class J3(Job):
            DEPENDENCIES = {J1: 'foo'}
            INTERVAL = '1h'

        class J4(J1):
            DEPENDENCIES = {J1}
            INTERVAL = '1h'

        class J5(J1):
            DEPENDENCIES = (J1,)
            INTERVAL = '1h'
            ARGUMENTS = {'foo': 1, 'scan': RangeScan(1, 10, 9)}

        test_data = [
            (J0, True, False),
            (J1, False, False),
            (J2, True, False),
            (J3, True, True),
            (J4, False, True),
            (J5, False, True),
        ]

        # Save an old timestamp
        old_time = time.time()

        for J, is_meta, is_timed in test_data:
            with self.subTest(job_class=J.__name__):
                j = J(s)
                self.assertEqual(j.is_meta(), is_meta)
                self.assertEqual(j.is_timed(), is_timed)

                for reset in [False, True]:  # reset=True must be last for the next test
                    with self.subTest(task='init', is_timed=is_timed, reset=reset):
                        # Test init
                        j.init(**_get_init_kwargs(s, reset=reset))
                        self.assertIsInstance(j._next_submit, float)
                        if is_timed:
                            self.assertLess(j._next_submit, float('inf'))
                            self.assertGreater(j._next_submit, 0.0)
                        else:
                            self.assertEqual(j._next_submit, float('inf'))

                with self.subTest(task='visit', is_timed=is_timed):
                    # Test visit
                    self.assertEqual(j.visit(wave=old_time), NodeAction.PASS)
                    new_time = time.time()
                    self.assertEqual(j.visit(wave=new_time), NodeAction.RUN if is_timed else NodeAction.PASS)

                with self.subTest(task='submit', is_meta=is_meta):
                    # Test submit
                    if not is_meta:
                        with self.assertLogs(j.logger, logging.INFO):
                            j.submit(wave=new_time, priority=0)
                    else:
                        j.submit(wave=new_time, priority=0)

                with self.subTest(task='cancel'):
                    j.cancel()

    def test_create_job_bad(self):
        s = _Scheduler(self.managers)

        class J0(Job):
            FILE = 'foo.py'

        class J1(Job):
            CLASS_NAME = 'Bar'

        class J2(Job):
            ARGUMENTS = {1: 1}

        class J3(Job):
            INTERVAL = 2.0

        class J4(Job):
            DEPENDENCIES = J3

        class J5(Job):
            DEPENDENCIES = ['J3']

        test_data = [
            (J0, dax.base.exceptions.BuildError),
            (J1, dax.base.exceptions.BuildError),
            (J2, dax.base.exceptions.BuildError),
            (J3, AssertionError),
            (J4, AssertionError),
            (J5, TypeError),  # Caused by failing issubclass() call
        ]

        for J, error_type in test_data:
            with self.subTest(job_class=J.__name__), self.assertRaises(error_type, msg='Bad job did not raise'):
                J(s)

    def test_create_node_bad_parent(self):
        class J0(Job):
            pass

        with self.assertRaises(TypeError, msg='Wrong job parent type did not raise'):
            # noinspection PyTypeChecker
            J0(self.managers)

        class T0(Trigger):
            pass

        with self.assertRaises(TypeError, msg='Wrong trigger parent type did not raise'):
            # noinspection PyTypeChecker
            T0(self.managers)

    def test_job_arguments(self):
        s = _Scheduler(self.managers)

        original_arguments = {'foo': 1,
                              'range': RangeScan(1, 10, 9),
                              'center': CenterScan(1, 10, 9),
                              'explicit': ExplicitScan([1, 10, 9]),
                              'no': NoScan(10)}

        class J0(Job):
            ARGUMENTS = original_arguments.copy()

        j = J0(s)
        arguments_ref = process_arguments(original_arguments)
        self.assertDictEqual(arguments_ref, j._arguments)
        self.assertDictEqual(original_arguments, J0.ARGUMENTS, 'Class arguments were mutated')

    def test_job_configurable_arguments(self):
        s = _Scheduler(self.managers)

        original_arguments: typing.Dict[str, typing.Any] = {
            'foo': 1,
            'range': RangeScan(1, 10, 9),
            'center': CenterScan(1, 10, 9),
            'explicit': ExplicitScan([1, 10, 9]),
            'no': NoScan(10)}
        keyword_arguments: typing.Dict[str, typing.Any] = {
            'bar': NumberValue(20),
            'baz': Scannable(CenterScan(100, 50, 2)),
            'foobar': Scannable(RangeScan(100, 300, 40)),
        }

        class J0(Job):
            ARGUMENTS = original_arguments.copy()

            def build_job(self) -> None:
                for key, argument in keyword_arguments.items():
                    self.ARGUMENTS[key] = self.get_argument(key, argument)

        j = J0(s)
        arguments_ref = original_arguments.copy()
        arguments_ref.update({k: v.default() for k, v in keyword_arguments.items()})
        arguments_ref = process_arguments(arguments_ref)

        self.assertDictEqual(arguments_ref, j._arguments)
        self.assertEqual(len(j._arguments), len(original_arguments) + len(keyword_arguments))
        self.assertDictEqual(original_arguments, J0.ARGUMENTS, 'Class arguments were mutated')
        for v in j._arguments.values():
            self.assertNotIsInstance(v, ScanObject)

    def test_job_reset(self):
        class J0(_Job):
            FILE = 'foo.py'
            CLASS_NAME = 'Bar'
            ARGUMENTS = {'foo': 1}
            INTERVAL = '1h'

        class S(_Scheduler):
            NODES = [J0]

        def start_scheduler(*, reset):
            # Prepare scheduler
            scheduler = S(self.managers)
            scheduler.prepare()
            # Extract job object
            job = scheduler._nodes[J0]
            # Initialize job
            job.init(**_get_init_kwargs(scheduler, reset=reset))
            # Return scheduler and job
            return scheduler, job

        # Start scheduler
        s, j = start_scheduler(reset=False)
        # First run, job will run because it has never before
        s.wave()
        self.assertDictEqual(j.counter, {'init': 1, 'visit': 2, 'submit': 1, 'schedule': 1})
        # Second run, job will not run because interval did not expire
        s.wave()
        self.assertDictEqual(j.counter, {'init': 1, 'visit': 3, 'submit': 1, 'schedule': 1})

        # Restart scheduler
        s, j = start_scheduler(reset=False)
        # First run, job will not run because interval did not expire
        s.wave()
        self.assertDictEqual(j.counter, {'init': 1, 'visit': 1})

        # Restart scheduler
        s, j = start_scheduler(reset=True)
        # First run, job will run because of reset
        s.wave()
        self.assertDictEqual(j.counter, {'init': 1, 'visit': 2, 'submit': 1, 'schedule': 1})

        # Restart scheduler
        s, j = start_scheduler(reset=False)
        # First run, job will not run because interval did not expire
        s.wave()
        self.assertDictEqual(j.counter, {'init': 1, 'visit': 1})

    def test_job_changed_arguments(self):
        class J0(_Job):
            FILE = 'foo.py'
            CLASS_NAME = 'Bar'
            ARGUMENTS = {'foo': 1}
            INTERVAL = '1h'

        class S(_Scheduler):
            NODES = [J0]

        def start_scheduler(*, reset):
            # Prepare scheduler
            scheduler = S(self.managers)
            scheduler.prepare()
            # Extract job object
            job = scheduler._nodes[J0]
            # Initialize job
            job.init(**_get_init_kwargs(scheduler, reset=reset))
            # Return scheduler and job
            return scheduler, job

        # Start scheduler
        s, j = start_scheduler(reset=False)
        # First run, job will run because it has never before
        s.wave()
        self.assertDictEqual(j.counter, {'init': 1, 'visit': 2, 'submit': 1, 'schedule': 1})
        # Second run, job will not run because interval did not expire
        s.wave()
        self.assertDictEqual(j.counter, {'init': 1, 'visit': 3, 'submit': 1, 'schedule': 1})

        # Restart scheduler
        s, j = start_scheduler(reset=False)
        # First run, job will not run because interval did not expire
        s.wave()
        self.assertDictEqual(j.counter, {'init': 1, 'visit': 1})

        # Change arguments
        J0.ARGUMENTS['bar'] = 2
        # Restart scheduler
        s, j = start_scheduler(reset=False)
        # First run, job will run because of changed arguments
        s.wave()
        self.assertDictEqual(j.counter, {'init': 1, 'visit': 2, 'submit': 1, 'schedule': 1})

        # Restart scheduler
        s, j = start_scheduler(reset=False)
        # First run, job will not run because interval did not expire and arguments did not change
        s.wave()
        self.assertDictEqual(j.counter, {'init': 1, 'visit': 1})

    def test_job_name(self):
        self.assertEqual(_JobA.get_name(), '_JobA')
        self.assertEqual(_JobA.__name__, '_JobA')

    def test_create_scheduler(self):
        with self.assertRaises(AssertionError, msg='Scheduler without name did not raise'):
            class S(DaxScheduler):
                NODES = {}

            S(self.managers)

        with self.assertRaises(AssertionError, msg='Scheduler without jobs did not raise'):
            class S(DaxScheduler):
                NAME = 'test_scheduler'

            S(self.managers)

        class S(DaxScheduler):
            NAME = 'test_scheduler'
            NODES = {}

        # Instantiate a well defined scheduler
        self.assertIsInstance(S(self.managers), DaxScheduler)

    def test_scheduler_pipeline(self):
        s = _Scheduler(self.managers)
        s.prepare()

        self.arguments['Job pipeline'] = 'main'

        with get_managers(arguments=self.arguments) as managers:
            with self.assertRaises(ValueError, msg='Pipeline conflict did not raise'):
                s = _Scheduler(managers)
                s.prepare()

    def test_duplicate_nodes(self):
        class SchedulerA(_Scheduler):
            NODES = [_Job1, _Job2, _Job3, _Job4, _JobA, _JobB, _JobC, _Job1]

            # noinspection PyMethodParameters
            def build(self_, *args: typing.Any, **kwargs: typing.Any) -> None:
                with self.assertLogs(self_.logger, logging.WARNING):
                    super(SchedulerA, self_).build(*args, **kwargs)

        class T(_Trigger):
            pass

        class SchedulerB(SchedulerA):
            NODES = [T, T]

        class SchedulerC(SchedulerA):
            NODES = SchedulerA.NODES + SchedulerB.NODES

        for S in [SchedulerA, SchedulerB, SchedulerC]:
            with self.subTest(cls=S.__name__):
                s = S(self.managers)
                self.assertLess(len(s._nodes), len(S.NODES), 'Number of nodes is expected to decrease')
                self.assertEqual(len(s._nodes), len(set(S.NODES)), 'Number of nodes does not match expected number')

    def test_node_name_conflict(self):
        class S(_Scheduler):
            # noinspection PyGlobalUndefined
            global _JobA
            NODES = [_JobA]

        # noinspection PyShadowingNames
        class _JobA(Job):
            pass

        S.NODES.append(_JobA)

        s = S(self.managers)
        with self.assertRaises(ValueError, msg='Node class name conflict did not raise'):
            s.prepare()

    def test_node_name_conflict2(self):
        class S(_Scheduler):
            # noinspection PyGlobalUndefined
            global _JobA
            NODES = [_JobA]

        # noinspection PyShadowingNames
        class _JobA(Trigger):
            pass

        S.NODES.append(_JobA)

        s = S(self.managers)
        with self.assertRaises(ValueError, msg='Node class name conflict did not raise'):
            s.prepare()

    def test_node_dependencies(self):
        class S(_Scheduler):
            NODES = {_JobA}

        with self.assertRaises(KeyError, msg='Dependency not in node set did not raise'):
            s = S(self.managers)
            try:
                s.prepare()
            except KeyError as e:
                self.assertIn(f'"{_JobB.get_name()}"', str(e), 'Job name not correctly displayed in error message')
                raise

    def test_contains(self):
        scheduler_nodes = {_Job4, _Job3, _JobB, _JobC}
        all_nodes = {_Job1, _Job2, _Job3, _Job4, _JobA, _JobB, _JobC}

        class S(_Scheduler):
            NODES = scheduler_nodes

        s = S(self.managers)
        s.prepare()

        for n in all_nodes:
            with self.subTest(node=n.get_name()):
                self.assertEqual(n in s, n in scheduler_nodes)
                self.assertEqual(n.get_name() in s, n in scheduler_nodes)

    def test_trigger_nodes(self):
        class T(_Trigger):
            NODES = [_JobA]

        class S(_Scheduler):
            NODES = {T}

        with self.assertRaises(KeyError, msg='Trigger node not in node set did not raise'):
            s = S(self.managers)
            try:
                s.prepare()
            except KeyError as e:
                self.assertIn(f'"{T.get_name()}"', str(e), 'Node name not correctly displayed in error message')
                raise

    def test_scheduler_dag(self):
        class JobA(Job):
            pass

        # Artificially create a self-loop
        JobA.DEPENDENCIES = {JobA}

        class S(_Scheduler):
            NODES = {JobA}

        with self.assertRaises(RuntimeError, msg='Non-DAG dependency graph did not raise'):
            s = S(self.managers)
            s.prepare()

    def test_scheduler_root_jobs(self):
        class JobZ(Job):
            pass

        class JobY(Job):
            DEPENDENCIES = [JobZ]

        class JobX(Job):
            DEPENDENCIES = [JobZ]

        job_sets = [
            ({_Job1, _Job2, _Job3, _Job4, _JobA, _JobB, _JobC},
             (_Job1, _JobA) if not self.REVERSE_WAVE else (_Job4, _JobC)),
            ({_JobA, _JobB, _JobC, JobX, JobY, JobZ},
             (_JobA, JobX, JobY) if not self.REVERSE_WAVE else (_JobC, JobZ)),
        ]

        for jobs, root_jobs in job_sets:
            class S(_Scheduler):
                NODES = jobs

            s = S(self.managers)
            s.prepare()

            self.assertEqual(len(s._root_nodes), len(root_jobs), 'Did not found expected number of root jobs')
            for j in s._root_nodes:
                self.assertIsInstance(j, root_jobs, 'Root jobs have an unexpected type')

    def test_scheduler_custom_root_jobs(self):
        class JobZ(Job):
            pass

        class JobY(Job):
            DEPENDENCIES = [JobZ]

        class JobX(Job):
            DEPENDENCIES = [JobZ]

        job_sets = [
            ({_Job1, _Job2, _Job3, _Job4, _JobA, _JobB, _JobC}, (_Job2, _JobB)),
            ({_JobA, _JobB, _JobC, JobX, JobY, JobZ}, (_JobB, JobY)),
        ]

        for jobs, root_jobs in job_sets:
            class S(_Scheduler):
                NODES = jobs
                ROOT_NODES = root_jobs

            s = S(self.managers)
            s.prepare()

            self.assertEqual(len(s._root_nodes), len(root_jobs), 'Did not found expected number of root jobs')
            for j in s._root_nodes:
                self.assertIsInstance(j, root_jobs, 'Root jobs have an unexpected type')

    def test_scheduler_custom_root_jobs_bad(self):
        class JobZ(Job):
            pass

        class JobY(Job):
            DEPENDENCIES = [JobZ]

        class JobX(Job):
            DEPENDENCIES = [JobZ]

        job_sets = [
            ({_Job1, _Job2, _Job3, _Job4, _JobA, _JobB, _JobC}, JobZ),
            ({_JobA, _JobB, _JobC, JobX, JobY, JobZ}, _Job4),
        ]

        for jobs, root_job in job_sets:
            class S(_Scheduler):
                NODES = jobs
                ROOT_NODES = {root_job}

            s = S(self.managers)
            with self.assertRaises(KeyError, msg='Root job outside job set did not raise'):
                try:
                    s.prepare()
                except KeyError as e:
                    self.assertIn(f'"{root_job.get_name()}"', str(e),
                                  'Job name not correctly displayed in error message')
                    raise

    def test_scheduler_unreachable_jobs(self):
        class S(_Scheduler):
            NODES = {_JobC}

        s = S(self.managers)
        with self.assertLogs(s.logger, logging.WARNING):
            s.prepare()

    def _check_scheduler(self, scheduler: DaxScheduler, reference: _REF_DICT_T) -> None:
        """Test scheduler outcome against a reference."""
        default = reference.get(None, {})
        for j in scheduler._graph:
            with self.subTest(job=j.get_name()):
                self.assertDictEqual(j.counter, reference.get(type(j), default),
                                     f'Job call pattern of job "{j.get_name()}" did not match reference')

    def test_scheduler_wave(self):
        class S(_Scheduler):
            NODES = {_Job1, _Job2, _Job3, _Job4, _JobA, _JobB, _JobC}

        s = S(self.managers)
        s.prepare()

        # Manually call init
        for j in s._graph:
            j.init(**_get_init_kwargs(s, reset=True))

        # Check data store calls
        self.assertEqual(len(s.data_store.method_calls), 0, 'Unexpected data store calls')

        # Wave
        s.wave()
        self._check_scheduler(s, self.WAVE_0_REF)

        # Wave
        s.wave(wave=time.time(), root_nodes=s._root_nodes, root_action=NodeAction.PASS, policy=s._policy)
        self._check_scheduler(s, self.WAVE_1_REF)

        # Check data store calls
        # noinspection PyTypeChecker
        self._check_scheduler_datastore_calls(s)

    WAVE_0_REF: _REF_DICT_T = {
        _Job1: {'init': 1, 'visit': 2, 'submit': 1, 'schedule': 1},
        _Job4: {'init': 1, 'visit': 2},
        _JobB: {'init': 1, 'visit': 2, 'submit': 1, 'schedule': 1},
        None: {'init': 1, 'visit': 1},
    }

    WAVE_1_REF: _REF_DICT_T = {
        _Job1: {'init': 1, 'visit': 3, 'submit': 1, 'schedule': 1},
        _Job4: {'init': 1, 'visit': 4},
        _JobB: {'init': 1, 'visit': 3, 'submit': 1, 'schedule': 1},
        None: {'init': 1, 'visit': 2},
    }

    def _check_scheduler_datastore_calls(self, scheduler):
        self.assertListEqual(scheduler.data_store.method_calls,
                             [call.append(scheduler.get_system_key(_Job1.get_name(), _Job1._RID_LIST_KEY), 1)])

    def test_scheduler_run(self):

        class S(_Scheduler):
            NODES = {_Job1, _Job2, _Job3, _Job4, _JobA, _JobB, _JobC}
            counter = 0

            def wave(self, **kwargs) -> None:
                super(S, self).wave(**kwargs)
                self.counter += 1
                if self.counter >= _NUM_WAVES:
                    raise TerminationRequested

        self.arguments.update(self.FAST_WAVE_ARGUMENTS)

        with get_managers(arguments=self.arguments) as managers:
            s = S(managers)
            s.prepare()

            # Run the scheduler
            s.run()
            self.assertEqual(s.testing_handled_requests, 0, 'Unexpected number of requests handled')
            self.assertEqual(s.testing_request_queue_qsize, 0, 'Request queue was not empty')
            self._check_scheduler(s, self.RUN_REF)

    RUN_REF: _REF_DICT_T = {
        _Job1: {'init': 1, 'visit': _NUM_WAVES + 1, 'submit': 1, 'schedule': 1},
        _Job4: {'init': 1, 'visit': _NUM_WAVES * 2},
        _JobB: {'init': 1, 'visit': _NUM_WAVES + 1, 'submit': 1, 'schedule': 1},
        None: {'init': 1, 'visit': _NUM_WAVES},
    }

    def test_scheduler_run_depth(self):

        class S(_Scheduler):
            NODES = {_Job1, _Job2, _Job3, _Job4, _JobA, _JobB, _JobC}
            counter = 0

            def wave(self, **kwargs) -> None:
                # Explicitly set depth
                kwargs['depth'] = 1
                super(S, self).wave(**kwargs)
                self.counter += 1
                if self.counter >= _NUM_WAVES:
                    raise TerminationRequested

        self.arguments.update(self.FAST_WAVE_ARGUMENTS)

        with get_managers(arguments=self.arguments) as managers:
            s = S(managers)
            s.prepare()

            # Run the scheduler
            s.run()
            self.assertEqual(s.testing_handled_requests, 0, 'Unexpected number of requests handled')
            self.assertEqual(s.testing_request_queue_qsize, 0, 'Request queue was not empty')
            self._check_scheduler(s, self.RUN_DEPTH_REF)

    RUN_DEPTH_REF: _REF_DICT_T = {
        _Job1: {'init': 1, 'visit': _NUM_WAVES + 1, 'submit': 1, 'schedule': 1},
        _Job2: {'init': 1, 'visit': _NUM_WAVES},
        _Job3: {'init': 1, 'visit': _NUM_WAVES},
        _JobA: {'init': 1, 'visit': _NUM_WAVES},
        _JobB: {'init': 1, 'visit': _NUM_WAVES + 1, 'submit': 1, 'schedule': 1},
        None: {'init': 1},
    }

    def test_scheduler_run_start_depth(self):

        class S(_Scheduler):
            NODES = {_Job1, _Job2, _Job3, _Job4, _JobA, _JobB, _JobC}
            counter = 0

            def wave(self, **kwargs) -> None:
                # Explicitly set start depth
                kwargs['start_depth'] = 1
                super(S, self).wave(**kwargs)
                self.counter += 1
                if self.counter >= _NUM_WAVES:
                    raise TerminationRequested

        self.arguments.update(self.FAST_WAVE_ARGUMENTS)

        with get_managers(arguments=self.arguments) as managers:
            s = S(managers)
            s.prepare()

            # Run the scheduler
            s.run()
            self.assertEqual(s.testing_handled_requests, 0, 'Unexpected number of requests handled')
            self.assertEqual(s.testing_request_queue_qsize, 0, 'Request queue was not empty')
            self._check_scheduler(s, self.RUN_START_DEPTH_REF)

    RUN_START_DEPTH_REF: _REF_DICT_T = {
        _Job1: {'init': 1},
        _Job4: {'init': 1, 'visit': _NUM_WAVES * 2},
        _JobA: {'init': 1},
        _JobB: {'init': 1, 'visit': _NUM_WAVES + 1, 'submit': 1, 'schedule': 1},
        None: {'init': 1, 'visit': _NUM_WAVES},
    }

    def _test_scheduler_controller(self, requests):
        class S(_Scheduler):
            NODES = {_Job1, _Job2, _Job3, _Job4, _JobA, _JobB, _JobC}
            CONTROLLER = 'dax_scheduler'

            counter = 0

            def wave(self, **kwargs) -> None:
                super(S, self).wave(**kwargs)

                self.counter += 1
                if self.counter >= _NUM_WAVES + len(requests):
                    raise TerminationRequested

            async def controller_callback(self, request_queue) -> None:
                # Construct a separate controller with the same queue
                # This is required because we can not use get_device() to obtain the controller
                # Because the server and the client are running on the same thread, the situation deadlocks
                controller = dax.base.scheduler.SchedulerController(self, request_queue)
                for args, kwargs in requests:
                    kwargs.setdefault('block', False)
                    await controller.submit(*args, **kwargs)

        self.arguments.update(self.FAST_WAVE_ARGUMENTS)

        with get_managers(_DEVICE_DB, arguments=self.arguments) as managers:
            s = S(managers)
            s.prepare()

            # Run the scheduler
            s.run()

        # Return the scheduler
        return s

    def test_scheduler_controller(self):
        requests = [
            ((_Job4.get_name(), _JobA.get_name()), {}),
            ((_JobC.get_name(),), {'action': str(NodeAction.PASS)}),
        ]
        s = self._test_scheduler_controller(requests)

        self.assertEqual(s.testing_handled_requests, len(requests), 'Unexpected number of requests handled')
        self.assertEqual(s.testing_request_queue_qsize, 0, 'Request queue was not empty')
        self._check_scheduler(s, self.CONTROLLER_REF)

    CONTROLLER_REF: _REF_DICT_T = {
        _Job1: {'init': 1, 'visit': _NUM_WAVES + 1, 'submit': 1, 'schedule': 1},
        _Job2: {'init': 1, 'visit': _NUM_WAVES},
        _Job3: {'init': 1, 'visit': _NUM_WAVES},
        _Job4: {'init': 1, 'visit': _NUM_WAVES * 2 + 1, 'submit': 1, 'schedule': 1},
        _JobA: {'init': 1, 'visit': _NUM_WAVES + 1, 'submit': 1, 'schedule': 1},
        _JobB: {'init': 1, 'visit': _NUM_WAVES + 1 + 1, 'submit': 1, 'schedule': 1},
        _JobC: {'init': 1, 'visit': _NUM_WAVES + 1 + 1},
    }

    def test_scheduler_controller_reversed(self):
        requests = [
            ((_Job4.get_name(), _JobB.get_name()), {'reverse': True}),
        ]
        s = self._test_scheduler_controller(requests)

        self.assertEqual(s.testing_handled_requests, len(requests), 'Unexpected number of requests handled')
        self.assertEqual(s.testing_request_queue_qsize, 0, 'Request queue was not empty')
        self._check_scheduler(s, self.CONTROLLER_REVERSED_REF)

    CONTROLLER_REVERSED_REF: _REF_DICT_T = {
        _Job1: {'init': 1, 'visit': _NUM_WAVES + 1 + 2, 'submit': 1, 'schedule': 1},
        _Job2: {'init': 1, 'visit': _NUM_WAVES + 1},
        _Job3: {'init': 1, 'visit': _NUM_WAVES + 1},
        _Job4: {'init': 1, 'visit': _NUM_WAVES * 2 + 1, 'submit': 1, 'schedule': 1},
        _JobA: {'init': 1, 'visit': _NUM_WAVES + 1},
        _JobB: {'init': 1, 'visit': _NUM_WAVES + 1 + 2, 'submit': 2, 'schedule': 2},
        _JobC: {'init': 1, 'visit': _NUM_WAVES},
    }

    def test_scheduler_controller_reversed_sequential(self):
        requests = [
            ((_Job4.get_name(),), {'reverse': True}),
            ((_JobB.get_name(),), {'reverse': True}),
        ]
        s = self._test_scheduler_controller(requests)

        self.assertEqual(s.testing_handled_requests, len(requests), 'Unexpected number of requests handled')
        self.assertEqual(s.testing_request_queue_qsize, 0, 'Request queue was not empty')
        self._check_scheduler(s, self.CONTROLLER_REVERSED_REF)

    def test_scheduler_controller_start_depth(self):
        requests = [
            ((_Job4.get_name(), _JobA.get_name()), {'start_depth': 1}),
            ((_JobC.get_name(),), {'action': str(NodeAction.PASS)}),
        ]
        s = self._test_scheduler_controller(requests)

        self.assertEqual(s.testing_handled_requests, len(requests), 'Unexpected number of requests handled')
        self.assertEqual(s.testing_request_queue_qsize, 0, 'Request queue was not empty')
        self._check_scheduler(s, self.CONTROLLER_START_DEPTH_REF)

    CONTROLLER_START_DEPTH_REF: _REF_DICT_T = {
        _Job1: {'init': 1, 'visit': _NUM_WAVES + 1, 'submit': 1, 'schedule': 1},
        _Job2: {'init': 1, 'visit': _NUM_WAVES},
        _Job3: {'init': 1, 'visit': _NUM_WAVES},
        _Job4: {'init': 1, 'visit': _NUM_WAVES * 2},
        _JobA: {'init': 1, 'visit': _NUM_WAVES},
        _JobB: {'init': 1, 'visit': _NUM_WAVES + 1 + 2, 'submit': 1 + 1, 'schedule': 1 + 1},
        _JobC: {'init': 1, 'visit': _NUM_WAVES + 1 + 1},
    }

    def test_scheduler_controller_foreign_key(self):
        test_data = [
            ('_Job1', ('foo',)),
            ('_Job2', ('foo',)),
            ('_Job3', ('foo',)),
            ('_Job4', ('foo',)),
            ('_JobA', ('foo',)),
            ('_JobB', ('foo',)),
            ('_JobC', ('foo',)),
            ('bar', ('baz',)),
            ('foo', ('bar', 'baz')),
            ('Job1', ('bar', 'baz')),
            ('S', ('foo',)),
            ('_Scheduler', ('foo',)),
        ]

        class S(_Scheduler):
            NODES = {_Job1, _Job2, _Job3, _Job4, _JobA, _JobB, _JobC}
            CONTROLLER = 'dax_scheduler'

            def wave(self, **kwargs) -> None:
                raise TerminationRequested

            # noinspection PyMethodParameters
            async def controller_callback(self_, request_queue) -> None:
                # Construct a separate controller with the same queue
                # This is required because we can not use get_device() to obtain the controller
                # Because the server and the client are running on the same thread, the situation deadlocks
                controller = dax.base.scheduler.SchedulerController(self_, request_queue)
                valid_keys = {n.get_name() for n in self_.NODES}
                for node, keys in test_data:
                    if node in valid_keys:
                        foreign_key = controller.get_foreign_key(node, *keys)
                        ref_key = dax.base.system._KEY_SEPARATOR.join([self_.NAME, node, *keys])
                        self.assertEqual(foreign_key, ref_key)
                    else:
                        with self.assertRaises(KeyError, msg='Node not in scheduling graph did not raise'):
                            controller.get_foreign_key(node, *keys)

        self.arguments.update(self.FAST_WAVE_ARGUMENTS)

        with get_managers(_DEVICE_DB, arguments=self.arguments) as managers:
            s = S(managers)
            s.prepare()

            # Run the scheduler and the tests
            s.run()

    def test_create_trigger(self):
        # asyncio entry point
        asyncio.run(self._test_create_trigger())

    async def _test_create_trigger(self):
        s = _Scheduler(self.managers)

        # Use a "stand-in" queue for testing
        request_queue = asyncio.Queue()
        queue_len = request_queue.qsize()
        self.assertEqual(queue_len, 0)

        class T0(Trigger):
            pass

        class T1(Trigger):
            NODES = [T0]

        class T2(Trigger):
            DEPENDENCIES = [T1]

        class T3(Trigger):
            DEPENDENCIES = {T1: 'foo'}
            INTERVAL = '1h'

        class T4(T1):
            DEPENDENCIES = {T1}
            INTERVAL = '1h'

        class T5(Trigger):
            NODES = {T1, T2}
            DEPENDENCIES = (T1,)
            INTERVAL = '1h'

        test_data = [
            (T0, True, False),
            (T1, False, False),
            (T2, True, False),
            (T3, True, True),
            (T4, False, True),
            (T5, False, True),
        ]

        # Save an old timestamp
        old_time = time.time()

        for T, is_meta, is_timed in test_data:
            with self.subTest(job_class=T.__name__):
                t = T(s)
                self.assertEqual(t.is_meta(), is_meta)
                self.assertEqual(t.is_timed(), is_timed)

                for reset in [False, True]:  # reset=True must be last for the next test
                    with self.subTest(task='init', is_timed=is_timed, reset=reset):
                        # Test init
                        t.init(**_get_init_kwargs(s, reset=reset, request_queue=request_queue))
                        self.assertIsInstance(t._next_submit, float)
                        if is_timed:
                            self.assertLess(t._next_submit, float('inf'))
                            self.assertGreater(t._next_submit, 0.0)
                        else:
                            self.assertEqual(t._next_submit, float('inf'))

                with self.subTest(task='visit', is_timed=is_timed):
                    # Test visit
                    self.assertEqual(t.visit(wave=old_time), NodeAction.PASS)
                    new_time = time.time()
                    self.assertEqual(t.visit(wave=new_time), NodeAction.RUN if is_timed else NodeAction.PASS)

                with self.subTest(task='submit', is_meta=is_meta):
                    # Test submit
                    if not is_meta:
                        with self.assertLogs(t.logger, logging.INFO):
                            t.submit(wave=new_time, priority=0)
                            queue_len += 1
                    else:
                        t.submit(wave=new_time, priority=0)
                    self.assertEqual(request_queue.qsize(), queue_len, 'Queue length did not match expected length')

                with self.subTest(task='cancel'):
                    t.cancel()

    def test_create_trigger_bad(self):
        s = _Scheduler(self.managers)

        class T0(Trigger):
            NODES = ['T0']

        class T1(Trigger):
            POLICY = 'Bar'

        class T2(Trigger):
            ACTION = 'foo'

        class T3(Trigger):
            INTERVAL = 2.0

        class T4(Trigger):
            DEPENDENCIES = T3

        class T5(Trigger):
            DEPENDENCIES = ['T3']

        class T6(Trigger):
            REVERSE = 'foo'

        test_data = [
            (T0, dax.base.exceptions.BuildError),
            (T1, dax.base.exceptions.BuildError),
            (T2, dax.base.exceptions.BuildError),
            (T3, AssertionError),
            (T4, AssertionError),
            (T5, TypeError),  # Caused by failing issubclass() call
            (T6, dax.base.exceptions.BuildError),
        ]

        for T, error_type in test_data:
            with self.subTest(job_class=T.__name__), self.assertRaises(error_type, msg='Bad trigger did not raise'):
                T(s)

    def test_scheduler_time_run(self):
        wave_interval = 1.0
        job_interval = (_NUM_WAVES * wave_interval) / 1.5

        class J0(_Job):
            INTERVAL = f'{job_interval}s'

        # There should be 2 submits, but there is only one if the wave interval is larger than the job interval
        num_submits = 2 if job_interval > wave_interval else 1

        class S(_Scheduler):
            NODES = {J0}
            counter = 0

            def wave(self, **kwargs) -> None:
                super(S, self).wave(**kwargs)
                self.counter += 1
                if self.counter >= _NUM_WAVES:
                    raise TerminationRequested

        self.arguments.update(self.FAST_WAVE_ARGUMENTS)

        with get_managers(arguments=self.arguments) as managers:
            s = S(managers)
            s.prepare()

            # Run the scheduler
            s.run()

            for j in s._graph:
                with self.subTest(job=j.get_name()):
                    if isinstance(j, J0):
                        ref_counter = {'init': 1, 'visit': _NUM_WAVES + num_submits,
                                       'submit': num_submits, 'schedule': num_submits}
                    else:
                        self.fail('Unexpected job type encountered')
                    self.assertDictEqual(j.counter, ref_counter,
                                         'Job call pattern did not match expected pattern')

    @unittest.skipUnless(CI_ENABLED, 'Not in a CI environment, skipping long test')
    def test_scheduler_run_long(self):
        num_waves = 10
        wave_interval = 1.0
        total_time = num_waves * wave_interval
        num_submits = 4

        class J1(_Job):
            pass

        class J0(_Job):
            DEPENDENCIES = [J1]

        class T0(_Trigger):
            NODES = [J0]
            INTERVAL = f'{total_time / (num_submits - 0.5)}s'

        class S(_Scheduler):
            NODES = {J0, J1, T0}
            counter = num_waves + num_submits

            def wave(self, **kwargs) -> None:
                super(S, self).wave(**kwargs)
                self.counter -= 1
                if self.counter <= 0:
                    raise TerminationRequested

        self.arguments.update(self.FAST_WAVE_ARGUMENTS)

        with get_managers(arguments=self.arguments) as managers:
            s = S(managers)
            s.prepare()

            # Run the scheduler
            s.run()

            for n in s._graph:
                with self.subTest(job=n.get_name()):
                    if isinstance(n, (T0, J0)):
                        ref_counter = {'init': 1, 'visit': num_waves + num_submits,
                                       'submit': num_submits, 'schedule': num_submits}
                    elif isinstance(n, J1):
                        if self.REVERSE_WAVE:
                            ref_counter = {'init': 1, 'visit': num_waves}
                        elif self.POLICY is Policy.LAZY:
                            ref_counter = {'init': 1, 'visit': num_waves + num_submits}
                        else:
                            ref_counter = {'init': 1, 'visit': num_waves + num_submits,
                                           'submit': num_submits, 'schedule': num_submits}
                    else:
                        self.fail('Unexpected job type encountered')
                    self.assertDictEqual(n.counter, ref_counter, 'Node call pattern did not match expected pattern')


class LazySchedulerReversedTestCase(LazySchedulerTestCase):
    POLICY = Policy.LAZY
    REVERSE_WAVE = True

    WAVE_0_REF: _REF_DICT_T = {
        _Job1: {'init': 1, 'visit': 3, 'submit': 1, 'schedule': 1},
        _JobB: {'init': 1, 'visit': 2, 'submit': 1, 'schedule': 1},
        None: {'init': 1, 'visit': 1},
    }

    WAVE_1_REF: _REF_DICT_T = {
        _Job1: {'init': 1, 'visit': 5, 'submit': 1, 'schedule': 1},
        _JobB: {'init': 1, 'visit': 3, 'submit': 1, 'schedule': 1},
        None: {'init': 1, 'visit': 2},
    }

    RUN_REF: _REF_DICT_T = {
        _Job1: {'init': 1, 'visit': _NUM_WAVES * 2 + 1, 'submit': 1, 'schedule': 1},
        _JobB: {'init': 1, 'visit': _NUM_WAVES + 1, 'submit': 1, 'schedule': 1},
        None: {'init': 1, 'visit': _NUM_WAVES},
    }

    RUN_DEPTH_REF: _REF_DICT_T = {
        _Job1: {'init': 1},
        _JobA: {'init': 1},
        _JobB: {'init': 1, 'visit': _NUM_WAVES + 1, 'submit': 1, 'schedule': 1},
        None: {'init': 1, 'visit': _NUM_WAVES},
    }

    RUN_START_DEPTH_REF: _REF_DICT_T = {
        _Job1: {'init': 1, 'visit': _NUM_WAVES * 2 + 1, 'submit': 1, 'schedule': 1},
        _Job4: {'init': 1},
        _JobB: {'init': 1, 'visit': _NUM_WAVES + 1, 'submit': 1, 'schedule': 1},
        _JobC: {'init': 1},
        None: {'init': 1, 'visit': _NUM_WAVES},
    }

    CONTROLLER_REF: _REF_DICT_T = {
        _Job1: {'init': 1, 'visit': (_NUM_WAVES + 1) * 2 + 1, 'submit': 1, 'schedule': 1},
        _Job4: {'init': 1, 'visit': _NUM_WAVES + 1, 'submit': 1, 'schedule': 1},
        _JobA: {'init': 1, 'visit': _NUM_WAVES + 1 + 1, 'submit': 1, 'schedule': 1},
        _JobB: {'init': 1, 'visit': _NUM_WAVES + 1 + 1, 'submit': 1, 'schedule': 1},
        None: {'init': 1, 'visit': _NUM_WAVES + 1},
    }

    CONTROLLER_REVERSED_REF: _REF_DICT_T = {
        _Job1: {'init': 1, 'visit': _NUM_WAVES * 2 + 1 + 2, 'submit': 1, 'schedule': 1},
        _Job4: {'init': 1, 'visit': _NUM_WAVES + 1, 'submit': 1, 'schedule': 1},
        _JobB: {'init': 1, 'visit': _NUM_WAVES + 1 + 2, 'submit': 2, 'schedule': 2},
        _JobC: {'init': 1, 'visit': _NUM_WAVES},
        None: {'init': 1, 'visit': _NUM_WAVES + 1},
    }

    CONTROLLER_START_DEPTH_REF: _REF_DICT_T = {
        _Job1: {'init': 1, 'visit': (_NUM_WAVES + 1) * 2 + 1, 'submit': 1, 'schedule': 1},
        _Job2: {'init': 1, 'visit': _NUM_WAVES + 1, 'submit': 1, 'schedule': 1},
        _Job3: {'init': 1, 'visit': _NUM_WAVES + 1, 'submit': 1, 'schedule': 1},
        _Job4: {'init': 1, 'visit': _NUM_WAVES},
        _JobB: {'init': 1, 'visit': _NUM_WAVES + 1 + 1, 'submit': 1, 'schedule': 1},
        None: {'init': 1, 'visit': _NUM_WAVES + 1},
    }


class GreedySchedulerTestCase(LazySchedulerTestCase):
    POLICY = Policy.GREEDY
    REVERSE_WAVE = False

    def test_scheduler_unreachable_jobs(self):
        with self.assertRaises(self.failureException, msg='Expected test failure did not happen'):
            # With a greedy policy, all jobs are reachable and the call to super will cause a test failure
            super(GreedySchedulerTestCase, self).test_scheduler_unreachable_jobs()

    WAVE_0_REF: _REF_DICT_T = {
        _Job1: {'init': 1, 'visit': 2, 'submit': 1, 'schedule': 1},
        _Job4: {'init': 1, 'visit': 2, 'submit': 1, 'schedule': 1},
        _JobA: {'init': 1, 'visit': 1},
        _JobB: {'init': 1, 'visit': 2, 'submit': 1, 'schedule': 1},
        None: {'init': 1, 'visit': 1, 'submit': 1, 'schedule': 1},
    }

    WAVE_1_REF: _REF_DICT_T = {
        _Job1: {'init': 1, 'visit': 3, 'submit': 1, 'schedule': 1},
        _Job4: {'init': 1, 'visit': 4, 'submit': 1, 'schedule': 1},
        _JobA: {'init': 1, 'visit': 2},
        _JobB: {'init': 1, 'visit': 3, 'submit': 1, 'schedule': 1},
        None: {'init': 1, 'visit': 2, 'submit': 1, 'schedule': 1},
    }

    def _check_scheduler_datastore_calls(self, scheduler):
        # Check data store calls (only jobs that are not meta-jobs perform a call)
        self.assertEqual(len(scheduler.data_store.method_calls), 3,
                         'Data store was called an unexpected number of times')

    CONTROLLER_REF: _REF_DICT_T = {
        _Job1: {'init': 1, 'visit': _NUM_WAVES + 1, 'submit': 1, 'schedule': 1},
        _Job2: {'init': 1, 'visit': _NUM_WAVES, 'submit': 1, 'schedule': 1},
        _Job3: {'init': 1, 'visit': _NUM_WAVES, 'submit': 1, 'schedule': 1},
        _Job4: {'init': 1, 'visit': _NUM_WAVES * 2 + 1, 'submit': 1 + 1, 'schedule': 1 + 1},
        _JobA: {'init': 1, 'visit': _NUM_WAVES + 1, 'submit': 1, 'schedule': 1},
        _JobB: {'init': 1, 'visit': _NUM_WAVES + 1 + 1 + 1, 'submit': 1 + 1, 'schedule': 1 + 1},
        _JobC: {'init': 1, 'visit': _NUM_WAVES + 1 + 1, 'submit': 1 + 1, 'schedule': 1 + 1},
    }

    CONTROLLER_REVERSED_REF: _REF_DICT_T = {
        _Job1: {'init': 1, 'visit': _NUM_WAVES + 1 + 3, 'submit': 2, 'schedule': 2},
        _Job2: {'init': 1, 'visit': _NUM_WAVES + 1, 'submit': 2, 'schedule': 2},
        _Job3: {'init': 1, 'visit': _NUM_WAVES + 1, 'submit': 2, 'schedule': 2},
        _Job4: {'init': 1, 'visit': _NUM_WAVES * 2 + 1, 'submit': 2, 'schedule': 2},
        _JobA: {'init': 1, 'visit': _NUM_WAVES + 1, 'submit': 1, 'schedule': 1},
        _JobB: {'init': 1, 'visit': _NUM_WAVES + 1 + 2, 'submit': 2, 'schedule': 2},
        _JobC: {'init': 1, 'visit': _NUM_WAVES, 'submit': 1, 'schedule': 1},
    }

    CONTROLLER_START_DEPTH_REF: _REF_DICT_T = {
        _Job1: {'init': 1, 'visit': _NUM_WAVES + 1, 'submit': 1, 'schedule': 1},
        _Job2: {'init': 1, 'visit': _NUM_WAVES, 'submit': 1, 'schedule': 1},
        _Job3: {'init': 1, 'visit': _NUM_WAVES, 'submit': 1, 'schedule': 1},
        _Job4: {'init': 1, 'visit': _NUM_WAVES * 2, 'submit': 1, 'schedule': 1},
        _JobA: {'init': 1, 'visit': _NUM_WAVES},
        _JobB: {'init': 1, 'visit': _NUM_WAVES + 1 + 1 + 1, 'submit': 1 + 1, 'schedule': 1 + 1},
        _JobC: {'init': 1, 'visit': _NUM_WAVES + 1 + 1, 'submit': 1 + 1, 'schedule': 1 + 1},
    }

    RUN_REF: _REF_DICT_T = {
        _Job1: {'init': 1, 'visit': _NUM_WAVES + 1, 'submit': 1, 'schedule': 1},
        _Job4: {'init': 1, 'visit': _NUM_WAVES * 2, 'submit': 1, 'schedule': 1},
        _JobA: {'init': 1, 'visit': _NUM_WAVES},
        _JobB: {'init': 1, 'visit': _NUM_WAVES + 1, 'submit': 1, 'schedule': 1},
        None: {'init': 1, 'visit': _NUM_WAVES, 'submit': 1, 'schedule': 1},
    }

    RUN_DEPTH_REF: _REF_DICT_T = {
        _Job1: {'init': 1, 'visit': _NUM_WAVES + 1, 'submit': 1, 'schedule': 1},
        _Job2: {'init': 1, 'visit': _NUM_WAVES, 'submit': 1, 'schedule': 1},
        _Job3: {'init': 1, 'visit': _NUM_WAVES, 'submit': 1, 'schedule': 1},
        _JobA: {'init': 1, 'visit': _NUM_WAVES},
        _JobB: {'init': 1, 'visit': _NUM_WAVES + 1, 'submit': 1, 'schedule': 1},
        None: {'init': 1},
    }

    RUN_START_DEPTH_REF: _REF_DICT_T = {
        _Job2: {'init': 1, 'visit': _NUM_WAVES},
        _Job3: {'init': 1, 'visit': _NUM_WAVES},
        _Job4: {'init': 1, 'visit': _NUM_WAVES * 2},
        _JobB: {'init': 1, 'visit': _NUM_WAVES + 1, 'submit': 1, 'schedule': 1},
        _JobC: {'init': 1, 'visit': _NUM_WAVES, 'submit': 1, 'schedule': 1},
        None: {'init': 1},
    }


class GreedySchedulerReversedTestCase(GreedySchedulerTestCase):
    POLICY = Policy.GREEDY
    REVERSE_WAVE = True

    WAVE_0_REF: _REF_DICT_T = {
        _Job1: {'init': 1, 'visit': 2 + 1, 'submit': 1, 'schedule': 1},
        _JobA: {'init': 1, 'visit': 1, 'submit': 1, 'schedule': 1},
        _JobB: {'init': 1, 'visit': 2, 'submit': 1, 'schedule': 1},
        None: {'init': 1, 'visit': 1},
    }

    WAVE_1_REF: _REF_DICT_T = {
        _Job1: {'init': 1, 'visit': 3 + 2, 'submit': 1, 'schedule': 1},
        _JobA: {'init': 1, 'visit': 2, 'submit': 1, 'schedule': 1},
        _JobB: {'init': 1, 'visit': 3, 'submit': 1, 'schedule': 1},
        None: {'init': 1, 'visit': 2},
    }

    def _check_scheduler_datastore_calls(self, scheduler):
        # Check data store calls (only jobs that are not meta-jobs perform a call)
        self.assertEqual(len(scheduler.data_store.method_calls), 2,
                         'Data store was called an unexpected number of times')

    CONTROLLER_REF: _REF_DICT_T = {
        _Job1: {'init': 1, 'visit': _NUM_WAVES * 2 + 1 + 2 + 1, 'submit': 2, 'schedule': 2},
        _JobA: {'init': 1, 'visit': _NUM_WAVES + 1 + 1, 'submit': 2, 'schedule': 2},
        _JobB: {'init': 1, 'visit': _NUM_WAVES + 1 + 1, 'submit': 1, 'schedule': 1},
        _JobC: {'init': 1, 'visit': _NUM_WAVES + 1},
        None: {'init': 1, 'visit': _NUM_WAVES + 1, 'submit': 1, 'schedule': 1},
    }

    CONTROLLER_REVERSED_REF: _REF_DICT_T = {
        _Job1: {'init': 1, 'visit': _NUM_WAVES * 2 + 1 + 2 + 1, 'submit': 2, 'schedule': 2},
        _JobA: {'init': 1, 'visit': _NUM_WAVES + 1, 'submit': 2, 'schedule': 2},
        _JobB: {'init': 1, 'visit': _NUM_WAVES + 1 + 2, 'submit': 2, 'schedule': 2},
        _JobC: {'init': 1, 'visit': _NUM_WAVES},
        None: {'init': 1, 'visit': _NUM_WAVES + 1, 'submit': 1, 'schedule': 1},
    }

    CONTROLLER_START_DEPTH_REF: _REF_DICT_T = {
        _Job1: {'init': 1, 'visit': _NUM_WAVES * 2 + 1 + 2 + 1, 'submit': 2, 'schedule': 2},
        _Job4: {'init': 1, 'visit': _NUM_WAVES},
        _JobB: {'init': 1, 'visit': _NUM_WAVES + 1 + 1, 'submit': 1, 'schedule': 1},
        _JobC: {'init': 1, 'visit': _NUM_WAVES + 1},
        None: {'init': 1, 'visit': _NUM_WAVES + 1, 'submit': 1, 'schedule': 1},
    }

    RUN_REF: _REF_DICT_T = {
        _Job1: {'init': 1, 'visit': _NUM_WAVES * 2 + 1, 'submit': 1, 'schedule': 1},
        _JobA: {'init': 1, 'visit': _NUM_WAVES, 'submit': 1, 'schedule': 1},
        _JobB: {'init': 1, 'visit': _NUM_WAVES + 1, 'submit': 1, 'schedule': 1},
        None: {'init': 1, 'visit': _NUM_WAVES},
    }

    RUN_DEPTH_REF: _REF_DICT_T = {
        _Job1: {'init': 1},
        _JobA: {'init': 1},
        _JobB: {'init': 1, 'visit': _NUM_WAVES + 1, 'submit': 1, 'schedule': 1},
        None: {'init': 1, 'visit': _NUM_WAVES},
    }

    RUN_START_DEPTH_REF: _REF_DICT_T = {
        _Job1: {'init': 1, 'visit': _NUM_WAVES * 2 + 1, 'submit': 1, 'schedule': 1},
        _Job2: {'init': 1, 'visit': _NUM_WAVES},
        _Job3: {'init': 1, 'visit': _NUM_WAVES},
        _JobA: {'init': 1, 'visit': _NUM_WAVES, 'submit': 1, 'schedule': 1},
        _JobB: {'init': 1, 'visit': _NUM_WAVES + 1, 'submit': 1, 'schedule': 1},
        None: {'init': 1},
    }


class _CalJob0(CalibrationJob):
    pass


@create_calibration
class _CalJob1(CalibrationJob):
    pass


class CalibrationJobTestCase(unittest.TestCase):
    POLICY = Policy.LAZY
    REVERSE_WAVE = False

    def setUp(self) -> None:
        self.arguments: typing.Dict[str, typing.Any] = {'Scheduling policy': str(self.POLICY),
                                                        'Reverse wave': self.REVERSE_WAVE,
                                                        'Job pipeline': 'test_pipeline',
                                                        'View graph': False}
        self.managers = get_managers(arguments=self.arguments)

    def tearDown(self) -> None:
        # Close managers
        self.managers.close()

    def test_create_job_bad(self):
        s = _Scheduler(self.managers)

        class J0(CalibrationJob):
            pass

        class J1(CalibrationJob):
            _META_EXP_FILE = '/home/foo/scheduler.py'
            CHECK_FILE = 'foo.py'

        class J2(CalibrationJob):
            _META_EXP_FILE = '/home/foo/scheduler.py'
            CHECK_FILE = 'foo.py'
            CALIBRATION_FILE = 'foo.py'

        class J3(CalibrationJob):
            _META_EXP_FILE = '/home/foo/scheduler.py'
            CHECK_FILE = 'foo.py'
            CALIBRATION_FILE = 'foo.py'
            CHECK_CLASS_NAME = 'Bar'

        class J4(CalibrationJob):
            _META_EXP_FILE = '/home/foo/scheduler.py'
            CHECK_FILE = 'foo.py'
            CALIBRATION_FILE = 'foo.py'
            CALIBRATION_CLASS_NAME = 'Bar'

        class J5(CalibrationJob):
            _META_EXP_FILE = '/home/foo/scheduler.py'
            CHECK_FILE = 'foo.py'
            CALIBRATION_FILE = 'foo.py'
            CHECK_CLASS_NAME = 'Bar'
            CALIBRATION_CLASS_NAME = 'Bar'
            CHECK_ARGUMENTS = {1: 1}

        class J6(CalibrationJob):
            _META_EXP_FILE = '/home/foo/scheduler.py'
            CHECK_FILE = 'foo.py'
            CALIBRATION_FILE = 'foo.py'
            CHECK_CLASS_NAME = 'Bar'
            CALIBRATION_CLASS_NAME = 'Bar'
            CALIBRATION_ARGUMENTS = {1: 1}

        class J7(CalibrationJob):
            _META_EXP_FILE = '/home/foo/scheduler.py'
            CHECK_FILE = 'foo.py'
            CALIBRATION_FILE = 'foo.py'
            CHECK_CLASS_NAME = 'Bar'
            CALIBRATION_CLASS_NAME = 'Bar'
            CALIBRATION_TIMEOUT = 2.0

        test_data = [
            (J0, dax.base.exceptions.BuildError),
            (J1, dax.base.exceptions.BuildError),
            (J2, dax.base.exceptions.BuildError),
            (J3, dax.base.exceptions.BuildError),
            (J4, dax.base.exceptions.BuildError),
            (J5, dax.base.exceptions.BuildError),
            (J6, dax.base.exceptions.BuildError),
            (J7, dax.base.exceptions.BuildError),
        ]

        for J, error_type in test_data:
            with self.subTest(job_class=J.__name__), self.assertRaises(error_type, msg='Bad job did not raise'):
                J(s)

    def test_create_node_bad_parent(self):
        class J0(CalibrationJob):
            pass

        with self.assertRaises(TypeError, msg='Wrong calibration job parent type did not raise'):
            # noinspection PyTypeChecker
            J0(self.managers)

    def test_job_arguments(self):
        s = _Scheduler(self.managers)

        original_check_arguments = {'foo': 1,
                                    'range': RangeScan(1, 10, 9),
                                    'center': CenterScan(1, 10, 9),
                                    'explicit': ExplicitScan([1, 10, 9]),
                                    'no': NoScan(10)}

        original_cal_arguments = {'foo': -1,
                                  'range': RangeScan(5, 10, 9),
                                  'center': CenterScan(1, 5, 9),
                                  'explicit': ExplicitScan([2, 10, 9]),
                                  'no': NoScan(3)}

        class J0(CalibrationJob):
            _META_EXP_FILE = '/home/foo/scheduler.py'
            CHECK_FILE = '/home/foo/foo.py'
            CALIBRATION_FILE = '/home/foo/foo.py'
            CHECK_CLASS_NAME = 'Bar'
            CALIBRATION_CLASS_NAME = 'Bar'
            REPOSITORY = False  # need this + abs file paths because init() tries to find repository dir otherwise
            CHECK_ARGUMENTS = original_check_arguments.copy()
            CALIBRATION_ARGUMENTS = original_cal_arguments.copy()

        j = J0(s)
        # CalibrationJob doesn't build _arguments until init()
        j.init(reset=False,
               job_pipeline='test_pipeline',
               request_queue=None,
               controller_key='dax_scheduler')
        check_arguments_ref = process_arguments(original_check_arguments)
        cal_arguments_ref = process_arguments(original_cal_arguments)
        self.assertDictEqual(check_arguments_ref, j._arguments['check'])
        self.assertDictEqual(cal_arguments_ref, j._arguments['calibration'])
        self.assertDictEqual(original_check_arguments, J0.CHECK_ARGUMENTS, 'Class check arguments were mutated')
        self.assertDictEqual(original_cal_arguments, J0.CALIBRATION_ARGUMENTS,
                             'Class calibration arguments were mutated')

    def test_job_configurable_arguments(self):
        s = _Scheduler(self.managers)

        original_check_arguments = {'foo': 1,
                                    'range': RangeScan(1, 10, 9),
                                    'center': CenterScan(1, 10, 9),
                                    'explicit': ExplicitScan([1, 10, 9]),
                                    'no': NoScan(10)}

        original_cal_arguments = {'foo': -1,
                                  'range': RangeScan(5, 10, 9),
                                  'center': CenterScan(1, 5, 9),
                                  'explicit': ExplicitScan([2, 10, 9]),
                                  'no': NoScan(3)}

        check_keyword_arguments: typing.Dict[str, typing.Any] = {
            'bar': NumberValue(20),
            'baz': Scannable(CenterScan(100, 50, 2)),
            'foobar': Scannable(RangeScan(100, 300, 40)),
        }

        cal_keyword_arguments: typing.Dict[str, typing.Any] = {
            'bar': NumberValue(0),
            'baz': Scannable(CenterScan(10, 50, 2)),
            'foobar': Scannable(RangeScan(10, 300, 40)),
        }

        class J0(CalibrationJob):
            _META_EXP_FILE = '/home/foo/scheduler.py'
            CHECK_FILE = '/home/foo/foo.py'
            CALIBRATION_FILE = '/home/foo/foo.py'
            CHECK_CLASS_NAME = 'Bar'
            CALIBRATION_CLASS_NAME = 'Bar'
            REPOSITORY = False  # need this + abs file paths because init() tries to find repository dir otherwise
            CHECK_ARGUMENTS = original_check_arguments.copy()
            CALIBRATION_ARGUMENTS = original_cal_arguments.copy()

            def build_job(self) -> None:
                for key, argument in check_keyword_arguments.items():
                    self.CHECK_ARGUMENTS[key] = self.get_argument(key, argument)
                for key, argument in cal_keyword_arguments.items():
                    self.CALIBRATION_ARGUMENTS[key] = self.get_argument(key, argument)

        j = J0(s)
        j.init(reset=False,
               job_pipeline='test_pipeline',
               request_queue=None,
               controller_key='dax_scheduler')
        check_arguments_ref = original_check_arguments.copy()
        check_arguments_ref.update({k: v.default() for k, v in check_keyword_arguments.items()})
        check_arguments_ref = process_arguments(check_arguments_ref)
        cal_arguments_ref = original_cal_arguments.copy()
        cal_arguments_ref.update({k: v.default() for k, v in cal_keyword_arguments.items()})
        cal_arguments_ref = process_arguments(cal_arguments_ref)

        self.assertDictEqual(check_arguments_ref, j._arguments['check'])
        self.assertDictEqual(cal_arguments_ref, j._arguments['calibration'])
        self.assertEqual(len(j._arguments['check']), len(original_check_arguments) + len(check_keyword_arguments))
        self.assertEqual(len(j._arguments['calibration']), len(original_cal_arguments) + len(cal_keyword_arguments))
        self.assertDictEqual(original_check_arguments, J0.CHECK_ARGUMENTS, 'Class check arguments were mutated')
        self.assertDictEqual(original_cal_arguments, J0.CALIBRATION_ARGUMENTS,
                             'Class calibration arguments were mutated')
        for v in j._arguments['check'].values():
            self.assertNotIsInstance(v, ScanObject)
        for v in j._arguments['calibration'].values():
            self.assertNotIsInstance(v, ScanObject)

    @staticmethod
    def test_decorator():
        assert _CalJob0._meta_exp_name() not in globals()
        assert _CalJob1._meta_exp_name() in globals()
        assert not hasattr(_CalJob0, '_META_EXP_FILE')
        assert hasattr(_CalJob1, '_META_EXP_FILE')


class OptimusCalibrationTestCase(unittest.TestCase):
    def setUp(self) -> None:
        # master subprocess needs to be able to import dax
        master_env = os.environ.copy()
        master_env['PYTHONPATH'] = f'{os.path.dirname(dax.__dax_dir__)}:{master_env.get("PYTHONPATH", "")}'
        # use temp_dir context for running experiments
        self.temp_dir = temp_dir()
        self.wd = self.temp_dir.__enter__()
        # write device_db to temp dir
        with open(os.path.join(self.wd, 'device_db.py'), 'x') as f:
            f.write(f'device_db = {str(_DEVICE_DB)}')
        # create repository
        os.mkdir(os.path.join(self.wd, 'repository'))
        # start master
        master_cmd = [sys.executable, '-u', '-m', 'artiq.frontend.artiq_master',
                      '--no-localhost-bind', '--bind', _LOCALHOST, '--port-notify', str(_ARTIQ_MASTER_NOTIFY_PORT)]
        self.master = subprocess.Popen(master_cmd, stdout=subprocess.PIPE, universal_newlines=True, env=master_env,
                                       bufsize=1)
        self.master.__enter__()
        master_ready = False
        assert isinstance(self.master.stdout, io.TextIOBase)
        for line in iter(self.master.stdout.readline, ''):
            sys.stdout.write(line)
            if line.rstrip() == 'ARTIQ master is now ready.':
                master_ready = True
                break
        if not master_ready:
            raise Exception('ARTIQ master failed to start')

    def tearDown(self) -> None:
        self.master.terminate()
        self.master.__exit__(None, None, None)
        time.sleep(5)  # wait a bit to make sure all subprocesses are closed
        self.temp_dir.__exit__(None, None, None)

    @staticmethod
    def _create_scheduler_exp(node_indices: typing.Set[int]) -> str:
        return textwrap.dedent(f'''
        class Scheduler(DaxScheduler, Experiment):
            NAME = 'scheduler'
            NODES = {{{', '.join(f'CalJob{i}' for i in node_indices)}, CalTrigger}}
            CONTROLLER = 'dax_scheduler'
            DEFAULT_SCHEDULING_POLICY = Policy.GREEDY
            DEFAULT_RESET_NODES = False
            DEFAULT_CANCEL_NODES = True
            LOG_LEVEL = logging.INFO

            def run(self):
                self.set_dataset('cal_order', [], archive=False, persist=True, broadcast=True)
                now: float = time.time()
                for node_name, node in self._node_name_map.items():
                    if node_name == 'CalTrigger':
                        self.set_dataset(f'{{self.NAME}}.{{node_name}}.{{node._LAST_SUBMIT_KEY}}', 0.0,
                                         archive=False, persist=True, broadcast=True)
                    else:
                        self.set_dataset(f'{{self.NAME}}.{{node_name}}.{{node.LAST_CHECK_KEY}}', now,
                                         archive=False, persist=True, broadcast=True)
                        self.set_dataset(f'{{self.NAME}}.{{node_name}}.{{node.LAST_CAL_KEY}}', now,
                                         archive=False, persist=True, broadcast=True)
                super(Scheduler, self).run()

            def _render_graph(self) -> None:
                # Skip render graph, prevents output and call to the renderer
                pass
        ''')

    @staticmethod
    def _create_trigger(root_indices: typing.Set[int]) -> str:
        return textwrap.dedent(f'''
        class CalTrigger(Trigger):
            NODES = {{{', '.join(f'CalJob{i}' for i in root_indices)}}}
            INTERVAL = '1h'
            POLICY = Policy.GREEDY

        ''')

    @staticmethod
    def _create_calibration_job(index: int, dep_indices: typing.Set[int], timeout: bool) -> str:
        return textwrap.dedent(f'''
        @create_calibration
        class CalJob{index}(CalibrationJob):
            CHECK_FILE = 'calibrations.py'
            CALIBRATION_FILE = 'calibrations.py'
            CHECK_CLASS_NAME = 'Check{index}'
            CALIBRATION_CLASS_NAME = 'Cal{index}'
            CALIBRATION_TIMEOUT = {None if timeout else '"1h"'}
            LOG_LEVEL = logging.INFO
            DEPENDENCIES = {{{', '.join(f'CalJob{i}' for i in dep_indices)}}}

            def submit(self, *args, **kwargs):
                if not self.get_dataset_sys(self.DIAGNOSE_FLAG_KEY, False):
                    self.append_to_dataset('cal_order', type(self).__name__)
                super(CalibrationJob, self).submit(*args, **kwargs)

        ''')

    @staticmethod
    def _create_check_exp(index: int, result: int) -> str:
        result_strings = ['return', 'raise OutOfSpecError', 'raise BadDataError']
        return textwrap.dedent(f'''
        class Check{index}(EnvExperiment):
            def run(self):
                order = self.get_dataset('cal_order', [], archive=False)
                order.append(type(self).__name__)
                self.set_dataset('cal_order', order, archive=False, persist=True, broadcast=True)
                {result_strings[result]}

        ''')

    @staticmethod
    def _create_cal_exp(index: int, result: int) -> str:
        result_strings = ['return', 'raise FailedCalibrationError']
        return textwrap.dedent(f'''
        class Cal{index}(EnvExperiment):
            def run(self):
                order = self.get_dataset('cal_order', [], archive=False)
                order.append(type(self).__name__)
                self.set_dataset('cal_order', order, archive=False, persist=True, broadcast=True)
                {result_strings[result]}

        ''')

    @staticmethod
    def _create_scheduler_imports() -> str:
        return textwrap.dedent('''
        from dax.scheduler import *
        import logging
        import time

        ''')

    @staticmethod
    def _create_exp_imports() -> str:
        return textwrap.dedent('''
        from artiq.experiment import *
        from dax.base.exceptions import OutOfSpecError, BadDataError, FailedCalibrationError

        ''')

    @staticmethod
    def _random_dag(num_nodes: int, p: float, rng: np.random.Generator) -> nx.DiGraph:  # type: ignore[name-defined]
        # method 1 from https://doi.org/10.1016/0167-6377(86)90066-0
        # no point in finding transitive closure since we want transitive reduction anyway
        assert isinstance(num_nodes, int) and num_nodes > 0
        assert isinstance(p, float) and 0 <= p <= 1
        adj = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                adj[i, j] = rng.choice([1, 0], p=[p, 1 - p])  # type: ignore
        # redundant since scheduler also calls this, but need to make sure we're working with the exact same graph
        g: nx.DiGraph = nx.convert_matrix.from_numpy_array(adj, create_using=nx.DiGraph)  # type: ignore
        g = nx.algorithms.transitive_reduction(g)
        if not nx.algorithms.is_directed_acyclic_graph(g):  # don't think this should happen, but can't hurt to check
            raise Exception('Non-DAG graph created')
        return g

    @staticmethod
    def _get_root_nodes(g: nx.DiGraph) -> typing.Set[int]:
        # noinspection PyTypeChecker
        return {node for node, degree in g.in_degree if degree == 0}

    @staticmethod
    def _wait_and_get_datasets(scheduler_rid: int) -> typing.Dict:
        async def get_datasets():
            d_ = {}

            def sub_init(x):
                d_.clear()
                d_.update(x)
                # note: this is not a real exception - it's a hack to break out of the Subscriber _receive_cr loop
                raise ConnectionError

            async def subscribe():
                s = Subscriber('datasets', sub_init)
                try:
                    await s.connect(_LOCALHOST, _ARTIQ_MASTER_NOTIFY_PORT)
                    await asyncio.wait_for(s.receive_task, None)
                finally:
                    await s.close()

            await subscribe()
            return d_

        async def wait_for_scheduler():
            d_ = {}

            def sub_init(x):
                d_.clear()
                d_.update(x)
                return d_

            # noinspection PyUnusedLocal
            def sub_mod(mod):
                if len(d_) <= 1:  # Also raise if there is no experiment in the schedule
                    # note: this is not a real exception - it's a hack to break out of the Subscriber _receive_cr loop
                    raise Exception('This exception is just used to exit the subscriber')

            async def subscribe():
                s = Subscriber('schedule', sub_init, notify_cb=sub_mod)
                try:
                    await s.connect(_LOCALHOST, _ARTIQ_MASTER_NOTIFY_PORT)
                    await asyncio.wait_for(s.receive_task, None)
                finally:
                    await s.close()

            await subscribe()
            assert scheduler_rid in d_, f'DAX Scheduler experiment not present in ARTIQ schedule: {d_}'

        asyncio.run(wait_for_scheduler())
        d: typing.Dict = asyncio.run(get_datasets())
        return d

    def _flatten(self, list_):
        for el in list_:
            if isinstance(el, list):
                yield from self._flatten(el)
            else:
                yield el

    def _verify_actions(self, expected: typing.List, actual: typing.List[str]) -> bool:
        if len(actual) != len(list(self._flatten(expected))):
            return False

        def _verify(expected_item: typing.List, cur_idx: int) -> typing.Tuple[bool, int]:
            expected_length = len(expected_item)
            if not expected_length:
                return True, cur_idx
            elif expected_length == 1:  # ['Check#']
                actual_item = actual[cur_idx]
                cur_idx += 1
                return bool(expected_item[0] == actual_item), cur_idx
            else:  # ['Check#', 'Cal#'] or ['Check#', <any number of nested lists>, 'Cal#']
                actual_item = actual[cur_idx]
                cur_idx += 1
                if expected_item[0] != actual_item:
                    return False, cur_idx
                exp_sublist = expected_item[1:-1]
                while exp_sublist:
                    actual_item = actual[cur_idx]
                    next_exp = []
                    for k, sub_item in enumerate(exp_sublist):
                        if sub_item[0] == actual_item:
                            next_exp = exp_sublist.pop(k)
                            break
                    if not next_exp:
                        return False, cur_idx
                    rv_, cur_idx = _verify(next_exp, cur_idx)
                    if not rv_:
                        return False, cur_idx
                actual_item = actual[cur_idx]
                cur_idx += 1
                return bool(actual_item == expected_item[-1]), cur_idx

        idx = 0
        for el in expected:
            rv, idx = _verify(el, idx)
            if not rv:
                return False
        return True

    @staticmethod
    def _submit_scheduler_exp() -> int:
        # submit scheduler experiment via artiq_client
        client_cmd = [sys.executable, '-u', '-m', 'artiq.frontend.artiq_client', '-s', _LOCALHOST,
                      'submit', '-p', 'scheduler', '-R', '-c', 'Scheduler', 'scheduler.py']
        client_res = subprocess.run(client_cmd, capture_output=True, universal_newlines=True)
        if client_res.returncode != 0:
            raise Exception(
                f'ARTIQ client submit exited with return code {client_res.returncode}: "{client_res.stderr}"')
        # normal output is of the format 'RID: \d+'
        # regex would probably be more robust, but slower(?)
        rid: int = int(client_res.stdout.strip().split(' ')[1])
        return rid

    @classmethod
    def _delete_exp(cls, rid: int) -> None:
        # delete scheduler experiment via artiq_client
        client_cmd = [sys.executable, '-u', '-m', 'artiq.frontend.artiq_client', '-s', _LOCALHOST,
                      'delete', '-g', str(rid)]
        client_res = subprocess.run(client_cmd, capture_output=True, universal_newlines=True)
        if client_res.returncode != 0:
            raise Exception(
                f'ARTIQ client delete exited with return code {client_res.returncode}: "{client_res.stderr}"')

    @classmethod
    def _delete_all_exp(cls, rids: typing.Collection[int]) -> None:
        for rid in rids:
            cls._delete_exp(rid)

    @unittest.skipUnless(CI_ENABLED, 'Not in a CI environment, skipping long test')
    def test_optimus(self):
        # test the algorithm without failed calibrations (which aren't really addressed in the initial algorithm)
        p = .5
        num_reps = 5
        for num_nodes in range(5, 10):
            for _ in range(num_reps):
                # initialize seed and rng for numpy.random calls
                seed: int = time.time_ns()
                rng: np.random.Generator = np.random.default_rng(seed)
                # create graph
                g: nx.DiGraph = self._random_dag(num_nodes, p, rng)
                root_nodes: typing.Set[int] = self._get_root_nodes(g)
                # randomly choose results for each experiment - no calibration failure though
                timeouts: typing.List[bool] = [rng.choice([True, False]) for _ in range(num_nodes)]
                results: typing.List[int] = [rng.choice([0, 1, 2]) for _ in range(num_nodes)]
                # create the scheduler/experiment files - use reversed topological sort so that jobs are written
                # in correct dependency order in scheduler file
                ts_rev = list(reversed(list(nx.topological_sort(g))))
                with open(os.path.join(self.wd, 'repository', 'scheduler.py'), 'w') as scheduler_file:
                    with open(os.path.join(self.wd, 'repository', 'calibrations.py'), 'w') as experiment_file:
                        scheduler_file.write(self._create_scheduler_imports())
                        experiment_file.write(self._create_exp_imports())
                        for idx in ts_rev:
                            scheduler_file.write(self._create_calibration_job(idx, g.successors(idx), timeouts[idx]))
                            experiment_file.write(self._create_check_exp(idx, results[idx]))
                            experiment_file.write(self._create_cal_exp(idx, 0))
                        scheduler_file.write(self._create_trigger(root_nodes))
                        scheduler_file.write(self._create_scheduler_exp(set(g.nodes)))

                rid: int = self._submit_scheduler_exp()
                time.sleep(5)  # need to wait a bit for the DAX scheduler to submit jobs
                datasets: typing.Dict[str, typing.Tuple[bool, typing.Any]] = self._wait_and_get_datasets(rid)
                cal_order: typing.List[str] = datasets['cal_order'][1]
                self._delete_exp(rid)
                # again, not as robust as a regex for picking the \d+ out of 'CalJob\d+', but faster
                initial_submit: typing.List[int] = [int(node_str[6:]) for node_str in cal_order[:num_nodes]]
                last_cals: typing.List[float] = [0.0] * num_nodes

                def get_actions(node: int, diagnose: bool = False) -> typing.List:
                    actions: typing.List = []
                    # sort according to initial_submit order to preserve graph structure (kind of, at least)
                    children = sorted(list(g.successors(node)), key=lambda item: initial_submit.index(item))
                    if timeouts[node] or diagnose or any([last_cals[n] > last_cals[node] for n in children]):
                        actions.append(f'Check{node}')
                        if results[node] == 2:
                            for n in children:
                                actions.append(get_actions(n, diagnose=True))
                        if results[node] != 0:
                            last_cals[node] = time.time()
                            actions.append(f'Cal{node}')
                    return actions

                expected_actions = [get_actions(node) for node in initial_submit]
                expected_actions_flat = list(self._flatten(expected_actions))
                actual_actions = cal_order[num_nodes:]
                error_str = textwrap.dedent(f'''
                Expected and actual traversals do not match:
                expected: {expected_actions}
                expected (flat): {expected_actions_flat}
                actual: {actual_actions}

                initial submit order: {initial_submit}
                root nodes: {root_nodes}
                deps: {' '.join([f'{node} -> {{{list(g.successors(node))}}}' for node in initial_submit])}
                timeouts: {timeouts}
                exp results: {results}
                rng seed: {seed}
                ''')
                with self.subTest(num_nodes=num_nodes, p=p, expected_actions=expected_actions,
                                  acutal_actions=actual_actions):
                    self.assertTrue(self._verify_actions(expected_actions, actual_actions), error_str)

    @unittest.skipUnless(CI_ENABLED, 'Not in a CI environment, skipping long test')
    def test_barrier(self):
        # verify that the barrier experiment is scheduled (and takes precedence) in the case of a failure
        with open(os.path.join(self.wd, 'repository', 'scheduler.py'), 'w') as scheduler_file:
            with open(os.path.join(self.wd, 'repository', 'calibrations.py'), 'w') as experiment_file:
                scheduler_file.write(self._create_scheduler_imports())
                experiment_file.write(self._create_exp_imports())
                scheduler_file.write(self._create_calibration_job(0, set(), True))
                experiment_file.write(self._create_check_exp(0, 1))  # out of spec to force calibration
                experiment_file.write(self._create_cal_exp(0, 1))  # calibration failure
                scheduler_file.write(self._create_trigger({0}))
                scheduler_file.write(self._create_scheduler_exp({0}))
        rid: int = self._submit_scheduler_exp()
        time.sleep(5)  # need to wait a bit for the DAX scheduler to submit jobs

        async def get_schedule():
            d_ = {}

            def sub_init(x):
                d_.clear()
                d_.update(x)
                raise ConnectionError

            async def subscribe():
                s = Subscriber('schedule', sub_init)
                try:
                    await s.connect(_LOCALHOST, _ARTIQ_MASTER_NOTIFY_PORT)
                    await asyncio.wait_for(s.receive_task, None)
                finally:
                    await s.close()

            await subscribe()
            return d_

        schedule = asyncio.run(get_schedule())
        self._delete_exp(rid)
        self._delete_all_exp(schedule)
        barriers = [s for s in schedule.values() if s['expid']['class_name'] == 'Barrier' and s['status'] == 'running']
        if len(barriers) != 1:
            self.fail(f'Barrier experiment is either not in the schedule or not running. Current schedule:\n{schedule}')
