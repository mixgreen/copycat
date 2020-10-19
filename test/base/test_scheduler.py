import unittest
import time
import logging
import typing
import collections
import contextlib
import socket
import abc
import asyncio
from unittest.mock import MagicMock, call

from artiq.language.scan import *
from artiq.experiment import TerminationRequested, NumberValue
import artiq.frontend.artiq_run  # type: ignore

from dax.base.scheduler import *
import dax.base.scheduler
from dax.util.artiq import get_managers
import dax.base.system
import dax.base.exceptions

from test.environment import *

_NUM_WAVES = 5 if CI_ENABLED else 1


def _find_free_port() -> int:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return int(s.getsockname()[1])


_DEVICE_DB: typing.Dict[str, typing.Any] = {
    'core': {
        'type': 'local',
        'module': 'artiq.coredevice.core',
        'class': 'Core',
        'arguments': {'host': '0.0.0.0', 'ref_period': 1e-9}
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
        'host': '127.0.0.1',  # IPv4 for CI
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
    NODES: typing.Set[typing.Type[dax.base.scheduler.Node]] = set()

    def __init__(self, *args, **kwargs):
        # Mock data store
        self._data_store = MagicMock(spec=dax.base.system.DaxDataStore)
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
             reverse=None) -> None:
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

        super(_Scheduler, self).wave(wave=wave, root_nodes=root_nodes, root_action=root_action,
                                     policy=policy, reverse=reverse)

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

    def _plot_graph(self) -> None:
        # Skip plotting graph, prevents output and call to the renderer
        pass


class SchedulerMiscTestCase(unittest.TestCase):
    def test_str_to_time(self):
        test_data = [
            ('', 0.0),
            ('0s', 0.0),
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

        # noinspection PyProtectedMember
        from dax.base.scheduler import _str_to_time as str_to_time

        for s, v in test_data:
            with self.subTest(string=s, value=v):
                self.assertEqual(str_to_time(s), v, 'Converted time string does not equal expected float value')

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
        # noinspection PyProtectedMember
        from dax.base.scheduler import _DaxSchedulerClient

        class S(_Scheduler):
            CONTROLLER = 'dax_scheduler'

        self.assertTrue(issubclass(dax_scheduler_client(S), _DaxSchedulerClient))

    def test_client_class_instantiation_bad(self):
        with self.assertRaises(AssertionError, msg='Lack of scheduler controller did not raise'):
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

        # noinspection PyArgumentList
        c = Client(get_managers(device_db, Node=_JobB.get_name()))
        c.prepare()
        with self.assertRaises(AttributeError, msg='Dummy device did not raise expected error'):
            c.run()


class LazySchedulerTestCase(unittest.TestCase):
    POLICY = Policy.LAZY
    REVERSE_GRAPH = False

    FAST_WAVE_ARGUMENTS = {
        'Wave interval': 1,
        'Clock period': 0.1,
    }

    def setUp(self) -> None:
        self.arguments: typing.Dict[str, typing.Any] = {'Scheduling policy': str(self.POLICY),
                                                        'Reverse dependencies': self.REVERSE_GRAPH,
                                                        'Job pipeline': 'test_pipeline',
                                                        'View graph': False}
        self.mop = get_managers(arguments=self.arguments)

    def test_create_job(self):
        s = _Scheduler(self.mop)

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
                            j.submit(wave=new_time)
                    else:
                        j.submit(wave=new_time)

                with self.subTest(task='cancel'):
                    j.cancel()

    def test_create_job_bad(self):
        from dax.base.exceptions import BuildError

        s = _Scheduler(self.mop)

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
            (J0, BuildError),
            (J1, BuildError),
            (J2, BuildError),
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
            J0(self.mop)

        class T0(Trigger):
            pass

        with self.assertRaises(TypeError, msg='Wrong trigger parent type did not raise'):
            T0(self.mop)

    def test_job_arguments(self):
        s = _Scheduler(self.mop)

        class J0(Job):
            ARGUMENTS = {'foo': 1,
                         'range': RangeScan(1, 10, 9),
                         'center': CenterScan(1, 10, 9),
                         'explicit': ExplicitScan([1, 10, 9]),
                         'no': NoScan(10)}

        j = J0(s)
        arguments = j._process_arguments()
        self.assertEqual(len(arguments), len(J0.ARGUMENTS))
        for v in arguments.values():
            self.assertNotIsInstance(v, ScanObject)
        self.assertDictEqual(arguments,
                             {k: v.describe() if isinstance(v, ScanObject) else v for k, v in J0.ARGUMENTS.items()})

    def test_job_configurable_arguments(self):
        s = _Scheduler(self.mop)

        class J0(Job):
            ARGUMENTS = {'foo': 1,
                         'range': RangeScan(1, 10, 9),
                         'center': CenterScan(1, 10, 9),
                         'explicit': ExplicitScan([1, 10, 9]),
                         'no': NoScan(10)}

            def build_job(self) -> None:
                self.ARGUMENTS['bar'] = self.get_argument('bar', NumberValue(20))
                self.ARGUMENTS['baz'] = self.get_argument('baz', Scannable(CenterScan(100, 50, 2)))
                self.ARGUMENTS['foobar'] = self.get_argument('foobar', Scannable(RangeScan(100, 300, 40)))

        configurable_arguments = ['bar', 'baz', 'foobar']
        j = J0(s)
        arguments = j._process_arguments()
        self.assertEqual(len(arguments), len(J0.ARGUMENTS) + len(configurable_arguments))
        for v in arguments.values():
            self.assertNotIsInstance(v, ScanObject)

        # Pop three configurable arguments before comparing
        for k in configurable_arguments:
            arguments.pop(k)
        self.assertDictEqual(arguments,
                             {k: v.describe() if isinstance(v, ScanObject) else v for k, v in J0.ARGUMENTS.items()})

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
            scheduler = S(self.mop)
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
            scheduler = S(self.mop)
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

            S(self.mop)

        with self.assertRaises(AssertionError, msg='Scheduler without jobs did not raise'):
            class S(DaxScheduler):
                NAME = 'test_scheduler'

            S(self.mop)

        class S(DaxScheduler):
            NAME = 'test_scheduler'
            NODES = {}

        # Instantiate a well defined scheduler
        self.assertIsInstance(S(self.mop), DaxScheduler)

    def test_scheduler_pipeline(self):
        s = _Scheduler(self.mop)
        s.prepare()

        self.arguments['Job pipeline'] = 'main'
        with self.assertRaises(ValueError, msg='Pipeline conflict did not raise'):
            s = _Scheduler(get_managers(arguments=self.arguments))
            s.prepare()

    def test_duplicate_nodes(self):
        class S(_Scheduler):
            NODES = [_Job1, _Job2, _Job3, _Job4, _JobA, _JobB, _JobC, _Job1]

        s = S(self.mop)
        with self.assertLogs(s.logger, logging.WARNING):
            s.prepare()

        class T(_Trigger):
            pass

        class S(_Scheduler):
            NODES = [T, T]

        s = S(self.mop)
        with self.assertLogs(s.logger, logging.WARNING):
            s.prepare()

    def test_node_name_conflict(self):
        class S(_Scheduler):
            # noinspection PyGlobalUndefined
            global _JobA
            NODES = [_JobA]

        # noinspection PyShadowingNames
        class _JobA(Job):
            pass

        S.NODES.append(_JobA)

        s = S(self.mop)
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

        s = S(self.mop)
        with self.assertRaises(ValueError, msg='Node class name conflict did not raise'):
            s.prepare()

    def test_node_dependencies(self):
        class S(_Scheduler):
            NODES = {_JobA}

        with self.assertRaises(KeyError, msg='Dependency not in node set did not raise'):
            s = S(self.mop)
            try:
                s.prepare()
            except KeyError as e:
                self.assertIn(f'"{_JobB.get_name()}"', str(e), 'Job name not correctly displayed in error message')
                raise

    def test_trigger_nodes(self):
        class T(_Trigger):
            NODES = [_JobA]

        class S(_Scheduler):
            NODES = {T}

        with self.assertRaises(KeyError, msg='Trigger node not in node set did not raise'):
            s = S(self.mop)
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
            s = S(self.mop)
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
             (_Job1, _JobA) if not self.REVERSE_GRAPH else (_Job4, _JobC)),
            ({_JobA, _JobB, _JobC, JobX, JobY, JobZ},
             (_JobA, JobX, JobY) if not self.REVERSE_GRAPH else (_JobC, JobZ)),
        ]

        for jobs, root_jobs in job_sets:
            class S(_Scheduler):
                NODES = jobs

            s = S(self.mop)
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

            s = S(self.mop)
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

            s = S(self.mop)
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

        s = S(self.mop)
        with self.assertLogs(s.logger, logging.WARNING):
            s.prepare()

    def test_scheduler_wave(self):
        class S(_Scheduler):
            NODES = {_Job1, _Job2, _Job3, _Job4, _JobA, _JobB, _JobC}

        s = S(self.mop)
        s.prepare()

        # Manually call init
        for j in s._graph:
            j.init(**_get_init_kwargs(s, reset=True))

        # Check data store calls
        self.assertEqual(len(s.data_store.method_calls), 0, 'Unexpected data store calls')

        # Wave
        s.wave()
        self._check_scheduler_wave_0(s)

        # Wave
        s.wave(wave=time.time(), root_nodes=s._root_nodes, root_action=NodeAction.PASS, policy=s._policy)
        self._check_scheduler_wave_1(s)

        # Check data store calls
        # noinspection PyTypeChecker
        self._check_scheduler_datastore_calls(s)

    def _check_scheduler_wave_0(self, scheduler):
        for j in scheduler._graph:
            with self.subTest(job=j.get_name()):
                if isinstance(j, (_Job1, _JobB)):
                    ref_counter = {'init': 1, 'visit': 2, 'submit': 1, 'schedule': 1}
                elif isinstance(j, (_Job2, _Job3)):
                    ref_counter = {'init': 1, 'visit': 1}
                elif isinstance(j, _Job4):
                    ref_counter = {'init': 1, 'visit': 2}
                else:
                    ref_counter = {'init': 1, 'visit': 1}
                self.assertDictEqual(j.counter, ref_counter,
                                     'Job call pattern did not match expected pattern')

    def _check_scheduler_wave_1(self, scheduler):
        for j in scheduler._graph:
            with self.subTest(job=j.get_name()):
                if isinstance(j, (_Job1, _JobB)):
                    ref_counter = {'init': 1, 'visit': 3, 'submit': 1, 'schedule': 1}
                elif isinstance(j, (_Job2, _Job3)):
                    ref_counter = {'init': 1, 'visit': 2}
                elif isinstance(j, _Job4):
                    ref_counter = {'init': 1, 'visit': 4}
                else:
                    ref_counter = {'init': 1, 'visit': 2}
                self.assertDictEqual(j.counter, ref_counter,
                                     'Job call pattern did not match expected pattern')

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
        s = S(get_managers(arguments=self.arguments))
        s.prepare()

        # Run the scheduler
        s.run()
        self.assertEqual(s.testing_handled_requests, 0, 'Unexpected number of requests handled')
        self.assertEqual(s.testing_request_queue_qsize, 0, 'Request queue was not empty')
        self._check_scheduler_run(s)

    def _check_scheduler_run(self, scheduler):
        for j in scheduler._graph:
            with self.subTest(job=j.get_name()):
                if isinstance(j, (_Job1, _JobB)):
                    ref_counter = {'init': 1, 'visit': _NUM_WAVES + 1, 'submit': 1, 'schedule': 1}
                elif isinstance(j, (_Job2, _Job3)):
                    ref_counter = {'init': 1, 'visit': _NUM_WAVES}
                elif isinstance(j, _Job4):
                    ref_counter = {'init': 1, 'visit': _NUM_WAVES * 2}
                else:
                    ref_counter = {'init': 1, 'visit': _NUM_WAVES}
                self.assertDictEqual(j.counter, ref_counter,
                                     'Job call pattern did not match expected pattern')

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
                # noinspection PyProtectedMember
                from dax.base.scheduler import _SchedulerController
                # Construct a separate controller with the same queue
                # This is required because we can not use get_device() to obtain the controller
                # Because the server and the client are running on the same thread, the situation deadlocks
                controller = _SchedulerController(request_queue)
                for args, kwargs in requests:
                    kwargs.setdefault('block', False)
                    await controller.submit(*args, **kwargs)

        self.arguments.update(self.FAST_WAVE_ARGUMENTS)
        s = S(get_managers(_DEVICE_DB, arguments=self.arguments))
        s.prepare()

        # Run the scheduler
        s.run()
        # Return the scheduler
        return s

    def test_scheduler_controller0(self):
        requests = [
            ((_Job4.get_name(), _JobA.get_name()), {}),
            ((_JobC.get_name(),), {'action': str(NodeAction.PASS)}),
        ]
        s = self._test_scheduler_controller(requests)

        self.assertEqual(s.testing_handled_requests, len(requests), 'Unexpected number of requests handled')
        self.assertEqual(s.testing_request_queue_qsize, 0, 'Request queue was not empty')
        self._check_scheduler_controller0(s)

    def _check_scheduler_controller0(self, scheduler):
        for j in scheduler._graph:
            with self.subTest(job=j.get_name()):
                if isinstance(j, _Job1):
                    ref_counter = {'init': 1, 'visit': _NUM_WAVES + 1, 'submit': 1, 'schedule': 1}
                elif isinstance(j, (_Job2, _Job3)):
                    ref_counter = {'init': 1, 'visit': _NUM_WAVES}
                elif isinstance(j, _Job4):
                    ref_counter = {'init': 1, 'visit': _NUM_WAVES * 2 + 1, 'submit': 1, 'schedule': 1}
                elif isinstance(j, _JobA):
                    ref_counter = {'init': 1, 'visit': _NUM_WAVES + 1, 'submit': 1, 'schedule': 1}
                elif isinstance(j, _JobB):
                    ref_counter = {'init': 1, 'visit': _NUM_WAVES + 1 + 1, 'submit': 1, 'schedule': 1}
                elif isinstance(j, _JobC):
                    ref_counter = {'init': 1, 'visit': _NUM_WAVES + 1 + 1}
                else:
                    self.fail('Statement should not be reachable')
                self.assertDictEqual(j.counter, ref_counter,
                                     'Job call pattern did not match expected pattern')

    def test_scheduler_controller1(self):
        requests = [
            ((_Job4.get_name(), _JobB.get_name()), {'reverse': True}),
        ]
        s = self._test_scheduler_controller(requests)

        self.assertEqual(s.testing_handled_requests, len(requests), 'Unexpected number of requests handled')
        self.assertEqual(s.testing_request_queue_qsize, 0, 'Request queue was not empty')
        self._check_scheduler_controller1(s)

    def _check_scheduler_controller1(self, scheduler):
        for j in scheduler._graph:
            with self.subTest(job=j.get_name()):
                if isinstance(j, _Job4):
                    ref_counter = {'init': 1, 'visit': _NUM_WAVES * 2 + 1, 'submit': 1, 'schedule': 1}
                elif isinstance(j, _Job1):
                    ref_counter = {'init': 1, 'visit': _NUM_WAVES + 1 + 2, 'submit': 1, 'schedule': 1}
                elif isinstance(j, (_Job2, _Job3, _JobA)):
                    ref_counter = {'init': 1, 'visit': _NUM_WAVES + 1}
                elif isinstance(j, _JobB):
                    ref_counter = {'init': 1, 'visit': _NUM_WAVES + 1 + 2, 'submit': 2, 'schedule': 2}
                elif isinstance(j, _JobC):
                    ref_counter = {'init': 1, 'visit': _NUM_WAVES}
                else:
                    self.fail('Statement should not be reachable')
                self.assertDictEqual(j.counter, ref_counter,
                                     'Job call pattern did not match expected pattern')

    def test_scheduler_controller1_sequential(self):
        requests = [
            ((_Job4.get_name(),), {'reverse': True}),
            ((_JobB.get_name(),), {'reverse': True}),
        ]
        s = self._test_scheduler_controller(requests)

        self.assertEqual(s.testing_handled_requests, len(requests), 'Unexpected number of requests handled')
        self.assertEqual(s.testing_request_queue_qsize, 0, 'Request queue was not empty')
        self._check_scheduler_controller1(s)

    def test_create_trigger(self):
        # asyncio entry point
        asyncio.run(self._test_create_trigger())

    async def _test_create_trigger(self):
        s = _Scheduler(self.mop)

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
                            t.submit(wave=new_time)
                            queue_len += 1
                    else:
                        t.submit(wave=new_time)
                    self.assertEqual(request_queue.qsize(), queue_len, 'Queue length did not match expected length')

                with self.subTest(task='cancel'):
                    t.cancel()

    def test_create_trigger_bad(self):
        from dax.base.exceptions import BuildError

        s = _Scheduler(self.mop)

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
            (T0, BuildError),
            (T1, BuildError),
            (T2, BuildError),
            (T3, AssertionError),
            (T4, AssertionError),
            (T5, TypeError),  # Caused by failing issubclass() call
            (T6, BuildError),
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
        s = S(get_managers(arguments=self.arguments))
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

    @unittest.skipUnless(CI_ENABLED, 'Skipping long test when not running in CI environment')
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
        s = S(get_managers(arguments=self.arguments))
        s.prepare()

        # Run the scheduler
        s.run()

        for n in s._graph:
            with self.subTest(job=n.get_name()):
                if isinstance(n, T0):
                    ref_counter = {'init': 1, 'visit': num_waves + num_submits,
                                   'submit': num_submits, 'schedule': num_submits}
                elif isinstance(n, J0):
                    ref_counter = {'init': 1, 'visit': num_waves + num_submits,
                                   'submit': num_submits, 'schedule': num_submits}
                elif isinstance(n, J1):
                    if self.REVERSE_GRAPH:
                        ref_counter = {'init': 1, 'visit': num_waves}
                    elif self.POLICY is Policy.LAZY:
                        ref_counter = {'init': 1, 'visit': num_waves + num_submits}
                    else:
                        ref_counter = {'init': 1, 'visit': num_waves + num_submits,
                                       'submit': num_submits, 'schedule': num_submits}
                else:
                    self.fail('Unexpected job type encountered')
                self.assertDictEqual(n.counter, ref_counter,
                                     'Node call pattern did not match expected pattern')


class LazySchedulerReversedTestCase(LazySchedulerTestCase):
    POLICY = Policy.LAZY
    REVERSE_GRAPH = True

    def _check_scheduler_wave_0(self, scheduler):
        for j in scheduler._graph:
            with self.subTest(job=j.get_name()):
                if isinstance(j, _Job1):
                    ref_counter = {'init': 1, 'visit': 3, 'submit': 1, 'schedule': 1}
                elif isinstance(j, (_Job2, _Job3, _Job4)):
                    ref_counter = {'init': 1, 'visit': 1}
                elif isinstance(j, _JobB):
                    ref_counter = {'init': 1, 'visit': 2, 'submit': 1, 'schedule': 1}
                else:
                    ref_counter = {'init': 1, 'visit': 1}
                self.assertDictEqual(j.counter, ref_counter,
                                     'Job call pattern did not match expected pattern')

    def _check_scheduler_wave_1(self, scheduler):
        for j in scheduler._graph:
            with self.subTest(job=j.get_name()):
                if isinstance(j, _Job1):
                    ref_counter = {'init': 1, 'visit': 5, 'submit': 1, 'schedule': 1}
                elif isinstance(j, (_Job2, _Job3, _Job4)):
                    ref_counter = {'init': 1, 'visit': 2}
                elif isinstance(j, _JobB):
                    ref_counter = {'init': 1, 'visit': 3, 'submit': 1, 'schedule': 1}
                else:
                    ref_counter = {'init': 1, 'visit': 2}
                self.assertDictEqual(j.counter, ref_counter,
                                     'Job call pattern did not match expected pattern')

    def _check_scheduler_run(self, scheduler):
        for j in scheduler._graph:
            with self.subTest(job=j.get_name()):
                if isinstance(j, _JobB):
                    ref_counter = {'init': 1, 'visit': _NUM_WAVES + 1, 'submit': 1, 'schedule': 1}
                elif isinstance(j, _Job1):
                    ref_counter = {'init': 1, 'visit': _NUM_WAVES * 2 + 1, 'submit': 1, 'schedule': 1}
                else:
                    ref_counter = {'init': 1, 'visit': _NUM_WAVES}
                self.assertDictEqual(j.counter, ref_counter,
                                     'Job call pattern did not match expected pattern')

    def _check_scheduler_controller0(self, scheduler):
        for j in scheduler._graph:
            with self.subTest(job=j.get_name()):
                if isinstance(j, _Job1):
                    ref_counter = {'init': 1, 'visit': (_NUM_WAVES + 1) * 2 + 1, 'submit': 1, 'schedule': 1}
                elif isinstance(j, (_Job2, _Job3)):
                    ref_counter = {'init': 1, 'visit': _NUM_WAVES + 1}
                elif isinstance(j, _Job4):
                    ref_counter = {'init': 1, 'visit': _NUM_WAVES + 1, 'submit': 1, 'schedule': 1}
                elif isinstance(j, (_JobA, _JobB)):
                    ref_counter = {'init': 1, 'visit': _NUM_WAVES + 1 + 1, 'submit': 1, 'schedule': 1}
                elif isinstance(j, _JobC):
                    ref_counter = {'init': 1, 'visit': _NUM_WAVES + 1}
                else:
                    self.fail('Statement should not be reachable')
                self.assertDictEqual(j.counter, ref_counter,
                                     'Job call pattern did not match expected pattern')

    def _check_scheduler_controller1(self, scheduler):
        for j in scheduler._graph:
            with self.subTest(job=j.get_name()):
                if isinstance(j, _Job4):
                    ref_counter = {'init': 1, 'visit': _NUM_WAVES + 1, 'submit': 1, 'schedule': 1}
                elif isinstance(j, _Job1):
                    ref_counter = {'init': 1, 'visit': _NUM_WAVES * 2 + 1 + 2, 'submit': 1, 'schedule': 1}
                elif isinstance(j, (_Job2, _Job3, _JobA)):
                    ref_counter = {'init': 1, 'visit': _NUM_WAVES + 1}
                elif isinstance(j, _JobB):
                    ref_counter = {'init': 1, 'visit': _NUM_WAVES + 1 + 2, 'submit': 2, 'schedule': 2}
                elif isinstance(j, _JobC):
                    ref_counter = {'init': 1, 'visit': _NUM_WAVES}
                else:
                    self.fail('Statement should not be reachable')
                self.assertDictEqual(j.counter, ref_counter,
                                     'Job call pattern did not match expected pattern')


class GreedySchedulerTestCase(LazySchedulerTestCase):
    POLICY = Policy.GREEDY
    REVERSE_GRAPH = False

    def test_scheduler_unreachable_jobs(self):
        with self.assertRaises(self.failureException, msg='Expected test failure did not happen'):
            # With a greedy policy, all jobs are reachable and the call to super will cause a test failure
            super(GreedySchedulerTestCase, self).test_scheduler_unreachable_jobs()

    def _check_scheduler_wave_0(self, scheduler):
        for j in scheduler._graph:
            with self.subTest(job=j.get_name()):
                if isinstance(j, (_Job1, _JobB)):
                    ref_counter = {'init': 1, 'visit': 2, 'submit': 1, 'schedule': 1}
                elif isinstance(j, (_Job2, _Job3, _JobC)):
                    ref_counter = {'init': 1, 'visit': 1, 'submit': 1, 'schedule': 1}
                elif isinstance(j, _Job4):
                    ref_counter = {'init': 1, 'visit': 2, 'submit': 1, 'schedule': 1}
                else:
                    ref_counter = {'init': 1, 'visit': 1}
                self.assertDictEqual(j.counter, ref_counter,
                                     'Job call pattern did not match expected pattern')

    def _check_scheduler_wave_1(self, scheduler):
        for j in scheduler._graph:
            with self.subTest(job=j.get_name()):
                if isinstance(j, (_Job1, _JobB)):
                    ref_counter = {'init': 1, 'visit': 3, 'submit': 1, 'schedule': 1}
                elif isinstance(j, (_Job2, _Job3, _JobC)):
                    ref_counter = {'init': 1, 'visit': 2, 'submit': 1, 'schedule': 1}
                elif isinstance(j, _Job4):
                    ref_counter = {'init': 1, 'visit': 4, 'submit': 1, 'schedule': 1}
                else:
                    ref_counter = {'init': 1, 'visit': 2}
                self.assertDictEqual(j.counter, ref_counter,
                                     'Job call pattern did not match expected pattern')

    def _check_scheduler_datastore_calls(self, scheduler):
        # Check data store calls (only jobs that are not meta-jobs perform a call)
        self.assertEqual(len(scheduler.data_store.method_calls), 3,
                         'Data store was called an unexpected number of times')

    def _check_scheduler_controller0(self, scheduler):
        for j in scheduler._graph:
            with self.subTest(job=j.get_name()):
                if isinstance(j, _Job1):
                    ref_counter = {'init': 1, 'visit': _NUM_WAVES + 1, 'submit': 1, 'schedule': 1}
                elif isinstance(j, (_Job2, _Job3)):
                    ref_counter = {'init': 1, 'visit': _NUM_WAVES, 'submit': 1, 'schedule': 1}
                elif isinstance(j, _Job4):
                    ref_counter = {'init': 1, 'visit': _NUM_WAVES * 2 + 1, 'submit': 1 + 1, 'schedule': 1 + 1}
                elif isinstance(j, _JobC):
                    ref_counter = {'init': 1, 'visit': _NUM_WAVES + 1 + 1, 'submit': 1 + 1, 'schedule': 1 + 1}
                elif isinstance(j, _JobB):
                    ref_counter = {'init': 1, 'visit': _NUM_WAVES + 1 + 1 + 1, 'submit': 1 + 1, 'schedule': 1 + 1}
                else:
                    ref_counter = {'init': 1, 'visit': _NUM_WAVES + 1, 'submit': 1, 'schedule': 1}
                self.assertDictEqual(j.counter, ref_counter,
                                     'Job call pattern did not match expected pattern')

    def _check_scheduler_controller1(self, scheduler):
        for j in scheduler._graph:
            with self.subTest(job=j.get_name()):
                if isinstance(j, _Job4):
                    ref_counter = {'init': 1, 'visit': _NUM_WAVES * 2 + 1, 'submit': 2, 'schedule': 2}
                elif isinstance(j, _Job1):
                    ref_counter = {'init': 1, 'visit': _NUM_WAVES + 1 + 3, 'submit': 2, 'schedule': 2}
                elif isinstance(j, (_Job2, _Job3)):
                    ref_counter = {'init': 1, 'visit': _NUM_WAVES + 1, 'submit': 2, 'schedule': 2}
                elif isinstance(j, _JobA):
                    ref_counter = {'init': 1, 'visit': _NUM_WAVES + 1, 'submit': 1, 'schedule': 1}
                elif isinstance(j, _JobB):
                    ref_counter = {'init': 1, 'visit': _NUM_WAVES + 1 + 2, 'submit': 2, 'schedule': 2}
                elif isinstance(j, _JobC):
                    ref_counter = {'init': 1, 'visit': _NUM_WAVES, 'submit': 1, 'schedule': 1}
                else:
                    self.fail('Statement should not be reachable')
                self.assertDictEqual(j.counter, ref_counter,
                                     'Job call pattern did not match expected pattern')

    def _check_scheduler_run(self, scheduler):
        for j in scheduler._graph:
            with self.subTest(job=j.get_name()):
                if isinstance(j, (_Job1, _JobB)):
                    ref_counter = {'init': 1, 'visit': _NUM_WAVES + 1, 'submit': 1, 'schedule': 1}
                elif isinstance(j, (_Job2, _Job3, _JobC)):
                    ref_counter = {'init': 1, 'visit': _NUM_WAVES, 'submit': 1, 'schedule': 1}
                elif isinstance(j, _Job4):
                    ref_counter = {'init': 1, 'visit': _NUM_WAVES * 2, 'submit': 1, 'schedule': 1}
                else:
                    ref_counter = {'init': 1, 'visit': _NUM_WAVES}
                self.assertDictEqual(j.counter, ref_counter,
                                     'Job call pattern did not match expected pattern')


class GreedySchedulerReversedTestCase(GreedySchedulerTestCase):
    POLICY = Policy.GREEDY
    REVERSE_GRAPH = True

    def _check_scheduler_wave_0(self, scheduler):
        for j in scheduler._graph:
            with self.subTest(job=j.get_name()):
                if isinstance(j, _Job1):
                    ref_counter = {'init': 1, 'visit': 2 + 1, 'submit': 1, 'schedule': 1}
                elif isinstance(j, _JobB):
                    ref_counter = {'init': 1, 'visit': 2, 'submit': 1, 'schedule': 1}
                elif isinstance(j, _JobA):
                    ref_counter = {'init': 1, 'visit': 1, 'submit': 1, 'schedule': 1}
                else:
                    ref_counter = {'init': 1, 'visit': 1}
                self.assertDictEqual(j.counter, ref_counter,
                                     'Job call pattern did not match expected pattern')

    def _check_scheduler_wave_1(self, scheduler):
        for j in scheduler._graph:
            with self.subTest(job=j.get_name()):
                if isinstance(j, _Job1):
                    ref_counter = {'init': 1, 'visit': 3 + 2, 'submit': 1, 'schedule': 1}
                elif isinstance(j, _JobB):
                    ref_counter = {'init': 1, 'visit': 3, 'submit': 1, 'schedule': 1}
                elif isinstance(j, _JobA):
                    ref_counter = {'init': 1, 'visit': 2, 'submit': 1, 'schedule': 1}
                else:
                    ref_counter = {'init': 1, 'visit': 2}
                self.assertDictEqual(j.counter, ref_counter,
                                     'Job call pattern did not match expected pattern')

    def _check_scheduler_datastore_calls(self, scheduler):
        # Check data store calls (only jobs that are not meta-jobs perform a call)
        self.assertEqual(len(scheduler.data_store.method_calls), 2,
                         'Data store was called an unexpected number of times')

    def _check_scheduler_controller0(self, scheduler):
        for j in scheduler._graph:
            with self.subTest(job=j.get_name()):
                if isinstance(j, _Job1):
                    ref_counter = {'init': 1, 'visit': _NUM_WAVES * 2 + 1 + 2 + 1, 'submit': 2, 'schedule': 2}
                elif isinstance(j, (_Job2, _Job3, _Job4)):
                    ref_counter = {'init': 1, 'visit': _NUM_WAVES + 1, 'submit': 1, 'schedule': 1}
                elif isinstance(j, _JobA):
                    ref_counter = {'init': 1, 'visit': _NUM_WAVES + 1 + 1, 'submit': 2, 'schedule': 2}
                elif isinstance(j, _JobB):
                    ref_counter = {'init': 1, 'visit': _NUM_WAVES + 1 + 1, 'submit': 1, 'schedule': 1}
                elif isinstance(j, _JobC):
                    ref_counter = {'init': 1, 'visit': _NUM_WAVES + 1}
                else:
                    self.fail('Statement should not be reachable')
                self.assertDictEqual(j.counter, ref_counter,
                                     'Job call pattern did not match expected pattern')

    def _check_scheduler_controller1(self, scheduler):
        for j in scheduler._graph:
            with self.subTest(job=j.get_name()):
                if isinstance(j, (_Job2, _Job3, _Job4)):
                    ref_counter = {'init': 1, 'visit': _NUM_WAVES + 1, 'submit': 1, 'schedule': 1}
                elif isinstance(j, _Job1):
                    ref_counter = {'init': 1, 'visit': _NUM_WAVES * 2 + 1 + 2 + 1, 'submit': 2, 'schedule': 2}
                elif isinstance(j, _JobA):
                    ref_counter = {'init': 1, 'visit': _NUM_WAVES + 1, 'submit': 2, 'schedule': 2}
                elif isinstance(j, _JobB):
                    ref_counter = {'init': 1, 'visit': _NUM_WAVES + 1 + 2, 'submit': 2, 'schedule': 2}
                elif isinstance(j, _JobC):
                    ref_counter = {'init': 1, 'visit': _NUM_WAVES}
                else:
                    self.fail('Statement should not be reachable')
                self.assertDictEqual(j.counter, ref_counter,
                                     'Job call pattern did not match expected pattern')

    def _check_scheduler_run(self, scheduler):
        for j in scheduler._graph:
            with self.subTest(job=j.get_name()):
                if isinstance(j, _JobB):
                    ref_counter = {'init': 1, 'visit': _NUM_WAVES + 1, 'submit': 1, 'schedule': 1}
                elif isinstance(j, _JobA):
                    ref_counter = {'init': 1, 'visit': _NUM_WAVES, 'submit': 1, 'schedule': 1}
                elif isinstance(j, _Job1):
                    ref_counter = {'init': 1, 'visit': _NUM_WAVES * 2 + 1, 'submit': 1, 'schedule': 1}
                else:
                    ref_counter = {'init': 1, 'visit': _NUM_WAVES}
                self.assertDictEqual(j.counter, ref_counter,
                                     'Job call pattern did not match expected pattern')
