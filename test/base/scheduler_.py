import unittest
import time
import logging
import typing
import collections
import contextlib
import os
import socket
from unittest.mock import MagicMock, call

from artiq.language.scan import *
from artiq.experiment import TerminationRequested, NumberValue

from dax.base.scheduler import *
from dax.util.artiq import get_managers
import dax.base.system
import dax.util.output


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


@contextlib.contextmanager
def _isolation() -> typing.Generator[None, None, None]:
    """Move into a temp dir and suppress stdout/stderr."""
    with dax.util.output.temp_dir(), open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


class _Job(Job):
    def __init__(self, *args, **kwargs):
        super(_Job, self).__init__(*args, **kwargs)
        self.counter = collections.Counter()

    def init(self, *args, **kwargs):
        self.counter['init'] += 1
        return super(_Job, self).init(*args, **kwargs)

    def visit(self, *args, **kwargs):
        self.counter['visit'] += 1
        return super(_Job, self).visit(*args, **kwargs)

    def submit(self, *args, **kwargs):
        self.counter['submit'] += 1
        return super(_Job, self).submit(*args, **kwargs)

    def schedule(self, *args, **kwargs):
        self.counter['schedule'] += 1
        return super(_Job, self).schedule(*args, **kwargs)

    def cancel(self):
        self.counter['cancel'] += 1
        return super(_Job, self).cancel()


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


class _JobA(_Job):
    DEPENDENCIES = {_JobB}
    FILE = 'foo.py'
    CLASS_NAME = 'Bar'


class _Scheduler(DaxScheduler):
    NAME = 'test_scheduler'
    JOBS: typing.Set[typing.Type[Job]] = set()

    # Modify graphviz format to prevent usage of visual render engines
    _GRAPHVIZ_FORMAT = 'gv'

    def __init__(self, *args, **kwargs):
        self._data_store = MagicMock(spec=dax.base.system.DaxDataStore)
        self.external_requests = 0
        super(_Scheduler, self).__init__(*args, **kwargs)

    @property
    def data_store(self):
        return self._data_store

    def wave(self, *,
             wave=None,
             root_jobs=None,
             root_action=None,
             policy=None) -> None:
        # Use defaults for simplicity
        if wave is None:
            wave = time.time()
        if root_jobs is None:
            root_jobs = self._root_jobs
        if root_action is None:
            root_action = JobAction.PASS
        if policy is None:
            policy = self._policy

        super(_Scheduler, self).wave(wave=wave, root_jobs=root_jobs, root_action=root_action, policy=policy)

    async def _run_scheduler(self, queue) -> None:
        self.queue = queue
        self.controller_callback()
        await super(_Scheduler, self)._run_scheduler(queue)

    def controller_callback(self) -> None:
        pass

    def _handle_external_request(self, queue) -> None:
        self.external_requests += 1
        super(_Scheduler, self)._handle_external_request(queue)


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
        for a in JobAction:
            with self.subTest(job_action=a):
                self.assertEqual(str(a), a.name)

    def test_job_action_submittable(self):
        submittable = {JobAction.RUN, JobAction.FORCE}

        for a in JobAction:
            with self.subTest(job_action=a):
                self.assertEqual(a in submittable, a.submittable())

    def test_policy_complete(self):
        for p in Policy:
            with self.subTest(policy=str(p)):
                self.assertEqual(len(p.value), len(JobAction) ** 2, 'Policy not fully implemented')

    def test_policy_differences(self):
        for a0 in JobAction:
            for a1 in JobAction:
                lazy = Policy.LAZY.action(a0, a1)
                greedy = Policy.GREEDY.action(a0, a1)

                if a0 is JobAction.RUN and a1 is JobAction.PASS:
                    # Only for this one scenario the policies are different
                    self.assertEqual(lazy, JobAction.PASS, 'Policy lazy returned an unexpected action')
                    self.assertEqual(greedy, JobAction.RUN, 'Policy greedy returned an unexpected action')
                else:
                    # All other cases the policies return the same actions
                    self.assertEqual(lazy, greedy, 'Policies unexpectedly returned different actions')


class LazySchedulerTestCase(unittest.TestCase):
    POLICY = Policy.LAZY

    def setUp(self) -> None:
        self.mop = get_managers(Policy=str(self.POLICY), Pipeline='test_pipeline',
                                **{'View graph': False})  # type: ignore[arg-type]

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
                        j.init(reset=reset)
                        self.assertIsInstance(j._next_submit, float)
                        if is_timed:
                            self.assertLess(j._next_submit, float('inf'))
                            self.assertGreater(j._next_submit, 0.0)
                        else:
                            self.assertEqual(j._next_submit, float('inf'))

                with self.subTest(task='visit', is_timed=is_timed):
                    # Test visit
                    self.assertEqual(j.visit(wave=old_time), JobAction.PASS)
                    new_time = time.time()
                    self.assertEqual(j.visit(wave=new_time), JobAction.RUN if is_timed else JobAction.PASS)

                with self.subTest(task='submit', is_meta=is_meta):
                    # Test submit
                    if not is_meta:
                        with self.assertLogs(j.logger, logging.INFO):
                            j.submit(wave=new_time, pipeline='main', priority=0)
                    else:
                        j.submit(wave=new_time, pipeline='main', priority=0)

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
            (J2, AssertionError),
            (J3, AssertionError),
            (J4, AssertionError),
            (J5, TypeError),  # Caused by failing issubclass() call
        ]

        for J, error_type in test_data:
            with self.subTest(job_class=J.__name__), self.assertRaises(error_type, msg='Bad job did not raise'):
                J(s)

    def test_create_job_bad_parent(self):
        class J0(Job):
            pass

        with self.assertRaises(TypeError, msg='Wrong job parent type did not raise'):
            J0(self.mop)

    def test_job_arguments(self):
        s = _Scheduler(self.mop)

        class J0(Job):
            ARGUMENTS = {'foo': 1,
                         'range': RangeScan(1, 10, 9),
                         'center': CenterScan(1, 10, 9),
                         'explicit': ExplicitScan([1, 10, 9]),
                         'no': NoScan(10),
                         }

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
                         'no': NoScan(10),
                         }

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
            JOBS = [J0]

        def start_scheduler(*, reset):
            # Prepare scheduler
            scheduler = S(self.mop)
            scheduler.prepare()
            # Extract job object
            job = scheduler._jobs[J0]
            # Initialize job
            job.init(reset=reset)
            # Return scheduler and job
            return scheduler, job

        with _isolation():
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
            JOBS = [J0]

        def start_scheduler(*, reset):
            # Prepare scheduler
            scheduler = S(self.mop)
            scheduler.prepare()
            # Extract job object
            job = scheduler._jobs[J0]
            # Initialize job
            job.init(reset=reset)
            # Return scheduler and job
            return scheduler, job

        with _isolation():
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
                JOBS = {}

            S(self.mop)

        with self.assertRaises(AssertionError, msg='Scheduler without jobs did not raise'):
            class S(DaxScheduler):
                NAME = 'test_scheduler'

            S(self.mop)

        class S(DaxScheduler):
            NAME = 'test_scheduler'
            JOBS = {}

        # Instantiate a well defined scheduler
        self.assertIsInstance(S(self.mop), DaxScheduler)

    def test_scheduler_pipeline(self):
        with _isolation():
            with self.assertRaises(ValueError, msg='Pipeline conflict did not raise'):
                s = _Scheduler(get_managers(Policy=str(self.POLICY), Pipeline='main', **{'View graph': False}))
                s.prepare()

            s = _Scheduler(self.mop)
            s.prepare()

    def test_scheduler_duplicate_jobs(self):
        class S(_Scheduler):
            JOBS = [_Job1, _Job2, _Job3, _Job4, _JobA, _JobB, _JobC, _Job1]

        s = S(self.mop)
        with self.assertLogs(s.logger, logging.WARNING), _isolation():
            s.prepare()

    def test_scheduler_job_name_conflict(self):
        class S(_Scheduler):
            # noinspection PyGlobalUndefined
            global _JobA
            JOBS = [_JobA]

        # noinspection PyShadowingNames
        class _JobA(Job):
            pass

        S.JOBS.append(_JobA)

        s = S(self.mop)
        with self.assertRaises(ValueError, msg='Job class name conflict did not raise'), _isolation():
            s.prepare()

    def test_scheduler_dependencies(self):
        class S(_Scheduler):
            JOBS = {_JobA}

        with self.assertRaises(KeyError, msg='Dependency not in job set did not raise'), _isolation():
            s = S(self.mop)
            try:
                s.prepare()
            except KeyError as e:
                self.assertIn(f'"{_JobB.get_name()}"', str(e), 'Job name not correctly displayed in error message')
                raise

    def test_scheduler_dag(self):
        class JobA(Job):
            pass

        # Artificially create a self-loop
        JobA.DEPENDENCIES = {JobA}

        class S(_Scheduler):
            JOBS = {JobA}

        with self.assertRaises(RuntimeError, msg='Non-DAG dependency graph did not raise'), _isolation():
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
            ({_Job1, _Job2, _Job3, _Job4, _JobA, _JobB, _JobC}, (_Job1, _JobA)),
            ({_JobA, _JobB, _JobC, JobX, JobY, JobZ}, (_JobA, JobX, JobY)),
        ]

        with _isolation():
            for jobs, root_jobs in job_sets:
                class S(_Scheduler):
                    JOBS = jobs

                s = S(self.mop)
                s.prepare()

                self.assertEqual(len(s._root_jobs), len(root_jobs), 'Did not found expected number of root jobs')
                for j in s._root_jobs:
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

        with _isolation():
            for jobs, root_jobs in job_sets:
                class S(_Scheduler):
                    JOBS = jobs
                    ROOT_JOBS = root_jobs

                s = S(self.mop)
                s.prepare()

                self.assertEqual(len(s._root_jobs), len(root_jobs), 'Did not found expected number of root jobs')
                for j in s._root_jobs:
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

        with _isolation():
            for jobs, root_job in job_sets:
                class S(_Scheduler):
                    JOBS = jobs
                    ROOT_JOBS = {root_job}

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
            JOBS = {_JobC}

        s = S(self.mop)
        with self.assertLogs(s.logger, logging.WARNING), _isolation():
            s.prepare()

    def test_scheduler_wave(self):
        class S(_Scheduler):
            JOBS = {_Job1, _Job2, _Job3, _Job4, _JobA, _JobB, _JobC}

        with _isolation():
            s = S(self.mop)
            s.prepare()

        # Manually call init
        for j in s._job_graph:
            j.init(reset=True)

        # Check data store calls
        self.assertEqual(len(s.data_store.method_calls), 0, 'Unexpected data store calls')

        # Wave
        s.wave()
        self._check_scheduler_wave_0(s)

        # Wave
        s.wave(wave=time.time(), root_jobs=s._root_jobs, root_action=JobAction.PASS, policy=s._policy)
        self._check_scheduler_wave_1(s)

        # Check data store calls
        # noinspection PyTypeChecker
        self._check_scheduler_datastore_calls(s)

    def _check_scheduler_wave_0(self, scheduler):
        for j in scheduler._job_graph:
            with self.subTest(job=j.get_name()):
                if isinstance(j, _Job1):
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
        for j in scheduler._job_graph:
            with self.subTest(job=j.get_name()):
                if isinstance(j, _Job1):
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
        waves = 3

        class S(_Scheduler):
            JOBS = {_Job1, _Job2, _Job3, _Job4, _JobA, _JobB, _JobC}
            counter = 0

            def wave(self, **kwargs) -> None:
                super(S, self).wave(**kwargs)
                self.counter += 1
                if self.counter >= waves:
                    raise TerminationRequested

        with _isolation():
            s = S(get_managers(Policy=str(self.POLICY), Pipeline='test_pipeline',
                               **{'Wave interval': 1.0, 'Clock period': 0.1, 'View graph': False}))
            s.prepare()

        # Run the scheduler
        s.run()
        self._check_scheduler_run(s, waves)

    def _check_scheduler_run(self, scheduler, waves):
        for j in scheduler._job_graph:
            with self.subTest(job=j.get_name()):
                if isinstance(j, _Job1):
                    ref_counter = {'init': 1, 'visit': waves + 1, 'submit': 1, 'schedule': 1}
                elif isinstance(j, (_Job2, _Job3)):
                    ref_counter = {'init': 1, 'visit': waves}
                elif isinstance(j, _Job4):
                    ref_counter = {'init': 1, 'visit': waves * 2}
                else:
                    ref_counter = {'init': 1, 'visit': waves}
                self.assertDictEqual(j.counter, ref_counter,
                                     'Job call pattern did not match expected pattern')

    def test_scheduler_controller(self):
        waves = 3

        class S(_Scheduler):
            JOBS = {_Job1, _Job2, _Job3, _Job4, _JobA, _JobB, _JobC}
            CONTROLLER = 'dax_scheduler'

            counter = 0

            def wave(self, **kwargs) -> None:
                super(S, self).wave(**kwargs)

                if kwargs['wave'] > 0.0:
                    self.counter += 1
                    if self.counter >= waves:
                        raise TerminationRequested

            def controller_callback(self) -> None:
                # noinspection PyProtectedMember
                from dax.base.scheduler import _SchedulerController
                # Construct a separate controller with the same queue
                # This is required because we can not use get_device() to obtain the controller
                # Because the server and the client are running on the same thread then, the situation deadlocks
                controller = _SchedulerController(self.queue)
                controller.submit(_Job4.get_name())
                controller.submit(_JobA.get_name())
                controller.submit(_JobC.get_name(), action=str(JobAction.PASS))

        with _isolation():
            s = S(get_managers(_DEVICE_DB, Policy=str(self.POLICY), Pipeline='test_pipeline',
                               **{'Wave interval': 1.0, 'Clock period': 0.1, 'View graph': False}))
            s.prepare()

        # Run the scheduler
        s.run()
        self._check_scheduler_controller(s, waves)

    def _check_scheduler_controller(self, scheduler, waves):
        for j in scheduler._job_graph:
            with self.subTest(job=j.get_name()):
                if isinstance(j, _Job1):
                    ref_counter = {'init': 1, 'visit': waves + 1, 'submit': 1, 'schedule': 1}
                elif isinstance(j, (_Job2, _Job3)):
                    ref_counter = {'init': 1, 'visit': waves}
                elif isinstance(j, _Job4):
                    ref_counter = {'init': 1, 'visit': waves * 2 + 1, 'submit': 1, 'schedule': 1}
                elif isinstance(j, _JobA):
                    ref_counter = {'init': 1, 'visit': waves + 1, 'submit': 1, 'schedule': 1}
                elif isinstance(j, _JobC):
                    ref_counter = {'init': 1, 'visit': waves + 1 + 1}
                else:
                    ref_counter = {'init': 1, 'visit': waves + 1}
                self.assertDictEqual(j.counter, ref_counter,
                                     'Job call pattern did not match expected pattern')


class GreedySchedulerTestCase(LazySchedulerTestCase):
    POLICY = Policy.GREEDY

    def test_scheduler_unreachable_jobs(self):
        with self.assertRaises(self.failureException, msg='Expected test failure did not happen'):
            # With a greedy policy, all jobs are reachable and the call to super will cause a test failure
            super(GreedySchedulerTestCase, self).test_scheduler_unreachable_jobs()

    def _check_scheduler_wave_0(self, scheduler):
        for j in scheduler._job_graph:
            with self.subTest(job=j.get_name()):
                if isinstance(j, _Job1):
                    ref_counter = {'init': 1, 'visit': 2, 'submit': 1, 'schedule': 1}
                elif isinstance(j, (_Job2, _Job3)):
                    ref_counter = {'init': 1, 'visit': 1, 'submit': 1, 'schedule': 1}
                elif isinstance(j, _Job4):
                    ref_counter = {'init': 1, 'visit': 2, 'submit': 1, 'schedule': 1}
                else:
                    ref_counter = {'init': 1, 'visit': 1}
                self.assertDictEqual(j.counter, ref_counter,
                                     'Job call pattern did not match expected pattern')

    def _check_scheduler_wave_1(self, scheduler):
        for j in scheduler._job_graph:
            with self.subTest(job=j.get_name()):
                if isinstance(j, _Job1):
                    ref_counter = {'init': 1, 'visit': 3, 'submit': 1, 'schedule': 1}
                elif isinstance(j, (_Job2, _Job3)):
                    ref_counter = {'init': 1, 'visit': 2, 'submit': 1, 'schedule': 1}
                elif isinstance(j, _Job4):
                    ref_counter = {'init': 1, 'visit': 4, 'submit': 1, 'schedule': 1}
                else:
                    ref_counter = {'init': 1, 'visit': 2}
                self.assertDictEqual(j.counter, ref_counter,
                                     'Job call pattern did not match expected pattern')

    def _check_scheduler_datastore_calls(self, scheduler):
        # Check data store calls (only jobs that are not meta-jobs perform a call)
        self.assertEqual(len(scheduler.data_store.method_calls), 2,
                         'Data store was called an unexpected number of times')

    def _check_scheduler_run(self, scheduler, waves):
        for j in scheduler._job_graph:
            with self.subTest(job=j.get_name()):
                if isinstance(j, _Job1):
                    ref_counter = {'init': 1, 'visit': waves + 1, 'submit': 1, 'schedule': 1}
                elif isinstance(j, (_Job2, _Job3)):
                    ref_counter = {'init': 1, 'visit': waves, 'submit': 1, 'schedule': 1}
                elif isinstance(j, _Job4):
                    ref_counter = {'init': 1, 'visit': waves * 2, 'submit': 1, 'schedule': 1}
                else:
                    ref_counter = {'init': 1, 'visit': waves}
                self.assertDictEqual(j.counter, ref_counter,
                                     'Job call pattern did not match expected pattern')

    def _check_scheduler_controller(self, scheduler, waves):
        for j in scheduler._job_graph:
            with self.subTest(job=j.get_name()):
                if isinstance(j, _Job1):
                    ref_counter = {'init': 1, 'visit': waves + 1, 'submit': 1, 'schedule': 1}
                elif isinstance(j, (_Job2, _Job3)):
                    ref_counter = {'init': 1, 'visit': waves, 'submit': 1, 'schedule': 1}
                elif isinstance(j, _Job4):
                    ref_counter = {'init': 1, 'visit': waves * 2 + 1, 'submit': 1 + 1, 'schedule': 1 + 1}
                elif isinstance(j, _JobC):
                    ref_counter = {'init': 1, 'visit': waves + 1 + 1, 'submit': 1, 'schedule': 1}
                else:
                    ref_counter = {'init': 1, 'visit': waves + 1, 'submit': 1, 'schedule': 1}
                self.assertDictEqual(j.counter, ref_counter,
                                     'Job call pattern did not match expected pattern')
