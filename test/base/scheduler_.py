import unittest
import time
import logging
import typing

from artiq.language.scan import *

from dax.base.scheduler import *
from dax.util.artiq import get_manager_or_parent


class _Job4(Job):
    pass


class _Job2(Job):
    DEPENDENCIES = {_Job4}


class _Job3(Job):
    DEPENDENCIES = {_Job4}


class _Job1(Job):
    DEPENDENCIES = {_Job2, _Job3}


class _JobC(Job):
    pass


class _JobB(Job):
    DEPENDENCIES = {_JobC}


class _JobA(Job):
    DEPENDENCIES = {_JobB}


class _Scheduler(DaxScheduler):
    NAME = 'test_scheduler'
    JOBS: typing.Set[typing.Type[Job]] = set()


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
        # noinspection PyProtectedMember
        from dax.base.scheduler import JobAction
        for a in JobAction:
            with self.subTest(job_action=a):
                self.assertEqual(str(a), a.name)

    def test_job_action_submittable(self):
        # noinspection PyProtectedMember
        from dax.base.scheduler import JobAction

        submittable = {JobAction.RUN}

        for a in JobAction:
            with self.subTest(job_action=a):
                self.assertEqual(a in submittable, a.submittable())

    def test_policy_complete(self):
        # noinspection PyProtectedMember
        from dax.base.scheduler import JobAction

        for p in Policy:
            with self.subTest(policy=str(p)):
                self.assertEqual(len(p.value), len(JobAction) ** 2, 'Policy not fully implemented')


class LazySchedulerTestCase(unittest.TestCase):
    POLICY = Policy.LAZY

    def setUp(self) -> None:
        self.mop = get_manager_or_parent(Policy=str(self.POLICY))

    def test_create_job(self):
        # noinspection PyProtectedMember
        from dax.base.scheduler import JobAction

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
        for v in j._process_arguments().values():
            self.assertNotIsInstance(v, ScanObject)
        self.assertDictEqual(j._process_arguments(),
                             {k: v.describe() if isinstance(v, ScanObject) else v for k, v in J0.ARGUMENTS.items()})

    # TODO: add test functions for the scheduler class


class GreedySchedulerTestCase(LazySchedulerTestCase):
    POLICY = Policy.GREEDY
