import unittest
import collections
import typing
import itertools
import numpy as np
import h5py

from artiq.experiment import *

from dax.base.scan import DaxScan, DaxScanReader, DaxScanChain
from dax.base.system import DaxSystem
from dax.util.artiq import get_managers
from dax.util.output import temp_dir

import test.helpers


class _MockSystem(DaxSystem):
    SYS_ID = 'test_system'
    SYS_VER = 0
    CORE_LOG_KEY = None
    DAX_INFLUX_DB_KEY = None


class _MockScan1(DaxScan, _MockSystem):
    FOO = 20

    def build_scan(self) -> None:
        # Counter
        self.counter: typing.Counter[str] = collections.Counter()

        # Scan
        self.add_scan('foo', 'foo', Scannable(RangeScan(1, self.FOO, self.FOO, randomize=False)))

    def init_scan_elements(self) -> None:
        super(_MockScan1, self).init_scan_elements()
        self.counter['init_scan_elements'] += 1

    def host_enter(self) -> None:
        self.counter['host_enter'] += 1

    def host_setup(self) -> None:
        self.counter['host_setup'] += 1

    def _dax_control_flow_setup(self) -> None:
        self.counter['_dax_control_flow_setup'] += 1

    def device_setup(self):  # type: () -> None
        self.counter['device_setup'] += 1

    def run_point(self, point, index):  # type: (typing.Any, typing.Any) -> None
        self.counter['run_point'] += 1

    def device_cleanup(self):  # type: () -> None
        self.counter['device_cleanup'] += 1

    def _dax_control_flow_cleanup(self) -> None:
        self.counter['_dax_control_flow_cleanup'] += 1

    def host_cleanup(self) -> None:
        self.counter['host_cleanup'] += 1

    def host_exit(self) -> None:
        self.counter['host_exit'] += 1


class _MockScanCallback(_MockScan1):

    def build_scan(self) -> None:
        self.callback()

    def callback(self):
        raise NotImplementedError


class _MockScanEmpty(_MockScan1):

    def build_scan(self) -> None:
        # Counter
        self.counter: typing.Counter[str] = collections.Counter()

    def run_point(self, point, index):  # type: (typing.Any, typing.Any) -> None
        raise NotImplementedError('The run_point() function should not have been reached')


class _MockScan2(_MockScan1):
    BAR = 30

    def build_scan(self) -> None:
        super(_MockScan2, self).build_scan()
        self._add_scan()

    def _add_scan(self):
        self.add_scan('bar', 'bar', Scannable(RangeScan(1, self.BAR, self.BAR, randomize=True)))


class _MockScan2Static(_MockScan2):
    BAR = 16
    BAR_POINTS = list(range(BAR))

    def build_scan(self) -> None:
        self.add_static_scan('bar', self.BAR_POINTS)
        super(_MockScan2Static, self).build_scan()

    def _add_scan(self):
        pass


class _MockScan2Iterator(_MockScan2):
    BAR = 16
    BAR_POINTS = list(range(BAR))

    def build_scan(self) -> None:
        self.add_iterator('bar', 'bar', self.BAR)
        super(_MockScan2Iterator, self).build_scan()

    def _add_scan(self):
        pass


class _MockScanTerminate(_MockScan1):
    TERMINATE = 5

    def run_point(self, point, index):  # type: (typing.Any, typing.Any) -> None
        if self.counter['run_point'] == self.TERMINATE:
            raise TerminationRequested
        self.counter['run_point'] += 1


class _MockScanStop(_MockScan1):
    STOP = 5

    def run_point(self, point, index):  # type: (typing.Any, typing.Any) -> None
        if self.counter['run_point'] == self.STOP:
            self.stop_scan()
        self.counter['run_point'] += 1


class _MockScanInfinite(_MockScan1):
    INFINITE_SCAN_ARGUMENT = True
    INFINITE_SCAN_DEFAULT = True

    STOP = 100

    def run_point(self, point, index):  # type: (typing.Any, typing.Any) -> None
        if self.counter['run_point'] == self.STOP:
            self.stop_scan()
        self.counter['run_point'] += 1


class _MockScanChain(_MockScan1):
    def build_scan(self) -> None:
        # Counter
        self.counter: typing.Counter[str] = collections.Counter()

        # Scan
        with DaxScanChain(self, 'foo', group='bar') as chain:
            chain.add_scan('1', Scannable(RangeScan(1, self.FOO, self.FOO, randomize=False)))
            chain.add_scan('2', Scannable(RangeScan(1, self.FOO, self.FOO, randomize=False)))


class _MockScanInfiniteNoArgument(_MockScanInfinite):
    INFINITE_SCAN_ARGUMENT = False
    INFINITE_SCAN_DEFAULT = True

    STOP = 100

    def run_point(self, point, index):  # type: (typing.Any, typing.Any) -> None
        if self.counter['run_point'] == self.STOP:
            self.stop_scan()
        self.counter['run_point'] += 1


class _MockScanDisableIndex(_MockScan1):
    ENABLE_SCAN_INDEX = False


class _IndexAttributeError(AttributeError):
    pass


class _MockScanDisableIndexBad(_MockScan1):
    ENABLE_SCAN_INDEX = False

    def run_point(self, point, index):  # type: (typing.Any, typing.Any) -> None
        assert point.foo is not None
        try:
            assert index.foo is not None  # Will raise AttributeError
        except AttributeError:
            # Change the exception type to specifically recognize this case
            raise _IndexAttributeError from AttributeError
        self.counter['run_point'] += 1


class _MockScan2ValueCheck(_MockScan2):

    def build_scan(self) -> None:
        super(_MockScan2ValueCheck, self).build_scan()

        # Iterators to check the values
        scan_values = self.get_scannables()
        self.scan_foo: typing.Iterator[typing.Any] = itertools.chain(
            *[itertools.repeat(v, self.BAR) for v in scan_values['foo']])
        self.scan_bar: typing.Iterator[typing.Any] = itertools.cycle(scan_values['bar'])

        # Iterators to check indices
        self.index_foo: typing.Iterator[typing.Any] = itertools.chain(
            *[itertools.repeat(v, self.BAR) for v in range(self.FOO)])
        self.index_bar: typing.Iterator[typing.Any] = itertools.cycle(range(self.BAR))

    def run_point(self, point, index):  # type: (typing.Any, typing.Any) -> None
        super(_MockScan2ValueCheck, self).run_point(point, index)

        # Check values of points
        point_foo = next(self.scan_foo)
        point_bar = next(self.scan_bar)
        assert point.foo == point_foo, f'{point.foo} != {point_foo}'
        assert point.bar == point_bar, f'{point.bar} != {point_bar}'

        # Check indices
        index_foo = next(self.index_foo)
        index_bar = next(self.index_bar)
        assert index.foo == index_foo, f'{index.foo} != {index_foo}'
        assert index.bar == index_bar, f'{index.bar} != {index_bar}'


class _MockScan2ValueCheckReordered(_MockScan2ValueCheck):
    def build_scan(self) -> None:
        super(_MockScan2ValueCheckReordered, self).build_scan()

        # Change the scan order
        self.set_scan_order('bar', 'foo')

        # Iterators to check the values
        scan_values = self.get_scannables()
        self.scan_bar = itertools.chain(*[itertools.repeat(v, self.FOO) for v in scan_values['bar']])
        self.scan_foo = itertools.cycle(scan_values['foo'])

        # Iterators to check indices
        self.index_bar = itertools.chain(*[itertools.repeat(v, self.FOO) for v in range(self.BAR)])
        self.index_foo = itertools.cycle(range(self.FOO))


class Scan1TestCase(unittest.TestCase):
    SCAN_CLASS = _MockScan1

    def setUp(self) -> None:
        self.managers = get_managers()
        self.scan = self.SCAN_CLASS(self.managers)

    def tearDown(self) -> None:
        # Close managers
        self.managers.close()

    def test_kernel_invariants(self):
        test.helpers.test_system_kernel_invariants(self, self.scan)

    def test_kernel_invariants_scan_elements(self):
        self.scan.run()

        num_scannables = len(self.scan.get_scannables())

        for p, i in self.scan._dax_scan_elements:
            self.assertEqual(len(p.kernel_invariants), num_scannables)
            test.helpers.test_kernel_invariants(self, p)

            if self.scan.ENABLE_SCAN_INDEX:
                self.assertEqual(len(i.kernel_invariants), num_scannables)
                test.helpers.test_kernel_invariants(self, i)
            else:
                self.assertEqual(len(i.kernel_invariants), 0)

    def test_is_infinite(self):
        self.assertFalse(self.scan.is_infinite_scan, 'Scan reported incorrectly it was infinite')

    def test_is_terminated(self):
        self.assertFalse(self.scan.is_terminated_scan)

    def test_call_counters(self):
        # Run the scan
        self.scan.run()

        # Verify counters
        counter_ref = {
            'init_scan_elements': 1,
            'host_enter': 1,
            'host_setup': 1,
            '_dax_control_flow_setup': 1,
            'device_setup': 1,
            'run_point': self.scan.FOO,
            'device_cleanup': 1,
            '_dax_control_flow_cleanup': 1,
            'host_cleanup': 1,
            'host_exit': 1,
        }
        self.assertDictEqual(self.scan.counter, counter_ref, 'Function counters did not match expected values')

    def test_early_scan_element_init(self):
        # Call element init before run
        self.scan.init_scan_elements()
        # Run the scan
        self.scan.run()
        # Verify init was only called once
        self.assertEqual(self.scan.counter['init_scan_elements'], 1,
                         'init_scan_elements counter did not match expected value')

    def test_raise_add_scan(self):
        with self.assertRaises(RuntimeError):
            self.scan.add_scan('bar', 'bar', Scannable(NoScan(1)))

    def test_raise_add_iterator(self):
        with self.assertRaises(RuntimeError):
            self.scan.add_iterator('bar', 'bar', 1)

    def test_raise_add_static_scan(self):
        with self.assertRaises(RuntimeError):
            self.scan.add_static_scan('bar', [])

    def test_get_scan_points_too_early(self):
        with self.assertRaises(AttributeError, msg='Scan point request before run did not raise'):
            self.scan.get_scan_points()
        self.scan.run()
        self.scan.get_scan_points()

    def test_get_scan_points(self):
        self.scan.run()
        points = self.scan.get_scan_points()
        self.assertIn('foo', points)

    def test_get_scannables(self):
        scannables = self.scan.get_scannables()
        self.assertIn('foo', scannables)
        self.assertEqual(len(scannables['foo']), self.scan.FOO)

    def test_scan_reader(self):
        self.scan.run()

        with temp_dir():
            # Write data to HDF5 file
            file_name = 'result.h5'
            with h5py.File(file_name, 'w') as f:
                self.managers.dataset_mgr.write_hdf5(f)

            # Read HDF5 file with scan reader
            r = DaxScanReader(file_name)

            # Verify if the data matches with the scan object
            scannables = self.scan.get_scannables()
            scan_points = self.scan.get_scan_points()
            keys = list(scannables.keys())
            self.assertSetEqual(set(r.keys), set(keys), 'Keys in reader did not match object keys')
            for k in keys:
                self.assertListEqual(scannables[k], list(r.scannables[k]),
                                     'Scannable in reader did not match object scannable')
                self.assertListEqual(scan_points[k], list(r.scan_points[k]),
                                     'Scan points in reader did not match object scan points')

            # Verify if the data matches with a scan reader using a different source
            r_ = DaxScanReader(self.scan)
            self.assertSetEqual(set(r.keys), set(r_.keys), 'Keys in readers did not match')
            for k in r.keys:
                self.assertListEqual(list(r_.scannables[k]), list(r.scannables[k]),
                                     'Scannable in readers did not match')
                self.assertListEqual(list(r_.scan_points[k]), list(r.scan_points[k]),
                                     'Scan points in readers did not match')


class ChainScanTestCase(Scan1TestCase):
    # Run all the same tests as before to ensure that nothing breaks
    # Note: chain was created with two of the same scannable
    SCAN_CLASS = _MockScanChain

    def test_get_scannables(self):
        scannables = self.scan.get_scannables()
        self.assertIn('foo', scannables)
        self.assertEqual(len(scannables['foo']), self.scan.FOO * 2)

    def test_call_counters(self):
        # Run the scan
        self.scan.run()

        # Verify counters
        counter_ref = {
            'init_scan_elements': 1,
            'host_enter': 1,
            'host_setup': 1,
            '_dax_control_flow_setup': 1,
            'device_setup': 1,
            'run_point': self.scan.FOO * 2,
            'device_cleanup': 1,
            '_dax_control_flow_cleanup': 1,
            'host_cleanup': 1,
            'host_exit': 1,
        }

        self.assertDictEqual(self.scan.counter, counter_ref, 'Function counters did not match expected values')

    def test_raise_reentrant_context_in_order(self):
        class MockChainScan(_MockScanCallback):
            def callback(self_scan):
                with DaxScanChain(self_scan, key='foo') as chain:
                    chain.add_scan('bar', Scannable(NoScan(1)))
                with self.assertRaises(RuntimeError, msg='Reentering scan context after exit did not raise'):
                    with chain:
                        pass

        MockChainScan(self.managers)

    def test_raise_reentrant_context_in_context(self):
        class MockChainScan(_MockScanCallback):
            def callback(self_scan):
                with DaxScanChain(self_scan, key='foo') as chain:
                    chain.add_scan('bar', Scannable(NoScan(1)))
                    with self.assertRaises(RuntimeError, msg='Reentering scan context in context did not raise'):
                        with chain:
                            pass

        MockChainScan(self.managers)

    def test_raise_duplicate_scan_key(self):
        class MockChainScan(_MockScanCallback):
            def callback(self_scan):
                self_scan.add_scan('foo', 'foo', Scannable(NoScan(1)))
                with self.assertRaises(LookupError):
                    self_scan.add_scan('foo', 'foo', Scannable(NoScan(1)))
                with self.assertRaises(LookupError):
                    self_scan.add_iterator('foo', 'foo', 1)
                with self.assertRaises(LookupError):
                    self_scan.add_static_scan('foo', [1])

        MockChainScan(self.managers)


class BuildScanTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.managers = get_managers()

    def tearDown(self) -> None:
        # Close managers
        self.managers.close()

    def test_raise_duplicate_scan_key(self):
        class MockScan(_MockScanCallback):
            # noinspection PyMethodParameters
            def callback(self_scan):
                self_scan.add_scan('foo', 'foo', Scannable(NoScan(1)))
                with self.assertRaises(LookupError):
                    self_scan.add_scan('foo', 'foo', Scannable(NoScan(1)))
                with self.assertRaises(LookupError):
                    self_scan.add_iterator('foo', 'foo', 1)
                with self.assertRaises(LookupError):
                    self_scan.add_static_scan('foo', [1])

        MockScan(self.managers)

    def test_raise_duplicate_static_scan_key(self):
        class MockScan(_MockScanCallback):
            # noinspection PyMethodParameters
            def callback(self_scan):
                self_scan.add_static_scan('foo', [])
                with self.assertRaises(LookupError):
                    self_scan.add_scan('foo', 'foo', Scannable(NoScan(1)))
                with self.assertRaises(LookupError):
                    self_scan.add_iterator('foo', 'foo', 1)
                with self.assertRaises(LookupError):
                    self_scan.add_static_scan('foo', [1])

        MockScan(self.managers)

    def test_raise_bad_scan_type(self):
        test_data = [
            NoScan(1),
            EnumerationValue('abc'),
            'aa',
            1,
        ]

        class MockScan(_MockScanCallback):
            # noinspection PyMethodParameters
            def callback(self_scan):
                for scannable in test_data:
                    with self.subTest(scannable=scannable):
                        with self.assertRaises(TypeError, msg='Bad scan type did not raise'):
                            # noinspection PyTypeChecker
                            self_scan.add_scan('foo', 'foo', scannable)

        MockScan(self.managers)

    def test_raise_bad_static_scan_type(self):
        test_data = [
            {1, 2},
            {1: 'a', 2: 'b'},
            (i ** 2 for i in range(5)),
            [1, 2, 3, 'a'],
            [1, 0.1],
            [1 + 4j],
            np.empty(4, dtype=np.complex_),
            np.empty(4, dtype=np.object_),
            np.empty((4, 2), dtype=np.int32),
        ]

        class MockScan(_MockScanCallback):
            # noinspection PyMethodParameters
            def callback(self_scan):
                for points in test_data:
                    with self.subTest(points=points):
                        with self.assertRaises(TypeError, msg='Bad scan type did not raise'):
                            self_scan.add_static_scan('foo', points)

        MockScan(self.managers)

    def test_good_static_scans(self):
        test_data = [
            [1, 2],
            [0.3],
            [True, False],
            ['foo', 'bar'],
            [],
            range(4),
            np.array([], dtype=np.int32),
            np.empty(4, dtype=np.int32),
            np.empty(4, dtype=np.int64),
            np.empty(4, dtype=np.float_),
            np.asarray(['foo', 'bar']),
        ]
        counter = itertools.count()

        class MockScan(_MockScanCallback):
            # noinspection PyMethodParameters
            def callback(self_scan):
                for points in test_data:
                    with self.subTest(points=points):
                        self_scan.add_static_scan(f'foo{next(counter)}', points)

        MockScan(self.managers)

    def test_raise_bad_scan_key(self):
        class MockScan(_MockScanCallback):
            # noinspection PyMethodParameters
            def callback(self_scan):
                keys = ['aa.', '9int', 'foo-bar']
                for k in keys:
                    with self.subTest(key=k):
                        with self.assertRaises(ValueError, msg='Bad scan key did not raise'):
                            self_scan.add_scan(k, 'some name', Scannable(NoScan(1)))

        MockScan(self.managers)

    def test_raise_bad_iterator_key(self):
        class MockScan(_MockScanCallback):
            # noinspection PyMethodParameters
            def callback(self_scan):
                keys = ['aa.', '9int', 'foo-bar']
                for k in keys:
                    with self.subTest(key=k):
                        with self.assertRaises(ValueError, msg='Bad iterator key did not raise'):
                            self_scan.add_iterator(k, 'some name', 1)

        MockScan(self.managers)

    def test_raise_bad_static_scan_key(self):
        class MockScan(_MockScanCallback):
            # noinspection PyMethodParameters
            def callback(self_scan):
                keys = ['aa.', '9int', 'foo-bar']
                for k in keys:
                    with self.subTest(key=k):
                        with self.assertRaises(ValueError, msg='Bad static scan key did not raise'):
                            self_scan.add_static_scan(k, [])

        MockScan(self.managers)

    def test_scan_build_arguments(self):
        test_args = (1, 2, 'd', 6.9)
        test_kwargs = {'foo': 4, 'foo bar': 6.6, 'bar': RangeScan(2, 5, 10)}

        class MockScan(_MockScan1):
            # noinspection PyMethodParameters
            def build_scan(self_scan, *args, **kwargs) -> None:
                self.assertTupleEqual(args, test_args, 'Positional arguments did not match')
                self.assertDictEqual(kwargs, test_kwargs, 'Keyword arguments did not match')

        MockScan(self.managers, scan_args=test_args, scan_kwargs=test_kwargs)


class EmptyScanTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.managers = get_managers()
        self.scan = _MockScanEmpty(self.managers)

    def tearDown(self) -> None:
        # Close managers
        self.managers.close()

    def test_function_calls(self):
        # Verify counters (object is an empty dict)
        self.assertDictEqual(self.scan.counter, {}, 'Function counters did not match expected values')

    def test_run_point_not_called(self):
        # The run function should exit early and run_point() is not called (which will raise if it does)
        self.scan.run()


class Scan2TestCase(Scan1TestCase):
    SCAN_CLASS = _MockScan2

    def test_call_counters(self):
        # Run the scan
        self.scan.run()

        # Verify counters
        counter_ref = {
            'init_scan_elements': 1,
            'host_enter': 1,
            'host_setup': 1,
            '_dax_control_flow_setup': 1,
            'device_setup': 1,
            'run_point': self.scan.FOO * self.scan.BAR,
            'device_cleanup': 1,
            '_dax_control_flow_cleanup': 1,
            'host_cleanup': 1,
            'host_exit': 1,
        }
        self.assertDictEqual(self.scan.counter, counter_ref, 'Function counters did not match expected values')

    def test_get_scan_points(self):
        self.scan.run()
        points = self.scan.get_scan_points()
        self.assertIn('foo', points)
        self.assertIn('bar', points)
        self.assertEqual(len(points['foo']), self.scan.FOO * self.scan.BAR)
        self.assertEqual(len(points['bar']), self.scan.FOO * self.scan.BAR)

    def test_get_scannables(self):
        scannables = self.scan.get_scannables()
        self.assertIn('foo', scannables)
        self.assertIn('bar', scannables)
        self.assertEqual(len(scannables['foo']), self.scan.FOO)
        self.assertEqual(len(scannables['bar']), self.scan.BAR)


class Scan2StaticTestCase(Scan2TestCase):
    SCAN_CLASS = _MockScan2Static


class Scan2IteratorTestCase(Scan2TestCase):
    SCAN_CLASS = _MockScan2Iterator


class ScanTerminateTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.managers = get_managers()
        self.scan = _MockScanTerminate(self.managers)

    def tearDown(self) -> None:
        # Close managers
        self.managers.close()

    def test_call_counters(self):
        # Verify flag
        self.assertFalse(self.scan.is_terminated_scan)
        # Run the scan
        self.scan.run()
        # Verify flag
        self.assertTrue(self.scan.is_terminated_scan)

        # Verify counters
        counter_ref = {
            'init_scan_elements': 1,
            'host_enter': 1,
            'host_setup': 1,
            '_dax_control_flow_setup': 1,
            'device_setup': 1,
            'run_point': self.scan.TERMINATE,
            'device_cleanup': 1,
            '_dax_control_flow_cleanup': 1,
            'host_cleanup': 1,
            # host_exit() was not called, hence the entry is not existing in the counter
        }
        self.assertDictEqual(self.scan.counter, counter_ref, 'Function counters did not match expected values')


class ScanStopTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.managers = get_managers()
        self.scan = _MockScanStop(self.managers)

    def tearDown(self) -> None:
        # Close managers
        self.managers.close()

    def test_call_counters(self):
        # Run the scan
        self.scan.run()

        # Verify counters
        counter_ref = {
            'init_scan_elements': 1,
            'host_enter': 1,
            'host_setup': 1,
            '_dax_control_flow_setup': 1,
            'device_setup': 1,
            'run_point': self.scan.STOP + 1,  # The last point is finished, so plus 1
            'device_cleanup': 1,
            '_dax_control_flow_cleanup': 1,
            'host_cleanup': 1,
            'host_exit': 1,  # host_exit() is called when using stop_scan()
        }
        self.assertDictEqual(self.scan.counter, counter_ref, 'Function counters did not match expected values')


class InfiniteScanTestCase(unittest.TestCase):
    SCAN_CLASS = _MockScanInfinite

    def setUp(self) -> None:
        self.managers = get_managers()
        self.scan: _MockScan1 = self.SCAN_CLASS(self.managers)

    def tearDown(self) -> None:
        # Close managers
        self.managers.close()

    def test_is_infinite(self):
        self.assertTrue(self.scan.is_infinite_scan, 'Scan reported incorrectly it was not infinite')

    def test_call_counters(self):
        # Run the scan
        self.scan.run()

        # Verify counters
        counter_ref = {
            'init_scan_elements': 1,
            'host_enter': 1,
            'host_setup': 1,
            '_dax_control_flow_setup': 1,
            'device_setup': 1,
            'run_point': self.scan.STOP + 1,  # The last point is finished, so plus 1
            'device_cleanup': 1,
            '_dax_control_flow_cleanup': 1,
            'host_cleanup': 1,
            'host_exit': 1,  # host_exit() is called when using stop_scan()
        }
        self.assertDictEqual(self.scan.counter, counter_ref, 'Function counters did not match expected values')


class InfiniteScanNoArgumentTestCase(InfiniteScanTestCase):
    SCAN_CLASS = _MockScanInfiniteNoArgument


class DisableIndexScanTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.managers = get_managers()

    def tearDown(self) -> None:
        # Close managers
        self.managers.close()

    def test_scan_length(self):
        # Create scan objects
        scan_w_index = _MockScan1(self.managers)
        scan_wo_index = _MockScanDisableIndex(self.managers)

        # Run both scans
        scan_w_index.run()
        scan_wo_index.run()

        # Verify if both scans had the same length and point values
        self.assertDictEqual(scan_w_index.get_scan_points(), scan_wo_index.get_scan_points(),
                             'Scan with index was not identical to scan without index')

        # Verify counters
        counter_ref = {
            'init_scan_elements': 1,
            'host_enter': 1,
            'host_setup': 1,
            '_dax_control_flow_setup': 1,
            'device_setup': 1,
            'run_point': scan_w_index.FOO,
            'device_cleanup': 1,
            '_dax_control_flow_cleanup': 1,
            'host_cleanup': 1,
            'host_exit': 1,
        }
        self.assertDictEqual(scan_w_index.counter, counter_ref, 'Function counters did not match expected values')
        self.assertDictEqual(scan_wo_index.counter, counter_ref, 'Function counters did not match expected values')

    def test_index_disabled(self):
        # Create scan object
        bad_scan = _MockScanDisableIndexBad(self.managers)

        # Run the scan, expecting a specific exception
        with self.assertRaises(_IndexAttributeError, msg='Accessing an index attribute did not raise'):
            bad_scan.run()


class ScanValueTestCase(Scan2TestCase):
    SCAN_CLASS = _MockScan2ValueCheck


class ScanValueReorderedTestCase(Scan2TestCase):
    SCAN_CLASS = _MockScan2ValueCheckReordered

    def test_raise_scan_order(self):
        with self.assertRaises(RuntimeError, msg='Reordering scan outside build did not raise'):
            self.scan.set_scan_order()
