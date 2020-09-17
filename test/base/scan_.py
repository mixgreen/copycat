import unittest
import collections
import typing
import itertools
import numpy as np
import h5py  # type: ignore

from artiq.experiment import *

from dax.base.scan import *
from dax.base.dax import DaxSystem
from dax.util.artiq import get_manager_or_parent
from dax.util.output import temp_dir


class _MockSystem(DaxSystem):
    SYS_ID = 'test_system'
    SYS_VER = 0


class _MockScan1(DaxScan, _MockSystem):
    FOO = 20

    def build_scan(self) -> None:
        # Counter
        self.counter: typing.Counter[str] = collections.Counter()

        # Scan
        self.add_scan('foo', 'foo', Scannable(RangeScan(1, self.FOO, self.FOO, randomize=False)))

    def host_enter(self) -> None:
        self.counter['host_enter'] += 1

    def host_setup(self) -> None:
        self.counter['host_setup'] += 1

    def device_setup(self):  # type: () -> None
        self.counter['device_setup'] += 1

    def run_point(self, point, index):  # type: (typing.Any, typing.Any) -> None
        self.counter['run_point'] += 1

    def device_cleanup(self):  # type: () -> None
        self.counter['device_cleanup'] += 1

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
    BAR_POINTS = list(range(16))
    BAR = len(BAR_POINTS)

    def build_scan(self) -> None:
        self.add_static_scan('bar', self.BAR_POINTS)
        super(_MockScan2Static, self).build_scan()

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


class _MockScanInfiniteNoArgument(_MockScan1):
    INFINITE_SCAN_ARGUMENT = False
    INFINITE_SCAN_DEFAULT = True

    STOP = 100

    def run_point(self, point, index):  # type: (typing.Any, typing.Any) -> None
        if self.counter['run_point'] == self.STOP:
            self.stop_scan()
        self.counter['run_point'] += 1


class _MockScanDisableIndex(_MockScan1):
    ENABLE_INDEX = False


class _IndexAttributeError(AttributeError):
    pass


class _MockScanDisableIndexBad(_MockScan1):
    ENABLE_INDEX = False

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

    def setUp(self) -> None:
        self.mop = get_manager_or_parent()
        self.scan = _MockScan1(self.mop)

    def test_is_infinite(self):
        self.assertFalse(self.scan.is_infinite_scan, 'Scan reported incorrectly it was infinite')

    def test_call_counters(self):
        # Run the scan
        self.scan.run()

        # Verify counters
        counter_ref = {
            'host_enter': 1,
            'host_setup': 1,
            'device_setup': 1,
            'run_point': self.scan.FOO,
            'device_cleanup': 1,
            'host_cleanup': 1,
            'host_exit': 1,
        }
        self.assertDictEqual(self.scan.counter, counter_ref, 'Function counters did not match expected values')

    def test_raise_add_scan(self):
        with self.assertRaises(TypeError, msg='Adding scan outside build did not raise'):
            self.scan.add_scan('bar', 'bar', Scannable(NoScan(1)))

    def test_raise_add_static_scan(self):
        with self.assertRaises(TypeError, msg='Adding scan outside build did not raise'):
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
            _, dataset_mgr, _, _ = self.mop
            with h5py.File(file_name, 'w') as f:
                dataset_mgr.write_hdf5(f)

            # Read HDF5 file with scan reader
            r = DaxScanReader(file_name)

            # Verify if the data matches
            scannables = self.scan.get_scannables()
            scan_points = self.scan.get_scan_points()
            keys = list(scannables.keys())
            self.assertSetEqual(set(r.keys), set(keys), 'Keys in reader did not match object keys')
            for k in keys:
                self.assertListEqual(scannables[k], list(r.scannables[k]),
                                     'Scannable in reader did not match object scannable')
                self.assertListEqual(scan_points[k], list(r.scan_points[k]),
                                     'Scan points in reader did not match object scan points')


class BuildScanTestCase(unittest.TestCase):

    def test_raise_duplicate_scan_key(self):
        class MockScan(_MockScanCallback):
            # noinspection PyMethodParameters
            def callback(self_scan):
                self_scan.add_scan('foo', 'foo', Scannable(NoScan(1)))
                with self.assertRaises(LookupError, msg='Reusing scan key did not raise'):
                    self_scan.add_scan('foo', 'foo', Scannable(NoScan(1)))
                with self.assertRaises(LookupError, msg='Reusing scan key for static scan did not raise'):
                    self_scan.add_static_scan('foo', [1])

        MockScan(get_manager_or_parent())

    def test_raise_duplicate_static_scan_key(self):
        class MockScan(_MockScanCallback):
            # noinspection PyMethodParameters
            def callback(self_scan):
                self_scan.add_static_scan('foo', [])
                with self.assertRaises(LookupError, msg='Reusing static scan key did not raise'):
                    self_scan.add_scan('foo', 'foo', Scannable(NoScan(1)))
                with self.assertRaises(LookupError, msg='Reusing static scan key for static scan did not raise'):
                    self_scan.add_static_scan('foo', [1])

        MockScan(get_manager_or_parent())

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

        MockScan(get_manager_or_parent())

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

        MockScan(get_manager_or_parent())

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
            np.empty(4, dtype=np.float),
            np.asarray(['foo', 'bar']),
        ]
        counter = itertools.count()

        class MockScan(_MockScanCallback):
            # noinspection PyMethodParameters
            def callback(self_scan):
                for points in test_data:
                    with self.subTest(points=points):
                        self_scan.add_static_scan(f'foo{next(counter)}', points)

        MockScan(get_manager_or_parent())

    def test_raise_bad_scan_key(self):
        class MockScan(_MockScanCallback):
            # noinspection PyMethodParameters
            def callback(self_scan):
                keys = ['aa.', '9int', 'foo-bar']
                for k in keys:
                    with self.subTest(key=k):
                        with self.assertRaises(ValueError, msg='Bad scan key did not raise'):
                            self_scan.add_scan(k, 'some name', Scannable(NoScan(1)))

        MockScan(get_manager_or_parent())

    def test_raise_bad_static_scan_key(self):
        class MockScan(_MockScanCallback):
            # noinspection PyMethodParameters
            def callback(self_scan):
                keys = ['aa.', '9int', 'foo-bar']
                for k in keys:
                    with self.subTest(key=k):
                        with self.assertRaises(ValueError, msg='Bad static scan key did not raise'):
                            self_scan.add_static_scan(k, [])

        MockScan(get_manager_or_parent())


class EmptyScanTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.scan = _MockScanEmpty(get_manager_or_parent())

        # Verify counters (object is an empty dict)
        self.assertDictEqual(self.scan.counter, {}, 'Function counters did not match expected values')

    def test_run_point_not_called(self):
        # The run function should exit early and run_point() is not called (which will raise if it does)
        self.scan.run()


class Scan2TestCase(Scan1TestCase):

    def setUp(self) -> None:
        self.mop = get_manager_or_parent()
        self.scan = _MockScan2(self.mop)

    def test_call_counters(self):
        # Run the scan
        self.scan.run()

        # Verify counters
        counter_ref = {
            'host_enter': 1,
            'host_setup': 1,
            'device_setup': 1,
            'run_point': self.scan.FOO * self.scan.BAR,
            'device_cleanup': 1,
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

    def setUp(self) -> None:
        self.mop = get_manager_or_parent()
        self.scan = _MockScan2Static(self.mop)


class ScanTerminateTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.scan = _MockScanTerminate(get_manager_or_parent())

    def test_call_counters(self):
        # Run the scan
        self.scan.run()

        # Verify counters
        counter_ref = {
            'host_enter': 1,
            'host_setup': 1,
            'device_setup': 1,
            'run_point': self.scan.TERMINATE,
            'device_cleanup': 1,
            'host_cleanup': 1,
            # host_exit() was not called, hence the entry is not existing in the counter
        }
        self.assertDictEqual(self.scan.counter, counter_ref, 'Function counters did not match expected values')


class ScanStopTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.scan = _MockScanStop(get_manager_or_parent())

    def test_call_counters(self):
        # Run the scan
        self.scan.run()

        # Verify counters
        counter_ref = {
            'host_enter': 1,
            'host_setup': 1,
            'device_setup': 1,
            'run_point': self.scan.STOP + 1,  # The last point is finished, so plus 1
            'device_cleanup': 1,
            'host_cleanup': 1,
            'host_exit': 1,  # host_exit() is called when using stop_scan()
        }
        self.assertDictEqual(self.scan.counter, counter_ref, 'Function counters did not match expected values')


class InfiniteScanTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.scan: _MockScan1 = _MockScanInfinite(get_manager_or_parent())

    def test_is_infinite(self):
        self.assertTrue(self.scan.is_infinite_scan, 'Scan reported incorrectly it was not infinite')

    def test_call_counters(self):
        # Run the scan
        self.scan.run()

        # Verify counters
        counter_ref = {
            'host_enter': 1,
            'host_setup': 1,
            'device_setup': 1,
            'run_point': self.scan.STOP + 1,  # The last point is finished, so plus 1
            'device_cleanup': 1,
            'host_cleanup': 1,
            'host_exit': 1,  # host_exit() is called when using stop_scan()
        }
        self.assertDictEqual(self.scan.counter, counter_ref, 'Function counters did not match expected values')


class InfiniteScanNoArgumentTestCase(InfiniteScanTestCase):

    def setUp(self) -> None:
        self.scan = _MockScanInfiniteNoArgument(get_manager_or_parent())


class DisableIndexScanTestCase(unittest.TestCase):

    def test_scan_length(self):
        # Create scan objects
        scan_w_index = _MockScan1(get_manager_or_parent())
        scan_wo_index = _MockScanDisableIndex(get_manager_or_parent())

        # Run both scans
        scan_w_index.run()
        scan_wo_index.run()

        # Verify if both scans had the same length and point values
        self.assertDictEqual(scan_w_index.get_scan_points(), scan_wo_index.get_scan_points(),
                             'Scan with index was not identical to scan without index')

        # Verify counters
        counter_ref = {
            'host_enter': 1,
            'host_setup': 1,
            'device_setup': 1,
            'run_point': scan_w_index.FOO,
            'device_cleanup': 1,
            'host_cleanup': 1,
            'host_exit': 1,
        }
        self.assertDictEqual(scan_w_index.counter, counter_ref, 'Function counters did not match expected values')
        self.assertDictEqual(scan_wo_index.counter, counter_ref, 'Function counters did not match expected values')

    def test_index_disabled(self):
        # Create scan object
        bad_scan = _MockScanDisableIndexBad(get_manager_or_parent())

        # Run the scan, expecting a specific exception
        with self.assertRaises(_IndexAttributeError, msg='Accessing an index attribute did not raise'):
            bad_scan.run()


class ScanValueTestCase(Scan2TestCase):

    def setUp(self) -> None:
        # Exceptions are raised if values don't match
        self.mop = get_manager_or_parent()
        self.scan = _MockScan2ValueCheck(self.mop)


class ScanValueReorderedTestCase(Scan2TestCase):

    def setUp(self) -> None:
        # Exceptions are raised if values don't match
        self.mop = get_manager_or_parent()
        self.scan = _MockScan2ValueCheckReordered(self.mop)

    def test_raise_scan_order(self):
        with self.assertRaises(TypeError, msg='Reordering scan outside build did not raise'):
            self.scan.set_scan_order()


if __name__ == '__main__':
    unittest.main()
