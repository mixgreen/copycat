import unittest
import collections
import typing
import itertools

from artiq.experiment import *

import dax.base.scan
from dax.base.scan import *
from dax.base.dax import DaxSystem
from dax.util.artiq_helpers import get_manager_or_parent


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

    @rpc
    def rpc_func(self):
        pass

    @portable
    def portable_func(self):
        pass

    @kernel
    def kernel_func(self):
        pass


class _MockScanCallback(_MockScan1):

    def build_scan(self) -> None:
        self.callback()

    def callback(self):
        raise NotImplementedError


class _MockScan2(_MockScan1):
    BAR = 30

    def build_scan(self) -> None:
        super(_MockScan2, self).build_scan()
        self.add_scan('bar', 'bar', Scannable(RangeScan(1, self.BAR, self.BAR, randomize=True)))


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
        self.scan_foo = itertools.chain(*[itertools.repeat(v, self.BAR) for v in scan_values['foo']])
        self.scan_bar = itertools.cycle(scan_values['bar'])

        # Iterators to check indices
        self.index_foo = itertools.chain(*[itertools.repeat(v, self.BAR) for v in range(self.FOO)])
        self.index_bar = itertools.cycle(range(self.BAR))

    def run_point(self, point, index):  # type: (typing.Any, typing.Any) -> None
        super(_MockScan2ValueCheck, self).run_point(point, index)

        # Check values of points
        point_foo = next(self.scan_foo)
        point_bar = next(self.scan_bar)
        assert point.foo == point_foo, '{} != {}'.format(point.foo, point_foo)
        assert point.bar == point_bar, '{} != {}'.format(point.bar, point_bar)

        # Check indices
        index_foo = next(self.index_foo)
        index_bar = next(self.index_bar)
        assert index.foo == index_foo, '{} != {}'.format(index.foo, index_foo)
        assert index.bar == index_bar, '{} != {}'.format(index.bar, index_bar)


class _MockScan2ValueCheckReordered(_MockScan2):
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

    def run_point(self, point, index):  # type: (typing.Any, typing.Any) -> None
        super(_MockScan2ValueCheckReordered, self).run_point(point, index)

        # Check values of points
        point_foo = next(self.scan_foo)
        point_bar = next(self.scan_bar)
        assert point.foo == point_foo, '{} != {}'.format(point.foo, point_foo)
        assert point.bar == point_bar, '{} != {}'.format(point.bar, point_bar)

        # Check indices
        index_foo = next(self.index_foo)
        index_bar = next(self.index_bar)
        assert index.foo == index_foo, '{} != {}'.format(index.foo, index_foo)
        assert index.bar == index_bar, '{} != {}'.format(index.bar, index_bar)


class Scan1TestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.scan = _MockScan1(get_manager_or_parent())

    def test_is_kernel(self):
        is_kernel = dax.base.scan._is_kernel

        self.assertFalse(is_kernel(self.scan.run_point), 'Undecorated function wrongly marked as a kernel function')
        self.assertFalse(is_kernel(self.scan.rpc_func), 'RPC function wrongly marked as a kernel function')
        self.assertFalse(is_kernel(self.scan.portable_func), 'Portable function wrongly marked as a kernel function')
        self.assertTrue(is_kernel(self.scan.kernel_func), 'Kernel function not correctly recognized as such')

    def test_call_counters(self):
        # Run the scan
        self.scan.run()

        # Verify counters
        counter_ref = {
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

    def test_get_scan_points(self):
        self.scan.run()
        points = self.scan.get_scan_points()
        self.assertIn('foo', points)

    def test_get_scannables(self):
        scannables = self.scan.get_scannables()
        self.assertIn('foo', scannables)
        self.assertEqual(len(scannables['foo']), self.scan.FOO)


class BuildScanTestCase(unittest.TestCase):

    def test_raise_duplicate_scan_key(self):
        class MockScan(_MockScanCallback):
            # noinspection PyMethodParameters
            def callback(self_scan):
                self_scan.add_scan('foo', 'foo', Scannable(NoScan(1)))
                with self.assertRaises(LookupError, msg='Reusing scan key did not raise'):
                    self_scan.add_scan('foo', 'foo', Scannable(NoScan(1)))

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


class Scan2TestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.scan = _MockScan2(get_manager_or_parent())

    def test_call_counters(self):
        # Run the scan
        self.scan.run()

        # Verify counters
        counter_ref = {
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


class ScanTerminateTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.scan = _MockScanTerminate(get_manager_or_parent())

    def test_call_counters(self):
        # Run the scan
        self.scan.run()

        # Verify counters
        counter_ref = {
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
        self.scan = _MockScanInfinite(get_manager_or_parent())

    def test_call_counters(self):
        # Run the scan
        self.scan.run()

        # Verify counters
        counter_ref = {
            'host_setup': 1,
            'device_setup': 1,
            'run_point': self.scan.STOP + 1,  # The last point is finished, so plus 1
            'device_cleanup': 1,
            'host_cleanup': 1,
            'host_exit': 1,  # host_exit() is called when using stop_scan()
        }
        self.assertDictEqual(self.scan.counter, counter_ref, 'Function counters did not match expected values')


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
        self.scan = _MockScan2ValueCheck(get_manager_or_parent())


class ScanValueReorderedTestCase(Scan2TestCase):

    def setUp(self) -> None:
        # Exceptions are raised if values don't match
        self.scan = _MockScan2ValueCheckReordered(get_manager_or_parent())

    def test_raise_scan_order(self):
        with self.assertRaises(TypeError, msg='Reordering scan outside build did not raise'):
            self.scan.set_scan_order()


if __name__ == '__main__':
    unittest.main()
