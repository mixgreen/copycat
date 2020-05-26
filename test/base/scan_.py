import unittest
import collections
import typing

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
        self.counter = collections.Counter()  # type: typing.Dict[str, int]

        # Scan
        self.add_scan('foo', 'foo', Scannable(RangeScan(1, self.FOO, self.FOO, randomize=True)))

    def host_setup(self) -> None:
        self.counter['host_setup'] += 1

    def device_setup(self):  # type: () -> None
        self.counter['device_setup'] += 1

    def run_point(self, point):  # type: (typing.Any) -> None
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

    def run_point(self, point):  # type: (typing.Any) -> None
        if self.counter['run_point'] == self.TERMINATE:
            raise TerminationRequested
        self.counter['run_point'] += 1


class _MockScanStop(_MockScan1):
    STOP = 5

    def run_point(self, point):  # type: (typing.Any) -> None
        if self.counter['run_point'] == self.STOP:
            self.stop_scan()
        self.counter['run_point'] += 1


class _MockScanInfinite(_MockScan1):
    INFINITE_SCAN_ARGUMENT = True
    INFINITE_SCAN_DEFAULT = True

    STOP = 100

    def run_point(self, point):  # type: (typing.Any) -> None
        if self.counter['run_point'] == self.STOP:
            self.stop_scan()
        self.counter['run_point'] += 1


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


if __name__ == '__main__':
    unittest.main()
