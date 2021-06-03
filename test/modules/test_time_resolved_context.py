import unittest
import numpy as np
import h5py

from dax.experiment import *
import dax.util.matplotlib_backend  # noqa: F401
from dax.modules.time_resolved_context import *
from dax.util.artiq import get_managers
from dax.util.output import temp_dir


class _TestSystem(DaxSystem):
    SYS_ID = 'unittest_system'
    SYS_VER = 0

    def build(self, default_dataset_key=None) -> None:  # type: ignore
        super(_TestSystem, self).build()
        self.time_resolved_context = TimeResolvedContext(self, 'context', default_dataset_key=default_dataset_key)


class TimeResolvedContextTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.managers = get_managers()
        self.s = _TestSystem(self.managers)
        self.s.dax_init()
        self.t = self.s.time_resolved_context

    def tearDown(self) -> None:
        # Close managers
        self.managers.close()

    def test_in_context(self):
        # Initially we are out of context
        self.assertFalse(self.t.in_context(), 'in_context() reported wrong value')
        with self.t:  # Without call we use the default values
            # In context
            self.assertTrue(self.t.in_context(), 'in_context() reported wrong value')

        # Out of context
        self.assertFalse(self.t.in_context(), 'in_context() reported wrong value')
        # Open context manually
        self.t.open()
        # In context
        self.assertTrue(self.t.in_context(), 'in_context() reported wrong value')
        # Close context manually
        self.t.close()
        # Out of context
        self.assertFalse(self.t.in_context(), 'in_context() reported wrong value')

    def test_append_out_of_context(self):
        # We can not call append out of context
        with self.assertRaises(TimeResolvedContextError, msg='Append out of context did not raise'):
            self.t.append([[]], 0.0, 0.0)

    def test_call_in_context(self):
        # We can not call the time resolved context out of context
        with self.t:
            with self.assertRaises(TimeResolvedContextError, msg='Config in context did not raise'):
                self.t.config_dataset()

    def test_nesting_exceptions(self):
        with self.assertRaises(TimeResolvedContextError, msg='Close out of context did not raise'):
            self.t.close()
        # Open the context
        self.t.open()
        with self.assertRaises(TimeResolvedContextError, msg='Open context in context did not raise'):
            self.t.open()
        # Close the context
        self.t.close()
        with self.assertRaises(TimeResolvedContextError, msg='Close out of context did not raise'):
            self.t.close()
        with self.t:
            with self.assertRaises(TimeResolvedContextError, msg='Nesting context did not raise'):
                with self.t:
                    pass

    def test_append_data(self):
        data = [
            ([[1, 2, 3, 1, 0, 0], [0, 0, 3, 7, 2, 0]], 180),
            ([[2, 3, 4, 2, 1, 0], [0, 0, 2, 4, 2, 1]], 1801),
            ([[3, 4, 5, 3, 2, 0], [0, 1, 3, 5, 3, 0]], 189),
            ([[4, 5, 6, 4, 2, 0], [0, 0, 4, 4, 3, 1]], 115),
            ([[5, 6, 7, 5, 2, 1], [0, 1, 5, 3, 1, 0]], 125),
            ([[6, 7, 8, 6, 3, 1], [0, 0, 0, 5, 6, 3]], 105),
            ([[1, 2, 3, 1, 0, 0], [0, 0, 3, 7, 2, 0]], -180),
            ([[2, 3, 4, 2, 1, 0], [0, 0, 2, 4, 2, 1]], -1801),
            ([[3, 4, 5, 3, 2, 0], [0, 1, 3, 5, 3, 0]], -189),
            ([[4, 5, 6, 4, 2, 0], [0, 0, 4, 4, 3, 1]], -115),
            ([[5, 6, 7, 5, 2, 1], [0, 1, 5, 3, 1, 0]], -125),
            ([[6, 7, 8, 6, 3, 1], [0, 0, 0, 5, 6, 3]], -105),
        ]

        with self.t:
            # Check buffer
            self.assertListEqual([], self.t._buffer_data, 'Buffer was not cleared when entering new context')
            for d in data:
                self.t.append_data(*d)
                r = (d[0], self.s.core.mu_to_seconds(d[1]))
                self.assertEqual(r, self.t._buffer_data[-1], 'Append did not appended data to buffer')
            # Check buffer
            r = [(d, self.s.core.mu_to_seconds(o)) for d, o in data]
            self.assertListEqual(r, self.t._buffer_data, 'Buffer did not contain expected data')

            # Add meta to prevent consistency errors
            for _ in range(len(data)):
                self.t.append_meta(2 * us, 0 * us, 0 * us)

        with self.t:
            # Check buffer
            self.assertListEqual([], self.t._buffer_data, 'Buffer was not cleared when entering new context')

    def test_data_meta_buffers_inconsistent(self):
        data = [
            ([[1, 2, 3, 1, 0, 0], [0, 0, 3, 7, 2, 0]], 180),
            ([[2, 3, 4, 2, 1, 0], [0, 0, 2, 4, 2, 1]], 1801),
        ]

        # Manually open context
        self.t.open()
        # Check buffer
        self.assertListEqual([], self.t._buffer_data, 'Buffer was not cleared when entering new context')
        for d in data:
            self.t.append_data(*d)
            r = (d[0], self.s.core.mu_to_seconds(d[1]))
            self.assertEqual(r, self.t._buffer_data[-1], 'Append did not appended data to buffer')
        # Check buffer
        r = [(d, self.s.core.mu_to_seconds(o)) for d, o in data]
        self.assertListEqual(r, self.t._buffer_data, 'Buffer did not contain expected data')
        # Add no meta to raise consistency errors

        with self.assertRaises(RuntimeError, msg='Inconsistent data and meta buffer did not raise'):
            self.t.close()

    def test_data_inconsistent(self):
        data = [
            ([[1, 2, 3, 1, 0, 0], [0, 0, 3, 7, 2, 0]], 180),
            ([[2, 3, 4, 2, 1, 0], [0, 0, 2, 4, 2]], 1801),
        ]

        # Manually open context
        self.t.open()
        # Check buffer
        self.assertListEqual([], self.t._buffer_data, 'Buffer was not cleared when entering new context')
        for d in data:
            self.t.append_data(*d)
            r = (d[0], self.s.core.mu_to_seconds(d[1]))
            self.assertEqual(r, self.t._buffer_data[-1], 'Append did not appended data to buffer')
        # Check buffer
        r = [(d, self.s.core.mu_to_seconds(o)) for d, o in data]
        self.assertListEqual(r, self.t._buffer_data, 'Buffer did not contain expected data')

        # Add meta to prevent consistency errors
        for _ in range(len(data)):
            self.t.append_meta(2 * us, 0 * us, 0 * us)

        with self.assertRaises(RuntimeError, msg='Inconsistent data in buffer did not raise'):
            self.t.close()

    def test_data_inconsistent_2(self):
        data = [
            ([[1, 2, 3, 1, 0, 0], [0, 0, 3, 7, 2, 0]], 180),
            ([[2, 3, 4, 2, 1, 0]], 1801),
        ]

        # Manually open context
        self.t.open()
        # Check buffer
        self.assertListEqual([], self.t._buffer_data, 'Buffer was not cleared when entering new context')
        for d in data:
            self.t.append_data(*d)
            r = (d[0], self.s.core.mu_to_seconds(d[1]))
            self.assertEqual(r, self.t._buffer_data[-1], 'Append did not appended data to buffer')
        # Check buffer
        r = [(d, self.s.core.mu_to_seconds(o)) for d, o in data]
        self.assertListEqual(r, self.t._buffer_data, 'Buffer did not contain expected data')

        # Add meta to prevent consistency errors
        for _ in range(len(data)):
            self.t.append_meta(2 * us, 0 * us, 0 * us)

        with self.assertRaises(RuntimeError, msg='Inconsistent data in buffer did not raise'):
            self.t.close()

    def test_data_empty(self):
        data = [
            ([], 180),
            ([], 1801),
        ]

        # Manually open context
        self.t.open()
        # Check buffer
        self.assertListEqual([], self.t._buffer_data, 'Buffer was not cleared when entering new context')
        for d in data:
            self.t.append_data(*d)
            r = (d[0], self.s.core.mu_to_seconds(d[1]))
            self.assertEqual(r, self.t._buffer_data[-1], 'Append did not appended data to buffer')
        # Check buffer
        r = [(d, self.s.core.mu_to_seconds(o)) for d, o in data]
        self.assertListEqual(r, self.t._buffer_data, 'Buffer did not contain expected data')

        # Add meta to prevent consistency errors
        for _ in range(len(data)):
            self.t.append_meta(2 * us, 0 * us, 0 * us)

        with self.assertRaises(RuntimeError, msg='Empty data in buffer did not raise'):
            self.t.close()

    def test_append_meta(self):
        data = [
            (2 * us, 0 * us, 0 * us),
            (2 * us, 1 * us, 0 * us),
            (3 * us, 2 * us, 10 * us),
            (3 * us, 3 * us, 10 * us),
            (4 * us, 0 * us, 20 * us),
        ]

        with self.t:
            # Check buffer
            self.assertListEqual([], self.t._buffer_meta, 'Buffer was not cleared when entering new context')
            for d in data:
                self.t.append_meta(*d)
                self.assertEqual(d, self.t._buffer_meta[-1], 'Append did not appended data to buffer')
            # Check buffer
            self.assertListEqual(data, self.t._buffer_meta, 'Buffer did not contain expected data')

            # Add data to prevent consistency errors
            for _ in range(len(data)):
                self.t.append_data([[]])

        with self.t:
            # Check buffer
            self.assertListEqual([], self.t._buffer_meta, 'Buffer was not cleared when entering new context')

    def test_append_meta_mu(self):
        data = [
            (2000, 0, 0),
            (2000, 1000, 0),
            (3000, 2000, 10000),
            (3000, 3000, 10000),
            (4000, 0000, 20000),
        ]
        converted_data = [tuple(self.s.core.mu_to_seconds(e) for e in d) for d in data]

        with self.t:
            # Check buffer
            self.assertListEqual([], self.t._buffer_meta, 'Buffer was not cleared when entering new context')
            for d, ref in zip(data, converted_data):
                self.t.append_meta_mu(*d)
                self.assertEqual(ref, self.t._buffer_meta[-1], 'Append did not appended data to buffer')
            # Check buffer
            self.assertListEqual(converted_data, self.t._buffer_meta, 'Buffer did not contain expected data')

            # Add data to prevent consistency errors
            for _ in range(len(data)):
                self.t.append_data([[]])

        with self.t:
            # Check buffer
            self.assertListEqual([], self.t._buffer_meta, 'Buffer was not cleared when entering new context')

    def test_remove_meta(self):
        num_points = 5

        with self.t:
            # Check buffer
            self.assertListEqual([], self.t._buffer_meta, 'Buffer was not cleared when entering new context')

            # Append meta
            for _ in range(num_points):
                self.t.append_meta(2 * us, 0 * us, 0 * us)
            # Remove meta
            for _ in range(num_points):
                self.t.remove_meta()

            # Check buffer
            self.assertListEqual([], self.t._buffer_meta, 'Buffer was not empty after removing all metadata')

    def test_archive(self):
        bin_width = 1 * us
        bin_spacing = 1 * ns
        offset = 5 * ns
        data = [[1, 2], [3, 4]]

        with self.t:
            self.t.append(data, bin_width, bin_spacing, offset)

        # Check traces data format
        trace = self.t.get_traces()
        self.assertEqual(len(trace), 1, 'Output did not match expected size')
        self.assertSetEqual(set(TimeResolvedContext.DATASET_COLUMNS), set(trace[0].keys()),
                            'Trace keys did not match')
        self.assertTrue((np.asarray(trace[0]['result']) == np.asarray(data)).all(),
                        'Trace result did not match expected outcome')
        self.assertTrue((trace[0]['width'] == np.full(2, bin_width)).all(),
                        'Trace width did not match expected outcome')
        r = np.arange(len(data)) * (bin_width + bin_spacing) + offset
        self.assertTrue(np.allclose(trace[0]['time'], r),
                        'Trace time did not match expected outcome')

    def test_multi_archive(self):
        bin_width = 1 * us
        bin_spacing = 1 * ns
        offset = 5 * ns
        data = [[1, 2], [3, 4]]
        dataset_key = 'some_key'

        # Store in a specific dataset
        self.t.config_dataset(dataset_key)
        with self.t:
            self.t.append(data, bin_width, bin_spacing, offset)

        # Store other data too in the default dataset
        self.t.config_dataset()
        with self.t:
            self.t.append_mu([[]], 0, 0)

        # Check traces data format for our specific key
        trace = self.t.get_traces(dataset_key)
        self.assertEqual(len(trace), 1, 'Output did not match expected size')
        self.assertSetEqual(set(TimeResolvedContext.DATASET_COLUMNS), set(trace[0].keys()),
                            'Trace keys did not match')
        self.assertTrue((np.asarray(trace[0]['result']) == np.asarray(data)).all(),
                        'Trace result did not match expected outcome')
        self.assertTrue((trace[0]['width'] == np.full(2, bin_width)).all(),
                        'Trace width did not match expected outcome')
        r = np.arange(len(data)) * (bin_width + bin_spacing) + offset
        self.assertTrue(np.allclose(trace[0]['time'], r),
                        'Trace time did not match expected outcome')

    def test_dataset_traces(self):
        bin_width = 1 * us
        bin_spacing = 1 * ns
        offset = 5 * ns
        data = [[1, 2], [3, 4]]

        # Store data
        with self.t:
            self.t.append(data, bin_width, bin_spacing, offset)

        # Datasets
        result_key = self.t.DATASET_KEY_FORMAT.format(dataset_key=self.t.DEFAULT_DATASET_KEY, index=0, column='result')
        width_key = self.t.DATASET_KEY_FORMAT.format(dataset_key=self.t.DEFAULT_DATASET_KEY, index=0, column='width')
        time_key = self.t.DATASET_KEY_FORMAT.format(dataset_key=self.t.DEFAULT_DATASET_KEY, index=0, column='time')

        # Check traces data format for our specific key
        self.assertTrue((np.asarray(self.s.get_dataset(result_key)) == np.asarray(data)).all(),
                        'Trace result did not match expected outcome')
        self.assertTrue((self.s.get_dataset(width_key) == np.full(2, bin_width)).all(),
                        'Trace width did not match expected outcome')
        r = np.arange(len(data)) * (bin_width + bin_spacing) + offset
        self.assertTrue(np.allclose(self.s.get_dataset(time_key), r),
                        'Trace time did not match expected outcome')

    def test_default_dataset_key(self):
        dataset_key = 'foo'
        self.t.config_dataset(dataset_key)  # Store in a specific dataset
        with self.t:
            self.assertEqual(self.t._dataset_key, dataset_key, 'Custom dataset key was not correctly stored')
        with self.t:
            # Dataset key should remain
            self.assertEqual(self.t._dataset_key, dataset_key, 'Custom dataset key was not correctly saved')

    def test_archive_keys(self):
        keys = ['a', 'foo', 'bar', 'foobar']
        for k in keys:
            self.t.config_dataset(k)
            with self.t:
                pass

        self.assertSetEqual(set(keys), set(self.t.get_keys()))

    def test_applets(self):
        # In simulation we can only call these functions, but nothing will happen
        self.t.plot()
        self.t.clear_plot()
        self.t.disable_plot()
        self.t.disable_all_plots()

    def test_partition_bins(self):
        data = [
            ((50, 64, 2 * us, 1 * us), [(50, 0 * us)]),
            ((64, 64, 2 * us, 1 * us), [(64, 0 * us)]),
            ((100, 64, 2 * us, 1 * us), [(64, 0 * us), (36, 64 * 3 * us)]),
            ((128, 64, 2 * us, 1 * us), [(64, 0 * us), (64, 64 * 3 * us)]),
            ((129, 64, 2 * us, 1 * us), [(64, 0 * us), (64, 64 * 3 * us), (1, 128 * 3 * us)]),
            ((0, 64, 2 * us, 1 * us), []),
        ]

        for i, ref in data:
            with self.subTest(input=i):
                partitions = self.t.partition_bins(*i)
                for n, o in partitions:
                    self.assertIsInstance(n, np.int32, 'Unexpected num bins type')
                    self.assertIsInstance(o, float, 'Unexpected offset type')
                self.assertEqual(partitions, ref, 'Partitioned output did not match reference')

    def test_partition_bins_mu(self):
        data = [
            ((50, 64, 2000, 1000), [(50, 0)]),
            ((64, 64, 2000, 1000), [(64, 0)]),
            ((100, 64, 2000, 1000), [(64, 0), (36, 64 * 3000)]),
            ((128, 64, 2000, 1000), [(64, 0), (64, 64 * 3000)]),
            ((129, 64, 2000, 1000), [(64, 0), (64, 64 * 3000), (1, 128 * 3000)]),
            ((0, 64, 2000, 1000), []),
        ]

        for i, ref in data:
            with self.subTest(input=i):
                partitions = self.t.partition_bins_mu(*i)
                for n, o in partitions:
                    self.assertIsInstance(n, np.int32, 'Unexpected num bins type')
                    self.assertIsInstance(o, np.int64, 'Unexpected offset type')
                self.assertEqual(partitions, ref, 'Partitioned output did not match reference')

    def test_partition_bins_ceil(self):
        data = [
            ((50, 64, 2 * us, 1 * us), [(64, 0 * us)]),
            ((64, 64, 2 * us, 1 * us), [(64, 0 * us)]),
            ((100, 64, 2 * us, 1 * us), [(64, 0 * us), (64, 64 * 3 * us)]),
            ((128, 64, 2 * us, 1 * us), [(64, 0 * us), (64, 64 * 3 * us)]),
            ((129, 64, 2 * us, 1 * us), [(64, 0 * us), (64, 64 * 3 * us), (64, 128 * 3 * us)]),
            ((0, 64, 2 * us, 1 * us), []),
        ]

        for i, ref in data:
            with self.subTest(input=i):
                partitions = self.t.partition_bins(*i, ceil=True)
                for n, o in partitions:
                    self.assertIsInstance(n, np.int32, 'Unexpected num bins type')
                    self.assertIsInstance(o, float, 'Unexpected offset type')
                self.assertEqual(partitions, ref, 'Partitioned output with ceil did not match reference')

    def test_partition_bins_ceil_mu(self):
        data = [
            ((50, 64, 2000, 1000), [(64, 0)]),
            ((64, 64, 2000, 1000), [(64, 0)]),
            ((100, 64, 2000, 1000), [(64, 0), (64, 64 * 3000)]),
            ((128, 64, 2000, 1000), [(64, 0), (64, 64 * 3000)]),
            ((129, 64, 2000, 1000), [(64, 0), (64, 64 * 3000), (64, 128 * 3000)]),
            ((0, 64, 2000, 1000), []),
        ]

        for i, ref in data:
            with self.subTest(input=i):
                partitions = self.t.partition_bins_mu(*i, ceil=True)
                for n, o in partitions:
                    self.assertIsInstance(n, np.int32, 'Unexpected num bins type')
                    self.assertIsInstance(o, np.int64, 'Unexpected offset type')
                self.assertEqual(partitions, ref, 'Partitioned output with ceil did not match reference')

    def test_partition_window(self):
        data = [
            ((50 * 3 * us, 64, 2 * us, 1 * us), [(50, 0 * us)]),
            ((64 * 3 * us, 64, 2 * us, 1 * us), [(64, 0 * us)]),
            ((100 * 3 * us, 64, 2 * us, 1 * us), [(64, 0 * us), (36, 64 * 3 * us)]),
            ((128 * 3 * us, 64, 2 * us, 1 * us), [(64, 0 * us), (64, 64 * 3 * us)]),
            ((129 * 3 * us, 64, 2 * us, 1 * us), [(64, 0 * us), (64, 64 * 3 * us), (1, 128 * 3 * us)]),
            ((0 * 3 * us, 64, 2 * us, 1 * us), []),
            ((1 * ns, 64, 2 * us, 1 * us), [(1, 0 * us)]),
        ]

        for i, ref in data:
            with self.subTest(input=i):
                partitions = self.t.partition_window(*i)
                for n, o in partitions:
                    self.assertIsInstance(n, np.int32, 'Unexpected num bins type')
                    self.assertIsInstance(o, float, 'Unexpected offset type')
                self.assertEqual(partitions, ref, 'Partitioned output did not match reference')

    def test_partition_window_mu(self):
        data = [
            ((50000 * 3, 64, 2000, 1000), [(50, 0)]),
            ((64000 * 3, 64, 2000, 1000), [(64, 0)]),
            ((100000 * 3, 64, 2000, 1000), [(64, 0), (36, 64 * 3000)]),
            ((128000 * 3, 64, 2000, 1000), [(64, 0), (64, 64 * 3000)]),
            ((129000 * 3, 64, 2000, 1000), [(64, 0), (64, 64 * 3000), (1, 128 * 3000)]),
            ((0 * 3, 64, 2000, 1000), []),
            ((1, 64, 2000, 1000), [(1, 0)]),
        ]

        for i, ref in data:
            with self.subTest(input=i):
                partitions = self.t.partition_window_mu(*i)
                for n, o in partitions:
                    self.assertIsInstance(n, np.int32, 'Unexpected num bins type')
                    self.assertIsInstance(o, np.int64, 'Unexpected offset type')
                self.assertEqual(partitions, ref, 'Partitioned output did not match reference')

    def test_partition_window_ceil(self):
        data = [
            ((50 * 3 * us, 64, 2 * us, 1 * us), [(64, 0 * us)]),
            ((64 * 3 * us, 64, 2 * us, 1 * us), [(64, 0 * us)]),
            ((100 * 3 * us, 64, 2 * us, 1 * us), [(64, 0 * us), (64, 64 * 3 * us)]),
            ((128 * 3 * us, 64, 2 * us, 1 * us), [(64, 0 * us), (64, 64 * 3 * us)]),
            ((129 * 3 * us, 64, 2 * us, 1 * us), [(64, 0 * us), (64, 64 * 3 * us), (64, 128 * 3 * us)]),
            ((0 * 3 * us, 64, 2 * us, 1 * us), []),
            ((1 * ns, 64, 2 * us, 1 * us), [(64, 0 * us)]),
        ]

        for i, ref in data:
            with self.subTest(input=i):
                partitions = self.t.partition_window(*i, ceil=True)
                for n, o in partitions:
                    self.assertIsInstance(n, np.int32, 'Unexpected num bins type')
                    self.assertIsInstance(o, float, 'Unexpected offset type')
                self.assertEqual(partitions, ref, 'Partitioned output with ceil did not match reference')

    def test_partition_window_ceil_mu(self):
        data = [
            ((50000 * 3, 64, 2000, 1000), [(64, 0)]),
            ((64000 * 3, 64, 2000, 1000), [(64, 0)]),
            ((100000 * 3, 64, 2000, 1000), [(64, 0), (64, 64 * 3000)]),
            ((128000 * 3, 64, 2000, 1000), [(64, 0), (64, 64 * 3000)]),
            ((129000 * 3, 64, 2000, 1000), [(64, 0), (64, 64 * 3000), (64, 128 * 3000)]),
            ((0 * 3, 64, 2000, 1000), []),
            ((1, 64, 2000, 1000), [(64, 0)]),
        ]

        for i, ref in data:
            with self.subTest(input=i):
                partitions = self.t.partition_window_mu(*i, ceil=True)
                for n, o in partitions:
                    self.assertIsInstance(n, np.int32, 'Unexpected num bins type')
                    self.assertIsInstance(o, np.int64, 'Unexpected offset type')
                self.assertEqual(partitions, ref, 'Partitioned output with ceil did not match reference')


class TimeResolvedAnalyzerTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.managers = get_managers()
        self.s = _TestSystem(self.managers)
        self.s.dax_init()
        self.t = self.s.time_resolved_context

    def tearDown(self) -> None:
        # Close managers
        self.managers.close()

    def test_analyzer_system(self):
        # The analyzer requests an output file which will trigger the creation of an experiment output dir
        # To prevent unnecessary directories after testing, we switch to a temp dir
        with temp_dir():
            TimeResolvedAnalyzer(self.s)

    def test_analyzer_module(self):
        # The analyzer requests an output file which will trigger the creation of an experiment output dir
        # To prevent unnecessary directories after testing, we switch to a temp dir
        with temp_dir():
            TimeResolvedAnalyzer(self.t)

    def test_hdf5_read(self):
        # Add data to the archive
        bin_width = 1 * us
        bin_spacing = 1 * ns
        offset = 5 * ns
        data_0 = [[1, 2], [3, 4], [2, 6], [4, 5], [9, 9], [9, 7], [7, 8]]
        data_1 = [[16, 25, 56], [66, 84, 83], [45, 77, 96], [88, 63, 79], [62, 93, 49], [29, 25, 7], [6, 17, 80]]

        # Store data
        with self.t:
            self.t.append(data_0, bin_width, bin_spacing, offset)
        self.t.config_dataset('foo')
        with self.t:
            self.t.append(data_1, bin_width, bin_spacing, offset)

        with temp_dir():
            # Write data to HDF5 file
            file_name = 'result.h5'
            with h5py.File(file_name, 'w') as f:
                self.managers.dataset_mgr.write_hdf5(f)

            # Read file with TimeResolvedAnalyzer
            a = TimeResolvedAnalyzer(file_name)

            # Compare results
            self.assertListEqual(a.keys, self.t.get_keys(), 'Keys did not match')
            for k in a.keys:
                for v, w in zip(a.traces[k], self.t.get_traces(k)):
                    for c in TimeResolvedContext.DATASET_COLUMNS:
                        self.assertIn(c, v, 'Did not found expected dataset columns')
                        self.assertIn(c, w, 'Did not found expected dataset columns')
                        self.assertTrue(np.array_equal(v[c], w[c]), f'Column/data "{c}" of trace did not match')

            # Compare to analyzer from object source
            b = TimeResolvedAnalyzer(self.s)
            self.assertListEqual(a.keys, b.keys, 'Keys did not match')
            for k in a.keys:
                for v, w in zip(a.traces[k], b.traces[k]):
                    for c in TimeResolvedContext.DATASET_COLUMNS:
                        self.assertIn(c, v, 'Did not found expected dataset columns')
                        self.assertIn(c, w, 'Did not found expected dataset columns')
                        self.assertTrue(np.array_equal(v[c], w[c]), f'Column/data "{c}" of trace did not match')

    def test_plot(self):
        # Add data to the archive
        bin_width = 1 * us
        bin_spacing = 1 * ns
        offset = 5 * ns
        data = [[16, 25, 56], [66, 84, 83], [45, 77, 96], [88, 63, 79]]

        # Store data
        with self.t:
            self.t.append(data, bin_width, bin_spacing, offset)

        with temp_dir():
            # Make analyzer object
            a = TimeResolvedAnalyzer(self.s)
            # Call plot functions to see if no exceptions occur
            a.plot_all_traces()

    def test_plot_hdf5(self):
        # Add data to the archive
        bin_width = 1 * us
        bin_spacing = 1 * ns
        offset = 5 * ns
        data = [[16, 25, 56], [66, 84, 83], [45, 77, 96], [88, 63, 79]]

        # Store data
        with self.t:
            self.t.append(data, bin_width, bin_spacing, offset)

        with temp_dir():
            # Write data to HDF5 file
            file_name = 'result.h5'
            with h5py.File(file_name, 'w') as f:
                self.managers.dataset_mgr.write_hdf5(f)

            # Make analyzer object
            a = TimeResolvedAnalyzer(file_name)
            # Call plot functions to see if no exceptions occur
            a.plot_all_traces()


if __name__ == '__main__':
    unittest.main()
