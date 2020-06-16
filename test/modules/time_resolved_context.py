import unittest
import numpy as np

from dax.experiment import *
from dax.modules.time_resolved_context import *
from dax.util.artiq_helpers import get_manager_or_parent
from dax.util.output import temp_dir
import dax.util.matplotlib_backend  # noqa: F401


class _TestSystem(DaxSystem):
    SYS_ID = 'unittest_system'
    SYS_VER = 0

    def build(self, default_dataset_key=None) -> None:  # type: ignore
        super(_TestSystem, self).build()
        self.time_resolved_context = TimeResolvedContext(self, 'context', default_dataset_key=default_dataset_key)


class TimeResolvedContextTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.s = _TestSystem(get_manager_or_parent())
        self.s.dax_init()
        self.t = self.s.time_resolved_context

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

            # Add meta to prevent consistency errors
            for _ in range(len(data)):
                self.t.append_data([[]])

        with self.t:
            # Check buffer
            self.assertListEqual([], self.t._buffer_meta, 'Buffer was not cleared when entering new context')

    def test_archive(self):
        bin_width = 1 * us
        bin_spacing = 1 * ns
        offset = 5 * ns
        offset_mu = 10
        data = [[1, 2], [3, 4]]

        with self.t:
            self.t.append(data, bin_width, bin_spacing, offset, offset_mu)

        # Check traces data format
        trace = self.t.get_traces()
        self.assertEqual(len(trace), 1, 'Output did not match expected size')
        self.assertSetEqual(set(TimeResolvedContext.DATASET_COLUMNS), set(trace[0].keys()),
                            'Trace keys did not match')
        self.assertTrue((np.asarray(trace[0]['result']) == np.asarray(data)).all(),
                        'Trace result did not match expected outcome')
        self.assertTrue((trace[0]['width'] == np.full(2, bin_width)).all(),
                        'Trace width did not match expected outcome')
        r = np.arange(len(data)) * (bin_width + bin_spacing) + offset + self.s.core.mu_to_seconds(offset_mu)
        self.assertTrue(np.allclose(trace[0]['time'], r),
                        'Trace time did not match expected outcome')

    def test_multi_archive(self):
        bin_width = 1 * us
        bin_spacing = 1 * ns
        offset = 5 * ns
        offset_mu = 10
        data = [[1, 2], [3, 4]]
        dataset_key = 'some_key'

        # Store in a specific dataset
        self.t.config_dataset(dataset_key)
        with self.t:
            self.t.append(data, bin_width, bin_spacing, offset, offset_mu)

        # Store other data too in the default dataset
        self.t.config_dataset()
        with self.t:
            self.t.append([[]], 0.0, 0.0, 0.0, 0)

        # Check traces data format for our specific key
        trace = self.t.get_traces(dataset_key)
        self.assertEqual(len(trace), 1, 'Output did not match expected size')
        self.assertSetEqual(set(TimeResolvedContext.DATASET_COLUMNS), set(trace[0].keys()),
                            'Trace keys did not match')
        self.assertTrue((np.asarray(trace[0]['result']) == np.asarray(data)).all(),
                        'Trace result did not match expected outcome')
        self.assertTrue((trace[0]['width'] == np.full(2, bin_width)).all(),
                        'Trace width did not match expected outcome')
        r = np.arange(len(data)) * (bin_width + bin_spacing) + offset + self.s.core.mu_to_seconds(offset_mu)
        self.assertTrue(np.allclose(trace[0]['time'], r),
                        'Trace time did not match expected outcome')

    def test_ddb_traces(self):
        bin_width = 1 * us
        bin_spacing = 1 * ns
        offset = 5 * ns
        offset_mu = 10
        data = [[1, 2], [3, 4]]

        # Store data
        with self.t:
            self.t.append(data, bin_width, bin_spacing, offset, offset_mu)

        # Datasets
        result_key = self.t.DATASET_KEY_FORMAT.format(dataset_key=self.t.DEFAULT_DATASET_KEY, index=0, column='result')
        width_key = self.t.DATASET_KEY_FORMAT.format(dataset_key=self.t.DEFAULT_DATASET_KEY, index=0, column='width')
        time_key = self.t.DATASET_KEY_FORMAT.format(dataset_key=self.t.DEFAULT_DATASET_KEY, index=0, column='time')

        # Check traces data format for our specific key
        self.assertTrue((np.asarray(self.s.get_dataset(result_key)) == np.asarray(data)).all(),
                        'Trace result did not match expected outcome')
        self.assertTrue((self.s.get_dataset(width_key) == np.full(2, bin_width)).all(),
                        'Trace width did not match expected outcome')
        r = np.arange(len(data)) * (bin_width + bin_spacing) + offset + self.s.core.mu_to_seconds(offset_mu)
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

        for i, o in data:
            with self.subTest(input=i):
                self.assertEqual(self.t.partition_bins(*i), o, 'Partitioned output did not match reference')

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

        for i, o in data:
            with self.subTest(input=i):
                self.assertEqual(self.t.partition_window(*i), o, 'Partitioned output did not match reference')


class TimeResolvedAnalyzerTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.s = _TestSystem(get_manager_or_parent())
        self.s.dax_init()
        self.t = self.s.time_resolved_context

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


if __name__ == '__main__':
    unittest.main()
