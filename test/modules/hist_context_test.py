import typing
import unittest
import collections
import numpy as np
import h5py

import artiq.coredevice

from dax.experiment import *
import dax.base.system
import dax.util.matplotlib_backend  # noqa: F401
from dax.modules.hist_context import *
from dax.interfaces.detection import DetectionInterface
from dax.interfaces.data_context import DataContextInterface, DataContextError, validate_interface
from dax.util.artiq import get_managers
from dax.util.output import temp_dir

from test.environment import CI_ENABLED
import test.helpers


class _MockDetectionModule(DaxModule, DetectionInterface):

    def build(self, state_detection_threshold: int):  # type: ignore[override]
        self.state_detection_threshold = state_detection_threshold

    def init(self) -> None:
        pass

    def post_init(self) -> None:
        pass

    def get_pmt_array(self) -> typing.List[artiq.coredevice.edge_counter.EdgeCounter]:
        raise NotImplementedError

    def get_state_detection_threshold(self) -> int:
        return self.state_detection_threshold

    def get_default_detection_time(self) -> float:
        return 100 * us


class _TestSystem(DaxSystem):
    SYS_ID = 'unittest_system'
    SYS_VER = 0
    CORE_LOG_KEY = None
    DAX_INFLUX_DB_KEY = None

    def build(self, state_detection_threshold=2, default_dataset_key=None) -> None:  # type: ignore[override]
        super(_TestSystem, self).build()
        self.detection = _MockDetectionModule(self, 'detection', state_detection_threshold=state_detection_threshold)
        self.hist_context = HistogramContext(self, 'hist_context', default_dataset_key=default_dataset_key)


class HistogramContextTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.managers = get_managers()
        self.s = _TestSystem(self.managers)
        self.s.dax_init()
        self.h = self.s.hist_context

    def tearDown(self) -> None:
        # Close managers
        self.managers.close()

    def test_data_context_types(self):
        self.assertIsInstance(self.h, DataContextInterface)
        self.assertIsInstance(HistogramContextError(), DataContextError)

    def test_data_context_valid_interface(self):
        self.assertTrue(validate_interface(self.h), 'HistogramContext is not a valid data context')

    def test_in_context(self):
        # Initially we are out of context
        self.assertFalse(self.h.in_context(), 'in_context() reported wrong value')
        with self.h:
            # In context
            self.assertTrue(self.h.in_context(), 'in_context() reported wrong value')

        # Out of context
        self.assertFalse(self.h.in_context(), 'in_context() reported wrong value')
        # Open context manually
        self.h.open()
        # In context
        self.assertTrue(self.h.in_context(), 'in_context() reported wrong value')
        # Close context manually
        self.h.close()
        # Out of context
        self.assertFalse(self.h.in_context(), 'in_context() reported wrong value')

    def test_append_out_of_context(self):
        for err_type in [HistogramContextError, DataContextError]:
            # We can not call append out of context
            with self.assertRaises(err_type, msg='Append out of context did not raise'):
                self.h.append([1])

    def test_call_in_context(self):
        for err_type in [HistogramContextError, DataContextError]:
            # We can not call the histogram context out of context
            with self.h:
                with self.assertRaises(err_type, msg='Config in context did not raise'):
                    self.h.config_dataset()

    def test_nesting_exceptions(self):
        for err_type in [HistogramContextError, DataContextError]:
            with self.assertRaises(err_type, msg='Close histogram out of context did not raise'):
                self.h.close()
            # Open the context
            self.h.open()
            with self.assertRaises(err_type, msg='Open context in context did not raise'):
                self.h.open()
            # Close the context
            self.h.close()
            with self.assertRaises(err_type, msg='Close histogram out of context did not raise'):
                self.h.close()
            with self.h:
                with self.assertRaises(err_type, msg='Nesting context did not raise'):
                    with self.h:
                        pass

    def test_append(self):
        data = [
            [1, 9],
            [2, 8],
            [2, 7],
            [3, 6],
            [3, 5],
            [3, 4],
        ]

        with self.h:
            # Check buffer
            self.assertListEqual([], self.h._buffer, 'Buffer was not cleared when entering new context')
            for d in data:
                self.h.append(d)
                self.assertEqual(d, self.h._buffer[-1], 'Append did not appended data to buffer')
            # Check buffer
            self.assertListEqual(data, self.h._buffer, 'Buffer did not contain expected data')

        with self.h:
            # Check buffer
            self.assertListEqual([], self.h._buffer, 'Buffer was not cleared when entering new context')

    def test_extend(self):
        data = [
            [1, 9],
            [2, 8],
            [2, 7],
            [3, 6],
            [3, 5],
            [3, 4],
        ]

        with self.h:
            # Check buffer
            self.assertListEqual([], self.h._buffer, 'Buffer was not cleared when entering new context')
            self.h.extend(data)
            # Check buffer
            self.assertListEqual(data, self.h._buffer, 'Buffer did not contain expected data')

        with self.h:
            # Check buffer
            self.assertListEqual([], self.h._buffer, 'Buffer was not cleared when entering new context')

    def test_empty_data(self):
        data = [[], [], []]

        # Open context manually
        self.h.open()
        # Check buffer
        self.assertListEqual([], self.h._buffer, 'Buffer was not cleared when entering new context')
        for d in data:
            self.h.append(d)
            self.assertEqual(d, self.h._buffer[-1], 'Append did not appended data to buffer')
        # Check buffer
        self.assertListEqual(data, self.h._buffer, 'Buffer did not contain expected data')
        with self.assertRaises(RuntimeError, msg='Submitting empty data did not raise'):
            self.h.close()

    def test_inconsistent_data(self):
        data = [[1, 2], [3, 4], [5, 6, 7]]

        # Open context manually
        self.h.open()
        # Check buffer
        self.assertListEqual([], self.h._buffer, 'Buffer was not cleared when entering new context')
        for d in data:
            self.h.append(d)
            self.assertEqual(d, self.h._buffer[-1], 'Append did not appended data to buffer')
        # Check buffer
        self.assertListEqual(data, self.h._buffer, 'Buffer did not contain expected data')
        with self.assertRaises(RuntimeError, msg='Submitting inconsistent data did not raise'):
            self.h.close()

    def test_raw_cache(self):
        # Add data to the cache
        num_histograms = 8
        data = [
            [1, 9],
            [2, 9],
            [2, 9],
            [3, 9],
            [3, 8],
            [3, 8],
        ]

        for _ in range(num_histograms):
            with self.h:
                for d in data:
                    self.h.append(d)

        # Check raw data
        raw = self.h.get_raw()
        self.assertEqual(raw, [data] * num_histograms, 'Obtained raw data did not meet expected data')

    def test_histogram_cache(self):
        # Add data to the cache
        num_histograms = 8
        data = [
            [1, 9],
            [2, 9],
            [2, 9],
            [3, 9],
            [3, 8],
            [3, 8],
        ]

        for _ in range(num_histograms):
            with self.h:
                for d in data:
                    self.h.append(d)

        # Check histograms data
        histograms = self.h.get_histograms()
        for h in histograms[0]:  # Channel 0
            self.assertDictEqual(h, {1: 1, 2: 2, 3: 3}, 'Obtained histograms did not meet expected format')
        for h in histograms[1]:  # Channel 1
            self.assertDictEqual(h, {8: 2, 9: 4}, 'Obtained histograms did not meet expected format')

    def test_multi_archive_histograms(self):
        # Add data to the archive
        num_histograms = 8
        dataset_key = 'foo'
        data_0 = [
            [1, 9],
            [2, 9],
            [2, 9],
            [3, 9],
            [3, 8],
            [3, 8],
        ]
        data_1 = [
            [4, 1, 0],
            [5, 2, 0],
            [6, 3, 0],
            [7, 4, 0],
            [8, 5, 0],
        ]

        # Store in a specific dataset
        self.h.config_dataset(dataset_key)
        for _ in range(num_histograms):
            with self.h:
                for d in data_0:
                    self.h.append(d)

        # Store other data too in the default dataset
        self.h.config_dataset()
        for _ in range(num_histograms):
            with self.h:
                for d in data_1:
                    self.h.append(d)

        # Check histograms data format for our specific key
        histograms = self.h.get_histograms(dataset_key)
        for h in histograms[0]:  # Channel 0
            self.assertDictEqual(h, {1: 1, 2: 2, 3: 3}, 'Obtained histograms did not meet expected format')
        for h in histograms[1]:  # Channel 1
            self.assertDictEqual(h, {8: 2, 9: 4}, 'Obtained histograms did not meet expected format')

    def test_dataset_raw(self):
        # Add data to the datasets
        num_histograms = 8
        data = [
            [1, 9],
            [2, 9],
            [2, 9],
            [3, 9],
            [3, 8],
            [3, 8],
        ]

        for _ in range(num_histograms):
            with self.h:
                for d in data:
                    self.h.append(d)

        # Get raw datasets
        key = '/'.join([self.h.DATASET_GROUP, self.h.DEFAULT_DATASET_KEY, self.h.RAW_DATASET_GROUP, '0'])
        dataset = self.s.get_dataset(key)
        self.assertEqual(dataset, data, 'Stored dataset does not match expected format')

    def test_get_probabilities(self):
        num_histograms = 8
        data = [
            [1, 9],
            [2, 9],
            [2, 9],
            [3, 8],
            [4, 7],
            [5, 6],
            [6, 5],
            [7, 2],
            [8, 1],
            [9, 0],
        ]

        for _ in range(num_histograms):
            with self.h:
                for d in data:
                    self.h.append(d)

        # Check data format
        probabilities = self.h.get_probabilities(state_detection_threshold=5)
        for p in probabilities[0]:  # Channel 0
            self.assertEqual(p, 4 / len(data), 'Obtained probabilities did not meet expected format')
        for p in probabilities[1]:  # Channel 1
            self.assertEqual(p, 6 / len(data), 'Obtained probabilities did not meet expected format')

    def test_get_mean_counts(self):
        num_histograms = 8
        data = [
            [1, 9],
            [2, 9],
            [2, 9],
            [3, 8],
            [4, 7],
            [5, 6],
            [6, 5],
            [7, 2],
            [8, 1],
            [9, 0],
        ]
        mean_count_ref = [np.mean(c) for c in zip(*data)]

        for _ in range(num_histograms):
            with self.h:
                for d in data:
                    self.h.append(d)

        # Check data
        mean_counts = self.h.get_mean_counts()
        for counts, ref in zip(mean_counts, mean_count_ref):
            self.assertEqual(len(counts), num_histograms)
            for c in counts:
                self.assertAlmostEqual(c, ref, msg='Obtained mean count does not match reference')

    def test_get_stdev_counts(self):
        num_histograms = 8
        data = [
            [1, 9],
            [2, 9],
            [2, 9],
            [3, 8],
            [4, 7],
            [5, 6],
            [6, 5],
            [7, 2],
            [8, 1],
            [9, 0],
        ]
        stdev_count_ref = [np.std(c) for c in zip(*data)]

        for _ in range(num_histograms):
            with self.h:
                for d in data:
                    self.h.append(d)

        # Check data
        stdev_counts = self.h.get_stdev_counts()
        for counts, ref in zip(stdev_counts, stdev_count_ref):
            self.assertEqual(len(counts), num_histograms)
            for c in counts:
                self.assertAlmostEqual(c, ref, msg='Obtained stdev count does not match reference')

    def test_default_dataset_key(self):
        dataset_key = 'foo'
        self.h.config_dataset(dataset_key)  # Store in a specific dataset
        with self.h:
            self.assertEqual(self.h._dataset_key, dataset_key, 'Custom dataset key was not correctly stored')
        with self.h:
            # Dataset key should remain
            self.assertEqual(self.h._dataset_key, dataset_key, 'Custom dataset key was not correctly saved')

    def test_archive_keys(self):
        keys = ['a', 'foo', 'bar', 'foobar']
        for k in keys:
            self.h.config_dataset(k)
            with self.h:
                pass

        self.assertSetEqual(set(keys), set(self.h.get_keys()))

    def test_applets(self):
        self.assertFalse(self.h._plot_state_probability)

        def plot_fn():
            # In simulation, we can only call these functions, but nothing will happen
            self.h.plot_histogram()
            self.h.plot_probability()
            self.h.plot_mean_count()
            self.h.plot_state_probability()
            self.h.clear_probability_plot()
            self.h.clear_mean_count_plot()
            self.h.disable_histogram_plot()
            self.h.disable_probability_plot()
            self.h.disable_mean_count_plot()
            self.h.disable_state_probability_plot()
            self.h.disable_all_plots()

        plot_fn()

        # Add data to the archive
        num_histograms = 4
        data = [
            [4, 1, 0],
            [5, 2, 0],
            [6, 3, 0],
            [7, 4, 0],
        ]

        # Store data
        for _ in range(num_histograms):
            with self.h:
                for d in data:
                    self.h.append(d)

        plot_fn()

    def test_kernel_invariants(self):
        # Test kernel invariants
        test.helpers.test_system_kernel_invariants(self, self.s)


class HistogramAnalyzerTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.managers = get_managers()
        self.s = _TestSystem(self.managers)
        self.s.dax_init()
        self.h = self.s.hist_context

    def tearDown(self) -> None:
        # Close managers
        self.managers.close()

    def test_histogram_analyzer_system(self):
        # The histogram analyzer requests an output file which will trigger the creation of an experiment output dir
        # To prevent unnecessary directories after testing, we switch to a temp dir
        with temp_dir():
            HistogramAnalyzer(self.s)

    def test_histogram_analyzer_module(self):
        # The histogram analyzer requests an output file which will trigger the creation of an experiment output dir
        # To prevent unnecessary directories after testing, we switch to a temp dir
        with temp_dir():
            HistogramAnalyzer(self.h)

    def test_counter_to_ndarray(self):
        c = collections.Counter([1, 2, 3, 3, 4, 4, 4, 9])
        a = np.asarray([0, 1, 1, 2, 3, 0, 0, 0, 0, 1])
        n = HistogramAnalyzer.counter_to_ndarray(c)
        self.assertListEqual(list(n), list(a), 'Counter did not convert correctly to ndarray')
        n = HistogramAnalyzer.counter_to_ndarray(c, max_count=max(c))
        self.assertListEqual(list(n), list(a), 'Counter did not convert correctly to ndarray with a given max')

    def test_counter_to_ndarray_max_count(self):
        c = collections.Counter([1, 2, 3, 3, 4, 4, 4, 9])
        a = np.asarray([0, 1, 1, 2, 3, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        for i in range(len(a)):
            n = HistogramAnalyzer.counter_to_ndarray(c, max_count=i)
            self.assertListEqual(list(n), list(a[:i + 1]),
                                 'Counter did not convert correctly to ndarray with a given max count')

    def test_ndarray_to_counter(self):
        c = collections.Counter([1, 2, 3, 3, 4, 4, 4, 9])
        a = np.asarray([0, 1, 1, 2, 3, 0, 0, 0, 0, 1])
        n = HistogramAnalyzer.ndarray_to_counter(a)
        self.assertEqual(n, c, 'ndarray did not convert correctly to Counter')

    def test_raw_to_histograms(self):
        raw = [
            [[1, 2, 0], [5, 5, 5], [6, 7, 0], ],
            [[6, 7, 0], [1, 2, 0], [5, 5, 5], ],
            [[True, True, False], [False, False, False], [True, True, True], ],
        ]
        ref = [
            [collections.Counter([1, 5, 6]), collections.Counter([2, 5, 7]), collections.Counter([0, 5, 0])],
            [collections.Counter([6, 1, 5]), collections.Counter([7, 2, 5]), collections.Counter([0, 0, 5])],
            [collections.Counter({True: 2, False: 1})] * 2 + [collections.Counter({True: 1, False: 2})],
        ]

        for rw, rf in zip(raw, ref):
            result = HistogramAnalyzer.raw_to_histograms(rw)
            self.assertListEqual(result, rf, 'Counters did not matched reference')

    def test_histogram_to_one_count(self):
        data = [
            (collections.Counter([3]), 0),
            (collections.Counter([5]), 1),
            (collections.Counter([1]), 0),
            (collections.Counter([4]), 1),
            (collections.Counter([0, 4]), 1),
            (collections.Counter([False, True]), 1),
            # Mixing bool and int counts is not expected to happen, but it should work fine
            (collections.Counter([False, True, 0, 4]), 2),
        ]
        threshold = 3

        for d, r in data:
            with self.subTest(histogram=d):
                self.assertEqual(HistogramAnalyzer.histogram_to_one_count(d, threshold), r,
                                 'Histogram to one count incorrect')

    def test_histogram_to_one_count_threshold_error(self):
        # No threshold provided, still works with binary data
        data = collections.Counter([False, True])
        ref = 1
        self.assertEqual(HistogramAnalyzer.histogram_to_one_count(data), ref)
        self.assertEqual(HistogramAnalyzer.histogram_to_one_count(data, -1), ref)
        self.assertEqual(HistogramAnalyzer.histogram_to_one_count(data, -100), ref)

        # No threshold provided, fails with integer data
        data_list = [
            (collections.Counter([3]), 0),
            (collections.Counter([5]), 1),
            (collections.Counter([1]), 0),
            (collections.Counter([4]), 1),
            (collections.Counter([0, 4]), 1),
            # Mixing bool and int counts is not expected to happen, but should still fail
            (collections.Counter([False, True, 0, 4]), 2),
        ]
        for data, ref in data_list:
            with self.assertRaises(TypeError, msg='Lack of threshold did not cause exception'):
                self.assertEqual(HistogramAnalyzer.histogram_to_one_count(data), ref)

    def test_histogram_to_probability(self):
        data = [(collections.Counter([3]), 0.0),
                (collections.Counter([5]), 1.0),
                (collections.Counter([1]), 0.0),
                (collections.Counter([4]), 1.0),
                (collections.Counter([0, 4]), 0.5), ]
        threshold = 3

        for d, r in data:
            with self.subTest(histogram=d):
                self.assertEqual(HistogramAnalyzer.histogram_to_probability(d, threshold), r,
                                 'Single histogram did not converted correctly to a probability')

    def test_histograms_to_probabilities(self):
        data = [[collections.Counter([3]), collections.Counter([5]), collections.Counter([1])],
                [collections.Counter([1]), collections.Counter([2]), collections.Counter([4])],
                [collections.Counter([1, 4]), collections.Counter([1, 2]), collections.Counter([5, 8])], ]
        threshold = 3
        reference = [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.5, 0.0, 1.0]]

        result = HistogramAnalyzer.histograms_to_probabilities(data, threshold)

        for ref, res in zip(reference, result):
            self.assertListEqual(ref, list(res), 'Histograms did not converted correctly to probabilities')

    def test_histogram_to_mean_count(self):
        data = [(collections.Counter(d), np.mean(d))
                for d in [[3], [5, 6, 7], [1, 2, 1, 2, 1, 2, 1, 2]]]

        for d, r in data:
            with self.subTest(histogram=d):
                self.assertAlmostEqual(HistogramAnalyzer.histogram_to_mean_count(d), r,
                                       msg='Single histogram did not converted correctly to mean count')

    def test_histogram_to_mean_stdev_count(self):
        data = [(collections.Counter(d), np.mean(d), np.std(d))
                for d in [[3], [5, 6, 7], [1, 2, 1, 2, 1, 2, 1, 2]]]

        for c, m, s in data:
            with self.subTest(histogram=c):
                mean, stdev = HistogramAnalyzer.histogram_to_mean_stdev_count(c)
                self.assertAlmostEqual(mean, m, msg='Single histogram did not converted correctly to mean/stdev count')
                self.assertAlmostEqual(stdev, s, msg='Single histogram did not converted correctly to mean/stdev count')

    def test_histogram_to_stdev_count(self):
        data = [(collections.Counter(d), np.std(d))
                for d in [[3], [5, 6, 7], [1, 2, 1, 2, 1, 2, 1, 2]]]

        for c, s in data:
            with self.subTest(histogram=c):
                self.assertAlmostEqual(HistogramAnalyzer.histogram_to_stdev_count(c), s,
                                       msg='Single histogram did not converted correctly to stdev count')

    def test_histograms_to_mean_counts(self):
        data = [[collections.Counter([3, 4, 5]), collections.Counter([5, 7, 9]), collections.Counter([1, 2, 1, 2])],
                [collections.Counter([1, 1, 2, 2]), collections.Counter([2, 2]), collections.Counter([4])], ]
        reference = [[4.0, 7.0, 1.5], [1.5, 2, 4]]

        result = HistogramAnalyzer.histograms_to_mean_counts(data)
        for ref, res in zip(reference, result):
            self.assertListEqual(ref, list(res), 'Histograms did not converted correctly to mean counts')

    def test_raw_to_states(self):
        num_bits = 3
        num_states = num_bits ** 2
        threshold = 2

        raw = [
            [[1, 2, 0], [5, 5, 5], [6, 7, 0], ],
            [[6, 7, 0], [1, 2, 0], [5, 5, 5], ],
            [[True, True, False], [False, False, False], [True, True, True], ],
        ]
        assert all(all(all(u < num_states for u in t) for t in s) for s in raw)

        # Calculate reference using the string conversion method
        ref = [[int(''.join('1' if p > threshold or p is True else '0' for p in reversed(points)), base=2)
                for points in hist] for hist in raw]

        result = HistogramAnalyzer.raw_to_states(raw, threshold)
        self.assertListEqual(result, ref, 'States did not matched reference')

    def test_raw_to_state_probabilities(self):
        num_bits = 3
        num_states = num_bits ** 2
        threshold = 2

        raw = [
            [[1, 2, 0], [5, 5, 5], [5, 5, 5], [6, 7, 0], ],
            [[6, 7, 0], [1, 2, 0], [1, 2, 0], [5, 5, 5], ],
            [[True, True, False], [False, False, False], [False, False, False], [True, True, True], ],
        ]
        assert all(all(all(u < num_states for u in t) for t in s) for s in raw)
        ref = [
            {0b000: 0.25, 0b111: 0.5, 0b011: 0.25},
            {0b011: 0.25, 0b000: 0.5, 0b111: 0.25},
            {0b011: 0.25, 0b000: 0.5, 0b111: 0.25},
        ]
        assert all(all(v < num_states for v in r.values()) for r in ref)

        result = HistogramAnalyzer.raw_to_state_probabilities(raw, threshold)
        self.assertListEqual(result, ref, 'State probabilities did not matched reference')

    def test_raw_to_flat_state_probabilities(self):
        num_bits = 3
        num_states = 2 ** num_bits
        threshold = 2

        raw = [
            [[1, 2, 0], [5, 5, 5], [5, 5, 5], [6, 7, 0], ],
            [[6, 7, 0], [1, 2, 0], [1, 2, 0], [5, 5, 5], ],
            [[True, True, False], [False, False, False], [False, False, False], [True, True, True], ],
        ]
        assert all(all(all(u < num_states for u in t) for t in s) for s in raw)

        ref_sparse = [
            {0b000: 0.25, 0b111: 0.5, 0b011: 0.25},
            {0b011: 0.25, 0b000: 0.5, 0b111: 0.25},
            {0b011: 0.25, 0b000: 0.5, 0b111: 0.25},
        ]
        assert all(all(v < num_states for v in r.values()) for r in ref_sparse)

        # Flatten sparse reference
        ref = [[r.get(i, 0.0) for i in range(num_states)] for r in ref_sparse]

        result = HistogramAnalyzer.raw_to_flat_state_probabilities(raw, threshold)
        for a, b in zip(result, ref):
            self.assertListEqual(a, b, 'Flat state probabilities did not matched reference')

    def _generate_hdf5_data(self, *, keep_raw=True, add_legacy=False, binary_measurements=False):
        num_histograms = 8
        all_data = {
            'foo': [[1, 9], [2, 9], [2, 9], [3, 9], [3, 8], [3, 8]],  # Specific dataset
            None: [[4, 1, 0], [5, 2, 0], [6, 3, 0], [7, 4, 0], [8, 5, 0]],  # Default dataset
        }

        if binary_measurements:
            all_data = {k: [[c > 2 for c in a] for a in v] for k, v in all_data.items()}

        for key, data in all_data.items():
            # Store data
            self.h.config_dataset(key)
            for i in range(num_histograms):
                with self.h:
                    for d in data:
                        self.h.append(d)

                # Reconstruct dataset key
                dataset_key = self.h.DEFAULT_DATASET_KEY if key is None else key
                dataset_key = self.h.DATASET_KEY_FORMAT.format(dataset_key=dataset_key, index=i)

                if add_legacy:
                    # Add histograms in legacy data format
                    histograms = self.h.get_histograms(key)
                    max_count = max(max(max(h) for h in c) for c in histograms)
                    flat_histograms = [HistogramAnalyzer.counter_to_ndarray(channel[-1], max_count=max_count)
                                       for channel in histograms]
                    self.h.set_dataset(dataset_key.replace('/raw/', '/'), flat_histograms, archive=True)

                if not keep_raw:
                    # Remove raw data from datasets
                    self.h.set_dataset(dataset_key, None, archive=False)

    def _compare_analyzer_to_context(self, a, *, has_raw=True):
        """"Compare histogram analyzer output to current histogram context."""

        self.assertListEqual(a.keys, self.h.get_keys(), 'Keys did not match')
        for key in a.keys:
            for v, w in zip(a.histograms[key], self.h.get_histograms(key)):
                self.assertListEqual(list(v), list(w), 'Histograms did not match')
            for v, w in zip(a.probabilities[key], self.h.get_probabilities(key)):
                self.assertListEqual(list(v), w, 'Probabilities did not match')
            for v, w in zip(a.mean_counts[key], self.h.get_mean_counts(key)):
                self.assertListEqual(list(v), w, 'Mean counts did not match')
            for v, w in zip(a.stdev_counts[key], self.h.get_stdev_counts(key)):
                self.assertListEqual(list(v), w, 'Stdev counts did not match')
            if has_raw:
                for v, w in zip(a.raw[key], self.h.get_raw(key)):
                    self.assertListEqual(v.tolist(), list(w), 'Raw counts did not match')

    def test_hdf5_read(self):
        self._generate_hdf5_data()

        with temp_dir():
            # Write data to HDF5 file
            file_name = 'result.h5'
            with h5py.File(file_name, 'w') as f:
                self.managers.dataset_mgr.write_hdf5(f)

            # Read file with HistogramAnalyzer
            a = HistogramAnalyzer(file_name, self.s.detection.get_state_detection_threshold())

            # Compare analyzer to histogram context
            self._compare_analyzer_to_context(a)

            # Compare to analyzer from object source
            b = HistogramAnalyzer(self.s, self.s.detection.get_state_detection_threshold())
            self.assertListEqual(a.keys, b.keys, 'Keys did not match')
            for key in a.keys:
                for v, w in zip(a.histograms[key], b.histograms[key]):
                    self.assertListEqual(list(v), list(w), 'Histograms did not match')
                for v, w in zip(a.probabilities[key], b.probabilities[key]):
                    self.assertListEqual(list(v), list(w), 'Probabilities did not match')
                for v, w in zip(a.mean_counts[key], b.mean_counts[key]):
                    self.assertListEqual(list(v), list(w), 'Mean counts did not match')
                for v, w in zip(a.stdev_counts[key], b.stdev_counts[key]):
                    self.assertListEqual(list(v), list(w), 'Stdev counts did not match')
                for v, w in zip(a.raw[key], b.raw[key]):
                    self.assertListEqual(v.tolist(), w.tolist(), 'Raw counts did not match')

    def test_hdf5_read_no_raw(self):
        self._generate_hdf5_data(keep_raw=False, add_legacy=True)

        with temp_dir():
            # Write data to HDF5 file
            file_name = 'result.h5'
            with h5py.File(file_name, 'w') as f:
                self.managers.dataset_mgr.write_hdf5(f)

            # Read file with HistogramAnalyzer
            a = HistogramAnalyzer(file_name, self.s.detection.get_state_detection_threshold())

            # Verify raw attribute is not available
            self.assertFalse(hasattr(a, 'raw'), 'Expected no attribute `raw`')

            # Compare analyzer to histogram context
            self._compare_analyzer_to_context(a, has_raw=False)

    def test_hdf5_read_raw_and_legacy(self):
        self._generate_hdf5_data(keep_raw=True, add_legacy=True)

        with temp_dir():
            # Write data to HDF5 file
            file_name = 'result.h5'
            with h5py.File(file_name, 'w') as f:
                self.managers.dataset_mgr.write_hdf5(f)

            # Read file with HistogramAnalyzer
            a = HistogramAnalyzer(file_name, self.s.detection.get_state_detection_threshold())

            # Verify raw attribute is available
            self.assertTrue(hasattr(a, 'raw'), 'Expected attribute `raw`')

            # Compare analyzer to histogram context
            self._compare_analyzer_to_context(a)

    def test_hdf5_read_no_histogram_data(self):
        self._generate_hdf5_data(keep_raw=False, add_legacy=False)

        with temp_dir():
            # Write data to HDF5 file
            file_name = 'result.h5'
            with h5py.File(file_name, 'w') as f:
                self.managers.dataset_mgr.write_hdf5(f)

            with self.assertRaises(KeyError, msg='Absence of histogram data did not raise'):
                HistogramAnalyzer(file_name)

    def test_hdf5_read_no_probabilities(self, binary_measurements=False):
        self._generate_hdf5_data(binary_measurements=binary_measurements)

        with temp_dir():
            # Write data to HDF5 file
            file_name = 'result.h5'
            with h5py.File(file_name, 'w') as f:
                self.managers.dataset_mgr.write_hdf5(f)

            # Read file with HistogramAnalyzer
            a = HistogramAnalyzer(file_name)

            # Verify probabilities attribute is not available
            self.assertEqual(binary_measurements, hasattr(a, 'probabilities'))

    def test_hdf5_read_binary_probabilities(self):
        self.test_hdf5_read_no_probabilities(binary_measurements=True)

    @unittest.skipUnless(CI_ENABLED, 'Not in a CI environment, skipping slow plotting test')
    def test_plot(self):
        # Add data to the archive
        num_histograms = 4
        data = [
            [4, 1, 0],
            [5, 2, 0],
            [6, 3, 0],
            [7, 4, 0],
        ]

        # Store data
        for _ in range(num_histograms):
            with self.h:
                for d in data:
                    self.h.append(d)

        with temp_dir():
            # Make analyzer object
            a = HistogramAnalyzer(self.s, self.s.detection.get_state_detection_threshold())
            # Call plot functions to see if no exceptions occur
            a.plot_all_histograms()
            a.plot_all_probabilities()
            a.plot_all_mean_counts()
            a.plot_all_state_probabilities()

    @unittest.skipUnless(CI_ENABLED, 'Not in a CI environment, skipping slow plotting test')
    def test_plot_hdf5(self):
        # Add data to the archive
        num_histograms = 4
        data = [
            [4, 1, 0],
            [5, 2, 0],
            [6, 3, 0],
            [7, 4, 0],
        ]

        # Store data
        for _ in range(num_histograms):
            with self.h:
                for d in data:
                    self.h.append(d)

        with temp_dir():
            # Write data to HDF5 file
            file_name = 'result.h5'
            with h5py.File(file_name, 'w') as f:
                self.managers.dataset_mgr.write_hdf5(f)

            # Make analyzer object
            a = HistogramAnalyzer(file_name, self.s.detection.get_state_detection_threshold())
            # Call plot functions to see if no exceptions occur
            a.plot_all_histograms()
            a.plot_all_probabilities()
            a.plot_all_mean_counts()
            a.plot_all_state_probabilities()
