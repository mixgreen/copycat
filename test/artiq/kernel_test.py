import unittest
import numpy as np

from artiq.experiment import *

import test.hw_testbench


class _ContextExperiment(HasEnvironment):
    class _Context:
        def __init__(self, name, call_list):
            self._name = name
            self._call_list = call_list

        @kernel
        def __enter__(self):
            self._append('enter')

        @kernel
        def __exit__(self, exc_type, exc_val, exc_tb):
            self._append('exit')

        def _append(self, msg):
            self._call_list.append(f'{self._name} {msg}')

    def build(self):
        self.setattr_device('core')
        self.call_list = []
        self.context_a = self._Context('a', self.call_list)
        self.context_b = self._Context('b', self.call_list)

    @kernel
    def multiple_item_context_test(self):
        with self.context_a, self.context_b:
            pass

    @kernel
    def nested_context_test(self):
        with self.context_a:
            with self.context_b:
                pass


class _BuiltinsExperiment(HasEnvironment):
    def build(self):
        self.setattr_device('core')

    @kernel
    def range_test(self, n):
        acc = 0
        for i in range(n):
            acc += i
        return acc

    @kernel
    def len_test(self, list_):
        return len(list_)

    @kernel
    def min_max_test(self, a, b):
        return min(a, b), max(a, b)

    @kernel
    def abs_test(self, a):
        return abs(a)

    @kernel
    def assert_test(self, a, b):
        assert a == b, 'Assert message'

    @kernel
    def sin_test(self, f):
        return np.sin(f)

    @kernel
    def rpc_args_kwargs_test(self):
        self._args_kwargs(1, 2, a=3, b=4)

    @rpc
    def _args_kwargs(self, *args, **kwargs):
        pass

    @kernel
    def array_test(self, a):
        a += 1
        acc = 0
        for e in a:
            acc += e
        return acc


class ArtiqKernelTestCase(test.hw_testbench.TestBenchCase):

    def test_nested_context(self):
        env = self.construct_env(_ContextExperiment)
        env.nested_context_test()
        self.assertListEqual(env.call_list, ['a enter', 'b enter', 'b exit', 'a exit'],
                             'nested context (two `with` statements) has incorrect behavior')

    @unittest.expectedFailure  # https://github.com/m-labs/artiq/issues/1478
    def test_multiple_item_context(self):
        env = self.construct_env(_ContextExperiment)
        env.multiple_item_context_test()
        self.assertListEqual(env.call_list, ['a enter', 'b enter', 'b exit', 'a exit'],
                             'multiple item context (single `with` statement) has incorrect behavior')

    def test_range(self):
        env = self.construct_env(_BuiltinsExperiment)
        acc = env.range_test(5)
        self.asserEqual(acc, sum(range(5)))

    def test_len(self):
        env = self.construct_env(_BuiltinsExperiment)
        list_ = [1, 2, 3, 4]
        length = env.len_test(list_)
        self.asserEqual(length, len(list_))

    def test_min_max(self):
        env = self.construct_env(_BuiltinsExperiment)
        a = 4
        b = 5
        min_, max_ = env.min_max_test(a, b)
        self.asserEqual(min_, min(a, b))
        self.asserEqual(max_, max(a, b))

    def test_abs(self):
        env = self.construct_env(_BuiltinsExperiment)
        for e in [-2, 4]:
            r = env.abs_test(e)
            self.asserEqual(r, abs(e))

    def test_assert(self):
        env = self.construct_env(_BuiltinsExperiment)
        with self.assertRaises(AssertionError):
            env.assert_test(3, 4)
        self.assertIsNone(env.assert_test(3, 3))

    def test_sin(self):
        env = self.construct_env(_BuiltinsExperiment)
        for e in [np.pi, np.pi * 1.5]:
            r = env.sin_test(e)
            self.assertAlmostEqual(r, np.sin(e))

    def test_rpc_args_kwargs(self):
        env = self.construct_env(_BuiltinsExperiment)
        self.assertIsNone(env.rpc_args_kwargs_test())

    def test_array(self):
        env = self.construct_env(_BuiltinsExperiment)
        arr = np.arange(5, dtype=np.int32)
        acc = env.array_test(arr)
        self.assertEqual(acc, sum(arr))
