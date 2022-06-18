import unittest
import numpy as np

from artiq.experiment import *
from artiq.coredevice.core import CompileError

from dax.base.system import DaxSystem

import test.hw_test


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


class _CompilerSupportExperiment(HasEnvironment):
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
    def array_test(self, a, adder):
        a += adder
        acc = 0
        for e in a:
            acc += e
        return acc

    @kernel
    def not_implemented_error_test(self):
        raise NotImplementedError

    @kernel
    def negative_delay_parallel_test(self) -> TInt64:
        t = now_mu()
        with parallel:
            delay_mu(-100)
        return now_mu() - t  # Should be 0


class _DaxSystemExperiment(DaxSystem):
    SYS_ID = 'unittest_system'
    SYS_VER = 0
    CORE_LOG_KEY = None
    DAX_INFLUX_DB_KEY = None

    @kernel
    def get_system_key_kernel(self, bar: TStr = 'bar') -> TStr:
        return self.get_system_key('foo', bar)

    @kernel
    def set_dataset_sys_kernel(self, key: TStr, value: TInt32) -> TNone:
        self.set_dataset_sys(key, value)
        return  # Added to make sure the IDE does not get confused by return type TNone

    @kernel
    def mutate_dataset_sys_kernel(self, key: TStr, index: TInt32, value: TInt32):
        self.mutate_dataset_sys(key, index, value)

    @kernel
    def append_to_dataset_sys_kernel(self, key: TStr, value: TInt32):
        self.append_to_dataset_sys(key, value)


class ArtiqKernelTestCase(test.hw_test.HardwareTestCase):

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
        env = self.construct_env(_CompilerSupportExperiment)
        acc = env.range_test(5)
        self.assertEqual(acc, sum(range(5)))

    def test_len(self):
        env = self.construct_env(_CompilerSupportExperiment)
        list_ = [1, 2, 3, 4]
        length = env.len_test(list_)
        self.assertEqual(length, len(list_))

    def test_min_max(self):
        env = self.construct_env(_CompilerSupportExperiment)
        a = 4
        b = 5
        min_, max_ = env.min_max_test(a, b)
        self.assertEqual(min_, min(a, b))
        self.assertEqual(max_, max(a, b))

    def test_abs(self):
        env = self.construct_env(_CompilerSupportExperiment)
        for e in [-2, 4]:
            r = env.abs_test(e)
            self.assertEqual(r, abs(e))

    def test_assert(self):
        env = self.construct_env(_CompilerSupportExperiment)
        with self.assertRaises(AssertionError):
            env.assert_test(3, 4)
        self.assertIsNone(env.assert_test(3, 3))

    def test_sin(self):
        env = self.construct_env(_CompilerSupportExperiment)
        for e in [np.pi, np.pi * 1.5]:
            r = env.sin_test(e)
            self.assertAlmostEqual(r, np.sin(e))

    def test_rpc_args_kwargs(self):
        env = self.construct_env(_CompilerSupportExperiment)
        self.assertIsNone(env.rpc_args_kwargs_test())

    def test_array(self):
        env = self.construct_env(_CompilerSupportExperiment)
        arr = np.arange(5, dtype=np.int32)
        adder = 1
        acc = env.array_test(arr, adder)
        self.assertEqual(acc, sum(arr) + len(arr) * adder)

    def test_not_implemented_error(self):
        env = self.construct_env(_CompilerSupportExperiment)
        with self.assertRaises(CompileError, msg='NotImplementedError does not result in a compile error'):
            env.not_implemented_error_test()

    def test_negative_delay_parallel(self):
        env = self.construct_env(_CompilerSupportExperiment)
        t = env.negative_delay_parallel_test()
        self.assertEqual(t, 0)

    def test_get_system_key_kernel(self):
        env = self.construct_env(_DaxSystemExperiment)
        r = env.get_system_key_kernel()
        self.assertEqual(r, env.get_system_key('foo', 'bar'))

    def test_set_dataset_sys_kernel(self):
        env = self.construct_env(_DaxSystemExperiment)
        key = 'foo'
        value = 3
        env.set_dataset_sys_kernel(key, value)
        self.assertEqual(value, env.get_dataset_sys(key))

    def test_mutate_dataset_sys_kernel(self):
        env = self.construct_env(_DaxSystemExperiment)
        key = 'foo'
        index = 1
        value = 3
        env.set_dataset_sys(key, [0, 0])
        env.mutate_dataset_sys_kernel(key, index, value)
        self.assertListEqual([0, 3], env.get_dataset_sys(key))

    def test_append_to_dataset_sys_kernel(self):
        env = self.construct_env(_DaxSystemExperiment)
        key = 'foo'
        value = 3
        env.set_dataset_sys(key, [])
        env.append_to_dataset_sys_kernel(key, value)
        self.assertListEqual([3], env.get_dataset_sys(key))
