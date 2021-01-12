import unittest

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
    def multiple_item_context(self):
        with self.context_a, self.context_b:  # noqa: ATQ901
            pass

    @kernel
    def nested_context(self):
        with self.context_a:
            with self.context_b:
                pass


class ArtiqKernelTestCase(test.hw_testbench.TestBenchCase):

    def test_nested_context(self):
        env = self.construct_env(_ContextExperiment)
        env.nested_context()
        self.assertListEqual(env.call_list, ['a enter', 'b enter', 'b exit', 'a exit'],
                             'nested context (two `with` statements) has incorrect behavior')

    @unittest.expectedFailure  # https://github.com/m-labs/artiq/issues/1478
    def test_multiple_item_context(self):
        env = self.construct_env(_ContextExperiment)
        env.multiple_item_context()
        self.assertListEqual(env.call_list, ['a enter', 'b enter', 'b exit', 'a exit'],
                             'multiple item context (single `with` statement) has incorrect behavior')
