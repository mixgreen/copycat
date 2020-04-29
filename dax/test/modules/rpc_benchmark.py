import unittest

from dax.test.helpers.mypy import type_check


class DaxModulesRpcBenchmarkStaticTyping(unittest.TestCase):

    def test_static_typing(self):
        # Type checking on DAX module RPC benchmark
        import dax.modules.rpc_benchmark as module
        type_check(self, module)


if __name__ == '__main__':
    unittest.main()
