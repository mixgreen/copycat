import unittest

from dax.test.helpers.mypy import type_check


class DaxModulesRtioBenchmarkStaticTyping(unittest.TestCase):

    def test_static_typing(self):
        # Type checking on DAX module RTIO benchmark
        import dax.modules.rtio_benchmark as module
        type_check(self, module)


if __name__ == '__main__':
    unittest.main()
