import unittest

from dax.test.helpers.mypy import type_check


class DaxSimTimeStaticTyping(unittest.TestCase):

    def test_time_static_typing(self):
        # Type checking on DAX sim time
        import dax.sim.time
        type_check(self, dax.sim.time, '--strict')


if __name__ == '__main__':
    unittest.main()
