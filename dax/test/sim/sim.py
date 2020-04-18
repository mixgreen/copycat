import unittest

from dax.test.helpers.mypy import type_check


class DaxSimStaticTyping(unittest.TestCase):

    def test_sim_static_typing(self):
        # Type checking on DAX sim
        import dax.sim.sim
        type_check(self, dax.sim.sim, '--strict')


if __name__ == '__main__':
    unittest.main()
