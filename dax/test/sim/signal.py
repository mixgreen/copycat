import unittest

from dax.test.helpers.mypy import type_check


class DaxSimSignalStaticTyping(unittest.TestCase):

    def test_signal_static_typing(self):
        # Type checking on DAX sim signal
        import dax.sim.signal
        type_check(self, dax.sim.signal, '--strict')


if __name__ == '__main__':
    unittest.main()
