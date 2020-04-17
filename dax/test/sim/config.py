import unittest

from dax.test.helpers.mypy import type_check


class DaxSimConfigStaticTyping(unittest.TestCase):

    def test_config_static_typing(self):
        # Type checking on DAX sim config
        import dax.sim.config
        type_check(self, dax.sim.config, '--strict')
