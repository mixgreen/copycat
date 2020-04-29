import unittest

from dax.test.helpers.mypy import type_check


class DaxModulesLedStaticTyping(unittest.TestCase):

    def test_static_typing(self):
        # Type checking on DAX module LED
        import dax.modules.led as module
        type_check(self, module)


if __name__ == '__main__':
    unittest.main()
