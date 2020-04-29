import unittest

from dax.test.helpers.mypy import type_check


class DaxModulesCpldInitStaticTyping(unittest.TestCase):

    def test_static_typing(self):
        # Type checking on DAX module CPLD init
        import dax.modules.cpld_init as module
        type_check(self, module)


if __name__ == '__main__':
    unittest.main()
