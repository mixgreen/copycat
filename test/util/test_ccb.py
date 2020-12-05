import unittest

from dax.experiment import DaxSystem
from dax.util.artiq import get_managers


class _TestSystem(DaxSystem):
    SYS_ID = 'unittest_system'
    SYS_VER = 0


class CcbTestCase(unittest.TestCase):

    def test_convert_group(self):
        # noinspection PyProtectedMember
        from dax.util.ccb import _convert_group
        sep = '.'
        g = 'this.is.a.group'
        self.assertEqual(_convert_group(g), g.split(sep), 'Convert group did not give expected result')
        g = g.split(sep)
        self.assertEqual(_convert_group(g), g, 'Convert group unintentionally altered list notation')

    def test_generate_command(self):
        # noinspection PyProtectedMember
        from dax.util.ccb import _generate_command

        data = {
            'base1': {'foo_bar': 'bar', 'baz': None},
            'base2': {'foo_bar': 1.5, 'baz': 'bar'},
            'base3': {'foo_bar': None, 'bar': None, 'baz': 1},
            'base4': {'foo_bar': True, 'baz': False},
        }

        for base, kwargs in data.items():
            with self.subTest(base_cmd=base, kwargs=kwargs):
                cmd = _generate_command(base, **kwargs)
                self.assertTrue(cmd.startswith(base), 'Command does not start with base command')
                self.assertNotIn('_', cmd, 'Underscores not filtered out of command')
                for k, v in kwargs.items():
                    if v in {None, False}:
                        self.assertNotIn(k, cmd, 'None or False valued argument found in command')
                    elif v is True:
                        self.assertIn(f'--{k.replace("_", "-")}', cmd, 'store_true argument not found in command')
                        self.assertNotIn('True', cmd, 'True string found in command')
                    else:
                        self.assertIn(f'--{k.replace("_", "-")}', cmd, 'Argument not found in command')
                        self.assertIn(f'"{v}"', cmd, 'Argument value not found in command')

    def test_ccb_tool(self):
        with get_managers() as managers:
            from dax.util.ccb import get_ccb_tool
            ccb = get_ccb_tool(_TestSystem(managers))

            # Just call methods to see if no errors occur
            self.assertIsNone(ccb.big_number('name', 'key'))
            self.assertIsNone(ccb.image('name', 'key'))
            self.assertIsNone(ccb.plot_xy('name', 'key'))
            self.assertIsNone(ccb.plot_xy_multi('name', 'key'))
            self.assertIsNone(ccb.plot_hist('name', 'key'))
            self.assertIsNone(ccb.plot_hist_artiq('name', 'key'))
            self.assertIsNone(ccb.plot_xy_hist('name', 'key', 'key', 'key'))
            self.assertIsNone(ccb.disable_applet('name'))
            self.assertIsNone(ccb.disable_applet_group('group'))


if __name__ == '__main__':
    unittest.main()
