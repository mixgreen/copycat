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
            'base1': (['y'], {'foo_bar': 'bar_', 'baz': None}, "'y' --foo-bar 'bar_'"),
            'base2': (['y'], {'foo_bar': 1.5, 'baz': '"bar"'}, "'y' --foo-bar 1.5 --baz '\"bar\"'"),
            'base3': (['y', 'fit'], {'foo_bar': None, 'bar': None, 'baz': 1}, "'y' 'fit' --baz 1"),
            'base4': (['y', 'fit'], {'foo_bar': True, 'baz': False}, "'y' 'fit' --foo-bar"),
            'base5': (['y', "'fit'"], {'foo_bar': 'bar', 'baz': "'baz'"}, "'y' 'fit' --foo-bar 'bar' --baz 'baz'"),
            'base6': (['y', 'fit'], {'foo_bar': 0, 'baz': 0.0}, "'y' 'fit' --foo-bar 0 --baz 0.0"),
            'base7': (['y'], {'foo_bar': ['a', 'b'], 'baz': [0, 1, 2, 3]}, "'y' --foo-bar 'a' 'b' --baz 0 1 2 3"),
        }

        for base, (args, kwargs, ref) in data.items():
            with self.subTest(base_cmd=base, args=args, kwargs=kwargs, ref=ref):
                cmd = _generate_command(base, *args, **kwargs)
                self.assertEqual(cmd, f"{base} {ref}")

    def test_generate_command_fail(self):
        # noinspection PyProtectedMember
        from dax.util.ccb import _generate_command

        data = {
            'base1': (['y'], {"foo_'bar": 'bar_', 'baz': None}),  # Single quote in optional argument name
            'base7': (['y'], {'baz': [[0, 1, 2, 3]]}),  # Nested list
        }

        for base, (args, kwargs) in data.items():
            with self.subTest(base_cmd=base, args=args, kwargs=kwargs):
                with self.assertRaises(ValueError):
                    _generate_command(base, *args, **kwargs)

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
