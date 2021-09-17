import unittest

from dax.experiment import DaxSystem
from dax.util.artiq import get_managers
from dax.util.output import temp_dir
from dax.util.ccb import CcbWrapper, CcbToolBase, CcbTool, get_ccb_tool
import dax.util.ccb
import dax.util.configparser


class _TestSystem(DaxSystem):
    SYS_ID = 'unittest_system'
    SYS_VER = 0
    CORE_LOG_KEY = None
    DAX_INFLUX_DB_KEY = None


def _clear_cache():
    dax.util.configparser._dax_config = None
    dax.util.ccb._ccb_tool = None


def _write_config(class_, *, no_module=False, no_class=False):
    # Write a config file
    lines = ['[dax.util.ccb]\n']
    if not no_module:
        lines.append(f'ccb_module = {class_.__module__}\n')
    if not no_class:
        lines.append(f'ccb_class = {class_.__name__}\n')
    with open('.dax', mode='w') as file:
        file.writelines(lines)
    # Clear cache
    _clear_cache()


class CcbTestCase(unittest.TestCase):

    def setUp(self) -> None:
        _clear_cache()

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
        from dax.util.ccb import generate_command

        data = {
            'base1': (['y'], {'foo_bar': 'bar_', 'baz': None}, "y --foo-bar bar_"),
            'base2': (['y'], {'foo_bar': 1.5, 'baz': '"bar"'}, "y --foo-bar 1.5 --baz '\"bar\"'"),
            'base3': (['y', 'fit'], {'foo_bar': None, 'bar': None, 'baz': 1}, "y fit --baz 1"),
            'base4': (['y', 'fit'], {'foo_bar': True, 'baz': False}, "y fit --foo-bar"),
            'base5': (['y', "'fit'"], {'foo_bar': 'bar'}, "y ''\"'\"\'fit'\"'\"'' --foo-bar bar"),
            'base6': (['y'], {'foo_bar': 'bar', 'baz': "'baz'"}, "y --foo-bar bar --baz ''\"'\"\'baz'\"'\"''"),
            'base7': (['y', 'fit'], {'foo_bar': 0, 'baz': 0.0}, "y fit --foo-bar 0 --baz 0.0"),
            'base8': (['y'], {'foo_bar': ['a', 'b'], 'baz': [0, 1, 2, 3]}, "y --foo-bar a b --baz 0 1 2 3"),
        }

        for base, (args, kwargs, ref) in data.items():
            with self.subTest(base_cmd=base, args=args, kwargs=kwargs, ref=ref):
                cmd = generate_command(base, *args, **kwargs)
                self.assertEqual(cmd, f"{base} {ref}")

    def test_generate_command_fail(self):
        # noinspection PyProtectedMember
        from dax.util.ccb import generate_command

        data = {
            'base1': (['y'], {"foo_'bar": 'bar_', 'baz': None}),  # Single quote in optional argument name
            'base2': (['y'], {"foo_\"bar": 'bar_', 'baz': None}),  # Double quote in optional argument name
            'base4': (['y'], {"0foo": 'bar_'}),  # Argument name is not an identifier
            'base7': (['y'], {'baz': [[0, 1, 2, 3]]}),  # Nested list
        }

        for base, (args, kwargs) in data.items():
            with self.subTest(base_cmd=base, args=args, kwargs=kwargs):
                with self.assertRaises(ValueError):
                    generate_command(base, *args, **kwargs)

    def _test_ccb_tool(self, ccb):
        # Check type
        self.assertIsInstance(ccb, CcbWrapper)
        self.assertIsInstance(ccb, CcbToolBase)
        # Just call methods to see if no errors occur
        self.assertIsNone(ccb.big_number('name', 'key'))
        self.assertIsNone(ccb.image('name', 'key'))
        self.assertIsNone(ccb.plot_xy('name', 'key'))
        self.assertIsNone(ccb.plot_xy_multi('name', 'key'))
        self.assertIsNone(ccb.plot_hist('name', 'key'))
        self.assertIsNone(ccb.plot_hist_multi('name', 'key'))
        self.assertIsNone(ccb.plot_xy_hist('name', 'key', 'key', 'key'))
        self.assertIsNone(ccb.disable_applet('name'))
        self.assertIsNone(ccb.disable_applet_group('group'))

    def test_ccb_tool(self):
        with get_managers() as managers:
            self._test_ccb_tool(CcbTool(_TestSystem(managers)))

    def test_get_ccb_tool(self):
        with get_managers() as managers:
            self._test_ccb_tool(get_ccb_tool(_TestSystem(managers)))

    def test_get_ccb_tool_config(self):
        with get_managers() as managers, temp_dir():
            for class_ in [_CcbTool, CcbTool]:
                # Write configuration
                _write_config(class_)
                # Get the CCB tool
                ccb = get_ccb_tool(_TestSystem(managers))
                self.assertIsInstance(ccb, class_)

    def test_get_ccb_tool_bad_config(self):
        with get_managers() as managers, temp_dir():
            for no_module, no_class in [(False, True), (True, False)]:
                # Write configuration
                _write_config(CcbTool, no_module=no_module, no_class=no_class)
                with self.assertRaises(LookupError):
                    # Get the CCB tool
                    get_ccb_tool(_TestSystem(managers))

    def test_get_ccb_tool_config_bad_type(self):
        with get_managers() as managers, temp_dir():
            for class_ in [dax.util.ccb.CcbWrapper, _TestSystem]:
                # Write configuration
                _write_config(class_)
                with self.assertRaises(TypeError):
                    # Get the CCB tool
                    get_ccb_tool(_TestSystem(managers))


class _CcbTool(CcbTool):
    pass
