import unittest
import dax.util.configparser


class ConfigParserTestCase(unittest.TestCase):

    def setUp(self) -> None:
        # Clear cache
        dax.util.configparser._dax_config = None

    def test_cache(self):
        # By default, use cache if available
        a = dax.util.configparser.get_dax_config()
        b = dax.util.configparser.get_dax_config()
        self.assertIs(a, b)

        # Use cache (force clearing and caching)
        a = dax.util.configparser.get_dax_config(clear_cache=True)
        b = dax.util.configparser.get_dax_config(clear_cache=False)
        self.assertIs(a, b)

        # Force clearing cache
        a = dax.util.configparser.get_dax_config(clear_cache=True)
        b = dax.util.configparser.get_dax_config(clear_cache=True)
        self.assertIsNot(a, b)
