import unittest


class OutputTestCase(unittest.TestCase):

    def test_base_name(self):
        from dax.util.output import get_base_name
        self.assertIsInstance(get_base_name(None), str)

    def test_file_name_generator(self):
        from dax.util.output import get_file_name_generator
        gen = get_file_name_generator(None)
        self.assertIn('pdf', gen('pdf'))
        self.assertIn('bar', gen('pdf', 'bar'))

    def test_file_name(self):
        from dax.util.output import get_file_name
        self.assertIn('pdf', get_file_name(None, 'pdf'))
        self.assertIn('bar', get_file_name(None, 'pdf', 'bar'))


if __name__ == '__main__':
    unittest.main()
