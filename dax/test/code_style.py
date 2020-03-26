import unittest
import pycodestyle
import os
import io
import contextlib

import dax


class TestCodeStyle(unittest.TestCase):

    def test_code_style(self):
        """Test that the code in the repository conforms to PEP-8."""

        # Get path to the root of DAX
        dax_path = os.path.dirname(os.path.realpath(dax.__file__))

        # Create a style object
        style = pycodestyle.StyleGuide(ignore=['E501'])  # Ignore line too long

        # Buffer to store stdout output
        buf = io.StringIO()

        with contextlib.redirect_stdout(buf):
            # Check all files
            result = style.check_files([dax_path])

        # Format message and assert
        msg = '\n\nCode style report:\n{:s}'.format(buf.getvalue())
        self.assertEqual(result.total_errors, 0, msg)


if __name__ == '__main__':
    unittest.main()
