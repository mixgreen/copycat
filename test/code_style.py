import unittest
import io
import contextlib


class TestCodeStyle(unittest.TestCase):

    def test_code_style(self):
        """Test that the code in the repository conforms to PEP-8."""

        try:
            import pycodestyle  # type: ignore
        except ImportError:
            self.skipTest('pycodestyle library not available')
        else:
            # Get DAX directory
            from dax import __dax_dir__ as dax_dir

            # Create a style object
            style = pycodestyle.StyleGuide(max_line_length=120)  # Increase line length

            # Buffer to store stdout output
            buf = io.StringIO()

            with contextlib.redirect_stdout(buf):
                # Check all files
                result = style.check_files([dax_dir])

            # Format message and assert
            msg = '\n\nCode style report:\n{:s}'.format(buf.getvalue())
            self.assertEqual(result.total_errors, 0, msg)


if __name__ == '__main__':
    unittest.main()
