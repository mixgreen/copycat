import unittest
import types
import os.path

import mypy.api


def type_check(test_case: unittest.TestCase, module):
    """Perform MyPy static type checking on the provided module."""

    assert isinstance(test_case, unittest.TestCase), 'The provided test case must be of type unittest.TestCase'
    assert isinstance(module, types.ModuleType), 'The provided module must be a types.ModuleType'

    # Get path to the module
    path = os.path.abspath(module.__file__)

    # Run MyPy static typing
    report, err_report, exit_status = mypy.api.run([path])

    # Assert
    err_report = '\nError report:\n{:s}\n'.format(err_report) if err_report else err_report
    msg = '\n\nType checking report:\n{:s}{:s}'.format(report, err_report)
    test_case.assertEqual(exit_status, 0, msg)
