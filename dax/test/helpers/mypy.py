import unittest
import types
import os

import mypy.api


def type_check(test_case: unittest.TestCase, module, *cmd_args: str) -> None:
    """Perform MyPy static type checking on the provided module."""

    assert isinstance(test_case, unittest.TestCase), 'The provided test case must be of type unittest.TestCase'
    assert isinstance(module, types.ModuleType), 'The provided module must be a types.ModuleType'
    assert all(isinstance(a, str) for a in cmd_args), 'Command line arguments must be strings'

    # Get path to the module
    module_path: str = os.path.realpath(module.__file__)

    # Run MyPy static typing
    report, err_report, exit_status = mypy.api.run([*cmd_args, module_path])

    # Format message and assert
    err_report: str = '\nError report:\n{:s}'.format(err_report) if err_report else err_report
    msg: str = '\n\nType checking report:\n{:s}{:s}'.format(report, err_report)
    test_case.assertEqual(exit_status, 0, msg)


def _set_env() -> None:
    """Set the MYPYPATH environmental variable to include stubs."""
    env_var: str = 'MYPYPATH'
    stubs_path: str = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'stubs')
    os.environ[env_var] = ':'.join(p for p in (stubs_path, os.environ.get(env_var)) if p)


# Set environment once
_set_env()
