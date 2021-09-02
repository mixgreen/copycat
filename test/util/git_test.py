import unittest
import importlib
import typing
import os

import pygit2

import dax.util.git
import dax.util.output


class GitTestCase(unittest.TestCase):

    def _test_not_in_repo(self, module: typing.Any = dax.util.git):
        # Do basic tests for not being in a Git repo
        self.assertIsNone(module._REPO_INFO)
        with self.assertRaises(module.NotInRepositoryError):
            module.get_repository_info()

    def test_repo_info(self):
        self.assertIsInstance(dax.util.git._REPO_INFO, (tuple, type(None)), 'Unexpected type for repo info')

        # Discover repo path
        # noinspection PyCallingNonCallable
        path = pygit2.discover_repository(os.getcwd())

        if path is None:
            # Skip remaining tests
            self._test_not_in_repo()
            self.skipTest('CWD currently not in a Git repo')

        # Reference repo
        repo = pygit2.Repository(path)
        # Get repo info, should not raise an exception
        repo_info = dax.util.git.get_repository_info()

        # Check types
        self.assertIsInstance(repo_info.path, str)
        self.assertIsInstance(repo_info.commit, str)
        self.assertIsInstance(repo_info.dirty, bool)

        # Check values
        self.assertTrue(path.endswith('.git/'))
        self.assertEqual(repo_info.path, str(path)[:-5], 'Git path did not match reference')
        self.assertEqual(repo_info.commit, '' if repo.is_empty else str(pygit2.Repository(path).head.target.hex),
                         'Git commit hash did not match reference')

    def test_not_in_repo(self):
        with dax.util.output.temp_dir():
            git = importlib.reload(dax.util.git)
            self._test_not_in_repo(git)

        # Reload module again to undo changes
        importlib.reload(dax.util.git)
