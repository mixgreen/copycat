import os
import typing
import pygit2

__all__ = ['RepositoryInfo', 'in_repository', 'NotInRepositoryError', 'get_repository_info']


class RepositoryInfo(typing.NamedTuple):
    """A named tuple with repository information."""

    path: str
    """The root path of the repository."""
    commit: str
    """The commit hash."""
    dirty: bool
    """:const:`True` if the repository is dirty."""

    def as_dict(self) -> typing.Dict[str, typing.Union[str, bool]]:
        """Return repository information as a dictionary with field names and values."""
        return self._asdict()


def _load() -> typing.Optional[RepositoryInfo]:
    # Discover repository path of current working directory, also looks in parent directories
    try:
        # noinspection PyCallingNonCallable
        path = pygit2.discover_repository(os.getcwd())
        if path is not None:
            # Obtain the repository object
            repo = pygit2.Repository(path)
            commit = '' if repo.is_empty else str(repo.head.target.hex)
            # Check if the repository is dirty
            dirty = any(s not in {pygit2.GIT_STATUS_CURRENT, pygit2.GIT_STATUS_IGNORED} for s in repo.status().values())
            # Return results
            parent_path = str(path).rsplit('.git')[0]  # Strip off ".git"
            return RepositoryInfo(parent_path, commit, dirty)
    except pygit2.GitError:
        pass

    # Not in a Git repository
    return None


# Load repository information
_REPO_INFO: typing.Optional[RepositoryInfo] = _load()
del _load  # Remove one-time function


def in_repository() -> bool:
    """Check if the current working directory is inside a Git repository.

    :return: :const:`True` if the current working directory is inside a Git repository
    """
    return _REPO_INFO is not None


class NotInRepositoryError(Exception):
    """An exception that is raised when repository interaction is requested while not inside a Git repository."""

    def __init__(self) -> None:
        super(NotInRepositoryError, self).__init__('Current working directory is not inside a Git repository')


def get_repository_info() -> RepositoryInfo:
    """Get repository information as a named tuple.

    :return: Repository information as a named tuple (see :class:`RepositoryInfo`)
    :raises NotInRepositoryError: Raised when the current working directory is not inside a Git repository
    """
    if _REPO_INFO is not None:
        return _REPO_INFO
    else:
        raise NotInRepositoryError
