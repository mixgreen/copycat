import typing
import os
import os.path
import pathlib
import contextlib
import tempfile

__all__ = ['temp_dir', 'get_base_path', 'FileNameGenerator', 'BaseFileNameGenerator', 'get_file_name']


@contextlib.contextmanager
def temp_dir() -> typing.Generator[str, None, None]:
    """Context manager to temporally change current working directory to a unique temp directory.

    Mainly used for testing.
    The temp directory is removed when the context is exited.
    """

    # Remember the original directory
    orig_dir = os.getcwd()

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Change the directory
        os.chdir(tmp_dir)
        try:
            yield tmp_dir
        finally:
            # Return to the original directory
            os.chdir(orig_dir)


def get_base_path(scheduler: typing.Any) -> pathlib.Path:
    """Generate an absolute base path using the experiment metadata.

    The base path includes unique experiment metadata and can be used to generate
    output file names for experiment output.

    :param scheduler: The scheduler object
    :return: The base path (absolute)
    """

    try:
        # Try to obtain information from the scheduler
        rid = int(scheduler.rid)
        class_name = str(scheduler.expid.get('class_name'))
    except AttributeError:
        # Reset to default values if data could not be obtained from the scheduler
        rid = 0
        class_name = str(None)

    # Make absolute base path
    base_path = pathlib.Path(os.path.abspath(f'{rid:09d}-{class_name}'))
    # Ensure directory exists
    base_path.mkdir(parents=True, exist_ok=True)

    # Return base path
    return base_path


class BaseFileNameGenerator:
    """A file name generator that generates file names in a base path."""

    _base_path: pathlib.Path
    _unique: bool

    def __init__(self, base: typing.Union[str, pathlib.Path] = '', *,
                 unique: bool = False):
        """Create a new base file name generator.

        :param base: The base path, current working directory by default
        :param unique: Force unique file names
        """
        assert isinstance(base, (str, pathlib.Path))
        assert isinstance(unique, bool)

        # Store attributes
        self._base_path = pathlib.Path(base) if isinstance(base, str) else base
        self._unique = unique

    def _generate_file_name(self, name: str, ext: typing.Optional[str], *,
                            count: int = 0) -> pathlib.Path:
        """Generate a file name.

        If file names must be unique, use sequence numbers if necessary.

        :param name: Name of the file
        :param ext: The extension of the file
        :param count: Current sequence number
        """

        # Generate count string
        count_str: str = '' if count == 0 else f' ({count})'
        # Generate ext string
        ext_str: str = '' if ext is None else f'.{ext}'
        # Join base path with assembled name
        file_name: pathlib.Path = self._base_path.joinpath(f'{name}{count_str}{ext_str}')

        if self._unique and file_name.exists():
            # File name exists, recurse
            return self._generate_file_name(name, ext, count=count + 1)
        else:
            # Return file name
            return file_name

    def __call__(self, name: str, ext: typing.Optional[str] = None) -> str:
        """Generate an output file name.

        :param name: Name of the file
        :param ext: The extension of the file
        :return: A complete file name
        :raises ValueError: Raised if the name is empty
        """
        assert isinstance(name, str), 'File name must be of type str'
        assert isinstance(ext, str) or ext is None, 'File extension must be of type str or None'

        if not name:
            raise ValueError('The given name can not be empty')

        # Return name
        return str(self._generate_file_name(name, ext))


class FileNameGenerator(BaseFileNameGenerator):
    """A file name generator that generates unique file names in a single path based on the experiment metadata."""

    def __init__(self, scheduler: typing.Any):
        """Create a new file name generator.

        :param scheduler: The ARTIQ scheduler object
        """
        super(FileNameGenerator, self).__init__(get_base_path(scheduler), unique=True)


def get_file_name(scheduler: typing.Any, name: str, ext: typing.Optional[str] = None) -> str:
    """Generate a single unique and uniformly styled output file name based on the experiment metadata.

    When using this function in combination with :func:`temp_dir`, make sure
    this function is called before leaving the context.

    When generating more file names, it is more efficient to use :class:`FileNameGenerator`.

    :param scheduler: The scheduler object
    :param name: Name of the file
    :param ext: The extension of the file
    :return: A complete file name
    :raises ValueError: Raised if the name is empty
    """

    # Return full file name
    return FileNameGenerator(scheduler)(name, ext)
