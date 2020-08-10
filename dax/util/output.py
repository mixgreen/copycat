import typing
import os
import pathlib
import contextlib
import tempfile

__all__ = ['temp_dir', 'get_base_path', 'get_file_name_generator', 'dummy_file_name_generator', 'get_file_name']


@contextlib.contextmanager
def temp_dir() -> typing.Generator[None, None, None]:
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
            yield
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
        # Reset to defaults if data could not be obtained from the scheduler
        rid = 0
        class_name = str(None)

    # Make absolute base path
    base_path = pathlib.Path(os.path.abspath(f'{rid:09d}-{class_name}'))
    # Ensure directory exists
    base_path.mkdir(parents=True, exist_ok=True)

    # Return base path
    return base_path


# Type ignore required to pass type checking without use of typing.Protocol
def get_file_name_generator(scheduler: typing.Any):  # type: ignore
    """Obtain a generator that generates file names in a single path based on the experiment metadata.

    :param scheduler: The scheduler object
    :return: Generator function for file names
    """

    # Get base path
    base_path = get_base_path(scheduler)

    # Generator function
    def file_name_generator(name: str, ext: typing.Optional[str] = None) -> str:
        """Generate a uniformly styled output file names.

        :param name: Name of the file
        :param ext: The extension of the file
        :return: A complete file name
        :raises ValueError: Raised if the name is empty
        """
        assert isinstance(name, str), 'File name must be of type str'
        assert isinstance(ext, str) or ext is None, 'File extension must be of type str or None'

        if not name:
            raise ValueError('The given name can not be empty')

        if ext is not None:
            # Add file extension
            name = f'{name}.{ext}'
        # Join base path and provided name
        file_name = base_path.joinpath(name)
        # Return full file name as a string
        return str(file_name)

    # Return generator function
    return file_name_generator


def dummy_file_name_generator(name: str, ext: typing.Optional[str] = None) -> str:
    """Generate output file names, dummy replacement for :func:`get_file_name_generator`.

    This function has the same interface as the generator returned by :func:`get_file_name_generator`.

    :param name: Name of the file
    :param ext: The extension of the file
    :return: A complete file name
    :raises ValueError: Raised if the name is empty
    """
    assert isinstance(name, str), 'File name must be of type str'
    assert isinstance(ext, str) or ext is None, 'File extension must be of type str or None'

    if not name:
        raise ValueError('The given name can not be empty')

    if ext is not None:
        # Add file extension
        name = f'{name}.{ext}'

    # Return name
    return name


def get_file_name(scheduler: typing.Any, name: str, ext: typing.Optional[str] = None) -> str:
    """Generate a single uniformly styled output file name based on the experiment metadata.

    When using this function in combination with :func:`temp_dir`, make sure
    this function is called before leaving the context.

    :param scheduler: The scheduler object
    :param name: Name of the file
    :param ext: The extension of the file
    :return: A complete file name
    :raises ValueError: Raised if the name is empty
    """

    # Get a generator (extra typing annotation required to pass type checking without use of typing.Protocol)
    gen: typing.Callable[[str, typing.Optional[str]], str] = get_file_name_generator(scheduler)
    # Return full file name
    return gen(name, ext)
