import types
import shutil
import os.path

import artiq.tools

import dax.util.output

__all__ = ['load_module']


def load_module(file_name: str, prefix: str) -> types.ModuleType:
    """Load a Python file or an archive with a ``main.py`` file.

    :param file_name: The file name
    :param prefix: The prefix used to make the module name unique (e.g. ``"dax_program_client_"``)
    :return: The loaded module
    :raises FileNotFoundError: Raised if the file does not exist or the archive is not correctly formatted
    """
    assert isinstance(file_name, str), 'File name must be of type str'
    assert isinstance(prefix, str), 'Prefix must be of type str'

    # Expand and check path
    file_name = os.path.expanduser(file_name)
    if not os.path.isfile(file_name):
        raise FileNotFoundError(f'No such file or path is a directory: "{file_name}"')

    if file_name.endswith('.py'):
        # Load file/module
        return artiq.tools.file_import(file_name, prefix=prefix)
    else:
        # We assume that we are dealing with an archive
        with dax.util.output.temp_dir() as temp_dir:
            # Unpack archive
            shutil.unpack_archive(file_name, extract_dir=temp_dir)  # Raises exception if format is not recognized
            unpacked_file_name = os.path.join(temp_dir, 'main.py')
            if not os.path.isfile(unpacked_file_name):
                raise FileNotFoundError(f'Archive "{file_name}" does not contain a main.py file')
            return artiq.tools.file_import(unpacked_file_name, prefix=prefix)
