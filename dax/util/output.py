import typing

__all__ = ['get_base_name', 'get_file_name_generator', 'get_file_name']


def get_base_name(scheduler: typing.Any) -> str:
    """Generate a base file name based on the experiment metadata.

    The base file name consists of experiment metadata and does not contain an extension yet.

    :param scheduler: The scheduler object
    :return: The base file name
    """

    try:
        # Try to obtain information from the scheduler
        rid = int(scheduler.rid)
        class_name = str(scheduler.expid.get('class_name'))
    except AttributeError:
        # Reset to defaults if data could not be obtained from the scheduler
        rid = 0
        class_name = str(None)

    # Make and return base file name
    base_name = '{:09d}-{:s}'.format(rid, class_name)
    return base_name


# Type ignore required to pass type checking without use of typing.Protocol
def get_file_name_generator(scheduler: typing.Any):  # type: ignore
    """Obtain a generator that generates uniform file names based on the experiment metadata.

    The base file name consists of experiment metadata and does not contain an extension yet.

    :param scheduler: The scheduler object
    :return: Generator function for file names
    """

    # Get base file name
    base_name = get_base_name(scheduler)

    # Generator function
    def file_name_generator(ext: str, postfix: typing.Optional[str] = None) -> str:
        """Generate a uniformly styled output file name based on the experiment metadata.

        :param ext: The extension of the file
        :param postfix: A postfix for the base file name
        :return: A complete file name
        """
        assert isinstance(ext, str), 'File extension must be of type str'
        assert isinstance(postfix, str) or postfix is None, 'Postfix must be of type str or None'

        # Add postfix if provided
        output_file_name = base_name if postfix is None else '-'.join((base_name, postfix))
        # Return full file name
        return '.'.join((output_file_name, ext))

    # Return generator function
    return file_name_generator


def get_file_name(scheduler: typing.Any, ext: str, postfix: typing.Optional[str] = None) -> str:
    """Generate a single uniformly styled output file name based on the experiment metadata.

    :param scheduler: The scheduler object
    :param ext: The extension of the file
    :param postfix: A postfix for the base file name
    :return: A complete file name
    """

    # Get a generator (extra typing annotation required to pass type checking without use of typing.Protocol)
    gen = get_file_name_generator(scheduler)  # type: typing.Callable[[str, typing.Optional[str]], str]
    # Return full file name
    return gen(ext, postfix)
