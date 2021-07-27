# mypy: no_warn_unused_ignores

import typing
import types
import os.path
import shutil
import shlex
import argparse

import artiq.tools

# No wildcard import to prevent aliasing with ``types``
from dax.experiment import DaxClient, dax_client_factory, Experiment, StringValue, NoDefault

import dax.base.program
import dax.util.artiq
import dax.util.output
import dax.interfaces.operation
import dax.interfaces.data_context

__all__ = ['ProgramClient']


def _import_file(file_name: str) -> types.ModuleType:
    return artiq.tools.file_import(file_name, prefix='dax_program_client_')


@dax_client_factory
class ProgramClient(DaxClient, Experiment):
    """Client to dynamically load and run a DAX program.

    To use this client, a system needs to have the following components available:

    - An :class:`dax.interfaces.operation.OperationInterface`
    - A :class:`dax.interfaces.data_context.DataContextInterface`

    Users can override the following attributes to change the default arguments of this client:

    - :attr:`DEFAULT_OPERATION_KEY`
    - :attr:`DEFAULT_DATA_CONTEXT_KEY`

    This class can be customized by overriding the :func:`add_arguments`, :func:`setup`,
    and :func:`cleanup` functions.
    """

    MANAGERS_KWARG = 'managers'

    DEFAULT_OPERATION_KEY: typing.ClassVar[typing.Union[str, typing.Type[NoDefault]]] = NoDefault
    """Key of the default operation interface."""
    DEFAULT_DATA_CONTEXT_KEY: typing.ClassVar[typing.Union[str, typing.Type[NoDefault]]] = NoDefault
    """Key of the default data context interface."""

    _managers: typing.Any
    _scheduler: typing.Any
    _file: str
    _class: str
    _arguments: str
    _operation_key: str
    _data_context_key: str
    _operation: dax.interfaces.operation.OperationInterface
    _data_context: dax.interfaces.data_context.DataContextInterface
    _program: Experiment

    def build(self, *, managers: typing.Any) -> None:  # type: ignore[override]
        assert isinstance(self.DEFAULT_OPERATION_KEY, str) or self.DEFAULT_OPERATION_KEY is NoDefault
        assert isinstance(self.DEFAULT_DATA_CONTEXT_KEY, str) or self.DEFAULT_DATA_CONTEXT_KEY is NoDefault

        # Store reference to ARTIQ managers
        self._managers = managers
        # Obtain the scheduler
        self._scheduler = self.get_device('scheduler')

        # Search for interfaces
        self._operation_interfaces = self.registry.search_interfaces(
            dax.interfaces.operation.OperationInterface)  # type: ignore[misc]
        if not self._operation_interfaces:
            raise LookupError('No operation interfaces available')
        self._data_context_interfaces = self.registry.search_interfaces(
            dax.interfaces.data_context.DataContextInterface)  # type: ignore[misc]
        if not self._data_context_interfaces:
            raise LookupError('No data context interfaces available')

        # Obtain arguments
        self._file = self.get_argument(
            'file', StringValue(), tooltip='File containing the program to run or an archive with a main.py file')
        self._class = self.get_argument(
            'class', StringValue(''), tooltip='Class name of the program to run (optional)')
        self._arguments = self.get_argument(
            'arguments', StringValue(''), tooltip='Command-line arguments (format: `[KEY=PYON_VALUE ...]`)')
        self._operation_key = self.get_argument(
            'operation', dax.util.artiq.default_enumeration_value(sorted(self._operation_interfaces),
                                                                  default=self.DEFAULT_OPERATION_KEY),
            tooltip='The operation interface to use')
        self._data_context_key = self.get_argument(
            'data_context', dax.util.artiq.default_enumeration_value(sorted(self._data_context_interfaces),
                                                                     default=self.DEFAULT_DATA_CONTEXT_KEY),
            tooltip='The data context interface to use')

        # Add custom arguments
        self.add_arguments()

    def prepare(self) -> None:
        # Load the module
        module: types.ModuleType = self._load_module()

        # Obtain class
        self.logger.debug('Loading program class%s', f' "{self._class}"' if self._class else '')
        program_cls = artiq.tools.get_experiment(module,
                                                 class_name=self._class if self._class else None)
        self.logger.info(f'Loaded program "{self._file}:{program_cls.__name__}"')
        self._class = program_cls.__name__  # Store class name in case none was given

        # Archive program metadata
        self.set_dataset('file', self._file)
        self.set_dataset('class', self._class)
        self.set_dataset('arguments', self._arguments)

        # Test class
        if not issubclass(program_cls, dax.base.program.DaxProgram):
            raise TypeError(f'Class "{self._file}:{self._class}" is not a DAX program')

        # Get interfaces
        self._operation = self._operation_interfaces[self._operation_key]
        self._data_context = self._data_context_interfaces[self._data_context_key]

        # Parse arguments
        if self._arguments:
            self.logger.debug(f'Parsing arguments: {self._arguments}')
            parser = argparse.ArgumentParser()
            parser.add_argument('args', nargs='*')
            try:
                arguments: typing.Dict[str, typing.Any] = artiq.tools.parse_arguments(
                    parser.parse_args(shlex.split(self._arguments, posix=False)).args)
            except Exception as e:
                raise RuntimeError('Exception occurred while parsing arguments') from e
        else:
            arguments = {}

        # Build the program
        self.logger.info(f'Building program "{self._class}"')
        self._isolated_managers = self._get_managers(arguments=arguments)
        self._program = program_cls(
            self._isolated_managers,
            core=self.core,
            operation=self._operation,
            data_context=self._data_context
        )

        # Prepare the program
        self.logger.info(f'Preparing program "{self._class}"')
        self._program.prepare()

    def run(self) -> None:
        # Validate interfaces
        assert dax.interfaces.operation.validate_interface(self._operation)
        assert dax.interfaces.data_context.validate_interface(self._data_context)

        try:
            # Perform setup
            self.setup()

            # Run the program
            self.logger.info(f'Running program "{self._class}"')
            self._program.run()
            self.logger.debug('Program finished')

        except:  # noqa: E722
            # write to hdf5 file even if run fails
            self._write_hdf5_file()

        finally:
            # Perform cleanup
            self.cleanup()

    def analyze(self) -> None:
        # Analyze the program
        self.logger.info(f'Analyzing program "{self._class}"')
        try:
            self._program.analyze()
        finally:
            # Write HDF5 file
            self._write_hdf5_file()

    def _load_module(self) -> types.ModuleType:
        # Expand and check path
        file_name = os.path.expanduser(self._file)
        if not os.path.isfile(file_name):
            raise FileNotFoundError(f'No such file or path is a directory: "{file_name}"')

        if self._file.endswith('.py'):
            # Load file/module
            self.logger.debug(f'Loading program file "{file_name}"')
            return _import_file(file_name)
        else:
            # We assume that we are dealing with an archive
            self.logger.debug(f'Unpacking and loading program archive "{file_name}"')
            with dax.util.output.temp_dir() as temp_dir:
                # Unpack archive
                shutil.unpack_archive(file_name, extract_dir=temp_dir)  # Raises exception of format is not recognized
                unpacked_file_name = os.path.join(temp_dir, 'main.py')
                if not os.path.isfile(unpacked_file_name):
                    raise FileNotFoundError(f'Archive "{file_name}" does not contain a main.py file')
                return _import_file(unpacked_file_name)

    def _get_managers(self, *, name: str = 'program',
                      arguments: typing.Dict[str, typing.Any]) -> dax.util.artiq.ManagersTuple:
        # Give a copy of managers, isolated
        return dax.util.artiq.isolate_managers(self._managers, name=name, arguments=arguments)

    def _write_hdf5_file(self) -> None:
        # Collect metadata
        metadata = {
            'rid': self._scheduler.rid,
            'file': self._file,
            'class': self._class,
            'arguments': self._arguments,
            'operation_key': self._operation_key,
            'data_context_key': self._data_context_key,
        }

        # Write a separate HDF5 file for the isolated datasets
        self.logger.debug('Writing HDF5 file for isolated dataset manager')
        self._isolated_managers.write_hdf5(dax.util.output.get_file_name(self._scheduler, 'program', 'h5'),
                                           metadata=metadata)

    """Customization functions"""

    def add_arguments(self) -> None:
        """Add custom arguments during the build phase."""
        pass

    def setup(self):  # type: () -> None
        """Setup on the host and/or the core device, called once at entry.

        Host and device setup are not separated for this client.
        """
        pass

    def cleanup(self):  # type: () -> None
        """Cleanup on the host and/or the core device, called once at exit.

        Host and device setup are not separated for this client.
        """
        pass
