import typing
import os.path

import artiq.tools

from dax.experiment import *
import dax.base.program
import dax.util.artiq
import dax.interfaces.operation

__all__ = ['ProgramClient']


@dax_client_factory
class ProgramClient(DaxClient, Experiment):
    """Client for dynamically loading and running a DAX program."""

    MANAGERS_KWARG = 'managers'

    def build(self, managers: typing.Any) -> None:  # type: ignore
        # Store reference to ARTIQ managers
        self._managers: typing.Any = managers

        # Obtain arguments
        self._program_file: str = self.get_argument('file', StringValue(),
                                                    tooltip='File containing the program to run')
        self._program_class: str = self.get_argument('class', StringValue(''),
                                                     tooltip='Class name of the program to run (optional)')

    def prepare(self) -> None:
        # Load file/module
        file_name = os.path.expanduser(self._program_file)
        self.logger.debug(f'Loading program file "{file_name}"')
        if not os.path.isfile(file_name):
            raise FileNotFoundError(f'No such file or path is a directory: "{file_name}"')
        module = artiq.tools.file_import(file_name, prefix='dax_program_client_')

        # Obtain class
        self.logger.debug('Loading program class%s', f' "{self._program_class}"' if self._program_class else '')
        program_cls = artiq.tools.get_experiment(module,
                                                 class_name=self._program_class if self._program_class else None)
        self.logger.info(f'Loaded program "{self._program_file}:{program_cls.__name__}"')

        # Test class
        if not issubclass(program_cls, dax.base.program.DaxProgram):
            raise TypeError(f'Class "{self._program_file}:{program_cls.__name__}" is not a DAX program')

        # Get interface
        self._interface: dax.interfaces.operation.OperationInterface
        self._interface = self.registry.find_interface(dax.interfaces.operation.OperationInterface)

        # Build the program
        self.logger.info('Building program')
        self._program: Experiment = program_cls(
            dax.util.artiq.isolate_managers(self._managers, name='program'),
            core=self.core,
            interface=self._interface
        )

        # Prepare the program
        self.logger.info('Preparing program')
        self._program.prepare()

    def run(self) -> None:
        # Validate the operation interface (should be done after DAX init)
        dax.interfaces.operation.validate_operation_interface(self._interface)
        # Run the program
        self.logger.info('Running program')
        self._program.run()

    def analyze(self) -> None:
        # Analyze the program
        self.logger.info('Analyzing program')
        self._program.analyze()
