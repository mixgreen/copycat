from __future__ import annotations  # Postponed evaluation of annotations

import logging
import importlib
import typing
import collections.abc
import dataclasses
import re

import sipyco.pyon

import dax.util.configparser
import dax.util.moninj

__all__ = ['DAX_SIM_CONFIG_KEY', 'enable_dax_sim']

_logger: logging.Logger = logging.getLogger(__name__)
"""The logger for this file."""

_DAX_COREDEVICE_PACKAGE: str = 'dax.sim.coredevice'
"""The path to the dax.sim coredevice package."""

_GENERIC_DEVICE: typing.Dict[str, str] = {
    'type': 'local',
    'module': f'{_DAX_COREDEVICE_PACKAGE}.generic',
    'class': 'Generic',
}
"""The properties of a generic device."""

_SIMULATION_ARGS: typing.List[str] = ['--simulation', '--no-localhost-bind']
"""The simulation options/arguments to add to controllers."""

_CONFIG_SECTION: str = 'dax.sim'
"""The section in the configuration file used by DAX.sim."""

DAX_SIM_CONFIG_KEY: str = '_dax_sim_config'
"""The key of the virtual simulation configuration device."""


@dataclasses.dataclass(frozen=True)
class _ConfigData:
    """Dataclass to hold configuration."""
    coredevice_packages: typing.List[str]
    exclude: typing.Sequence[typing.Pattern[str]]
    core_device: str
    localhost: str
    _config: dax.util.configparser.DaxConfigParser

    def get_args(self, key: str) -> typing.Dict[str, typing.Any]:
        """Get additional simulation arguments provided through the config file.

        Additional arguments are decoded as PYON values.

        :param key: The key of the device
        :return: A dict with simulation arguments
        """
        section: str = f'{_CONFIG_SECTION}.{key}'
        if section in self._config:
            return {k: sipyco.pyon.decode(v) for k, v in self._config.items(section)}
        else:
            return {}

    def is_excluded(self, key: str) -> bool:
        """Check if a device is excluded.

        :param key: The key of the device
        :return: :const:`True` if the device is excluded
        """
        return any(p.fullmatch(key) for p in self.exclude)

    @classmethod
    def create(cls, config: dax.util.configparser.DaxConfigParser, *,
               exclude: typing.Collection[str] = ()) -> _ConfigData:
        """Create a configuration dataclass given a config parser."""

        # Get coredevice packages and append the DAX coredevice package
        coredevice_packages: typing.List[str] = config.get(_CONFIG_SECTION, 'coredevice_packages', fallback='').split()
        coredevice_packages.append(_DAX_COREDEVICE_PACKAGE)

        # Join exclude patterns
        exclude = set(exclude)  # Use a set to get rid of duplicates
        exclude.update(config.get(_CONFIG_SECTION, 'exclude', fallback='').split())

        # Create and return the dataclass object
        return cls(
            coredevice_packages=coredevice_packages,
            exclude=[re.compile(p) for p in exclude],
            core_device=config.get(_CONFIG_SECTION, 'core_device', fallback='core'),
            localhost=config.get(_CONFIG_SECTION, 'localhost', fallback='::1'),
            _config=config
        )


def enable_dax_sim(ddb: typing.Dict[str, typing.Any], *,
                   enable: typing.Optional[bool] = None,
                   logging_level: typing.Union[int, str] = logging.NOTSET,
                   output: str = 'vcd',
                   exclude: typing.Collection[str] = (),
                   moninj_service: bool = True,
                   **signal_mgr_kwargs: typing.Any) -> typing.Dict[str, typing.Any]:
    """Enable the DAX simulation package by applying this function on your device DB.

    This function will modify your device DB in-place to configure it for simulation.

    The simulation can be configured through the function parameters or by using the
    configuration files. If given, function parameters are always prioritized over
    configuration file parameters. The possible configuration files are listed in
    the class variable :attr:`dax.util.configparser.DaxConfigParser.CONFIG_FILES`.

    The following options can currently be set through the configuration files
    using the section ``[dax.sim]``:

     - ``enable``, required if not provided as a function parameter
     - ``coredevice_packages``, additional packages to search for coredevice drivers (in order of priority)
     - ``exclude``, regex patterns to match excluded keys in the device DB (full match), merged with exclude patterns
       provided through the ``exclude`` argument
     - ``config_module``, the module of the simulation configuration class (defaults to DAX.sim config module)
     - ``config_class``, the class of the simulation configuration object (defaults to DAX.sim config class)
     - ``core_device``, the name of the core device (defaults to ``'core'``)
     - ``localhost``, the address to use to refer to localhost (defaults to IPv6 address ``'::1'``)

    If supported by a specific simulated device driver, extra simulation-specific arguments
    can be added by adding a ``'sim_args'`` key with a dict value to the device entry in the device DB.
    The ``'arguments'`` dict of the device will be updated with the contents of the ``'sim_args'`` dict.
    Extra simulation-specific arguments can also be passed through the configuration file.
    To add arguments for a device with key ``'device_key'`` add a section ``[dax.sim.device_key]``.
    Values in the section are decoded as PYON values.
    Excluded keys will not be mutated at all, and users are responsible for providing the correct arguments.

    The DAX.sim package provides a limited list of simulated coredevice drivers.
    Additional packages with simulated coredevice drivers can be added using the configuration files.
    If no coredevice driver was found, the device will be assigned a generic driver.
    Note that custom simulated coredevice drivers need to be a subclass of
    :class:`dax.sim.device.DaxSimDevice` to be compatible with other DAX.sim components.

    :param ddb: The device DB (will be updated if simulation is enabled)
    :param enable: Flag to enable DAX simulation
    :param logging_level: The logging level
    :param output: Simulation output type (``'null'``, ``'vcd'``, or ``'peek'``)
    :param exclude: Regex patterns to match excluded keys in the device DB (full match)
    :param moninj_service: Start the dummy MonInj service for the dashboard to connect to
    :param signal_mgr_kwargs: Arguments for the signal manager if output is enabled
    :return: The updated device DB
    :raises FileNotFoundError: Raised if configuration files are required but none are found
    """

    assert isinstance(ddb, dict), 'The device DB argument must be a dict'
    assert isinstance(enable, bool) or enable is None, 'The enable flag must be None or of type bool'
    assert isinstance(logging_level, (int, str)), 'Logging level must be of type int or str'
    assert isinstance(output, str), 'Output parameter must be of type str'
    assert isinstance(exclude, collections.abc.Collection), 'Exclude must be a collection'
    assert all(isinstance(s, str) for s in exclude), 'All exclude patterns must be of type str'
    assert isinstance(moninj_service, bool), 'MonInj service flag must be of type bool'

    # Set the logging level to the given value
    _logger.setLevel(logging_level)

    # Get configuration file
    _logger.debug('Obtaining configuration')
    config: dax.util.configparser.DaxConfigParser = dax.util.configparser.get_dax_config()

    if not config.used_config_files and enable is None:
        # No files were successfully read but one or more fields require a configuration file
        _logger.error(f'Could not find a configuration file at any of the following '
                      f'locations: {", ".join(config.CONFIG_FILES)}')
        raise FileNotFoundError('Configuration file not found')

    if enable is None:
        # Get the boolean value (can raise various exceptions)
        enable = config.getboolean(_CONFIG_SECTION, 'enable')

    if enable:
        # Log that DAX.sim was enabled
        _logger.info('DAX simulation enabled in device DB')

        if DAX_SIM_CONFIG_KEY not in ddb:
            # Convert the device DB
            _logger.debug('Converting device DB')

            # Construct configuration data object
            config_data: _ConfigData = _ConfigData.create(config, exclude=exclude)

            # Check core device in the device DB
            if config_data.core_device not in ddb:
                raise KeyError(f'Core device key "{config_data.core_device}" not found in the device DB')
            if not isinstance(ddb[config_data.core_device], dict):
                raise ValueError(f'Core device key "{config_data.core_device}" can not be an alias')

            try:
                # Set with port numbers used by controllers
                used_ports: typing.Set[int] = set()

                for k, v in ddb.items():
                    alias = get_unique_device_root_alias(k, ddb)
                    if config_data.is_excluded(k):
                        _logger.debug(f'Excluded entry "{k}"')
                    else:
                        # Mutate entry in-place
                        _mutate_ddb_entry(k, v, config=config_data, used_ports=used_ports, alias=alias)
            except Exception as e:
                # Log exception to provide more context
                _logger.exception(e)
                raise
        else:
            # Device DB was already converted
            _logger.debug('Device DB was already converted')

        # Add virtual device used for passing simulation configuration to device DB (overwrites existing configuration)
        _logger.debug('Updating simulation configuration in device DB')
        ddb[DAX_SIM_CONFIG_KEY] = {
            'type': 'local',
            'module': config.get(_CONFIG_SECTION, 'config_module', fallback='dax.sim.config'),
            'class': config.get(_CONFIG_SECTION, 'config_class', fallback='DaxSimConfig'),
            # Simulation configuration is passed through the arguments
            'arguments': {'logging_level': logging_level,
                          'output': output,
                          'signal_mgr_kwargs': signal_mgr_kwargs},
        }

        if moninj_service:
            # Get port
            moninj_port = ddb.get('core_moninj', {}).get('port_proxy', dax.util.moninj.MonInjDummyService.DEFAULT_PORT)
            # Start MonInj dummy service
            _logger.debug(f'Starting MonInj dummy service on port {moninj_port}')
            _start_moninj_service(port=moninj_port)

        # Return the device DB
        return ddb

    else:
        # Return the unmodified device DB
        _logger.debug('DAX simulation disabled')
        return ddb


def _mutate_ddb_entry(key: str, value: typing.Any, *,
                      config: _ConfigData,
                      used_ports: typing.Set[int], alias: str) -> typing.Any:
    """Mutate a device DB entry to use it for simulation."""

    assert isinstance(key, str), 'The key must be of type str'

    if isinstance(value, dict):  # If value is a dict, further processing is needed
        # Get the type entry of this value
        type_ = value.get('type')
        if not isinstance(type_, str):
            raise TypeError(f'The type key of local device "{key}" must be of type str')

        # Mutate entry
        if type_ == 'local':
            _mutate_local(key, value, config=config, alias=alias)
        elif type_ == 'controller':
            _mutate_controller(key, value, config=config, used_ports=used_ports)
        else:
            _logger.debug(f'Skipped entry "{key}" with unknown type "{type_}"')

    # Return the potentially modified value
    return value


def _mutate_local(key: str, value: typing.Dict[str, typing.Any], *, config: _ConfigData, alias: str) -> None:
    """Mutate a device DB local entry to use it for simulation."""

    # Add simulation arguments to normal arguments
    arguments = value.setdefault('arguments', {})
    if not isinstance(arguments, dict):
        raise TypeError(f'The arguments key of local device "{key}" must be of type dict')
    sim_args = value.setdefault('sim_args', {})
    if not isinstance(sim_args, dict):
        raise TypeError(f'The sim_args key of local device "{key}" must be of type dict')
    arguments.update(sim_args)

    # Add simulation arguments passed through the config file
    arguments.update(config.get_args(key))

    # Add key of the device to the device arguments
    arguments['_key'] = key

    # Add alias of the device to the device arguments
    arguments['_alias'] = alias

    if key == config.core_device:
        # Set the host of the core device to localhost
        if 'host' not in arguments:
            raise KeyError(f'No host argument present for core device "{key}"')
        arguments['host'] = config.localhost

    # Update the module of the current device to a simulation-capable coredevice driver
    _update_module(key, value, config=config)

    # Debug message
    _logger.debug(f'Local device "{key}": class "{value["module"]}.{value["class"]}", arguments {arguments}')


def _update_module(key: str, value: typing.Dict[str, typing.Any], *, config: _ConfigData) -> None:
    """Update the module of a local device to a simulation-capable coredevice driver."""

    # Get the module of the device
    module = value.get('module')
    if not isinstance(module, str):
        raise TypeError(f'The module key of local device "{key}" must be of type str')

    # Keep the tail of the module
    tail = module.rsplit('.', maxsplit=1)[-1]

    for package in config.coredevice_packages:
        # Convert module name based on the current package
        module = f'{package}.{tail}'

        try:
            # Check if the module exists by importing it
            m = importlib.import_module(module)
        except ImportError:
            # Module was not found, continue to next package
            continue
        else:
            # Get the class of the device
            class_ = value.get('class')
            if not isinstance(class_, str):
                raise TypeError(f'The class key of local device "{key}" must be of type str')

            if hasattr(m, class_):
                # Both module and class were found, update module and return
                value['module'] = module
                return
            else:
                # Class was not found in module, continue to next package
                continue

    # Module was not found in any package, fall back on generic device
    value.update(_GENERIC_DEVICE)


def _mutate_controller(key: str, value: typing.Dict[str, typing.Any], *,
                       config: _ConfigData, used_ports: typing.Set[int]) -> None:
    """Mutate a device DB controller entry to use it for simulation."""

    # Get the command of this controller
    command: typing.Any = value.get('command')

    if command is None:
        # No command was set
        _logger.debug(f'Controller "{key}": no command found')
    elif isinstance(command, str):
        # Check for the bind and port argument
        if any(s not in command for s in ['--bind', '{bind}']):
            raise ValueError(f'Controller "{key}" is missing the "--bind {{bind}}" argument')
        if '{port}' not in command:
            _logger.warning(f'Controller "{key}" is missing the "--port {{port}}" argument')
        # See which simulation arguments are not present
        args: typing.List[str] = [a for a in _SIMULATION_ARGS if a not in command]
        if args:
            # Add simulation arguments
            sim_args: str = ' '.join(args)
            value['command'] = f'{command} {sim_args}'
            _logger.debug(f'Controller "{key}": added simulation argument(s) "{sim_args}" to command')
        else:
            # No simulation arguments added
            _logger.debug(f'Controller "{key}": command not modified')
    else:
        # Command was not of type str
        raise TypeError(f'The command key of controller "{key}" must be of type str')

    # Set controller to run on localhost
    if 'host' not in value:
        raise KeyError(f'No host field present for controller "{key}"')
    value['host'] = config.localhost

    # Check that there are no port conflicts and add port to used_ports
    if 'port' not in value:
        raise KeyError(f'No port field present for controller "{key}"')
    port: int = value['port']
    if isinstance(port, int):
        if port in used_ports:
            raise ValueError(f'Port {port} used by controller "{key}" has already been used')
        used_ports.add(port)
    else:
        raise TypeError(f'The port key of controller "{key}" must be of type int')


def _start_moninj_service(*, port: int = dax.util.moninj.MonInjDummyService.DEFAULT_PORT) -> None:
    """Start the MonInj dummy service as an external process.

    If the MonInj dummy service was already started, it will exit silently.
    The current Python interpreter is used for the subprocess.
    """
    import subprocess
    import sys
    subprocess.Popen([sys.executable, '-m', 'dax.util.moninj', '--port', f'{port}', '--auto-close', '1'],
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     close_fds=True, start_new_session=True, creationflags=getattr(subprocess, 'DETACHED_PROCESS', 0))


def get_unique_device_root_alias(value: str, ddb: typing.Dict[str, typing.Any]) -> str:
    """Get the unique device root alias by resolving it recursively in the device DB.

    :param value: The value to resolve
    :param ddb: The device.db to resolve with
    :return: The resolved unique device root alias
    :raises LookupError: Raised if an alias loop was detected in the trace while resolving
    :raises TypeError: Raised if a value returned an unexpected type
    :raises ValueError: Raised if multiple aliases are found for the same value
    """

    assert isinstance(value, str), 'Value must be of type str'
    assert isinstance(ddb, dict), 'The device DB argument must be a dict'

    return _resolve_unique_device_root_alias(value, set(), ddb)


def _resolve_unique_device_root_alias(value: str, trace: typing.Set[str],
                                      ddb: typing.Dict[str, typing.Any]) -> str:
    """Recursively resolve aliases until we find the unique device name.

    :param value: The value to resolve
    :param ddb: The device.db to resolve with
    :return: The resolved unique device root alias
    :raises LookupError: Raised if an alias loop was detected in the trace while resolving
    :raises TypeError: Raised if a value returned an unexpected type
    :raises ValueError: Raised if multiple aliases are found for the same value
    """

    # Check if we are not stuck in a loop
    if value in trace:
        # We are in an alias loop
        raise LookupError(f'Key "{value}" caused an alias loop')
    # Add key to the trace
    trace.add(value)

    # Get all possible aliases
    keys: typing.Any = [k for k, v in ddb.items() if v == value]

    if len(keys) == 0:
        # No keys were found, value is the root alias
        return value
    elif not len(keys) == 1:
        # Multiple keys found for the same value
        raise ValueError
    elif isinstance(keys[0], str):
        # Recurse if we are still dealing with an alias
        return _resolve_unique_device_root_alias(keys[0], trace, ddb)
    else:
        # We ended up with an unexpected type
        raise TypeError(f'Value "{value}" returned an unexpected type')
