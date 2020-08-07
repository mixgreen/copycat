import logging
import importlib
import typing
import configparser

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

_DUMMY_DEVICE: typing.Dict[str, str] = {
    'type': 'local',
    'module': f'{_DAX_COREDEVICE_PACKAGE}.dummy',
    'class': 'Dummy',
}
"""The properties of a dummy device."""

_SPECIAL_ENTRIES: typing.Dict[str, typing.Callable[[typing.Dict[str, typing.Any]], typing.Any]] = {
    # Set core host address to ::1 to prevent any undesired connections
    'core': lambda d: d.get('arguments', {}).update({'host': '::1'}),
    # Core log controller should not start in simulation, replace with dummy device
    'core_log': lambda d: d.update(_DUMMY_DEVICE),
}
"""Special keys/entries in the device DB that will be replaced."""

_SIMULATION_ARG: str = '--simulation'
"""The simulation argument/option for controllers as proposed by the ARTIQ manual."""

_CONFIG_FILES: typing.List[str] = ['setup.cfg', '.dax']
"""Configuration file locations in reverse order of priority."""

DAX_SIM_CONFIG_KEY: str = '_dax_sim_config'
"""The key of the virtual simulation configuration device."""


def enable_dax_sim(ddb: typing.Dict[str, typing.Any], *,
                   enable: typing.Optional[bool] = None,
                   logging_level: typing.Union[int, str] = logging.NOTSET,
                   output: str = 'vcd',
                   moninj_service: bool = True,
                   **signal_mgr_kwargs: typing.Any) -> typing.Dict[str, typing.Any]:
    """Enable the DAX simulation package by applying this function on your device DB.

    This function will modify your device DB in-place to configure it for simulation.

    The simulation can be configured through the function parameters or by using the
    configuration files. If given, function parameters are always prioritized over
    configuration file parameters. The possible configuration files in order of
    priority currently are `'.dax'` and `'setup.cfg'`.

    The following options can currently be set through the configuration files
    using the section `[dax.sim]`:

     - `enable`, required if not provided as a function parameter
     - `coredevice_packages`, additional packages to search for coredevice drivers (in order of priority)
     - `config_module`, the module of the simulation configuration class (defaults to DAX.sim config module)
     - `config_class`, the class of the simulation configuration object (defaults to DAX.sim config class)

    If supported by a specific simulated device driver, extra simulation-specific arguments
    can be added by adding a `sim_args` dict to the device entry in the device DB.
    The `arguments` dict of the device will be updated with the contents of the `sim_args` dict.

    The DAX.sim package provides a limited list of simulated coredevice drivers.
    Additional packages with simulated coredevice drivers can be added using the configuration files.
    If no coredevice driver was found, the device will be assigned a generic driver.
    Note that custom simulated coredevice drivers need to be a subclass of
    :class:`dax.sim.device.DaxSimDevice` to be compatible with other DAX.sim components.

    :param ddb: The device DB (will be updated if simulation is enabled)
    :param enable: Flag to enable DAX simulation
    :param logging_level: The logging level
    :param output: Simulation output type (`'null'`, `'vcd'`, or `'peek'`)
    :param moninj_service: Start the dummy MonInj service for the dashboard to connect to
    :param signal_mgr_kwargs: Arguments for the signal manager if output is enabled
    :return: The updated device DB
    :raises FileNotFoundError: Raised if configuration files are required but none are found
    """

    assert isinstance(ddb, dict), 'The device DB argument must be a dict'
    assert isinstance(enable, bool) or enable is None, 'The enable flag must be None or of type bool'
    assert isinstance(logging_level, (int, str)), 'Logging level must be of type int or str'
    assert isinstance(output, str), 'Output parameter must be of type str'
    assert isinstance(moninj_service, bool), 'MonInj service flag must be of type bool'

    # Set the logging level to the given value
    _logger.setLevel(logging_level)

    # Read configuration file
    _logger.debug('Reading configuration file')
    config: configparser.ConfigParser = configparser.ConfigParser()

    if not config.read(_CONFIG_FILES) and enable is None:
        # No files were successfully read but one or more fields require a configuration file
        _logger.error(f'Could not find a configuration file at any of the following '
                      f'locations: {", ".join(_CONFIG_FILES)}')
        raise FileNotFoundError('Configuration file not found')

    # Get coredevice packages from config file
    coredevice_packages: typing.List[str] = config.get('dax.sim', 'coredevice_packages', fallback='').split()
    # Append the DAX coredevice package
    coredevice_packages.append(_DAX_COREDEVICE_PACKAGE)

    if enable is None:
        # Get the boolean value (can raise various exceptions)
        enable = config.getboolean('dax.sim', 'enable')

    if enable:
        # Log that dax.sim was enabled
        _logger.info('DAX simulation enabled in device DB')

        if DAX_SIM_CONFIG_KEY not in ddb:
            # Convert the device DB
            _logger.debug('Converting device DB')
            try:
                used_ports: typing.Set[int] = set()
                for k, v in ddb.items():
                    # Mutate every entry in-place
                    _mutate_ddb_entry(k, v, coredevice_packages, used_ports)
            except Exception as e:
                # Log exception to provide more context
                _logger.exception(e)
                raise
        else:
            # Device DB was already converted
            _logger.debug('Device DB was already converted')

        # Prepare virtual device used for passing simulation configuration
        sim_config = {DAX_SIM_CONFIG_KEY: {
            'type': 'local',
            'module': config.get('dax.sim', 'config_module', fallback='dax.sim.config'),
            'class': config.get('dax.sim', 'config_class', fallback='DaxSimConfig'),
            # Simulation configuration is passed through the arguments
            'arguments': {'logging_level': logging_level,
                          'output': output,
                          'signal_mgr_kwargs': signal_mgr_kwargs},
        }}

        # Add simulation configuration to device DB
        _logger.debug('Updating simulation configuration in device DB')
        ddb.update(sim_config)

        if moninj_service:
            # Start MonInj dummy service
            _logger.debug('Starting MonInj dummy service')
            _start_moninj_service()

        # Return the device DB
        return ddb

    else:
        # Return the unmodified device DB
        _logger.debug('DAX simulation disabled')
        return ddb


def _mutate_ddb_entry(key: str, value: typing.Any, coredevice_packages: typing.List[str],
                      used_ports: typing.Set[int]) -> typing.Any:
    """Mutate a device DB entry to use it for simulation."""

    assert isinstance(key, str), 'The key must be of type str'

    if key in _SPECIAL_ENTRIES:
        # Special entries receive pre-processing
        _SPECIAL_ENTRIES[key](value)

    if isinstance(value, dict):  # If value is a dict, further processing is needed
        # Get the type entry of this value
        type_ = value.get('type')
        if not isinstance(type_, str):
            raise TypeError(f'The type key of local device "{key}" must be of type str')

        # Mutate entry
        if type_ == 'local':
            _mutate_local(key, value, coredevice_packages)
        elif type_ == 'controller':
            _mutate_controller(key, value, used_ports)
        else:
            _logger.debug(f'Skipped entry "{key}"')
    else:
        # Value is not a dict, it can be ignored
        pass

    # Return the potentially modified value
    return value


def _mutate_local(key: str, value: typing.Dict[str, typing.Any], coredevice_packages: typing.List[str]) -> None:
    """Mutate a device DB local entry to use it for simulation."""

    # Update the module of the current device to a simulation-capable coredevice driver
    _update_module(key, value, coredevice_packages)

    # Add key of the device to the device arguments
    arguments = value.setdefault('arguments', {})
    if not isinstance(arguments, dict):
        raise TypeError(f'The arguments key of local device "{key}" must be of type dict')
    arguments.update(_key=key)

    # Add simulation arguments to normal arguments
    sim_args = value.setdefault('sim_args', {})
    if not isinstance(sim_args, dict):
        raise TypeError(f'The sim_args key of local device "{key}" must be of type dict')
    arguments.update(sim_args)

    # Debug message
    _logger.debug(f'Converted local device "{key}" to class "{value["module"]}.{value["class"]}"')


def _update_module(key: str, value: typing.Dict[str, typing.Any], coredevice_packages: typing.List[str]) -> None:
    """Update the module of a local device to a simulation-capable coredevice driver."""

    # Get the module of the device
    module = value.get('module')
    if not isinstance(module, str):
        raise TypeError(f'The module key of local device "{key}" must be of type str')

    # Keep the tail of the module
    tail = module.rsplit('.', maxsplit=1)[-1]

    for package in coredevice_packages:
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


def _mutate_controller(key: str, value: typing.Dict[str, typing.Any], used_ports: typing.Set[int]) -> None:
    """Mutate a device DB controller entry to use it for simulation."""

    # Get the command of this controller
    command = value.get('command')

    if command is None:
        # No command was set
        _logger.debug(f'No command found for controller "{key}"')
    elif isinstance(command, str):
        # Check if the controller was already set to simulation mode
        if _SIMULATION_ARG not in command:
            # Simulation argument not found, append it
            _logger.debug(f'Added simulation argument to command for controller "{key}"')
            value['command'] = ' '.join([command, _SIMULATION_ARG])
        else:
            # Debug message
            _logger.debug(f'Controller "{key}" was not modified')
    else:
        # Command was not of type str
        raise TypeError(f'The command key of controller "{key}" must be of type str')

    # Set controller to run on localhost
    if 'host' not in value:
        raise KeyError(f'No host field present for controller "{key}"')
    value['host'] = '::1'
    _logger.debug(f'Controller "{key}" set to run on ::1')

    # Check that there are no port conflicts and add port to used_ports
    if isinstance(value['port'], int):
        if value['port'] in used_ports:
            raise ValueError(f'Port {value["port"]} used by controller "{key}" has already been used')
        used_ports.add(value['port'])
    else:
        raise TypeError(f'The port key of controller "{key}" must be of type int')


def _start_moninj_service() -> None:
    """Start the MonInj dummy service as an external process.

    If the MonInj dummy service was already started, it will exit silently.
    The current Python interpreter is used for the subprocess.
    """
    import subprocess
    import sys
    subprocess.Popen([sys.executable, '-m', 'dax.util.moninj', '--auto-close', '1'])
