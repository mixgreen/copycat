"""
This module can be run directly to start the MonInj dummy service.

The MonInj dummy service can be used during simulation to reduce the number of error messages observed
in the ARTIQ dashboard. When using DAX.sim, the MonInj dummy service is started automatically.
"""

import logging
import asyncio

_logger: logging.Logger = logging.getLogger('MonInjDummyService')
"""The logger for this file."""

__all__ = ['MonInjDummyService', 'MonInjBindError']


class MonInjBindError(OSError):
    """Exception class raised if an OS error occurs while starting the MonInj server.

    If an OS error occurs when starting a server, it is probably due to a bind error.
    """
    pass


class MonInjDummyService:
    """MonInj dummy service class.

    This class manages a server that can be used to set up a dummy MonInj service.
    It can be used during simulation such that the ARTIQ dashboard can connect to this
    MonInj dummy service instead of periodically raising exceptions.
    """

    ARTIQ_HELLO: bytes = b"ARTIQ moninj\n"
    """Hello message from ARTIQ dashboard."""

    def __init__(self, host: str, port: int):
        """Instantiate a new MonInj dummy service.

        :param host: The host to bind to
        :param port: The port to bind to
        """
        assert isinstance(host, str), 'Host most be of type str'
        assert isinstance(port, int), 'Port must be of type int'

        # Store attributes
        self._host: str = host
        self._port: int = port

    def run(self) -> None:
        """Regular method to run the server in an infinite loop.

        Uses `asyncio.run()` to start this service.

        If the server is not able to bind, it might already be running.

        :raises MonInjBindError: Raised if the server could not bind
        """
        asyncio.run(self.async_run())

    async def async_run(self) -> None:
        """Async method to run the server in an infinite loop.

        If the server is not able to bind, it might already be running.

        :raises MonInjBindError: Raised if the server could not bind
        """

        _logger.info(f'Binding to ({self._host!r}, {self._port:d})')
        try:
            # Start server
            server = await asyncio.start_server(self._handler, self._host, self._port)
        except OSError as e:
            # Binding error, server is probably already running
            raise MonInjBindError from e

        if server.sockets:
            # Report socket information
            for socket in server.sockets:
                _logger.debug(f'Serving on {socket.getsockname()}')
        else:
            # No socket available for unknown reason
            raise RuntimeError('Unable to open socket for unknown reason')

        async with server:
            # Run server forever
            await server.serve_forever()

    async def _handler(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        # Wait for hello message
        _logger.debug('Connection opened, expecting hello message')
        hello = await reader.readuntil()

        # Retrieve peer information and report
        peer = writer.get_extra_info("peername")
        if hello == self.ARTIQ_HELLO:
            _logger.info(f'Established connection with {peer}')
        else:
            _logger.warning(f'Received unexpected message {hello.decode()!r} from {peer}, '
                            f'maintaining connection anyway')

        # Dashboard is waiting for a byte of data
        # We go in read mode resulting in both the dashboard and us waiting forever while maintaining the connection
        await reader.read()

        # The connection was closed from the dashboard side, close it from this side too
        _logger.info(f'Closing connection with {peer}')
        writer.close()


if __name__ == '__main__':
    import argparse

    # Parse arguments
    parser = argparse.ArgumentParser(description='Start MonInj dummy service')
    parser.add_argument('--host', default='::1', type=str, help='The host to bind to')
    parser.add_argument('--port', default=1383, type=int, help='The port to bind to')
    parser.add_argument('-q', '--quiet', default=0, action='count', help='Decrease verbosity')
    parser.add_argument('-v', '--verbose', default=0, action='count', help='Increase verbosity')
    args = parser.parse_args()

    # Configure logger
    logging.basicConfig()
    _logger.setLevel(logging.WARNING + logging.DEBUG * (args.quiet - args.verbose))

    try:
        # Create and run service
        MonInjDummyService(args.host, args.port).run()
    except MonInjBindError:
        # Could not bind, exit silently
        _logger.info('Could not bind to address, service might already be running')
