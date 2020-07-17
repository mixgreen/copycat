"""
This module can be run directly to start the MonInj dummy service.

The MonInj dummy service can be used during simulation to reduce the number of error messages observed
in the ARTIQ dashboard. When using DAX.sim, the MonInj dummy service is started automatically.

Note that the MonInj dummy service does not implement the SiPyCo interface and is therefore
not implemented as an ARTIQ controller.
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

    def __init__(self, host: str, port: int, auto_close: int = 0):
        """Instantiate a new MonInj dummy service.

        Set `auto_close` to `0` to not serve connections forever.

        :param host: The host to bind to
        :param port: The port to bind to
        :param auto_close: Automatically close server after a number of connections
        """
        assert isinstance(host, str), 'Host most be of type str'
        assert isinstance(port, int), 'Port must be of type int'
        assert isinstance(auto_close, int), 'Auto close must be of type int'
        assert auto_close >= 0, 'Auto close must be greater or equal to 0'

        # Store attributes
        self._host: str = host
        self._port: int = port
        self._auto_close: int = auto_close

        if self._auto_close:
            _logger.debug(f'Server is configured to automatically close after {self._auto_close} connection(s)')

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

        _logger.info(f'Binding to ({self._host!r}, {self._port})')
        try:
            # Start server
            self._server = await asyncio.start_server(self._handler, self._host, self._port)
        except OSError as e:
            # Binding error, server might already be running
            raise MonInjBindError from e

        if self._server.sockets:
            # Report socket information
            for socket in self._server.sockets:
                _logger.debug(f'Serving on {socket.getsockname()}')
        else:
            # No socket available for unknown reason
            raise RuntimeError('Unable to open socket for unknown reason')

        async with self._server:
            # Run server forever
            asyncio.create_task(self._server.serve_forever())
            await self._server.wait_closed()

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

        if self._auto_close == 1:
            # Close the server
            _logger.info('Automatically closing server as maximum number of connections was reached')
            self._server.close()
        elif self._auto_close > 1:
            # Decrement counter
            self._auto_close -= 1
            _logger.debug(f'Serving {self._auto_close} more connection(s) before automatically closing')


if __name__ == '__main__':
    import argparse

    # Parse arguments
    parser = argparse.ArgumentParser(description='Start MonInj dummy service')
    parser.add_argument('--host', default='::1', type=str, help='The host to bind to')
    parser.add_argument('--port', default=1383, type=int, help='The port to bind to')
    parser.add_argument('-q', '--quiet', default=0, action='count', help='Decrease verbosity')
    parser.add_argument('-v', '--verbose', default=0, action='count', help='Increase verbosity')
    parser.add_argument('--auto-close', default=0, type=int,
                        help='Automatically close server after a number of connections')
    args = parser.parse_args()

    # Configure logger
    logging.basicConfig()
    _logger.setLevel(logging.WARNING + logging.DEBUG * (args.quiet - args.verbose))

    try:
        # Create and run service
        MonInjDummyService(host=args.host, port=args.port, auto_close=args.auto_close).run()
    except MonInjBindError:
        # Could not bind, exit silently
        _logger.info('Could not bind to address (service might already be running)')
