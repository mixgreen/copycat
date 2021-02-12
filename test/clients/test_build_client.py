import typing
import unittest

import artiq.coredevice.edge_counter

from dax.experiment import *
from dax.sim import enable_dax_sim
from dax.util.artiq import get_managers

import dax.modules.rtio_benchmark
import dax.modules.rpc_benchmark
import dax.interfaces.detection

import dax.clients.gtkwave
import dax.clients.introspect
import dax.clients.pmt_monitor
import dax.clients.program
import dax.clients.rpc_benchmark
import dax.clients.rtio_benchmark
import dax.clients.system_benchmark


def _get_managers(**kwargs):
    return get_managers(enable_dax_sim(ddb=_DEVICE_DB, enable=True, logging_level=30,
                                       output='null', moninj_service=False), **kwargs)


class _TestDetectionModule(DaxModule, dax.interfaces.detection.DetectionInterface):

    def build(self):
        self.pmt_array: typing.List[artiq.coredevice.edge_counter.EdgeCounter] = \
            [self.get_device('ec0', artiq.coredevice.edge_counter.EdgeCounter)]

    def init(self) -> None:
        pass

    def post_init(self) -> None:
        pass

    def get_pmt_array(self) -> typing.List[artiq.coredevice.edge_counter.EdgeCounter]:
        return self.pmt_array

    def get_state_detection_threshold(self) -> int:
        return 2

    def get_default_detection_time(self) -> float:
        return 100 * us


class _TestSystem(DaxSystem):
    SYS_ID = 'unittest_system'
    SYS_VER = 0

    def build(self, *args: typing.Any, **kwargs: typing.Any) -> None:
        # Call super
        super(_TestSystem, self).build(*args, **kwargs)
        # Create modules
        _TestDetectionModule(self, 'detection')
        dax.modules.rtio_benchmark.RtioLoopBenchmarkModule(self, 'rtio_bench', ttl_out='ttl0', ttl_in='ttl1')
        dax.modules.rpc_benchmark.RpcBenchmarkModule(self, 'rpc_bench')


class BuildClientTestCase(unittest.TestCase):
    """Test case that builds and initializes clients as a basic test."""

    _CLIENTS: typing.List[typing.Tuple[typing.Any, typing.Dict[str, typing.Any], bool]] = [
        (dax.clients.gtkwave.GTKWaveSaveGenerator, {}, True),
        (dax.clients.introspect.Introspect, {}, True),
        (dax.clients.pmt_monitor.PmtMonitor, {}, True),
        (dax.clients.pmt_monitor.MultiPmtMonitor, {}, True),
        (dax.clients.program.ProgramClient, {'file': ''}, False),
        (dax.clients.rpc_benchmark.RpcBenchmarkLatency, {}, True),
        (dax.clients.rpc_benchmark.RpcBenchmarkAsyncThroughput, {}, True),
        (dax.clients.rtio_benchmark.RtioBenchmarkEventThroughput, {}, True),
        (dax.clients.rtio_benchmark.RtioBenchmarkEventBurst, {}, True),
        (dax.clients.rtio_benchmark.RtioBenchmarkDmaThroughput, {}, True),
        (dax.clients.rtio_benchmark.RtioBenchmarkLatencyCoreRtio, {}, True),
        (dax.clients.rtio_benchmark.RtioBenchmarkInputBufferSize, {}, True),
        (dax.clients.rtio_benchmark.RtioBenchmarkLatencyRtioCore, {}, True),
        (dax.clients.rtio_benchmark.RtioBenchmarkLatencyRtt, {}, True),
        (dax.clients.system_benchmark.SystemBenchmarkDaxInit, {}, True),
        (dax.clients.system_benchmark.SystemBenchmarkDaxInitProfile, {}, True),
    ]
    """List of client types."""

    _CUSTOM_CLIENTS: typing.List[typing.Tuple[typing.Any, typing.Dict[str, typing.Any]]] = [
        (dax.clients.system_benchmark.SystemBenchmarkBuildProfile, {}),
    ]
    """List of custom client types (not subclasses of DaxClient)."""

    def test_build_client(self) -> None:
        for client_type, kwargs, prepare in self._CLIENTS:
            with self.subTest(client_type=client_type.__name__):
                # noinspection PyTypeChecker
                class _InstantiatedClient(client_type(_TestSystem)):  # type: ignore[misc]
                    pass

                with _get_managers(**kwargs) as managers:
                    # Create client
                    client = _InstantiatedClient(managers)
                    self.assertIsInstance(client, DaxClient)
                    self.assertIsInstance(client, Experiment)

                    # Get system
                    system = client.registry.find_module(DaxSystem)
                    self.assertIsInstance(system, _TestSystem)

                    if prepare and client.DAX_INIT:
                        # Call the prepare function
                        client.prepare()
                        # Initialize system
                        self.assertIsNone(system.dax_init())

    def test_build_custom_clients(self) -> None:
        for client_type, kwargs in self._CUSTOM_CLIENTS:
            with self.subTest(client_type=client_type.__name__):
                # noinspection PyTypeChecker
                class _InstantiatedClient(client_type(_TestSystem)):  # type: ignore[misc]
                    pass

                with _get_managers(**kwargs) as managers:
                    # Create client
                    client = _InstantiatedClient(managers)
                    self.assertIsInstance(client, Experiment)


_DEVICE_DB = {
    # Core device
    'core': {
        'type': 'local',
        'module': 'artiq.coredevice.core',
        'class': 'Core',
        'arguments': {'host': None, 'ref_period': 1e-9}
    },
    'core_cache': {
        'type': 'local',
        'module': 'artiq.coredevice.cache',
        'class': 'CoreCache'
    },
    'core_dma': {
        'type': 'local',
        'module': 'artiq.coredevice.dma',
        'class': 'CoreDMA'
    },

    # Generic TTL
    'ttl0': {
        'type': 'local',
        'module': 'artiq.coredevice.ttl',
        'class': 'TTLInOut',
        'arguments': {'channel': 0},
    },
    'ttl1': {
        'type': 'local',
        'module': 'artiq.coredevice.ttl',
        'class': 'TTLInOut',
        'arguments': {'channel': 1},
    },

    # Edge counters
    'ec0': {
        'type': 'local',
        'module': 'artiq.coredevice.edge_counter',
        'class': 'EdgeCounter',
        'arguments': {'channel': 2},
    }
}

if __name__ == '__main__':
    unittest.main()
