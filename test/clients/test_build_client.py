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
import dax.clients.rpc_benchmark
import dax.clients.rtio_benchmark
import dax.clients.system_benchmark


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

    _CLIENTS = [
        dax.clients.gtkwave.GTKWaveSaveGenerator,
        dax.clients.introspect.Introspect,
        dax.clients.pmt_monitor.PmtMonitor,
        dax.clients.pmt_monitor.MultiPmtMonitor,
        dax.clients.rpc_benchmark.RpcBenchmarkLatency,
        dax.clients.rpc_benchmark.RpcBenchmarkAsyncThroughput,
        dax.clients.rtio_benchmark.RtioBenchmarkEventThroughput,
        dax.clients.rtio_benchmark.RtioBenchmarkEventBurst,
        dax.clients.rtio_benchmark.RtioBenchmarkDmaThroughput,
        dax.clients.rtio_benchmark.RtioBenchmarkLatencyCoreRtio,
        dax.clients.rtio_benchmark.RtioBenchmarkInputBufferSize,
        dax.clients.rtio_benchmark.RtioBenchmarkLatencyRtioCore,
        dax.clients.rtio_benchmark.RtioBenchmarkLatencyRtt,
        dax.clients.system_benchmark.SystemBenchmarkDaxInit,
        dax.clients.system_benchmark.SystemBenchmarkDaxInitProfile,
    ]
    """List of client types."""

    _CUSTOM_CLIENTS = [
        dax.clients.system_benchmark.SystemBenchmarkBuildProfile,
    ]
    """List of custom client types (not subclasses of DaxClient)."""

    def test_build_client(self):
        for client_type in self._CLIENTS:
            with self.subTest(client_type=client_type.__name__):
                class _InstantiatedClient(client_type(_TestSystem)):
                    pass

                # Create client
                manager = get_managers(
                    enable_dax_sim(ddb=_device_db, enable=True, logging_level=30, output='null', moninj_service=False))
                client = _InstantiatedClient(manager)
                self.assertIsInstance(client, DaxClient)

                # Get system
                system = client.registry.find_module(DaxSystem)
                self.assertIsInstance(system, _TestSystem)

                if client.DAX_INIT:
                    # Call the prepare function
                    client.prepare()
                    # Initialize system
                    self.assertIsNone(system.dax_init())

    def test_build_custom_clients(self):
        for client_type in self._CUSTOM_CLIENTS:
            with self.subTest(client_type=client_type.__name__):
                class _InstantiatedClient(client_type(_TestSystem)):
                    pass

                # Create client
                manager = get_managers(
                    enable_dax_sim(ddb=_device_db, enable=True, logging_level=30, output='null', moninj_service=False))
                _InstantiatedClient(manager)


_device_db = {
    # Core device
    'core': {
        'type': 'local',
        'module': 'artiq.coredevice.core',
        'class': 'Core',
        'arguments': {'host': '0.0.0.0', 'ref_period': 1e-9}
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
