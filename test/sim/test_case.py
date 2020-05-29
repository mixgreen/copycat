import unittest
import typing
import numpy as np

import artiq.coredevice.ttl  # type: ignore
import artiq.coredevice.edge_counter
import artiq.coredevice.ad9912  # type: ignore

from dax.experiment import DaxSystem
from dax.sim.signal import SignalNotSet
from dax.sim.test_case import PeekTestCase


class PeekTestCaseTestCase(PeekTestCase):

    def setUp(self) -> None:
        # Construct environment
        self.sys = self.construct_env(_TestSystem, device_db=_DEVICE_DB)

    def test_expect_assertion(self):
        test_data = {
            self.sys.ttl0: ('state', [0, 1, 'z', 'Z', True, False, np.int32(0), np.int64(1)]),  # bool
            self.sys.ec: ('count', [0, 1, 'x', 'X', True, False, 99, -34, np.int32(655), np.int64(7)]),  # int
            self.sys.ad9912: ('freq', [1.7, -8.2, 7.7, np.float(300)]),  # float
        }

        for scope, (signal, values) in test_data.items():
            for v in values:
                with self.subTest(scope=scope, signal=signal, value=v):
                    with self.assertRaises(self.failureException, msg='Expect wrong value did not assert'):
                        self.expect(scope, signal, v)

    def test_expect_notset(self):
        test_data = {
            self.sys.ttl0: ('state', SignalNotSet),  # bool
            self.sys.ec: ('count', 'z'),  # int
            self.sys.ad9912: ('freq', SignalNotSet),  # float
        }

        for scope, (signal, v) in test_data.items():
            with self.subTest(scope=scope, signal=signal):
                self.assertIsNone(self.expect(scope, signal, v))

    def test_expect_signal_type_error(self):
        test_data = {
            'record': self.sys.core_dma,
            'play': self.sys.core_dma,
        }

        for signal, scope in test_data.items():
            with self.subTest(scope=scope, signal=signal):
                with self.assertRaises(TypeError, msg='Non-expect signal type did not raise'):
                    self.expect(scope, signal, None)

    def test_special_values(self):
        test_data = {
            self.sys.ttl0: ('state', ['x', 'X', SignalNotSet]),  # bool
            self.sys.ec: ('count', ['z', 'Z']),  # int
        }

        for scope, (signal, values) in test_data.items():
            for v in values:
                with self.subTest(scope=scope, signal=signal, value=v):
                    self.assertIsNone(self.expect(scope, signal, v))


class _TestSystem(DaxSystem):
    SYS_ID = 'unittest_system'
    SYS_VER = 0

    def build(self, *args, **kwargs) -> None:
        super(_TestSystem, self).build(*args, **kwargs)

        self.ttl0 = self.get_device('ttl0', artiq.coredevice.ttl.TTLInOut)
        self.ttl1 = self.get_device('ttl1', artiq.coredevice.ttl.TTLInOut)
        self.ttls = [self.ttl0, self.ttl1]

        self.ec = self.get_device('ec', artiq.coredevice.edge_counter.EdgeCounter)

        self.ad9912 = self.get_device('ad9912', artiq.coredevice.ad9912.AD9912)


# Device DB
_DEVICE_DB = {
    # Core devices
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
    'ec': {
        'type': 'local',
        'module': 'artiq.coredevice.edge_counter',
        'class': 'EdgeCounter',
        'arguments': {},
    },
    "cpld": {
        "type": "local",
        "module": "artiq.coredevice.urukul",
        "class": "CPLD",
        "arguments": {
            "spi_device": "spi_urukul1",
            "sync_device": None,
            "io_update_device": "ttl_urukul1_io_update",
            "refclk": 1e9,
            "clk_sel": 1,
            "clk_div": 3
        }
    },
    "ad9912": {
        "type": "local",
        "module": "artiq.coredevice.ad9912",
        "class": "AD9912",
        "arguments": {
            "pll_n": 4,
            "chip_select": 5,
            "cpld_device": "cpld",
        }
    },
}

if __name__ == '__main__':
    unittest.main()
