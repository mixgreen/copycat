import os.path
import numpy as np

import artiq.coredevice.ttl  # type: ignore[import]
import artiq.coredevice.edge_counter
import artiq.coredevice.ad9910  # type: ignore[import]
import artiq.coredevice.ad9912  # type: ignore[import]

from dax.experiment import *
import dax.sim.test_case
from dax.sim.test_case import SignalNotSet
from dax.sim.signal import get_signal_manager
from dax.util.output import temp_dir


class PeekTestCaseTestCase(dax.sim.test_case.PeekTestCase):

    def setUp(self) -> None:
        # Construct environment
        self.sys = self.construct_env(_TestSystem, device_db=_DEVICE_DB)

    def test_peek(self):
        test_signals = {
            self.sys.ttl0: {'state': [0, 1, 'x', 'z']},
            self.sys.ttl1: {'input_freq': [10.0, 20.0], 'input_stdev': [1.0, 2.0], 'input_prob': [0.0, 0.5, 1.0]},
            self.sys.ec: {'count': [300, 500], 'input_freq': [10.0, 20.0], 'input_stdev': [1.0, 2.0]},
            self.sys.ad9910: {'freq': [200.4, 300.0], 'phase_mode': ['00', 'xx', 'z1']},
        }

        self.sys.ttl0.output()
        self.sys.ttl1.output()

        for scope, signals in test_signals.items():
            for name, values in signals.items():
                for v in values:
                    self.push(scope, name, v)
                    self.assertEqual(self.peek(scope, name), v)
                    self.expect(scope, name, v)
                    delay_mu(1)

    def test_expect_bool(self):
        zero_values = [0, False, np.int32(0), np.int64(0), 0.0]
        one_values = [1, True, np.int32(1), np.int64(1), 1.0]
        x_values = ['x', 'X', SignalNotSet]
        z_values = ['z', 'Z']
        test_data = [(v, values) for values in [zero_values, one_values] for v in values]

        scope = self.sys.ttl0
        name = 'state'

        # Test starting values
        for n in [name, 'direction', 'sensitivity']:
            for v in x_values:
                self.expect(scope, n, v)

        # Initialize device
        scope.output()
        for v in x_values:
            self.expect(scope, name, v)
        for v in one_values:
            self.expect(scope, 'direction', v)
        for v in z_values:
            self.expect(scope, 'sensitivity', v)

        for val, ref in test_data:
            with self.subTest(value=val, reference=ref):
                # Set new value
                delay(1 * us)
                scope.set_o(val)
                for _ in range(2):
                    # Test against reference values
                    for r in ref:
                        self.expect(scope, name, r)
                        self.assertEqual(self.peek(scope, name), r)
                    delay(1 * us)

    def test_expect_bool_bad_value(self):
        scope = self.sys.ttl0
        name = 'state'

        for val in ['a', None, 1.1, 0.5]:
            with self.subTest(value=val), self.assertRaises(ValueError):
                self.expect(scope, name, val)

    def test_expect_bool_vector(self):
        test_data = [
            ('XX', 'XX'),
            ('xx', 'xx'),
            ('xZ', 'Xz'),
            ('ZX', 'zx'),
            ('zx', 'ZX'),
            ('Z0', 'z0'),
            ('10', '10'),
            ('01', '01'),
            ('11', '11'),
            ('00', '00'),
        ]

        scope = self.sys.ad9910
        name = 'phase_mode'

        # Test starting values
        self.expect(scope, name, 'x')

        for val, ref in test_data:
            with self.subTest(value=val, reference=ref):
                # Set new value
                delay(1 * ns)
                self.push(scope, name, val)
                # Test value
                self.expect(scope, name, ref)
                self.assertEqual(self.peek(scope, name), val.lower())
                delay(1 * us)
                self.expect(scope, name, ref)

    def test_expect_bool_vector_bad_value(self):
        scope = self.sys.ad9910
        name = 'phase_mode'

        for val in [None, 1, 1.1, True, False, 'foo', '000']:
            with self.subTest(value=val), self.assertRaises(ValueError):
                self.expect(scope, name, val)

    def test_expect_int(self):
        test_data = [
            (1, 1),
            (np.int32(4), 4),
            (1, np.int32(1)),
            (np.int64(1), 1),
            (1, np.int64(1)),
            (np.int32(1), np.int64(1)),
            ('x', 'X'),
            ('x', SignalNotSet),
            ('Z', 'z'),
        ]

        scope = self.sys.ec
        name = 'count'
        signal = get_signal_manager().signal(scope, name)  # Get signal to easily change signal

        for v in ['z', 'Z']:
            self.expect(scope, name, v)

        for val, ref in test_data:
            with self.subTest(value=val, reference=ref):
                # Set new value
                delay(1 * us)
                signal.push(val)
                for _ in range(2):
                    # Test against reference values
                    self.expect(scope, name, ref)
                    self.assertEqual(self.peek(scope, name), val)
                    delay(1 * us)

    def test_expect_int_bad_value(self):
        scope = self.sys.ec
        name = 'count'

        for val in ['a', None, 1.1, 0.5]:
            with self.subTest(value=val), self.assertRaises(ValueError):
                self.expect(scope, name, val)

    def test_expect_float(self):
        test_data = [
            (1.0, 1.00),
            (99.2, 99.2),
            (-99.2, -99.2),
            (-99.2, np.float_(-99.2)),
            (np.float_(3), 3.0),
        ]

        scope = self.sys.ttl_clk  # This driver has no checks on its set() function
        name = 'freq'

        # Test starting values
        self.expect(scope, name, SignalNotSet)

        for val, ref in test_data:
            with self.subTest(value=val, reference=ref):
                # Set new value
                delay(1 * us)
                scope.set(val)
                # Test value
                self.expect(scope, name, ref)
                self.expect_close(scope, name, ref, places=7)
                delay(1 * us)
                self.expect(scope, name, ref)
                self.expect_close(scope, name, ref, places=7)

    def test_expect_float_bad_value(self):
        scope = self.sys.ttl_clk
        name = 'freq'

        for val in [True, False, None, '0.1']:
            with self.subTest(val=val), self.assertRaises(ValueError):
                self.expect(scope, name, val)

    def test_expect_is_close(self):
        test_data = [
            (99.2004, 99.2, 3),
            (-99.2004, -99.2, 3),
            (np.float_(99.2004), 99.2, 3),
            (99.2004, np.float_(99.2), 3),
            (np.float_(99.2004), np.float_(99.2), 3),
            (99.0004, 99, 3),
            (99.0004, np.int32(99), 3),
            (99.0004, np.int64(99), 3),
            (np.float_(99.0004), np.int32(99), 3),
            (99.00000004, 99, 7),
        ]

        scope = self.sys.ttl_clk  # This driver has no checks on its set() function
        name = 'freq'

        # Test starting values
        self.expect(scope, name, SignalNotSet)
        self.assertIs(self.peek(scope, name), SignalNotSet)

        for val, ref, places in test_data:
            with self.subTest(value=val, reference=ref, places=places):
                # Set new value
                delay(1 * us)
                scope.set(val)
                # Test value
                self.expect_close(scope, name, ref, places=places)
                delay(1 * us)
                self.expect_close(scope, name, ref, places=places)
                # Make the test fail
                with self.assertRaises(self.failureException, msg='expect() did not fail on almost equality'):
                    self.expect_close(scope, name, ref, places=places + 1)

    def test_expect_is_close_notset(self):
        scope = self.sys.ttl_clk  # This driver has no checks on its set() function
        name = 'freq'

        # Test starting values
        self.expect(scope, name, SignalNotSet)
        self.assertIs(self.peek(scope, name), SignalNotSet)

        with self.assertRaises(self.failureException, msg='expect() did not fail on almost equality'):
            # Fail on signal not set
            self.expect_close(scope, name, 0.1)

    def test_expect_is_close_bad_value(self):
        scope = self.sys.ttl_clk
        name = 'freq'

        for v in [SignalNotSet, 'x', 'z']:
            with self.assertRaises(TypeError, msg='Non-numerical value did not raise'):
                self.expect_close(scope, name, v, places=1)

    def test_expect_is_close_signal_type_error(self):
        signals = [
            (self.sys.ttl0, 'state'),  # bool
            (self.sys.ec, 'count'),  # int
            (self.sys.core_dma, 'play'),  # object
            (self.sys.core_dma, 'play_name'),  # str
        ]

        for scope, name in signals:
            with self.assertRaises(TypeError, msg='Non-float/int signal type did not raise'):
                self.expect_close(scope, name, 0, places=1)

    def test_expect_assertion(self):
        test_data = {
            self.sys.ttl0: ('state', [0, 1, 'z', 'Z', True, False, np.int32(0), np.int64(1)]),  # bool
            self.sys.ec: ('count', [0, 1, 'x', 'X', True, False, 99, -34, np.int32(655), np.int64(7)]),  # int
            self.sys.ad9912: ('freq', [1.7, -8.2, float(7.7), np.float_(300), np.float_(200)]),  # float
        }

        for scope, (name, values) in test_data.items():
            for v in values:
                with self.subTest(scope=scope, name=name, value=v):
                    with self.assertRaises(self.failureException, msg='Expect wrong value did not assert'):
                        self.expect(scope, name, v)

    def test_expect_notset(self):
        test_data = [
            (self.sys.ttl0, 'state', SignalNotSet),  # bool
            (self.sys.ttl0, 'state', 'x'),  # bool
            (self.sys.ttl0, 'state', 'X'),  # bool
            (self.sys.ec, 'count', 'z'),  # int with initialization value
            (self.sys.ec, 'count', 'Z'),  # int with initialization value
            (self.sys.ad9912, 'freq', SignalNotSet),  # float
            (self.sys.ad9912, 'freq', 'x'),  # float
            (self.sys.ad9912, 'freq', 'X'),  # float
        ]

        for scope, name, v in test_data:
            with self.subTest(scope=scope, name=name):
                self.expect(scope, name, v)

    def test_expect_signal_type_error(self):
        test_data = {
            'record': self.sys.core_dma,
            'play': self.sys.core_dma,
        }

        for name, scope in test_data.items():
            with self.subTest(scope=scope, name=name):
                with self.assertRaises(TypeError, msg='Non-expect signal type did not raise'):
                    self.expect(scope, name, None)

    def test_special_values(self):
        test_data = {
            self.sys.ttl0: ('state', ['x', 'X', SignalNotSet]),  # bool
            self.sys.ec: ('count', ['z', 'Z']),  # int
        }

        for scope, (name, values) in test_data.items():
            for v in values:
                with self.subTest(scope=scope, name=name, value=v):
                    self.expect(scope, name, v)

    def test_sequential(self):
        test_data = [1.0, 0.0, 1.0, 99.2, -99.2, np.float_(4)]

        scope = self.sys.ttl_clk  # This driver has no checks on its set() function
        name = 'freq'

        # Test starting values
        self.expect(scope, name, SignalNotSet)
        self.assertIs(self.peek(scope, name), SignalNotSet)

        with parallel:
            with sequential:
                for val in test_data:
                    # Set new value
                    delay(1 * us)
                    scope.set(val)

            with sequential:
                for val in test_data:
                    with self.subTest(msg='Test in parallel on exact time', value=val):
                        # Test value
                        delay(1 * us)
                        self.expect(scope, name, val)
                        self.assertEqual(self.peek(scope, name), val)

            with sequential:
                delay(0.5 * us)  # Shift time
                for val in test_data:
                    with self.subTest(msg='Test in parallel with delayed time', value=val):
                        # Test value
                        delay(1 * us)
                        self.expect(scope, name, val)
                        self.assertEqual(self.peek(scope, name), val)

    def test_push_buffer(self):
        test_data = [1 * kHz, 0 * kHz, 99.2 * kHz, 100 * kHz]

        scope = self.sys.ec
        name = 'input_freq'

        # Push the whole buffer
        self.push_buffer(scope, name, test_data)

        for _ in range(len(test_data)):
            delay_mu(1000)
            self.sys.ec.gate_rising_mu(self.sys.core.seconds_to_mu(1 * s))

        for freq in test_data:
            self.assertAlmostEqual(self.sys.ec.fetch_count(), freq, delta=1)
        self.assertEqual(self.peek(scope, name), test_data[-1])

    def test_write_vcd(self):
        file_name = 'foo.vcd'
        with temp_dir():
            self.write_vcd(file_name, self.sys.core)
            self.assertTrue(os.path.isfile(file_name))


class _TestSystem(DaxSystem):
    SYS_ID = 'unittest_system'
    SYS_VER = 0
    CORE_LOG_KEY = None
    DAX_INFLUX_DB_KEY = None

    def build(self, *args, **kwargs) -> None:
        super(_TestSystem, self).build(*args, **kwargs)

        self.ttl0 = self.get_device('ttl0', artiq.coredevice.ttl.TTLInOut)
        self.ttl1 = self.get_device('ttl1', artiq.coredevice.ttl.TTLInOut)
        self.ttls = [self.ttl0, self.ttl1]
        self.ec = self.get_device('ec', artiq.coredevice.edge_counter.EdgeCounter)
        self.ttl_clk = self.get_device('ttl_clk', artiq.coredevice.ttl.TTLInOut)

        self.ad9910 = self.get_device('ad9910', artiq.coredevice.ad9910.AD9910)
        self.ad9912 = self.get_device('ad9912', artiq.coredevice.ad9912.AD9912)


# Device DB
_DEVICE_DB = {
    # Core devices
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

    # TTL devices
    'ttl0': {
        'type': 'local',
        'module': 'artiq.coredevice.ttl',
        'class': 'TTLInOut',
        'arguments': {},
    },
    'ttl1': {
        'type': 'local',
        'module': 'artiq.coredevice.ttl',
        'class': 'TTLInOut',
        'arguments': {},
    },
    'ec': {
        'type': 'local',
        'module': 'artiq.coredevice.edge_counter',
        'class': 'EdgeCounter',
        'arguments': {},
    },
    'ttl_clk': {
        'type': 'local',
        'module': 'artiq.coredevice.ttl',
        'class': 'TTLClockGen',
        'arguments': {},
    },

    # CPLD and DDS devices
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
    "cpld10": {
        "type": "local",
        "module": "artiq.coredevice.urukul",
        "class": "CPLD",
        "arguments": {
            "refclk": 1e9,
            "clk_div": 1
        }
    },
    "ad9910": {
        "type": "local",
        "module": "artiq.coredevice.ad9910",
        "class": "AD9910",
        "arguments": {
            "pll_en": 0,
            "chip_select": 6,
            "cpld_device": "cpld10",
        }
    },
}
