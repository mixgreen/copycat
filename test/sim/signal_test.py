import unittest
import typing
import os.path
import numpy as np

from artiq.language.core import now_mu, at_mu, delay, delay_mu, parallel, sequential
from artiq.language.units import *
import artiq.coredevice.ttl  # type: ignore[import]
import artiq.coredevice.edge_counter
import artiq.coredevice.ad9910  # type: ignore[import]
import artiq.coredevice.ad9912  # type: ignore[import]
import artiq.coredevice.ad53xx  # type: ignore[import]
import artiq.coredevice.zotino  # type: ignore[import]

from dax.experiment import DaxSystem
from dax.sim import enable_dax_sim
from dax.sim.device import DaxSimDevice
from dax.sim.signal import get_signal_manager, Signal, SignalNotSetError, SignalNotFoundError, \
    DaxSignalManager, NullSignalManager, VcdSignalManager, PeekSignalManager
from dax.sim.coredevice.core import BaseCore
from dax.util.artiq import get_managers
from dax.util.output import temp_dir


class NullSignalManagerTestCase(unittest.TestCase):
    SIGNAL_MANAGER: typing.ClassVar[str] = 'null'
    SIGNAL_MANAGER_CLASS: typing.ClassVar[typing.Type[DaxSignalManager]] = NullSignalManager
    MIN_HORIZON: typing.ClassVar[typing.Optional[int]]
    MIN_HORIZON = 0  # The null signal manager starts with a minimum horizon of 0 due to init events

    def setUp(self) -> None:
        ddb = enable_dax_sim(_DEVICE_DB.copy(), enable=True, output=self.SIGNAL_MANAGER, moninj_service=False)
        self.managers = get_managers(ddb)
        self.sys = _TestSystem(self.managers)
        self.sm = get_signal_manager()

    def tearDown(self) -> None:
        self.sm.close()
        self.managers.close()

    def test_signal_manager_type(self):
        # Verify the signal manager type
        self.assertIsInstance(self.sm, self.SIGNAL_MANAGER_CLASS)

    def test_signal_manager_reentrant_close(self) -> None:
        self.sm.close()
        self.sm.close()  # Close twice, should not raise an exception

    def _get_signal_registrations(self):
        ad53xx_signals = {'init'} | {f'v_out_{i}' for i in range(32)} | {f'v_offset_{i}' for i in range(32)} | {
            f'gain_{i}' for i in range(32)}
        return {
            self.sys.core: {'reset'},
            self.sys.core_dma: {'record', 'play', 'play_name'},
            self.sys.ttl0: {'state', 'direction', 'sensitivity', 'input_freq', 'input_stdev', 'input_prob'},
            self.sys.ttl1: {'state', 'direction', 'sensitivity', 'input_freq', 'input_stdev', 'input_prob'},
            self.sys.ec: {'count', 'input_freq', 'input_stdev'},
            self.sys.ad9910.cpld: {'init', 'init_att', 'sw'} | {f'att_{i}' for i in range(4)},
            self.sys.ad9910: {'init', 'freq', 'phase', 'phase_mode', 'amp'},
            self.sys.ad9912: {'init', 'freq', 'phase'},
            self.sys.ad53xx: ad53xx_signals,
            self.sys.zotino: ad53xx_signals | {'led'},
        }

    def test_signal_repr_str(self):
        key = '_device_key'
        scope = DaxSimDevice(self.managers.device_mgr, _key=key)
        name = 'foo'
        signal = self.sm.register(scope, name, type_=object)

        ref = f'{key}.{name}'
        for val in [str(signal), repr(signal), f'{signal}']:
            self.assertEqual(val, ref)

    def test_register(self):
        scope = DaxSimDevice(self.managers.device_mgr, _key='_device_key')
        name = 'foo'

        with self.assertRaises(SignalNotFoundError):
            self.sm.signal(scope, name)

        signal = self.sm.register(scope, name, type_=object)
        self.assertIs(self.sm.signal(scope, name), signal)

        with self.assertRaises(LookupError):
            self.sm.register(scope, name, type_=object)

    def test_registered_devices(self):
        for scope, names in self._get_signal_registrations().items():
            for n in names:
                s = self.sm.signal(scope, n)
                self.assertIs(s, self.sm.signal(scope, n), 'Function returned a different signal object')

    def test_iter(self):
        self.assertGreater(len(self.sm), 200)  # Test that the iterator is not empty (exact number is not important)
        self.assertEqual(len(list(self.sm)), len(self.sm))

    def test_pull_not_set(self):
        for ttl in self.sys.ttl_list:
            for s in ['state', 'direction', 'sensitivity']:
                with self.assertRaises(SignalNotSetError):
                    self.sm.signal(ttl, s).pull()

    def test_pull_init(self):
        ttl_initialized = {'input_freq': 0.0, 'input_stdev': 0.0, 'input_prob': 0.0}
        initialized = {
            self.sys.ttl0: ttl_initialized,
            self.sys.ttl1: ttl_initialized,
            self.sys.ec: {'count': 'z', 'input_freq': 0.0, 'input_stdev': 0.0},
        }

        for scope, names in self._get_signal_registrations().items():
            init = initialized.get(scope, ())
            for n in names:
                signal = self.sm.signal(scope, n)
                if n in init:
                    self.assertEqual(signal.pull(), init[n])
                else:
                    with self.assertRaises(SignalNotSetError, msg=f'Signal: {signal}'):
                        signal.pull()

    def test_push(self, *, pull=False):
        test_data = {
            self.sys.ttl0._state: [0, 1, 'x', 'X', 'z', 'Z', True, False, np.int32(0), np.int64(1)],  # bool
            # Python hash(0) == hash(0.0), see https://docs.python.org/3/library/functions.html#hash
            self.sys.ttl1._state: [0.0, 1.0],  # bool, side effect of Python hash()
            self.sys.ec._count: [0, 1, 'x', 'X', 'z', 'Z', True, False, 99, -34, np.int32(655), np.int64(7)],  # int
            self.sys.ad9912._freq: [1.7, -8.2, 7.7, np.float_(300), np.float_(200)],  # float
            self.sys.core_dma._dma_record: ['foo', 'bar', ''],  # str
            self.sys.core_dma._dma_play: [True],  # object
        }

        for signal, values in test_data.items():
            with self.subTest(signal=signal):
                for v in values:
                    self.assertIsNone(signal.push(v)),

                    if pull:
                        self.assertEqual(signal.pull(), v)
                        self.assertEqual(signal.pull(offset=1), v)
                        with self.assertRaises(SignalNotSetError):
                            signal.pull(offset=-1)

    def test_push_bool_vector(self, *, pull=False):
        test_data = {
            self.sys.ad9910._phase_mode: ['xx', '10', '1z', 'XX', '00', 'ZZ'],  # bool vector
        }

        for signal, values in test_data.items():
            for v in values:
                with self.subTest(signal=signal, value=v):
                    signal.push(v)

                    if pull:
                        ref = v.lower()  # string is lowered when stored
                        self.assertEqual(signal.pull(), ref)
                        self.assertEqual(signal.pull(offset=1), ref)
                        with self.assertRaises(SignalNotSetError):
                            signal.pull(offset=-1)

    def test_flush_close(self):
        # Push signals
        self.test_push()
        self.test_push_bool_vector()
        # Flush and close
        self.sm.flush(self.sys.core.ref_period)
        self.sm.close()

    def test_push_bad(self):
        test_data = {
            self.sys.ttl0._state: ['foo', '00', np.int32(9), np.int64(-1), 0.4, None, '0', '1'],  # bool
            self.sys.ec._count: ['foo', 0.3, object, complex(6, 7), None, '0', '1'],  # int
            self.sys.ad9912._freq: [True, 1, object, complex(6, 7), None, '1'],  # float
            self.sys.core_dma._dma_record: [True, 1, object, complex(6, 7), 1.1, None],  # str
            self.sys.core_dma._dma_play: [3, 4.4, 'a', object, None],  # object
        }

        for signal, values in test_data.items():
            for v in values:
                with self.subTest(signal=signal, value=v):
                    with self.assertRaises(ValueError, msg='Bad push value for signal did not raise'):
                        signal.push(v),

    def test_push_bool_vector_bad(self):
        test_data = {
            self.sys.ad9910._phase_mode: ['foo', 0.3, object, complex(6, 7), None, 4, 9, -1, 1.0, 1, 2, 3, 0, True,
                                          False, np.int64(2), 'x', 'z', '000', '10z', '0', 'a0', '1g'],  # bool vector
        }

        for signal, values in test_data.items():
            for v in values:
                with self.subTest(signal=signal, value=v):
                    with self.assertRaises(ValueError, msg='Bad push value for signal did not raise'):
                        signal.push(v)

    def test_horizon_no_events(self):
        self.assertEqual(self.sm.horizon(), 0)
        t_sum = 0

        for t in [99, 1000000, -1000, -930]:  # Do not go negative, init events will limit the horizon to >=0
            delay_mu(t)
            t_sum += t
            self.assertEqual(self.sm.horizon(), t_sum)

    def test_horizon_reset_negative_timestamp(self, t=-100):
        min_horizon = t if self.MIN_HORIZON is None else max(t, self.MIN_HORIZON)

        delay_mu(t)
        self.assertEqual(self.sm.horizon(), min_horizon)
        self.assertEqual(now_mu(), t)
        self.sys.core.reset()
        self.assertEqual(self.sm.horizon(), min_horizon + BaseCore.DEFAULT_RESET_TIME_MU)
        self.assertEqual(now_mu(), min_horizon + BaseCore.DEFAULT_RESET_TIME_MU)

    def test_horizon_with_event(self, t=1000):  # Test disabled by default, must be called manually
        # Forward and reverse time, horizon will move along
        delay_mu(t)
        self.assertEqual(self.sm.horizon(), t)
        delay_mu(-t)
        self.assertEqual(self.sm.horizon(), 0)

        # Forward time, event, and reverse time, horizon will stay
        delay_mu(t)
        self.assertEqual(self.sm.horizon(), t)
        self.sys.ttl0.on()
        delay_mu(-t)
        self.assertEqual(self.sm.horizon(), t)

    def test_horizon_reset(self, t=1000):  # Test disabled by default, must be called manually
        # Forward and reverse time, horizon will move along
        delay_mu(t)
        self.assertEqual(self.sm.horizon(), t)
        delay_mu(-t)
        self.assertEqual(self.sm.horizon(), 0)
        self.assertEqual(now_mu(), 0)
        # Reset, which inserts events
        self.sys.core.reset()
        self.assertEqual(self.sm.horizon(), BaseCore.DEFAULT_RESET_TIME_MU)
        self.assertEqual(now_mu(), BaseCore.DEFAULT_RESET_TIME_MU)
        # Reverts still works, but the horizon will stay
        at_mu(0)
        self.assertEqual(self.sm.horizon(), BaseCore.DEFAULT_RESET_TIME_MU)
        self.assertEqual(now_mu(), 0)

        # Reset works from the horizon
        self.sys.core.reset()
        self.assertEqual(self.sm.horizon(), BaseCore.DEFAULT_RESET_TIME_MU * 2)
        self.assertEqual(now_mu(), BaseCore.DEFAULT_RESET_TIME_MU * 2)


class VcdSignalManagerTestCase(NullSignalManagerTestCase):
    SIGNAL_MANAGER = 'vcd'
    SIGNAL_MANAGER_CLASS = VcdSignalManager
    MIN_HORIZON = 0  # The VCD signal manager starts with a fixed horizon of 0 (in addition to init events)

    def setUp(self) -> None:
        # Enter temp dir
        self._temp_dir = temp_dir()
        self._temp_dir.__enter__()
        # Call super
        super(VcdSignalManagerTestCase, self).setUp()

    def tearDown(self) -> None:
        # Call super
        super(VcdSignalManagerTestCase, self).tearDown()
        # Exit temp dir
        self._temp_dir.__exit__(None, None, None)

    def test_horizon_break_realtime(self, t=1000):
        # Forward and reverse time, horizon will move along
        delay_mu(t)
        self.assertEqual(self.sm.horizon(), t)
        delay_mu(-t)
        self.assertEqual(self.sm.horizon(), 0)
        self.assertEqual(now_mu(), 0)
        # Break realtime, which does NOT insert an event but does flush
        self.sys.core.break_realtime()
        self.assertEqual(self.sm.horizon(), BaseCore.DEFAULT_RESET_TIME_MU)
        self.assertEqual(now_mu(), BaseCore.DEFAULT_RESET_TIME_MU)
        # Reverts still works, but the horizon does not shift due to the flush
        at_mu(0)
        self.assertEqual(self.sm.horizon(), BaseCore.DEFAULT_RESET_TIME_MU)
        self.assertEqual(now_mu(), 0)

        # Forward time, event, and reverse time, horizon will stay at the event
        at_mu(BaseCore.DEFAULT_RESET_TIME_MU)
        delay_mu(t)
        self.assertEqual(self.sm.horizon(), t + BaseCore.DEFAULT_RESET_TIME_MU)
        self.assertEqual(now_mu(), t + BaseCore.DEFAULT_RESET_TIME_MU)
        self.sys.ttl0.on()
        delay_mu(-t)
        self.assertEqual(self.sm.horizon(), t + BaseCore.DEFAULT_RESET_TIME_MU)
        self.assertEqual(now_mu(), BaseCore.DEFAULT_RESET_TIME_MU)
        # Break realtime works from the horizon
        self.sys.core.break_realtime()
        self.assertEqual(self.sm.horizon(), t + BaseCore.DEFAULT_RESET_TIME_MU * 2)
        self.assertEqual(now_mu(), t + BaseCore.DEFAULT_RESET_TIME_MU * 2)

    def test_signal_types(self):
        import dax.sim.signal
        self.assertSetEqual(set(dax.sim.signal.VcdSignal._VCD_TYPE), set(Signal._SIGNAL_TYPES))


class PeekSignalManagerTestCase(NullSignalManagerTestCase):
    SIGNAL_MANAGER = 'peek'
    SIGNAL_MANAGER_CLASS = PeekSignalManager
    MIN_HORIZON = 0  # The Peek signal manager starts with a minimum horizon of 0 due to init events

    def test_horizon_no_events(self):
        super(PeekSignalManagerTestCase, self).test_horizon_no_events()

        at_mu(0)
        self.assertEqual(self.sm.horizon(), 0)
        t_sum = 0

        for t in [-200, 99, 1000000, -1000, -930, -100000000]:  # Go negative, init events will limit the horizon to >=0
            delay_mu(t)
            t_sum += t
            self.assertEqual(self.sm.horizon(), max(0, t_sum))

    def test_horizon_break_realtime(self, t=1000):
        # Forward and reverse time, horizon will move along
        delay_mu(t)
        self.assertEqual(self.sm.horizon(), t)
        delay_mu(-t)
        self.assertEqual(self.sm.horizon(), 0)
        self.assertEqual(now_mu(), 0)
        # Break realtime, which does NOT insert an event but does flush
        self.sys.core.break_realtime()
        self.assertEqual(self.sm.horizon(), BaseCore.DEFAULT_RESET_TIME_MU)
        self.assertEqual(now_mu(), BaseCore.DEFAULT_RESET_TIME_MU)
        # Reverts still work
        at_mu(0)
        self.assertEqual(self.sm.horizon(), 0)
        self.assertEqual(now_mu(), 0)

        # Forward time, event, and reverse time, horizon will stay at the event
        delay_mu(t)
        self.assertEqual(self.sm.horizon(), t)
        self.assertEqual(now_mu(), t)
        self.sys.ttl0.on()
        delay_mu(-t)
        self.assertEqual(self.sm.horizon(), t)
        self.assertEqual(now_mu(), 0)
        # Break realtime works from the horizon
        self.sys.core.break_realtime()
        self.assertEqual(self.sm.horizon(), t + BaseCore.DEFAULT_RESET_TIME_MU)
        self.assertEqual(now_mu(), t + BaseCore.DEFAULT_RESET_TIME_MU)

    def test_push_pull(self):
        self.test_push(pull=True)

    def test_push_pull_bool_vector(self):
        self.test_push_bool_vector(pull=True)

    def test_pull_1(self):
        delay(1 * us)
        self.test_pull_not_set()

        # Set direction
        for ttl in self.sys.ttl_list:
            ttl.input()

        for ttl in self.sys.ttl_list:
            self.assertEqual(self.sm.signal(ttl, 'direction').pull(), 0)
            self.assertEqual(self.sm.signal(ttl, 'sensitivity').pull(), 0)
            self.assertEqual(self.sm.signal(ttl, 'state').pull(), 'z')

    def test_pull_2(self):
        delay(1 * us)
        self.test_pull_not_set()

        # Set direction
        for ttl in self.sys.ttl_list:
            ttl.output()

        for ttl in self.sys.ttl_list:
            self.assertEqual(self.sm.signal(ttl, 'direction').pull(), 1)
            self.assertEqual(self.sm.signal(ttl, 'sensitivity').pull(), 'z')
            self.assertEqual(self.sm.signal(ttl, 'state').pull(), 'x')

    def test_pull_after_delay(self):
        delay(1 * us)
        self.test_pull_not_set()

        # Set direction
        for ttl in self.sys.ttl_list:
            ttl.output()

        delay(10 * us)

        for ttl in self.sys.ttl_list:
            self.assertEqual(self.sm.signal(ttl, 'direction').pull(), 1)
            self.assertEqual(self.sm.signal(ttl, 'sensitivity').pull(), 'z')
            self.assertEqual(self.sm.signal(ttl, 'state').pull(), 'x')

    def test_pull_negative_delay(self):
        delay(10 * us)
        self.test_pull_not_set()

        # Set direction
        for ttl in self.sys.ttl_list:
            ttl.output()

        delay_mu(-1)
        self.test_pull_not_set()

        delay_mu(1)
        for ttl in self.sys.ttl_list:
            self.assertEqual(self.sm.signal(ttl, 'direction').pull(), 1)
            self.assertEqual(self.sm.signal(ttl, 'sensitivity').pull(), 'z')
            self.assertEqual(self.sm.signal(ttl, 'state').pull(), 'x')

    def test_pull_negative_delay_arg(self):
        delay(10 * us)
        self.test_pull_not_set()

        # Set direction
        for ttl in self.sys.ttl_list:
            ttl.output()

        for ttl in self.sys.ttl_list:
            for s in ['state', 'direction', 'sensitivity']:
                with self.assertRaises(SignalNotSetError):
                    self.sm.signal(ttl, s).pull(time=now_mu() - 1)

        for ttl in self.sys.ttl_list:
            self.assertEqual(self.sm.signal(ttl, 'direction').pull(), 1)
            self.assertEqual(self.sm.signal(ttl, 'sensitivity').pull(), 'z')
            self.assertEqual(self.sm.signal(ttl, 'state').pull(), 'x')

    def test_pull_overwrite(self):
        delay(10 * us)
        self.test_pull_not_set()

        # Set direction
        for ttl in self.sys.ttl_list:
            ttl.output()
        for ttl in self.sys.ttl_list:
            ttl.input()

        for ttl in self.sys.ttl_list:
            self.assertEqual(self.sm.signal(ttl, 'direction').pull(), 0)
            self.assertEqual(self.sm.signal(ttl, 'sensitivity').pull(), 0)
            self.assertEqual(self.sm.signal(ttl, 'state').pull(), 'z')

    def test_pull_many_changes(self):
        delay(10 * us)
        self.test_pull_not_set()

        # Set direction
        for ttl in self.sys.ttl_list:
            ttl.output()
        for ttl in self.sys.ttl_list:
            ttl.input()
        delay(3 * us)
        for ttl in self.sys.ttl_list:
            ttl.output()

        for ttl in self.sys.ttl_list:
            for i in range(10):
                delay(2 * us)
                ttl.set_o(i % 2)
                self.assertEqual(self.sm.signal(ttl, 'direction').pull(), 1)
                self.assertEqual(self.sm.signal(ttl, 'sensitivity').pull(), 'z')
                self.assertEqual(self.sm.signal(ttl, 'state').pull(), i % 2)

    def test_pull_parallel(self):
        delay(10 * us)
        self.test_pull_not_set()

        # Set direction
        for ttl in self.sys.ttl_list:
            ttl.output()

        for ttl in self.sys.ttl_list:
            with parallel:
                with sequential:
                    for i in range(10):
                        delay(2 * us)
                        ttl.set_o(i % 2)
                with sequential:
                    for i in range(10):
                        delay(2 * us)
                        self.assertEqual(self.sm.signal(ttl, 'direction').pull(), 1)
                        self.assertEqual(self.sm.signal(ttl, 'sensitivity').pull(), 'z')
                        self.assertEqual(self.sm.signal(ttl, 'state').pull(), i % 2)

    def test_push_buffer(self):
        test_data = {
            self.sys.ttl0._state: [0, 1, 'x', 'X', 'z', 'Z', True, False, np.int32(0), np.int64(1)],  # bool
            self.sys.ec._count: [0, 1, 'x', 'X', 'z', 'Z', True, False, 99, -34, np.int32(655), np.int64(7)],  # int
            self.sys.ad9912._freq: [1.7, -8.2, 7.7, np.float_(300), np.float_(200)],  # float
            self.sys.core_dma._dma_record: ['foo', 'bar', ''],  # str
            self.sys.core_dma._dma_play: [True],  # object
        }
        delay_t = 100

        for signal, buffer in test_data.items():
            with self.subTest(signal=signal):
                signal.push_buffer(buffer)
                for v in buffer:
                    self.assertEqual(signal.pull(), v)
                    delay_mu(delay_t)
                end_t = now_mu()

                for v in reversed(buffer):
                    delay_mu(-delay_t)
                    self.assertEqual(signal.pull(), v)

                # Restore time
                at_mu(end_t)

    def test_write_vcd(self):
        file_name = 'foo.vcd'
        with temp_dir():
            self.sm.write_vcd(file_name, self.sys.core.ref_period)
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
        self.ttl_list = [self.ttl0, self.ttl1]
        self.ec = self.get_device('ec', artiq.coredevice.edge_counter.EdgeCounter)

        self.ad9910 = self.get_device('ad9910', artiq.coredevice.ad9910.AD9910)
        self.ad9912 = self.get_device('ad9912', artiq.coredevice.ad9912.AD9912)

        self.ad53xx = self.get_device('ad53xx', artiq.coredevice.ad53xx.AD53xx)
        self.zotino = self.get_device('zotino', artiq.coredevice.zotino.Zotino)


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

    # TTL and edge counter
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

    # Urukul CPLD and DDS devices
    'io_update': {
        'type': 'local',
        'module': 'artiq.coredevice.ttl',
        'class': 'TTLOut',
        'arguments': {},
    },
    "cpld": {
        "type": "local",
        "module": "artiq.coredevice.urukul",
        "class": "CPLD",
        "arguments": {
            "spi_device": "spi_urukul1",
            "sync_device": None,
            "io_update_device": "io_update",
            "refclk": 1e9,
            "clk_sel": 1,
            "clk_div": 3
        }
    },
    "ad9910": {
        "type": "local",
        "module": "artiq.coredevice.ad9910",
        "class": "AD9910",
        "arguments": {
            "pll_en": 0,
            "chip_select": 4,
            "cpld_device": "cpld",
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

    # Multi-channel DAC
    "ad53xx": {
        "type": "local",
        "module": "artiq.coredevice.ad53xx",
        "class": "AD53xx",
        "arguments": {}
    },
    "zotino": {
        "type": "local",
        "module": "artiq.coredevice.zotino",
        "class": "Zotino",
        "arguments": {}
    },
}
