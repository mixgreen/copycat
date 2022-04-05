import unittest
import unittest.mock
import typing

from dax.base.system import DaxSystem
from dax.util.artiq import get_managers
from dax.util.output import temp_dir
from dax.sim import enable_dax_sim
from dax.sim.signal import get_signal_manager, set_signal_manager, Signal, DaxSignalManager
from dax.util.gtkwave import GTKWSaveGenerator

_DEVICE_DB: typing.Dict[str, typing.Any] = {
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
}


class _TestSystem(DaxSystem):
    SYS_ID = 'unittest_system'
    SYS_VER = 0
    CORE_LOG_KEY = None
    DAX_INFLUX_DB_KEY = None


class GTKWaveTestCase(unittest.TestCase):

    def test_signal_types(self):
        self.assertSetEqual(set(Signal._SIGNAL_TYPES), set(GTKWSaveGenerator._GTKW_TYPE))

    def _test_gtk_wave_save_generator(self, *, signal_manager):
        with temp_dir():
            ddb = enable_dax_sim(ddb=_DEVICE_DB.copy(), enable=True, output=signal_manager, moninj_service=False)

            with get_managers(ddb) as managers:
                system = _TestSystem(managers)
                self.assertTrue(system.dax_sim_enabled)

                # Create GTKWave save generator object, which immediately writes the waves file
                GTKWSaveGenerator(system)
                # Manually close signal manager before leaving temp dir
                get_signal_manager().close()

    def test_gtk_wave_save_generator_null(self):
        self._test_gtk_wave_save_generator(signal_manager='null')

    def test_gtk_wave_save_generator_vcd(self):
        self._test_gtk_wave_save_generator(signal_manager='vcd')

    def test_gtk_wave_save_generator_peek(self):
        self._test_gtk_wave_save_generator(signal_manager='peek')

    def test_gtk_wave_save_generator_invalid_signal_manager(self):
        with temp_dir():
            set_signal_manager(unittest.mock.Mock(spec=DaxSignalManager))

            with get_managers(_DEVICE_DB.copy()) as managers:
                system = _TestSystem(managers)

                with self.assertRaises(RuntimeError, msg='Not using DAX.sim did not raise'):
                    GTKWSaveGenerator(system)

                # Manually close signal manager before leaving temp dir
                get_signal_manager().close()
