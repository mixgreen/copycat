import unittest
import typing

from dax.base.system import DaxSystem
from dax.util.artiq import get_managers
from dax.util.output import temp_dir
from dax.sim import enable_dax_sim
from dax.sim.signal import get_signal_manager, VcdSignalManager
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


class GTKWaveTestCase(unittest.TestCase):

    def test_signal_types(self):
        self.assertSetEqual(set(VcdSignalManager._CONVERT_TYPE), set(GTKWSaveGenerator._CONVERT_TYPE),
                            'Signal types did not match VCD signal types.')

    def test_gtk_wave_save_generator(self):
        with temp_dir():
            ddb = enable_dax_sim(ddb=_DEVICE_DB.copy(), enable=True, output='vcd', moninj_service=False)

            with get_managers(ddb) as managers:
                system = _TestSystem(managers)
                self.assertTrue(system.dax_sim_enabled)

                # Create GTKWave save generator object, which immediately writes the waves file
                GTKWSaveGenerator(system)

                # Manually close signal manager before leaving temp dir
                get_signal_manager().close()

    def test_gtk_wave_save_generator_invalid_signal_manager(self):
        with temp_dir():
            ddb = enable_dax_sim(ddb=_DEVICE_DB.copy(), enable=True, output='null', moninj_service=False)

            with get_managers(ddb) as managers:
                system = _TestSystem(managers)
                self.assertTrue(system.dax_sim_enabled)

                with self.assertRaises(RuntimeError, msg='Not using VCD signal manager did not raise'):
                    # Create GTKWave save generator object, which immediately writes the waves file
                    GTKWSaveGenerator(system)

                # Manually close signal manager before leaving temp dir
                get_signal_manager().close()


if __name__ == '__main__':
    unittest.main()
