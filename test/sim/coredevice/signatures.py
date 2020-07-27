import unittest
import inspect

import artiq.coredevice.cache  # type: ignore
import artiq.coredevice.core  # type: ignore
import artiq.coredevice.dma  # type: ignore
import artiq.coredevice.ttl  # type: ignore
import artiq.coredevice.edge_counter
import artiq.coredevice.ad9910  # type: ignore
import artiq.coredevice.ad9912  # type: ignore
import artiq.coredevice.urukul  # type: ignore

import dax.sim.coredevice.cache
import dax.sim.coredevice.core
import dax.sim.coredevice.dma
import dax.sim.coredevice.ttl
import dax.sim.coredevice.edge_counter
import dax.sim.coredevice.ad9910
import dax.sim.coredevice.ad9912
import dax.sim.coredevice.urukul


class CoredeviceSignatureTestCase(unittest.TestCase):
    class_list = [
        (dax.sim.coredevice.cache.CoreCache, artiq.coredevice.cache.CoreCache),
        (dax.sim.coredevice.core.Core, artiq.coredevice.core.Core),
        (dax.sim.coredevice.dma.CoreDMA, artiq.coredevice.dma.CoreDMA),
        (dax.sim.coredevice.ttl.TTLOut, artiq.coredevice.ttl.TTLOut),
        (dax.sim.coredevice.ttl.TTLInOut, artiq.coredevice.ttl.TTLInOut),
        (dax.sim.coredevice.ttl.TTLClockGen, artiq.coredevice.ttl.TTLClockGen),
        (dax.sim.coredevice.edge_counter.EdgeCounter, artiq.coredevice.edge_counter.EdgeCounter),
        (dax.sim.coredevice.ad9910.AD9910, artiq.coredevice.ad9910.AD9910),
        (dax.sim.coredevice.ad9912.AD9912, artiq.coredevice.ad9912.AD9912),
        (dax.sim.coredevice.urukul.CPLD, artiq.coredevice.urukul.CPLD),
    ]

    def test_methods(self):
        """Test if simulated coredevice drivers have the same public functions as ARTIQ coredevice drivers."""
        for d, a in self.class_list:
            with self.subTest('Class signature comparison', dax_class=d, artiq_class=a):
                # Get function lists
                d_m = [(n, f) for n, f in inspect.getmembers(d, inspect.isfunction) if not n.startswith('_')]  # DAX
                a_m = [(n, f) for n, f in inspect.getmembers(a, inspect.isfunction) if not n.startswith('_')]  # ARTIQ
                a_m_set = {n for n, _ in a_m}  # Set of real driver functions

                # Verify if public functions are the same
                self.assertGreaterEqual({n for n, _ in d_m}, a_m_set,
                                        'Simulated driver class does not implement all functions of real driver class')

                # Filter DAX class function list to match the ARTIQ class function list
                d_m = [(n, f) for n, f in d_m if n in a_m_set]

                for d_f, a_f in zip((f for _, f in d_m), (f for _, f in a_m)):
                    with self.subTest('Class function signature comparison',
                                      dax_func=d_f.__qualname__, artiq_func=a_f.__qualname__):
                        # Get function parameters
                        d_p = set(inspect.signature(d_f).parameters.keys())
                        a_p = set(inspect.signature(a_f).parameters.keys())

                        # Verify parameters
                        self.assertLessEqual(d_p, a_p, 'Simulated driver function signature requires more '
                                                       'parameters than real driver class')
                        if d_p < a_p:
                            # Parameters are a strict subset, then kwargs must be included to accept other arguments
                            self.assertIn('kwargs', d_p, 'Simulated driver function signature requires less parameters '
                                                         'than real driver, but does not support kwargs')


if __name__ == '__main__':
    unittest.main()
