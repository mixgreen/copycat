import unittest
import inspect

import artiq.coredevice.ad53xx  # type: ignore[import]
import artiq.coredevice.ad9910  # type: ignore[import]
import artiq.coredevice.ad9912  # type: ignore[import]
import artiq.coredevice.cache  # type: ignore[import]
import artiq.coredevice.comm_kernel
import artiq.coredevice.core
import artiq.coredevice.dma  # type: ignore[import]
import artiq.coredevice.edge_counter
import artiq.coredevice.spi2  # type: ignore[import]
import artiq.coredevice.ttl  # type: ignore[import]
import artiq.coredevice.urukul  # type: ignore[import]
import artiq.coredevice.zotino  # type: ignore[import]

import dax.sim.coredevice.ad53xx
import dax.sim.coredevice.ad9910
import dax.sim.coredevice.ad9912
import dax.sim.coredevice.cache
import dax.sim.coredevice.comm_kernel
import dax.sim.coredevice.core
import dax.sim.coredevice.dma
import dax.sim.coredevice.edge_counter
import dax.sim.coredevice.spi2
import dax.sim.coredevice.ttl
import dax.sim.coredevice.urukul
import dax.sim.coredevice.zotino


class CoredeviceSignatureTestCase(unittest.TestCase):
    class_list = [
        (dax.sim.coredevice.ad53xx.AD53xx, artiq.coredevice.ad53xx.AD53xx),
        (dax.sim.coredevice.ad9910.AD9910, artiq.coredevice.ad9910.AD9910),
        (dax.sim.coredevice.ad9912.AD9912, artiq.coredevice.ad9912.AD9912),
        (dax.sim.coredevice.cache.CoreCache, artiq.coredevice.cache.CoreCache),
        (dax.sim.coredevice.comm_kernel.CommKernelDummy, artiq.coredevice.comm_kernel.CommKernelDummy),
        (dax.sim.coredevice.core.Core, artiq.coredevice.core.Core),
        (dax.sim.coredevice.dma.CoreDMA, artiq.coredevice.dma.CoreDMA),
        (dax.sim.coredevice.edge_counter.EdgeCounter, artiq.coredevice.edge_counter.EdgeCounter),
        (dax.sim.coredevice.spi2.SPIMaster, artiq.coredevice.spi2.SPIMaster),
        (dax.sim.coredevice.ttl.TTLOut, artiq.coredevice.ttl.TTLOut),
        (dax.sim.coredevice.ttl.TTLInOut, artiq.coredevice.ttl.TTLInOut),
        (dax.sim.coredevice.ttl.TTLClockGen, artiq.coredevice.ttl.TTLClockGen),
        (dax.sim.coredevice.urukul.CPLD, artiq.coredevice.urukul.CPLD),
        (dax.sim.coredevice.zotino.Zotino, artiq.coredevice.zotino.Zotino),
    ]

    def test_functions(self):
        """Test if simulated coredevice drivers have the same public functions as ARTIQ coredevice drivers."""
        for dax_cls, artiq_cls in self.class_list:
            with self.subTest('Class signature comparison', dax_cls=dax_cls, artiq_cls=artiq_cls):
                self._test_function_signatures(dax_cls=dax_cls, artiq_cls=artiq_cls)

    def _test_function_signatures(self, *, dax_cls, artiq_cls):
        """This function does not use `subTest` to make sure exceptions fall through."""

        # Get function lists
        dax_m = [(n, f) for n, f in inspect.getmembers(dax_cls, inspect.isfunction) if not n.startswith('_')]
        artiq_m = [(n, f) for n, f in inspect.getmembers(artiq_cls, inspect.isfunction) if not n.startswith('_')]
        artiq_m_set = {n for n, _ in artiq_m}  # Set of real driver functions

        # Verify if public functions are the same
        self.assertGreaterEqual({n for n, _ in dax_m}, artiq_m_set,
                                f'Simulated driver class {dax_cls.__qualname__} does not implement all functions of '
                                f'the ARTIQ driver class {artiq_cls.__qualname__}')

        # Filter DAX class function list to match the ARTIQ class function list
        dax_m = [(n, f) for n, f in dax_m if n in artiq_m_set]

        for dax_fn, artiq_fn in zip((f for _, f in dax_m), (f for _, f in artiq_m)):
            # Get function signatures
            dax_sig = inspect.signature(dax_fn)
            artiq_sig = inspect.signature(artiq_fn)

            # Get function parameters
            dax_p = set(dax_sig.parameters.keys())
            artiq_p = set(artiq_sig.parameters.keys())
            # Take out kwargs
            has_kwargs = 'kwargs' in dax_p
            dax_p.discard('kwargs')

            # Verify parameter existence
            self.assertLessEqual(dax_p, artiq_p,
                                 f'Simulated driver function signature {dax_fn.__qualname__} requires more '
                                 f'parameters than the ARTIQ driver function {artiq_fn.__qualname__}')
            if dax_p < artiq_p:
                # Parameters are a strict subset, then kwargs must be included to accept other arguments
                self.assertTrue(has_kwargs,
                                f'Simulated driver function signature {dax_fn.__qualname__} requires less parameters '
                                f'than the ARTIQ driver function {artiq_fn.__qualname__}, but does not support kwargs')

    def test_verification(self):
        # Classes match
        for cls in [_MatchedClass0, _MatchedClass1]:
            with self.subTest(cls=cls):
                self._test_function_signatures(dax_cls=cls, artiq_cls=_ReferenceClass)

        # Classes do not match
        for cls in [_BadClass0, _BadClass1, _BadClass2]:
            with self.subTest(cls=cls):
                with self.assertRaises(self.failureException, msg='Class signature mismatch did not raise'):
                    self._test_function_signatures(dax_cls=cls, artiq_cls=_ReferenceClass)


class _ReferenceClass:
    def foo(self, a, b):
        """Function with reference signature."""
        pass


class _MatchedClass0:
    def foo(self, a, b):
        """Exact match with reference."""
        pass


class _MatchedClass1:
    def foo(self, **kwargs):
        """Missing parameters but with keyword arguments."""
        pass


class _BadClass0:
    """Missing function"""
    pass


class _BadClass1:
    def foo(self, a):
        """Missing parameter."""
        pass


class _BadClass2:
    def foo(self, a, b, z):
        """Additional parameter."""
        pass
