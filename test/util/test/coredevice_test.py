import unittest

from artiq.language.core import kernel, portable
import artiq.coredevice.core

import dax.util.test.coredevice

from test.environment import CI_ENABLED


class _Driver:
    kernel_invariants = {'core'}

    def __init__(self, dmgr, *, core_device='core'):
        self.core = dmgr.get(core_device)

    @kernel
    def not_implemented(self):
        raise NotImplementedError

    @portable
    def portable_(self):
        return 1

    @portable
    def portable_fail(self):
        return {1: 1}  # Dict unsupported

    @kernel
    def kernel_(self):
        return self.portable_()

    @kernel
    def kernel_fail(self):
        return self.portable_fail()


def _run_test(cls):
    obj = cls()
    obj.setUpClass()
    try:
        obj.setUp()
        obj.test_compile_functions()
    finally:
        obj.tearDown()
    return obj


@unittest.skipUnless(CI_ENABLED, 'Not in a CI environment, skipping compilation test')
class CompileTestCase2(unittest.TestCase):

    def test_pass(self):
        class Test(dax.util.test.coredevice.CompileTestCase):
            DEVICE_CLASS = _Driver
            FN_EXCLUDE = {'portable_fail', 'kernel_fail'}
            SIM_DEVICE = False

        self.assertIsInstance(_run_test(Test), Test)

    def test_fail(self):
        for exclude in ['portable_fail', 'kernel_fail']:
            class Test(dax.util.test.coredevice.CompileTestCase):
                DEVICE_CLASS = _Driver
                FN_EXCLUDE = {exclude}
                SIM_DEVICE = False

            with self.assertRaises(artiq.coredevice.core.CompileError):
                _run_test(Test)
