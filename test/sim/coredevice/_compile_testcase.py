import unittest

import dax.util.test.coredevice  # type: ignore[import]
from test.environment import CI_ENABLED

__all__ = ['CoredeviceCompileTestCase']


class _SkipCompilerTest(RuntimeError):
    pass


@unittest.skipUnless(CI_ENABLED, 'Not in a CI environment, skipping compilation test')
class CoredeviceCompileTestCase(dax.util.test.coredevice.CompileTestCase):
    pass
