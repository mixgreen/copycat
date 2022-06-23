import typing
import contextlib

from artiq.experiment import *
from dax.servo import DaxServo

import test.hw_test


class _ServoExperiment(DaxServo, HasEnvironment):

    def build_servo(self) -> None:
        self.setattr_device('core')

        self.add_servo('foo', 1)
        self.result = 0
        self.host_setup_done = False
        self.device_setup_done = False
        self.device_cleanup_done = False
        self.host_cleanup_done = False

    def host_setup(self):
        self.host_setup_done = True

    @kernel
    def device_setup(self):
        self.device_setup_done = True

    @kernel
    def run_point(self, point, index):  # type: (typing.Any, typing.Any) -> None  # pragma: no cover
        self.result += point.foo
        self.stop_servo()

    @kernel
    def device_cleanup(self):
        self.device_cleanup_done = True

    def host_cleanup(self):
        self.host_cleanup_done = True

    def get_identifier(self):
        return self.__class__.__name__


class ServoKernelTestCase(test.hw_test.HardwareTestCase):

    def test_run(self, env_cls=_ServoExperiment, result=1, *, host=True, device=True, exception=False):
        env = self.construct_env(env_cls)
        with (self.assertRaises(RuntimeError) if exception else contextlib.nullcontext()):
            env.run()
        self.assertEqual(env.result, result)
        self.assertEqual(env.host_setup_done, host)
        self.assertEqual(env.device_setup_done, device)
        self.assertEqual(env.device_cleanup_done, device)
        self.assertEqual(env.host_cleanup_done, host)

    def test_run_exception(self):
        class Env(_ServoExperiment):
            @kernel
            def run_point(self, point, index):  # type: (typing.Any, typing.Any) -> None  # pragma: no cover
                raise RuntimeError

        # Attributes only sync if there is no exception in the kernel
        self.test_run(Env, 0, device=False, exception=True)

    def test_host_setup_exception(self):
        class Env(_ServoExperiment):
            def host_setup(self):
                super(Env, self).host_setup()
                raise RuntimeError

        self.test_run(Env, 0, device=False, exception=True)

    def test_device_setup_exception(self):
        class Env(_ServoExperiment):
            @kernel
            def device_setup(self):
                raise RuntimeError

        # Attributes only sync if there is no exception in the kernel
        self.test_run(Env, 0, device=False, exception=True)

    def test_device_cleanup_exception(self):
        class Env(_ServoExperiment):
            @kernel
            def device_cleanup(self):
                self.device_cleanup_done = True
                raise RuntimeError

        # Attributes only sync if there is no exception in the kernel
        self.test_run(Env, 0, device=False, exception=True)

    def test_host_cleanup_exception(self):
        class Env(_ServoExperiment):
            def host_cleanup(self):
                super(Env, self).host_cleanup()
                raise RuntimeError

        self.test_run(Env, device=True, exception=True)
