import random

import artiq.coredevice.ttl  # type: ignore[import]

from dax.experiment import *
import dax.sim.test_case

from test.timer import Timer


class PeekBenchmarkTestCase(dax.sim.test_case.PeekTestCase):
    N = 10000
    SEED = None

    def setUp(self) -> None:
        # Construct environment
        self.sys = self.construct_env(_TestSystem, device_db=_DEVICE_DB)
        # Set up rng
        self.rng = random.Random(self.SEED)

    def _sequential_insertion(self, *, step=100):
        time_mu = now_mu()

        for _ in range(self.N):
            time_mu += step
            at_mu(time_mu)
            self.sys.ttl.on()

    def test_sequential_insertion(self):
        self.sys.ttl.output()

        with Timer() as t:
            self._sequential_insertion()
        t.u_print()

    def test_random_insertion(self):
        self.sys.ttl.output()
        max_time_mu = self.N ** 2

        with Timer() as t:
            for _ in range(self.N):
                at_mu(self.rng.randrange(max_time_mu))
                self.sys.ttl.on()
        t.u_print()

    def test_tail_peek(self):
        self.sys.ttl.output()
        self._sequential_insertion()

        with Timer() as t:
            # Tail peek
            for _ in range(self.N):
                self.peek(self.sys.ttl, 'state')
        t.u_print()

    def test_random_peek(self):
        self.sys.ttl.output()
        self._sequential_insertion()
        max_time_mu = now_mu()
        t = Timer()

        # Random peek
        for _ in range(self.N):
            at_mu(self.rng.randrange(max_time_mu))
            with t:
                self.peek(self.sys.ttl, 'state')
        t.u_print()


class _TestSystem(DaxSystem):
    SYS_ID = 'unittest_system'
    SYS_VER = 0

    def build(self, *args, **kwargs) -> None:
        super(_TestSystem, self).build(*args, **kwargs)
        self.ttl = self.get_device('ttl0', artiq.coredevice.ttl.TTLInOut)


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
}
