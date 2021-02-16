from dax.experiment import *

__all__ = ['RandomizedBenchmarkingSQ']


# TODO: inform the user that pygsti is an optional dependency and these clients require the package
# TODO: shared util functions can be privately stored in this class or in a new module dax.util.pygsti

@dax_client_factory
class RandomizedBenchmarkingSQ(DaxClient, Experiment):
    """Single-qubit randomized benchmarking using pyGSTi."""

    def run(self) -> None:
        pass
