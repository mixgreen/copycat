import artiq.coredevice.comm_kernel


class CommKernelDummy(artiq.coredevice.comm_kernel.CommKernelDummy):
    """Extended ARTIQ CommKernelDummy class to match CommKernel signature.

    This class is used by the core and is intended as a mock object.
    Hence, it does not inherit :class:`dax.sim.device.SimDevice`.
    """

    def __init__(self) -> None:
        pass

    def open(self) -> None:
        pass

    def close(self) -> None:
        pass
