import artiq.coredevice.comm_kernel


class CommKernelDummy(artiq.coredevice.comm_kernel.CommKernelDummy):
    """Extended ARTIQ CommKernelDummy class to match CommKernel signature.

    This class is used by the core and is intended as a mock object.
    Hence, it does not inherit from SimDevice.
    """

    def __init__(self) -> None:
        pass

    def open(self, **kwargs):
        pass

    def close(self):
        pass

    def read(self, length):
        raise NotImplementedError

    def write(self, data):
        raise NotImplementedError

    def reset_session(self):
        raise NotImplementedError
