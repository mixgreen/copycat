from artiq.coredevice.exceptions import DMAError

from dax.sim.coredevice import *


class _DMARecordContext:

    def __init__(self, core, name, epoch):
        # Store fixed variables
        self.core = core
        self.name = name
        self.epoch = epoch

        # Duration will be recorded using enter and exit
        self.duration = 0

    @kernel
    def __enter__(self):
        # Save current time
        self.duration = now_mu()

    @kernel
    def __exit__(self, type_, value, traceback):
        # Store duration
        self.duration = now_mu() - self.duration


class CoreDMA(DaxSimDevice):

    def __init__(self, dmgr, **kwargs):
        # Call super
        super(CoreDMA, self).__init__(dmgr, **kwargs)

        # Initialize epoch to zero
        self._epoch = 0
        # Dict for DMA traces
        self._dma_traces = dict()

        # Register signal
        self._signal_manager = get_signal_manager()
        self._dma = self._signal_manager.register(self.key, 'play', str)

    @kernel
    def record(self, name):
        """Returns a context manager that will record a DMA trace called ``name``.
        Any previously recorded trace with the same name is overwritten.
        The trace will persist across kernel switches."""
        self._epoch += 1
        recorder = _DMARecordContext(self.core, name, self._epoch)

        # Store DMA trace
        self._dma_traces[name] = recorder

        # Return the record context
        return recorder

    @kernel
    def erase(self, name):
        """Removes the DMA trace with the given name from storage."""
        self._epoch += 1
        self._dma_traces.pop(name)

    @kernel
    def playback(self, name):
        # Playback DMA trace
        self._playback(name)

    @kernel
    def _playback(self, name):
        # Get recorder
        recorder = self._dma_traces[name]
        # Place an event for the DMA playback
        self._signal_manager.event(self._dma, recorder.name)
        # Forward time by the duration of the DMA trace
        delay_mu(recorder.duration)

    @kernel
    def get_handle(self, name):
        # There are no handles in the simulated DMA controller, so we just return the name
        return name

    @kernel
    def playback_handle(self, handle):
        # Get recorder
        recorder = self._dma_traces[handle]
        # Check if it was the last recording, since playback_handle() is only possible with the latest trace
        if self._epoch != recorder.epoch:
            raise DMAError('Invalid handle')

        # Playback the trace (handle == name)
        self._playback(handle)
