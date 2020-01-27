from artiq.language.core import *
from artiq.language.types import *
from artiq.language.units import *
from artiq.coredevice.exceptions import DMAError

import dax.sim.time as time


class DMARecordContext:

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


class CoreDMA:
    kernel_invariants = {'core'}

    def __init__(self, dmgr, name='core_dma', core_device='core'):
        self.name = name
        self.core = dmgr.get(core_device)
        self.epoch = 0

        # Dict for DMA traces
        self._dma_traces = dict()

        # Register variables
        self._dma = time.manager.register(self.name, 'play', 'string')

    @kernel
    def record(self, name):
        """Returns a context manager that will record a DMA trace called ``name``.
        Any previously recorded trace with the same name is overwritten.
        The trace will persist across kernel switches."""
        self.epoch += 1
        recorder = DMARecordContext(self.core, name, self.epoch)

        # Store DMA trace
        self._dma_traces[name] = recorder

        # TODO, events should actually be captured so they can be replayed at playback

        # Return the record context
        return recorder

    @kernel
    def erase(self, name):
        """Removes the DMA trace with the given name from storage."""
        self.epoch += 1
        self._dma_traces.pop(name)

    @kernel
    def playback(self, name):
        """Replays a previously recorded DMA trace. This function blocks until
        the entire trace is submitted to the RTIO FIFOs."""

        # Playback DMA trace
        self._playback(name)

    @kernel
    def _playback(self, name):
        # Get recorder
        recorder = self._dma_traces[name]
        # Place an event for the DMA playback
        time.manager.event(self._dma, recorder.name)
        # Forward time by the duration of the DMA trace
        delay_mu(recorder.duration)

    @kernel
    def get_handle(self, name):
        """Returns a handle to a previously recorded DMA trace. The returned handle
        is only valid until the next call to :meth:`record` or :meth:`erase`."""

        # There are no handles in the simulated DMA controller, so we just return the name
        return name

    @kernel
    def playback_handle(self, handle):
        """Replays a handle obtained with :meth:`get_handle`. Using this function
        is much faster than :meth:`playback` for replaying a set of traces repeatedly,
        but incurs the overhead of managing the handles onto the programmer."""

        # Get recorder
        recorder = self._dma_traces[handle]
        # Check if it was the last recording, since playback_handle() is only possible with the latest trace
        if self.epoch != recorder.epoch:
            raise DMAError('Invalid handle')

        # Playback the trace (handle == name)
        self._playback(handle)
