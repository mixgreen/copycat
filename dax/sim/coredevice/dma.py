import numpy as np

from artiq.coredevice.exceptions import DMAError

from artiq.language.core import *
from artiq.language.units import *

from dax.sim.device import DaxSimDevice
from dax.sim.signal import get_signal_manager


class _DMARecordContext:

    def __init__(self, core, name: str, epoch: int, record_signal):
        # Store attributes
        self._core = core
        self._name = name
        self._epoch = epoch

        # Signals
        self._signal_manager = get_signal_manager()
        self._record_signal = record_signal

        # Duration will be recorded using enter and exit
        self._duration = np.int64(0)  # type: np.int64

    @property
    def core(self):
        return self._core

    @property
    def name(self) -> str:
        return self._name

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def duration(self) -> np.int64:
        return self._duration

    @kernel
    def __enter__(self):
        # Save current time
        self._duration = now_mu()
        # Set record signal
        self._signal_manager.event(self._record_signal, self.name)

    @kernel
    def __exit__(self, type_, value, traceback):
        # Store duration
        self._duration = now_mu() - self.duration
        # Reset record signal
        self._signal_manager.event(self._record_signal, None)  # Shows up as Z in the graphical interface


class CoreDMA(DaxSimDevice):

    def __init__(self, dmgr, **kwargs):
        # Call super
        super(CoreDMA, self).__init__(dmgr, **kwargs)

        # Initialize epoch to zero
        self._epoch = 0  # type: int
        # Dict for DMA traces
        self._dma_traces = dict()

        # Register signal
        self._signal_manager = get_signal_manager()
        self._dma_record = self._signal_manager.register(self.key, 'record', str)
        self._dma_play = self._signal_manager.register(self.key, 'play', object)
        self._dma_play_name = self._signal_manager.register(self.key, 'play_name', str)

    @kernel
    def record(self, name: str):
        assert isinstance(name, str), 'DMA trace name must be of type str'

        # Increment epoch
        self._epoch += 1

        # Create and store new DMA trace
        recorder = _DMARecordContext(self.core, name, self._epoch, self._dma_record)
        self._dma_traces[name] = recorder

        # Return the record context
        return recorder

    @kernel
    def erase(self, name):
        assert isinstance(name, str), 'DMA trace name must be of type str'

        if name not in self._dma_traces:
            raise KeyError('DMA trace "{:s}" does not exist, can not be erased'.format(name))
        self._dma_traces.pop(name)

        # Increment epoch
        self._epoch += 1

    @kernel
    def playback(self, name: str):
        assert isinstance(name, str), 'DMA trace name must be of type str'

        # Get handle
        if name not in self._dma_traces:
            raise KeyError('DMA trace "{:s}" does not exist, can not be played'.format(name))

        # Playback DMA trace
        self.playback_handle(self._dma_traces[name])

    @kernel
    def get_handle(self, name: str):
        assert isinstance(name, str), 'DMA trace name must be of type str'

        if name not in self._dma_traces:
            raise KeyError('DMA trace "{:s}" does not exist, can not obtain handle'.format(name))

        # Return the record context as the handle
        return self._dma_traces[name]

    @kernel
    def playback_handle(self, handle: _DMARecordContext):
        assert isinstance(handle, _DMARecordContext), 'DMA handle has an incorrect type'

        # Verify handle
        if self._epoch != handle.epoch:
            # An epoch mismatch occurs when adding or erasing a DMA trace after obtaining the handle
            raise DMAError('Invalid DMA handle "{:s}", epoch mismatch'.format(handle.name))

        # Place events for DMA playback
        self._signal_manager.event(self._dma_play, True)  # Represents the event of playing a trace
        self._signal_manager.event(self._dma_play_name, handle.name)  # Represents the duration of the event

        # Forward time by the duration of the DMA trace
        delay_mu(handle.duration)

        # Record ending of DMA trace (shows up as Z in the graphical interface)
        self._signal_manager.event(self._dma_play_name, None)
