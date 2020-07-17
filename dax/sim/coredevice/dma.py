import typing
import numpy as np

from artiq.coredevice.exceptions import DMAError

from artiq.language.core import *
from artiq.language.units import *

from dax.sim.device import DaxSimDevice
from dax.sim.signal import get_signal_manager


class _DMARecordContext:
    """DMA recording class, which is used as a context handler."""

    def __init__(self, core: typing.Any, name: str, record_signal: typing.Any):
        assert isinstance(name, str)

        # Store attributes
        self._core: typing.Any = core
        self._name: str = name

        # Signals
        self._signal_manager = get_signal_manager()
        self._record_signal: typing.Any = record_signal

        # Duration will be recorded using enter and exit
        self._duration: np.int64 = np.int64(0)

    @property
    def core(self) -> typing.Any:
        return self._core

    @property
    def name(self) -> str:
        return self._name

    @property
    def duration(self) -> np.int64:
        return self._duration

    @kernel
    def __enter__(self) -> None:
        # Save current time
        self._duration = now_mu()
        # Set record signal
        self._signal_manager.event(self._record_signal, self.name)

    @kernel
    def __exit__(self, type_: typing.Any, value: typing.Any, traceback: typing.Any) -> None:
        # Store duration
        self._duration = now_mu() - self.duration
        # Reset record signal
        self._signal_manager.event(self._record_signal, None)  # Shows up as Z in the graphical interface


class _DMAHandle:
    """DMA handle class."""

    def __init__(self, dma_recording: _DMARecordContext, epoch: int):
        assert isinstance(dma_recording, _DMARecordContext)
        assert isinstance(epoch, int) and epoch > 0

        # Store attributes
        self._recording: _DMARecordContext = dma_recording
        self._epoch: int = epoch

    @property
    def recording(self) -> _DMARecordContext:
        return self._recording

    @property
    def epoch(self) -> int:
        return self._epoch


class CoreDMA(DaxSimDevice):

    def __init__(self, dmgr: typing.Any, **kwargs: typing.Any):
        # Call super
        super(CoreDMA, self).__init__(dmgr, **kwargs)

        # Initialize epoch to zero
        self._epoch: int = 0
        # Dict for DMA traces
        self._dma_traces: typing.Dict[str, _DMARecordContext] = {}

        # Register signal
        self._signal_manager = get_signal_manager()
        self._dma_record: typing.Any = self._signal_manager.register(self, 'record', str)
        self._dma_play: typing.Any = self._signal_manager.register(self, 'play', object)
        self._dma_play_name: typing.Any = self._signal_manager.register(self, 'play_name', str)

    @kernel
    def record(self, name: str) -> _DMARecordContext:
        assert isinstance(name, str), 'DMA trace name must be of type str'

        # Increment epoch
        self._epoch += 1

        # Create and store new DMA trace
        recorder = _DMARecordContext(self.core, name, self._dma_record)
        self._dma_traces[name] = recorder

        # Return the record context
        return recorder

    @kernel
    def erase(self, name: str) -> None:
        assert isinstance(name, str), 'DMA trace name must be of type str'

        if name not in self._dma_traces:
            raise KeyError(f'DMA trace "{name}" does not exist, can not be erased')
        self._dma_traces.pop(name)

        # Increment epoch
        self._epoch += 1

    @kernel
    def playback(self, name: str) -> None:
        assert isinstance(name, str), 'DMA trace name must be of type str'

        # Get handle
        if name not in self._dma_traces:
            raise KeyError(f'DMA trace "{name}" does not exist, can not be played')

        # Playback recording
        self._playback_recording(self._dma_traces[name])

    @kernel
    def get_handle(self, name: str) -> _DMAHandle:
        assert isinstance(name, str), 'DMA trace name must be of type str'

        if name not in self._dma_traces:
            raise KeyError(f'DMA trace "{name}" does not exist, can not obtain handle')

        # Return the record context as the handle
        return _DMAHandle(self._dma_traces[name], self._epoch)

    @kernel
    def playback_handle(self, handle: _DMAHandle) -> None:
        assert isinstance(handle, _DMAHandle), 'DMA handle has an incorrect type'

        # Verify handle
        if self._epoch != handle.epoch:
            # An epoch mismatch occurs when adding or erasing a DMA trace after obtaining the handle
            raise DMAError(f'Invalid DMA handle for recording "{handle.recording.name}", epoch mismatch')

        # Playback recording
        self._playback_recording(handle.recording)

    def _playback_recording(self, recording: _DMARecordContext) -> None:
        assert isinstance(recording, _DMARecordContext), 'DMA recording has an incorrect type'

        # Place events for DMA playback
        self._signal_manager.event(self._dma_play, True)  # Represents the event of playing a trace
        self._signal_manager.event(self._dma_play_name, recording.name)  # Represents the duration of the event

        # Forward time by the duration of the DMA trace
        delay_mu(recording.duration)

        # Record ending of DMA trace (shows up as Z in the graphical interface)
        self._signal_manager.event(self._dma_play_name, None)
