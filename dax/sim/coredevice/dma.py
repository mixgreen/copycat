import typing
import numpy as np

from artiq.coredevice.exceptions import DMAError

from artiq.language.core import *
from artiq.language.units import *

from dax.sim.device import DaxSimDevice
from dax.sim.signal import get_signal_manager, Signal


class _DMARecordContext:
    """DMA recording class, which is used as a context handler."""

    _core: typing.Any
    _name: str
    _record_signal: Signal
    _duration: np.int64

    def __init__(self, core: typing.Any, name: str, record_signal: Signal):
        assert isinstance(name, str)

        # Store attributes
        self._core = core
        self._name = name

        # Signals
        self._record_signal = record_signal

        # Duration will be recorded using enter and exit
        self._duration = np.int64(0)

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
    def __enter__(self):  # type: () -> None
        # Save current time
        self._duration = now_mu()
        # Set record signal
        self._record_signal.push(self.name)

    @kernel
    def __exit__(self, type_, value, traceback):  # type: (typing.Any, typing.Any, typing.Any) -> None
        # Store duration
        self._duration = now_mu() - self.duration
        # Reset record signal
        self._record_signal.push('')


class _DMAHandle:
    """DMA handle class."""

    _recording: _DMARecordContext
    _epoch: int

    def __init__(self, dma_recording: _DMARecordContext, epoch: int):
        assert isinstance(dma_recording, _DMARecordContext)
        assert isinstance(epoch, int) and epoch > 0

        # Store attributes
        self._recording = dma_recording
        self._epoch = epoch

    @property
    def recording(self) -> _DMARecordContext:
        return self._recording

    @property
    def epoch(self) -> int:
        return self._epoch


class CoreDMA(DaxSimDevice):
    _epoch: int
    _dma_traces: typing.Dict[str, _DMARecordContext]
    _dma_record: Signal
    _dma_play: Signal
    _dma_play_name: Signal

    def __init__(self, dmgr: typing.Any, **kwargs: typing.Any):
        # Call super
        super(CoreDMA, self).__init__(dmgr, **kwargs)

        # Initialize epoch to zero
        self._epoch = 0
        # Dict for DMA traces
        self._dma_traces = {}

        # Register signal
        signal_manager = get_signal_manager()
        self._dma_record = signal_manager.register(self, 'record', str)
        self._dma_play = signal_manager.register(self, 'play', object)
        self._dma_play_name = signal_manager.register(self, 'play_name', str)

    @kernel
    def record(self, name):  # type: (str) -> _DMARecordContext
        assert isinstance(name, str), 'DMA trace name must be of type str'

        # Increment epoch
        self._epoch += 1

        # Create and store new DMA trace
        recorder = _DMARecordContext(self.core, name, self._dma_record)
        self._dma_traces[name] = recorder

        # Return the record context
        return recorder

    @kernel
    def erase(self, name):  # type: (str) -> None
        assert isinstance(name, str), 'DMA trace name must be of type str'

        if name not in self._dma_traces:
            raise KeyError(f'DMA trace "{name}" does not exist, can not be erased')
        self._dma_traces.pop(name)

        # Increment epoch
        self._epoch += 1

    @kernel
    def playback(self, name):  # type: (str) -> None
        assert isinstance(name, str), 'DMA trace name must be of type str'

        # Get handle
        if name not in self._dma_traces:
            raise KeyError(f'DMA trace "{name}" does not exist, can not be played')

        # Playback recording
        self._playback_recording(self._dma_traces[name])

    @kernel
    def get_handle(self, name):  # type: (str) -> _DMAHandle
        assert isinstance(name, str), 'DMA trace name must be of type str'

        if name not in self._dma_traces:
            raise KeyError(f'DMA trace "{name}" does not exist, can not obtain handle')

        # Return the record context as the handle
        return _DMAHandle(self._dma_traces[name], self._epoch)

    @kernel
    def playback_handle(self, handle):  # type: (_DMAHandle) -> None
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
        self._dma_play.push(True)  # Represents the event of playing a trace
        self._dma_play_name.push(recording.name)  # Represents the duration of the event

        # Forward time by the duration of the DMA trace
        delay_mu(recording.duration)

        # Record ending of DMA trace
        self._dma_play_name.push('')
