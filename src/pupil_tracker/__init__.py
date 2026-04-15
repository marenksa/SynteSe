"""Pupil Core color-to-music tracker - map gaze colors to musical notes."""

from pupil_tracker.analyzer import ColorAnalyzer, ColorReading, Note, NoteEvent
from pupil_tracker.client import (
    FixationData,
    FrameData,
    GazeData,
    Message,
    PupilCaptureClient,
)
from pupil_tracker.output import (
    ColorConsoleSink,
    MultiSink,
    NoteEventSink,
    OutputSink,
    PureDataSink,
)
from pupil_tracker.processor import FrameProcessor, GazeRegion
from pupil_tracker.recording import (
    BlinkSample,
    EyeClosureEvent,
    EyeClosureType,
    FixationSample,
    GazeSample,
    Recording,
    RecordingInfo,
)

__version__ = "0.1.0"

__all__ = [
    # Client
    "PupilCaptureClient",
    "GazeData",
    "FrameData",
    "FixationData",
    "Message",
    # Processor
    "FrameProcessor",
    "GazeRegion",
    # Analyzer
    "ColorAnalyzer",
    "ColorReading",
    "NoteEvent",
    "Note",
    # Output
    "OutputSink",
    "NoteEventSink",
    "MultiSink",
    "ColorConsoleSink",
    "PureDataSink",
    # Recording playback
    "Recording",
    "RecordingInfo",
    "GazeSample",
    "BlinkSample",
    "FixationSample",
    "EyeClosureEvent",
    "EyeClosureType",
]
