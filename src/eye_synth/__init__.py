"""Pupil Core color-to-music tracker - map gaze colors to musical notes."""

from eye_synth.input.live import (
    FixationData,
    FrameData,
    GazeData,
    Message,
    PupilCaptureClient,
)
from eye_synth.input.recording import (
    BlinkSample,
    BlinkType,
    FixationSample,
    FlutterEvent,
    GazeSample,
    Recording,
    RecordingInfo,
)
from eye_synth.output import (
    ColorConsoleSink,
    MultiSink,
    NoteEventSink,
    OutputSink,
    PureDataSink,
)
from eye_synth.signals.env_color import (
    ColorAnalyzer,
    ColorReading,
    FrameProcessor,
    GazeRegion,
    Note,
    NoteEvent,
)

__version__ = "0.1.0"

__all__ = [
    # Live input
    "PupilCaptureClient",
    "GazeData",
    "FrameData",
    "FixationData",
    "Message",
    # Recording input
    "Recording",
    "RecordingInfo",
    "GazeSample",
    "BlinkSample",
    "FixationSample",
    "BlinkType",
    "FlutterEvent",
    # Signals
    "FrameProcessor",
    "GazeRegion",
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
]
