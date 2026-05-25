"""Pupil Core eye-tracking music system — map gaze signals to sound."""

from base.input.live import (
    FixationData,
    FrameData,
    GazeData,
    Message,
    PupilCaptureClient,
)
from base.input.recording import (
    BlinkSample,
    BlinkType,
    FixationSample,
    FlutterEvent,
    GazeSample,
    Recording,
    RecordingInfo,
)
from base.output import (
    ColorConsoleSink,
    MultiSink,
    OutputSink,
    PureDataSink,
)
from base.signals.env_color import (
    ColorAnalyzer,
    ColorReading,
    FrameProcessor,
    GazeRegion,
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
    # Output
    "OutputSink",
    "MultiSink",
    "ColorConsoleSink",
    "PureDataSink",
]
