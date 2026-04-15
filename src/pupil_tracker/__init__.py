"""Pupil Core color-to-music tracker - map gaze colors to musical notes."""

from pupil_tracker.analyzer import ColorAnalyzer, ColorReading, Note
from pupil_tracker.client import FrameData, GazeData, Message, PupilCaptureClient
from pupil_tracker.output import (
    ColorConsoleSink,
    MultiSink,
    OutputSink,
    PureDataFUDISink,
)
from pupil_tracker.processor import FrameProcessor, GazeRegion
from pupil_tracker.recording import GazeSample, Recording, RecordingInfo

__version__ = "0.1.0"

__all__ = [
    # Client
    "PupilCaptureClient",
    "GazeData",
    "FrameData",
    "Message",
    # Processor
    "FrameProcessor",
    "GazeRegion",
    # Analyzer
    "ColorAnalyzer",
    "ColorReading",
    "Note",
    # Output
    "OutputSink",
    "MultiSink",
    "ColorConsoleSink",
    "PureDataFUDISink",
    # Recording playback
    "Recording",
    "RecordingInfo",
    "GazeSample",
]
