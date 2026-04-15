"""Pupil Core brightness tracker - stream gaze data and analyze brightness at gaze point."""

from pupil_tracker.analyzer import BrightnessAnalyzer, BrightnessReading
from pupil_tracker.client import FrameData, GazeData, Message, PupilCaptureClient
from pupil_tracker.output import (
    ConsoleSink,
    ConsoleThresholdSink,
    FileSink,
    MultiSink,
    OutputSink,
    ThresholdSink,
)
from pupil_tracker.processor import FrameProcessor, GazeRegion

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
    "BrightnessAnalyzer",
    "BrightnessReading",
    # Output
    "OutputSink",
    "ConsoleSink",
    "FileSink",
    "MultiSink",
    "ThresholdSink",
    "ConsoleThresholdSink",
]
