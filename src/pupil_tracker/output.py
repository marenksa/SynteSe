"""Output interfaces for streaming brightness data to various sinks."""

import json
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Protocol, TextIO

from pupil_tracker.analyzer import BrightnessReading


class OutputSink(Protocol):
    """Protocol for output sinks that receive brightness readings."""

    def emit(self, reading: BrightnessReading) -> None:
        """Emit a brightness reading to the sink.

        Args:
            reading: The brightness reading to emit.
        """
        ...

    def close(self) -> None:
        """Close the sink and release any resources."""
        ...


class ConsoleSink:
    """Output sink that prints brightness readings to the console."""

    def __init__(self, verbose: bool = False) -> None:
        """Initialize the console sink.

        Args:
            verbose: If True, print all details. If False, print compact output.
        """
        self._verbose = verbose

    def emit(self, reading: BrightnessReading) -> None:
        """Print a brightness reading to the console."""
        if self._verbose:
            print(
                f"[{reading.timestamp:.3f}] "
                f"Brightness: {reading.brightness:.1f} "
                f"(smoothed: {reading.smoothed_brightness:.1f}) "
                f"@ ({reading.center_x}, {reading.center_y}) "
                f"conf: {reading.confidence:.2f}"
            )
        else:
            # Compact visualization with brightness bar
            bar_length = 30
            filled = int(reading.smoothed_brightness / 255 * bar_length)
            bar = "█" * filled + "░" * (bar_length - filled)
            print(
                f"\rBrightness: [{bar}] {reading.smoothed_brightness:5.1f}/255",
                end="",
                flush=True,
            )

    def close(self) -> None:
        """Print a newline on close to clean up the output."""
        print()


class FileSink:
    """Output sink that writes brightness readings to a file."""

    def __init__(
        self,
        output_path: Path | str | None = None,
        format: str = "jsonl",  # noqa: A002
    ) -> None:
        """Initialize the file sink.

        Args:
            output_path: Path to the output file. If None, generates timestamped name.
            format: Output format, either "jsonl" or "csv".
        """
        self._format = format

        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            extension = "jsonl" if format == "jsonl" else "csv"
            output_path = Path(f"brightness_{timestamp}.{extension}")
        else:
            output_path = Path(output_path)

        self._path = output_path
        self._file: TextIO | None = None
        self._header_written = False

    def _ensure_open(self) -> TextIO:
        """Ensure the file is open and return it."""
        if self._file is None:
            self._file = open(self._path, "w", encoding="utf-8")
            if self._format == "csv" and not self._header_written:
                self._file.write(
                    "timestamp,brightness,smoothed_brightness,center_x,center_y,confidence\n"
                )
                self._header_written = True
        return self._file

    def emit(self, reading: BrightnessReading) -> None:
        """Write a brightness reading to the file."""
        file = self._ensure_open()

        if self._format == "jsonl":
            record = {
                "timestamp": reading.timestamp,
                "brightness": reading.brightness,
                "smoothed_brightness": reading.smoothed_brightness,
                "center_x": reading.center_x,
                "center_y": reading.center_y,
                "confidence": reading.confidence,
            }
            file.write(json.dumps(record) + "\n")
        else:
            file.write(
                f"{reading.timestamp},{reading.brightness},{reading.smoothed_brightness},"
                f"{reading.center_x},{reading.center_y},{reading.confidence}\n"
            )

    def close(self) -> None:
        """Close the output file."""
        if self._file is not None:
            self._file.close()
            self._file = None
            print(f"[FileSink] Data written to {self._path}")


class MultiSink:
    """Output sink that broadcasts readings to multiple sinks."""

    def __init__(self, sinks: list[OutputSink] | None = None) -> None:
        """Initialize with a list of sinks.

        Args:
            sinks: List of output sinks to broadcast to.
        """
        self._sinks: list[OutputSink] = sinks or []

    def add_sink(self, sink: OutputSink) -> None:
        """Add a sink to the broadcast list."""
        self._sinks.append(sink)

    def emit(self, reading: BrightnessReading) -> None:
        """Emit a reading to all sinks."""
        for sink in self._sinks:
            sink.emit(reading)

    def close(self) -> None:
        """Close all sinks."""
        for sink in self._sinks:
            sink.close()


class ThresholdSink(ABC):
    """Abstract base for sinks that trigger on brightness thresholds."""

    def __init__(
        self,
        low_threshold: float = 50.0,
        high_threshold: float = 200.0,
    ) -> None:
        """Initialize threshold sink.

        Args:
            low_threshold: Brightness below this triggers 'dark' signal.
            high_threshold: Brightness above this triggers 'bright' signal.
        """
        self._low_threshold = low_threshold
        self._high_threshold = high_threshold
        self._last_state: str = "normal"

    def emit(self, reading: BrightnessReading) -> None:
        """Check thresholds and emit appropriate signals."""
        brightness = reading.smoothed_brightness

        if brightness < self._low_threshold:
            new_state = "dark"
        elif brightness > self._high_threshold:
            new_state = "bright"
        else:
            new_state = "normal"

        # Only trigger on state change
        if new_state != self._last_state:
            self._on_state_change(self._last_state, new_state, reading)
            self._last_state = new_state

    @abstractmethod
    def _on_state_change(
        self,
        old_state: str,
        new_state: str,
        reading: BrightnessReading,
    ) -> None:
        """Handle state change event.

        Args:
            old_state: Previous brightness state.
            new_state: New brightness state.
            reading: The reading that triggered the change.
        """
        ...

    def close(self) -> None:
        """Default close implementation does nothing."""
        pass


class ConsoleThresholdSink(ThresholdSink):
    """Threshold sink that prints state changes to console."""

    def _on_state_change(
        self,
        old_state: str,
        new_state: str,
        reading: BrightnessReading,
    ) -> None:
        """Print state change to console."""
        print(
            f"\n[THRESHOLD] {old_state} -> {new_state} "
            f"(brightness: {reading.smoothed_brightness:.1f})"
        )

