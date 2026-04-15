"""Output interfaces for streaming brightness data to various sinks."""

import json
import socket
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Protocol, TextIO

from pythonosc import udp_client

from pupil_tracker.analyzer import BrightnessReading, ColorReading, Note


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


class PureDataSink:
    """Output sink that streams brightness to Pure Data via OSC.

    Sends OSC messages to Pure Data for real-time sound synthesis.
    The brightness is normalized to 0.0-1.0 range for easier mapping in Pd.

    Default address: /brightness (float 0-1)
    Optional: /gaze (x, y), /confidence (float)
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9000,
        send_gaze: bool = False,
        send_confidence: bool = False,
    ) -> None:
        """Initialize the Pure Data OSC sink.

        Args:
            host: IP address where Pure Data is running.
            port: UDP port Pure Data is listening on (default 9000).
            send_gaze: Also send gaze position as /gaze message.
            send_confidence: Also send confidence as /confidence message.
        """
        self._host = host
        self._port = port
        self._send_gaze = send_gaze
        self._send_confidence = send_confidence
        self._client = udp_client.SimpleUDPClient(host, port)
        print(f"[PureDataSink] Sending OSC to {host}:{port}")

    def emit(self, reading: BrightnessReading) -> None:
        """Send brightness reading to Pure Data via OSC.

        Args:
            reading: The brightness reading to send.
        """
        # Normalize brightness to 0.0-1.0 range
        normalized = reading.smoothed_brightness / 255.0
        self._client.send_message("/brightness", normalized)

        if self._send_gaze:
            self._client.send_message("/gaze", [reading.center_x, reading.center_y])

        if self._send_confidence:
            self._client.send_message("/confidence", reading.confidence)

    def close(self) -> None:
        """Close the sink (UDP is connectionless, nothing to do)."""
        print("[PureDataSink] Closed")


class PureDataFUDISink:
    """Output sink that streams brightness to Pure Data via FUDI (TCP).

    FUDI is Pure Data's native protocol - simpler than OSC, no externals needed.
    Messages are sent as: "brightness <value>;" over TCP.

    Use this if you don't want to install the mrpeach OSC externals in Pd.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9001,
    ) -> None:
        """Initialize the Pure Data FUDI sink.

        Args:
            host: IP address where Pure Data is running.
            port: TCP port Pure Data is listening on (default 9001).
        """
        self._host = host
        self._port = port
        self._socket: socket.socket | None = None
        self._connected = False

    def _ensure_connected(self) -> bool:
        """Ensure TCP connection to Pd is established."""
        if self._connected and self._socket:
            return True

        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.connect((self._host, self._port))
            self._connected = True
            print(f"[PureDataFUDISink] Connected to {self._host}:{self._port}")
            return True
        except (ConnectionRefusedError, OSError) as e:
            print(f"[PureDataFUDISink] Connection failed: {e}")
            self._socket = None
            self._connected = False
            return False

    def emit(self, reading: BrightnessReading) -> None:
        """Send brightness reading to Pure Data via FUDI.

        Args:
            reading: The brightness reading to send.
        """
        if not self._ensure_connected():
            return

        # Normalize and format as FUDI message
        normalized = reading.smoothed_brightness / 255.0
        message = f"brightness {normalized:.4f};\n"

        try:
            if self._socket:
                self._socket.send(message.encode("utf-8"))
        except (BrokenPipeError, ConnectionResetError):
            self._connected = False
            self._socket = None

    def close(self) -> None:
        """Close the TCP connection."""
        if self._socket:
            self._socket.close()
            self._socket = None
            self._connected = False
        print("[PureDataFUDISink] Closed")


# Color name lookup for display
NOTE_COLORS = {
    Note.C: "Red",
    Note.D: "Orange",
    Note.E: "Yellow",
    Note.F: "Green",
    Note.G: "Cyan",
    Note.A: "Blue",
    Note.B: "Violet",
}


class ColorConsoleSink:
    """Output sink that prints color readings to the console."""

    def __init__(self, verbose: bool = False) -> None:
        """Initialize the console sink.

        Args:
            verbose: If True, print all details. If False, print compact output.
        """
        self._verbose = verbose

    def emit(self, reading: ColorReading) -> None:
        """Print a color reading to the console."""
        note_name = reading.note.name
        color_name = NOTE_COLORS.get(reading.note, "Unknown")

        if self._verbose:
            hue_str = f"{reading.hue:.0f}" if reading.hue is not None else "N/A"
            print(
                f"[{reading.timestamp:.3f}] "
                f"Note: {note_name}{reading.octave} (MIDI {reading.midi_note}) "
                f"Color: {color_name} (H:{hue_str} S:{reading.saturation:.0f}) "
                f"Brightness: {reading.brightness:.1f} "
                f"@ ({reading.center_x}, {reading.center_y}) "
                f"conf: {reading.confidence:.2f}"
            )
        else:
            # Compact visualization
            print(
                f"\rNote: {note_name}{reading.octave} ({color_name:7}) "
                f"MIDI: {reading.midi_note:3d} "
                f"Brightness: {reading.smoothed_brightness:5.1f}",
                end="",
                flush=True,
            )

    def close(self) -> None:
        """Print a newline on close to clean up the output."""
        print()


class PureDataColorSink:
    """Output sink that streams color/note data to Pure Data via OSC.

    Sends MIDI note number for easy frequency conversion using Pd's mtof object.
    Also sends separate note and octave for flexibility.

    Messages:
        /midinote <int>: MIDI note number (36-83 for octaves 2-6)
        /note <int>: Note index (0=C, 1=D, ..., 6=B)
        /octave <int>: Octave number (2-6)
        /brightness <float>: Normalized brightness (0-1)
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9000,
    ) -> None:
        """Initialize the Pure Data color OSC sink.

        Args:
            host: IP address where Pure Data is running.
            port: UDP port Pure Data is listening on (default 9000).
        """
        self._host = host
        self._port = port
        self._client = udp_client.SimpleUDPClient(host, port)
        print(f"[PureDataColorSink] Sending OSC to {host}:{port}")

    def emit(self, reading: ColorReading) -> None:
        """Send color reading to Pure Data via OSC.

        Args:
            reading: The color reading to send.
        """
        # Send MIDI note (most useful for synthesis)
        self._client.send_message("/midinote", reading.midi_note)

        # Send individual components for flexibility
        self._client.send_message("/note", int(reading.note))
        self._client.send_message("/octave", reading.octave)
        self._client.send_message("/brightness", reading.smoothed_brightness / 255.0)

    def close(self) -> None:
        """Close the sink (UDP is connectionless, nothing to do)."""
        print("[PureDataColorSink] Closed")


class PureDataColorFUDISink:
    """Output sink that streams color/note data to Pure Data via FUDI (TCP).

    Sends MIDI note number for easy frequency conversion using Pd's mtof object.

    Messages (FUDI format):
        midinote <int>;
        note <int>;
        octave <int>;
        brightness <float>;
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9001,
    ) -> None:
        """Initialize the Pure Data color FUDI sink.

        Args:
            host: IP address where Pure Data is running.
            port: TCP port Pure Data is listening on (default 9001).
        """
        self._host = host
        self._port = port
        self._socket: socket.socket | None = None
        self._connected = False

    def _ensure_connected(self) -> bool:
        """Ensure TCP connection to Pd is established."""
        if self._connected and self._socket:
            return True

        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.connect((self._host, self._port))
            self._connected = True
            print(f"[PureDataColorFUDISink] Connected to {self._host}:{self._port}")
            return True
        except (ConnectionRefusedError, OSError) as e:
            print(f"[PureDataColorFUDISink] Connection failed: {e}")
            self._socket = None
            self._connected = False
            return False

    def emit(self, reading: ColorReading) -> None:
        """Send color reading to Pure Data via FUDI.

        Args:
            reading: The color reading to send.
        """
        if not self._ensure_connected():
            return

        # Send all values in a single message batch
        normalized_brightness = reading.smoothed_brightness / 255.0
        message = (
            f"midinote {reading.midi_note};\n"
            f"note {int(reading.note)};\n"
            f"octave {reading.octave};\n"
            f"brightness {normalized_brightness:.4f};\n"
        )

        try:
            if self._socket:
                self._socket.send(message.encode("utf-8"))
        except (BrokenPipeError, ConnectionResetError):
            self._connected = False
            self._socket = None

    def close(self) -> None:
        """Close the TCP connection."""
        if self._socket:
            self._socket.close()
            self._socket = None
            self._connected = False
        print("[PureDataColorFUDISink] Closed")

