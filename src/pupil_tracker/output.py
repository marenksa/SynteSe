"""Output interfaces for streaming color data to various sinks."""

import socket
from typing import Protocol

from pupil_tracker.analyzer import ColorReading, Note


class OutputSink(Protocol):
    """Protocol for output sinks that receive color readings."""

    def emit(self, reading: ColorReading) -> None:
        """Emit a color reading to the sink.

        Args:
            reading: The color reading to emit.
        """
        ...

    def close(self) -> None:
        """Close the sink and release any resources."""
        ...


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

    def emit(self, reading: ColorReading) -> None:
        """Emit a reading to all sinks."""
        for sink in self._sinks:
            sink.emit(reading)

    def close(self) -> None:
        """Close all sinks."""
        for sink in self._sinks:
            sink.close()


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


class PureDataFUDISink:
    """Output sink that streams color/note data to Pure Data via FUDI (TCP).

    FUDI is Pure Data's native protocol - simpler than OSC, no externals needed.
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
            print(f"[PureData FUDI] Connected to {self._host}:{self._port}")
            return True
        except (ConnectionRefusedError, OSError) as e:
            print(f"[PureData FUDI] Connection failed: {e}")
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
        print("[PureData FUDI] Closed")
