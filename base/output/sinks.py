"""Output sinks for streaming data to external consumers."""

import socket
from typing import Protocol

from base.signals.env_color import ColorReading


class OutputSink(Protocol):
    def emit(self, reading: ColorReading) -> None: ...
    def close(self) -> None: ...


class MultiSink:
    """Broadcasts ColorReadings to multiple sinks."""

    def __init__(self, sinks: list[OutputSink] | None = None) -> None:
        self._sinks: list[OutputSink] = sinks or []

    def add_sink(self, sink: OutputSink) -> None:
        self._sinks.append(sink)

    def emit(self, reading: ColorReading) -> None:
        for sink in self._sinks:
            sink.emit(reading)

    def close(self) -> None:
        for sink in self._sinks:
            sink.close()


class ColorConsoleSink:
    """Prints colour readings to the console."""

    def __init__(self, verbose: bool = False) -> None:
        self._verbose = verbose

    def emit(self, reading: ColorReading) -> None:
        hue_str = f"{reading.hue:.0f}" if reading.hue is not None else "N/A"
        if self._verbose:
            print(
                f"[{reading.timestamp:.3f}] "
                f"H:{hue_str} S:{reading.saturation:.0f} V:{reading.smoothed_brightness:.0f} "
                f"@ ({reading.center_x}, {reading.center_y}) "
                f"conf: {reading.confidence:.2f}"
            )
        else:
            print(
                f"\rHue:{hue_str:>5} Sat:{reading.saturation:5.1f} Brightness:{reading.smoothed_brightness:5.1f}",
                end="", flush=True,
            )

    def close(self) -> None:
        print()


class PureDataSink:
    """Sends messages to Pure Data via FUDI (TCP).

    Use send() for generic messages from patches:
        sink.send("note_on", 60, 0.8)
        sink.send("effect", 0.8)
        sink.send("confidence", 0.95)
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 9001) -> None:
        self._host = host
        self._port = port
        self._socket: socket.socket | None = None
        self._connected = False
        self._warned = False

    def _ensure_connected(self) -> bool:
        if self._connected and self._socket:
            return True
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.connect((self._host, self._port))
            self._connected = True
            self._warned = False
            print(f"[PureData] Connected to {self._host}:{self._port}")
            return True
        except (ConnectionRefusedError, OSError):
            if not self._warned:
                print(f"[PureData] Not available at {self._host}:{self._port} — will retry silently")
                self._warned = True
            self._socket = None
            self._connected = False
            return False

    def _send(self, message: str) -> None:
        if not self._ensure_connected():
            return
        try:
            if self._socket:
                self._socket.send(message.encode("utf-8"))
        except (BrokenPipeError, ConnectionResetError):
            self._connected = False
            self._socket = None

    def send(self, key: str, *values) -> None:
        """Send a generic FUDI message: 'key val1 val2 ...;'"""
        if not values:
            self._send(f"{key};\n")
            return
        parts = " ".join(str(v) if isinstance(v, int) else f"{v:.4f}" for v in values)
        self._send(f"{key} {parts};\n")

    def close(self) -> None:
        if self._socket and self._connected:
            self._socket.close()
            self._socket = None
            self._connected = False
            print("[PureData] Closed")
