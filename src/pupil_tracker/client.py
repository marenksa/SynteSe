"""ZMQ client for connecting to Pupil Capture and receiving gaze/video data."""

from dataclasses import dataclass
from typing import Iterator

import msgpack
import numpy as np
import zmq
from numpy.typing import NDArray


@dataclass(frozen=True)
class GazeData:
    """Gaze data from Pupil Capture."""

    timestamp: float
    norm_pos: tuple[float, float]  # Normalized position (0-1)
    confidence: float
    topic: str


@dataclass(frozen=True)
class FrameData:
    """Video frame data from Pupil Capture."""

    timestamp: float
    width: int
    height: int
    data: NDArray[np.uint8]  # BGR image array
    topic: str


@dataclass(frozen=True)
class FixationData:
    """Fixation data from Pupil Capture's fixation detector."""

    id: int
    timestamp: float
    duration: float  # Duration in ms
    norm_pos: tuple[float, float]  # Normalized position (0-1)
    dispersion: float  # Dispersion in degrees
    confidence: float
    topic: str


@dataclass(frozen=True)
class Message:
    """Union type for messages received from Pupil Capture."""

    gaze: GazeData | None = None
    frame: FrameData | None = None
    fixation: FixationData | None = None


class PupilCaptureClient:
    """Client for connecting to Pupil Capture via ZMQ."""

    DEFAULT_HOST: str = "127.0.0.1"
    DEFAULT_PORT: int = 50020
    DEFAULT_TIMEOUT_MS: int = 5000  # 5 second timeout

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
    ) -> None:
        """Initialize the Pupil Capture client.

        Args:
            host: The host where Pupil Capture is running.
            port: The control port for Pupil Capture (default 50020).
            timeout_ms: Timeout in milliseconds for ZMQ operations.
        """
        self._host = host
        self._port = port
        self._timeout_ms = timeout_ms
        self._context: zmq.Context[zmq.Socket[bytes]] | None = None
        self._remote: zmq.Socket[bytes] | None = None
        self._subscriber: zmq.Socket[bytes] | None = None
        self._sub_port: int | None = None

    def connect(self) -> None:
        """Connect to Pupil Capture and set up data subscription."""
        self._context = zmq.Context()

        # Connect to the control socket (REQ) with timeout
        self._remote = self._context.socket(zmq.REQ)
        self._remote.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
        self._remote.setsockopt(zmq.SNDTIMEO, self._timeout_ms)
        self._remote.connect(f"tcp://{self._host}:{self._port}")

        # Request the subscription port
        try:
            self._remote.send_string("SUB_PORT")
            self._sub_port = int(self._remote.recv_string())
        except zmq.Again as e:
            raise ConnectionError(
                f"Timeout connecting to Pupil Capture at {self._host}:{self._port}. "
                "Make sure Pupil Capture is running."
            ) from e

        print(f"[PupilClient] Connected. Data streaming on port: {self._sub_port}")

        # Connect to the data socket (SUB) optimized for REAL-TIME (zero buffer)
        self._subscriber = self._context.socket(zmq.SUB)
        self._subscriber.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
        # CRITICAL: Minimize all buffering for real-time
        self._subscriber.setsockopt(zmq.RCVHWM, 1)  # Receive high water mark = 1
        self._subscriber.setsockopt(zmq.LINGER, 0)  # Don't wait on close
        self._subscriber.connect(f"tcp://{self._host}:{self._sub_port}")

        # Subscribe to gaze, frame, and fixation topics
        # For gaze, prefer combined (01) over single eye to avoid binocular flicker
        # We subscribe to all and filter in _get_latest_messages
        self._subscriber.subscribe("gaze")
        self._subscriber.subscribe("frame.world")
        self._subscriber.subscribe("fixations")

        # Drain any buffered messages to start fresh
        self._drain_buffer()

    def _drain_buffer(self, silent: bool = False) -> int:
        """Drain any buffered messages to prevent lag.

        Returns:
            Number of messages drained.
        """
        if self._subscriber is None:
            return 0

        count = 0
        while True:
            try:
                self._subscriber.recv_multipart(flags=zmq.NOBLOCK)
                count += 1
            except zmq.Again:
                break

        if count > 0 and not silent:
            print(f"[PupilClient] Drained {count} buffered messages.")
        return count

    # Gaze topic priority (prefer combined; for single eye, we pick by confidence)
    GAZE_COMBINED_TOPICS = {"gaze.3d.01.", "gaze.2d.01."}

    def _is_combined_gaze(self, topic: str) -> bool:
        """Check if topic is combined (binocular) gaze."""
        return any(topic.startswith(t) for t in self.GAZE_COMBINED_TOPICS)

    def _get_latest_messages(
        self,
    ) -> tuple[list[bytes] | None, list[bytes] | None, list[bytes] | None]:
        """Get the latest gaze, frame, and fixation messages, discarding older ones.

        This drains the buffer and returns only the most recent of each type.
        For gaze, prefers combined gaze; for single-eye, picks higher confidence.

        Returns:
            Tuple of (latest_gaze_parts, latest_frame_parts, latest_fixation_parts).
        """
        if self._subscriber is None:
            return None, None, None

        latest_gaze: list[bytes] | None = None
        latest_gaze_combined: bool = False
        latest_gaze_confidence: float = -1.0
        latest_frame: list[bytes] | None = None
        latest_fixation: list[bytes] | None = None

        def should_replace_gaze(topic: str, confidence: float) -> bool:
            """Check if new gaze should replace current best."""
            nonlocal latest_gaze_combined, latest_gaze_confidence
            is_combined = self._is_combined_gaze(topic)

            # Always prefer combined over single-eye
            if is_combined and not latest_gaze_combined:
                return True
            if not is_combined and latest_gaze_combined:
                return False

            # Same type: prefer higher confidence (or newer if equal)
            return confidence >= latest_gaze_confidence

        def update_gaze(parts: list[bytes], topic: str) -> None:
            """Update latest gaze tracking state."""
            nonlocal latest_gaze, latest_gaze_combined, latest_gaze_confidence
            # Parse confidence from payload
            payload = msgpack.loads(parts[1], raw=False)
            confidence = float(payload.get("confidence", 0.0))

            if should_replace_gaze(topic, confidence):
                latest_gaze = parts
                latest_gaze_combined = self._is_combined_gaze(topic)
                latest_gaze_confidence = confidence

        # First, do one blocking receive to ensure we have at least one message
        try:
            parts = self._subscriber.recv_multipart()
            topic = parts[0].decode("utf-8")
            if topic.startswith("gaze"):
                update_gaze(parts, topic)
            elif topic.startswith("frame.world"):
                latest_frame = parts
            elif topic.startswith("fixations"):
                latest_fixation = parts
        except zmq.Again:
            return None, None, None

        # Now drain all remaining messages, keeping only the best of each type
        while True:
            try:
                parts = self._subscriber.recv_multipart(flags=zmq.NOBLOCK)
                topic = parts[0].decode("utf-8")
                if topic.startswith("gaze"):
                    update_gaze(parts, topic)
                elif topic.startswith("frame.world"):
                    latest_frame = parts
                elif topic.startswith("fixations"):
                    latest_fixation = parts
            except zmq.Again:
                break

        return latest_gaze, latest_frame, latest_fixation

    def disconnect(self) -> None:
        """Disconnect from Pupil Capture and clean up resources."""
        if self._subscriber is not None:
            self._subscriber.close()
            self._subscriber = None
        if self._remote is not None:
            self._remote.close()
            self._remote = None
        if self._context is not None:
            self._context.term()
            self._context = None
        print("[PupilClient] Disconnected.")

    def _parse_gaze(self, topic: str, payload: dict) -> GazeData:
        """Parse a gaze message from the payload."""
        norm_pos = payload.get("norm_pos", (0.0, 0.0))
        return GazeData(
            timestamp=payload.get("timestamp", 0.0),
            norm_pos=(float(norm_pos[0]), float(norm_pos[1])),
            confidence=float(payload.get("confidence", 0.0)),
            topic=topic,
        )

    def _parse_fixation(self, topic: str, payload: dict) -> FixationData:
        """Parse a fixation message from the payload."""
        norm_pos = payload.get("norm_pos", (0.0, 0.0))
        return FixationData(
            id=int(payload.get("id", 0)),
            timestamp=payload.get("timestamp", 0.0),
            duration=float(payload.get("duration", 0.0)),
            norm_pos=(float(norm_pos[0]), float(norm_pos[1])),
            dispersion=float(payload.get("dispersion", 0.0)),
            confidence=float(payload.get("confidence", 0.0)),
            topic=topic,
        )

    def _parse_frame(self, topic: str, payload: dict) -> FrameData | None:
        """Parse a frame message from the payload.

        Handles both raw BGR frames and JPEG-encoded frames.
        """
        import cv2

        raw_data = payload.get("__raw_data__", [])
        if not raw_data:
            # Try alternative key
            raw_data = payload.get("data", [])
        if not raw_data:
            return None

        # Handle both list of bytes and bytes object
        if isinstance(raw_data, list) and len(raw_data) > 0:
            # First element is typically the frame data
            frame_bytes = raw_data[0] if isinstance(raw_data[0], bytes) else bytes(raw_data)
        elif isinstance(raw_data, bytes):
            frame_bytes = raw_data
        else:
            return None

        width = payload.get("width", 0)
        height = payload.get("height", 0)
        frame_format = payload.get("format", "bgr")

        if width == 0 or height == 0:
            return None

        try:
            # Convert bytes to numpy array
            image_data = np.frombuffer(frame_bytes, dtype=np.uint8)

            # Handle JPEG-encoded frames
            if frame_format == "jpeg":
                frame = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
                if frame is None:
                    return None
            else:
                # Raw BGR format
                expected_size = height * width * 3
                if image_data.size < expected_size:
                    return None
                frame = image_data[:expected_size].reshape((height, width, 3))

            return FrameData(
                timestamp=payload.get("timestamp", 0.0),
                width=frame.shape[1],
                height=frame.shape[0],
                data=frame,
                topic=topic,
            )
        except (ValueError, RuntimeError, cv2.error):
            pass
        return None

    def _parse_parts(self, parts: list[bytes]) -> Message:
        """Parse message parts into a Message object."""
        if len(parts) < 2:
            return Message()

        topic_bytes = parts[0]
        payload_bytes = parts[1]
        extra_data = parts[2] if len(parts) > 2 else None

        topic = topic_bytes.decode("utf-8")
        payload = msgpack.loads(payload_bytes, raw=False)

        if extra_data is not None and topic.startswith("frame"):
            payload["__raw_data__"] = [extra_data]

        if topic.startswith("gaze"):
            return Message(gaze=self._parse_gaze(topic, payload))
        elif topic.startswith("frame.world"):
            frame = self._parse_frame(topic, payload)
            if frame is not None:
                return Message(frame=frame)
        elif topic.startswith("fixations"):
            return Message(fixation=self._parse_fixation(topic, payload))

        return Message()

    def stream_realtime(self) -> Iterator[Message]:
        """Stream ONLY the latest messages from Pupil Capture (zero buffer).

        This method always drains the buffer and yields only the most recent
        gaze and frame data. This ensures real-time performance with no lag.

        Yields:
            Message objects containing the latest gaze and/or frame data.
        """
        if self._subscriber is None:
            raise RuntimeError("Client not connected. Call connect() first.")

        timeout_count = 0
        max_consecutive_timeouts = 3

        while True:
            try:
                # Get the LATEST messages, discarding any buffered old ones
                latest_gaze_parts, latest_frame_parts, latest_fixation_parts = (
                    self._get_latest_messages()
                )
                timeout_count = 0

                # Parse fixation if we got one
                fixation_data = None
                if latest_fixation_parts is not None:
                    fix_msg = self._parse_parts(latest_fixation_parts)
                    fixation_data = fix_msg.fixation

                # Yield frame message if we got one
                if latest_frame_parts is not None:
                    msg = self._parse_parts(latest_frame_parts)
                    if msg.frame is not None:
                        # Include latest gaze and fixation in the same message
                        gaze_data = None
                        if latest_gaze_parts is not None:
                            gaze_msg = self._parse_parts(latest_gaze_parts)
                            gaze_data = gaze_msg.gaze
                        yield Message(
                            gaze=gaze_data,
                            frame=msg.frame,
                            fixation=fixation_data,
                        )
                elif latest_gaze_parts is not None:
                    # Only gaze (and possibly fixation), no frame this cycle
                    gaze_msg = self._parse_parts(latest_gaze_parts)
                    yield Message(gaze=gaze_msg.gaze, fixation=fixation_data)
                elif fixation_data is not None:
                    # Only fixation
                    yield Message(fixation=fixation_data)

            except zmq.Again:
                timeout_count += 1
                print(
                    f"\n[PupilClient] Timeout waiting for data "
                    f"({timeout_count}/{max_consecutive_timeouts})..."
                )
                if timeout_count >= max_consecutive_timeouts:
                    print("[PupilClient] Too many timeouts. Check Pupil Capture status.")
                    print("  - Is gaze mapping enabled?")
                    print("  - Is Frame Publisher plugin enabled?")
                    break
            except zmq.ZMQError as e:
                print(f"[PupilClient] ZMQ Error: {e}")
                break
            except KeyboardInterrupt:
                break

    def stream(self) -> Iterator[Message]:
        """Stream messages from Pupil Capture.

        Note: For real-time performance, use stream_realtime() instead.

        Yields:
            Message objects containing either gaze or frame data.
        """
        # Delegate to real-time streaming
        yield from self.stream_realtime()

    def __enter__(self) -> "PupilCaptureClient":
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, *_: object) -> None:
        """Context manager exit."""
        self.disconnect()

