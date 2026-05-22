"""Blink, flutter, and intentional closure detection.

Handles everything related to eyelid events:
- Pairing onset/offset events from Pupil Capture into complete blinks
- Classifying blinks by duration (normal, intentional, ambiguous)
- Detecting flutter bursts from rapid blink sequences
- Tracking whether eyes are currently closed (between onset and offset)

The StreamingBlinkTracker is used by tracker.py for live tracking.
The same types and constants are used by input/recording.py for playback.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from enum import Enum

# --- Thresholds ---

BLINK_MAX_MS = 400           # Blinks shorter than this are normal blinks
INTENTIONAL_MIN_MS = 500     # Blinks longer than this are intentional closures
INTENTIONAL_GESTURE_MS = 600 # Eyes closed this long = intentional closure gesture
FLUTTER_WINDOW_S = 1.5    # Sliding window for flutter detection
FLUTTER_MIN_BLINKS = 3    # Minimum blinks in window to qualify as flutter
FLUTTER_END_TIMEOUT_S = 0.3  # Flutter ends when no new blink arrives within this time
ONSET_DEBOUNCE_S = 0.05      # Consecutive onsets within this window = same blink (binocular duplicate)


# --- Types ---

class BlinkType(Enum):
    """Classification of a blink by its duration."""
    BLINK = "blink"            # < 400ms
    INTENTIONAL = "intentional"  # >= 500ms
    AMBIGUOUS = "ambiguous"    # 400–500ms


@dataclass
class BlinkSample:
    """A blink event paired from onset/offset detections."""
    timestamp: float          # Onset timestamp
    duration_ms: float        # Duration in ms (-1 if no offset detected)
    confidence: float         # Average confidence of onset/offset
    blink_type: BlinkType = BlinkType.BLINK


@dataclass
class FlutterEvent:
    """A rapid eye flutter burst detected from rapid blink onsets."""
    timestamp: float    # Start of the flutter burst
    duration_s: float   # How long the flutter pattern lasted
    blink_count: int    # Number of blinks in the burst


def classify_blink(duration_ms: float) -> BlinkType:
    """Classify a blink by its duration."""
    if duration_ms < 0:
        return BlinkType.BLINK  # Unpaired onset, assume normal blink
    if duration_ms <= BLINK_MAX_MS:
        return BlinkType.BLINK
    if duration_ms >= INTENTIONAL_MIN_MS:
        return BlinkType.INTENTIONAL
    return BlinkType.AMBIGUOUS


# --- Detector ---

class StreamingBlinkTracker:
    """Tracks blink onset/offset events in real time.

    - Pairs onset/offset events to compute duration
    - Classifies each blink by duration
    - Detects flutter as FLUTTER_MIN_BLINKS+ blinks within FLUTTER_WINDOW_S
    - Tracks whether eyes are currently closed (between onset and offset)
    """

    def __init__(self) -> None:
        self._pending_onset_ts: float | None = None
        self._pending_onset_conf: float = 0.0
        self._last_onset_mono: float = 0.0

        self._blink_count = 0

        self._blink_times: deque[float] = deque()
        self._flutter_start: float | None = None
        self._flutter_start_mono: float = 0.0
        self._flutter_count = 0
        self._flutter_blink_count = 0
        self._last_blink_mono: float = 0.0

    @property
    def blink_count(self) -> int:
        return self._blink_count

    @property
    def flutter_count(self) -> int:
        return self._flutter_count

    @property
    def is_flutter_active(self) -> bool:
        return self._flutter_start is not None

    @property
    def active_flutter_blink_count(self) -> int:
        return self._flutter_blink_count

    @property
    def is_eyes_closed(self) -> bool:
        """True between blink onset and offset (eyes currently closed)."""
        return self._pending_onset_ts is not None

    @property
    def eyes_closed_elapsed_ms(self) -> float:
        """Milliseconds since the most recent onset. 0.0 if eyes are open.

        Resets on every new onset, so rapid blinking never accumulates time
        across multiple blinks the way a patch-side timer would.
        """
        if self._pending_onset_ts is None:
            return 0.0
        return (time.monotonic() - self._last_onset_mono) * 1000

    def update(
        self, blink_type: str, timestamp: float, confidence: float
    ) -> tuple[BlinkSample | None, FlutterEvent | None]:
        """Feed a blink onset or offset event.

        Args:
            blink_type: "onset" or "offset" from Pupil Capture's blink detector.
            timestamp: Event timestamp.
            confidence: Detection confidence.

        Returns:
            Tuple of (completed_blink, flutter_event).
        """
        blink: BlinkSample | None = None

        if blink_type == "onset":
            now_mono = time.monotonic()
            if self._pending_onset_ts is not None:
                # Count the previous onset as a blink only if enough wall time
                # has passed — within ONSET_DEBOUNCE_S it's the other eye for
                # the same binocular blink, not a new blink.
                if now_mono - self._last_onset_mono >= ONSET_DEBOUNCE_S:
                    self._blink_count += 1
                    self._blink_times.append(self._pending_onset_ts)
                    self._last_blink_mono = now_mono
                    if self._flutter_start is not None:
                        self._flutter_blink_count += 1
            self._pending_onset_ts = timestamp
            self._pending_onset_conf = confidence
            self._last_onset_mono = now_mono

        elif blink_type == "offset":
            if self._pending_onset_ts is not None:
                duration_ms = (timestamp - self._pending_onset_ts) * 1000
                avg_conf = (self._pending_onset_conf + confidence) / 2
                blink = BlinkSample(
                    timestamp=self._pending_onset_ts,
                    duration_ms=duration_ms,
                    confidence=avg_conf,
                    blink_type=classify_blink(duration_ms),
                )
                self._blink_count += 1
                self._blink_times.append(self._pending_onset_ts)
                self._last_blink_mono = time.monotonic()
                if self._flutter_start is not None:
                    self._flutter_blink_count += 1
                self._pending_onset_ts = None

        self._check_flutter(timestamp)
        return blink, None

    def tick(self, now_mono: float) -> FlutterEvent | None:
        """Check for flutter end by timeout. Call every loop iteration.

        Returns a FlutterEvent when flutter ends, None otherwise.
        """
        if self._flutter_start is not None:
            if now_mono - self._last_blink_mono > FLUTTER_END_TIMEOUT_S:
                event = FlutterEvent(
                    timestamp=self._flutter_start,
                    duration_s=self._last_blink_mono - self._flutter_start_mono,
                    blink_count=self._flutter_blink_count,
                )
                self._flutter_start = None
                self._flutter_start_mono = 0.0
                self._flutter_blink_count = 0
                self._flutter_count += 1
                return event
        return None

    def _check_flutter(self, now: float) -> None:
        window_start = now - FLUTTER_WINDOW_S
        while self._blink_times and self._blink_times[0] < window_start:
            self._blink_times.popleft()

        count = len(self._blink_times)
        if count >= FLUTTER_MIN_BLINKS and self._flutter_start is None:
            self._flutter_start = now
            self._flutter_start_mono = time.monotonic()
            self._flutter_blink_count = count
