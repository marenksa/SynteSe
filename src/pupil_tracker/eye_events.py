"""Streaming blink event processing for real-time use.

Pairs blink onset/offset events from Pupil Capture's built-in detector,
classifies by duration, and detects flutter from rapid blink bursts.
"""

from __future__ import annotations

import time
from collections import deque

from pupil_tracker.recording import (
    BLINK_MAX_MS,
    FLUTTER_END_TIMEOUT_S,
    FLUTTER_MIN_BLINKS,
    FLUTTER_WINDOW_S,
    INTENTIONAL_MIN_MS,
    BlinkSample,
    BlinkType,
    FlutterEvent,
    classify_blink,
)


class StreamingBlinkTracker:
    """Tracks blink onset/offset events in real time.

    - Pairs onset→offset to compute duration
    - Classifies each blink by duration (blink/intentional/ambiguous)
    - Detects flutter as 3+ blinks within a 2s sliding window
    """

    def __init__(self) -> None:
        # Pending onset waiting for its offset
        self._pending_onset_ts: float | None = None
        self._pending_onset_conf: float = 0.0

        # Completed blinks
        self._blink_count = 0

        # Sliding window of blink timestamps for flutter detection
        self._blink_times: deque[float] = deque()
        self._flutter_start: float | None = None
        self._flutter_start_mono: float = 0.0  # monotonic time of flutter onset
        self._flutter_count = 0
        self._flutter_blink_count = 0  # Blinks accumulated during current flutter
        self._last_blink_mono: float = 0.0  # monotonic time of most recent blink

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
            - completed_blink is set when an onset→offset pair completes.
            - flutter_event is set when a flutter burst ends.
        """
        blink: BlinkSample | None = None

        if blink_type == "onset":
            # If we had a pending onset without offset, emit it as unpaired
            if self._pending_onset_ts is not None:
                blink = BlinkSample(
                    timestamp=self._pending_onset_ts,
                    duration_ms=-1,
                    confidence=self._pending_onset_conf,
                    blink_type=BlinkType.BLINK,
                )
                self._blink_count += 1
                self._blink_times.append(self._pending_onset_ts)
                self._last_blink_mono = time.monotonic()
                if self._flutter_start is not None:
                    self._flutter_blink_count += 1

            # Start new pending onset
            self._pending_onset_ts = timestamp
            self._pending_onset_conf = confidence

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

        # Check for flutter onset (sliding window)
        self._check_flutter(timestamp)

        return blink, None

    def tick(self, now_mono: float) -> FlutterEvent | None:
        """Check for flutter end by timeout. Call every loop iteration.

        Flutter ends when no new blink arrives within FLUTTER_END_TIMEOUT_S.
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
        """Detect flutter onset from recent blinks (sliding window)."""
        # Expire old blinks outside the window
        window_start = now - FLUTTER_WINDOW_S
        while self._blink_times and self._blink_times[0] < window_start:
            self._blink_times.popleft()

        count = len(self._blink_times)

        if count >= FLUTTER_MIN_BLINKS and self._flutter_start is None:
            # Flutter onset: becomes active immediately at the Nth blink
            self._flutter_start = now
            self._flutter_start_mono = time.monotonic()
            self._flutter_blink_count = count
