"""Detection pipeline: owns all detectors and populates SignalBus.

Two processing paths share the same detectors:

    process_live(message, now)              — fed from the ZMQ stream (tracker.py)
    process_recording_frame(...)            — fed frame-by-frame from a recording (player.py)

Both return the same SignalBus instance (updated in place) so callers can
immediately pass it to patch.update(signals, outputs).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from eye_synth.input.live import FrameData
from eye_synth.signals.bus import SignalBus
from eye_synth.signals.env_color import ColorAnalyzer, ColorReading, FrameProcessor, GazeRegion
from eye_synth.signals.env_scene_change import SceneChangeDetector
from eye_synth.signals.eye_blinks import FlutterEvent, StreamingBlinkTracker
from eye_synth.signals.eye_gaze import GazeVelocityTracker

if TYPE_CHECKING:
    from eye_synth.input.live import Message
    from eye_synth.input.recording import BlinkSample, FixationSample, GazeSample


class Pipeline:
    """Owns all detectors and populates SignalBus each iteration.

    Attributes:
        signals:            The SignalBus updated by each process_* call.
        blink_tracker:      Live blink/flutter detector (exposed for display state in tracker.py).
        processor:          Live frame/gaze processor (exposed for display in tracker.py).
        last_color_reading: Most recent ColorReading; None until the first gaze region is analyzed.
    """

    def __init__(
        self,
        region_size: int = 50,
        smoothing: int = 3,
    ) -> None:
        self.region_size = region_size

        # Shared detectors
        self.analyzer = ColorAnalyzer(smoothing_window=smoothing)
        self.scene_detector = SceneChangeDetector()
        self.gaze_vel = GazeVelocityTracker()

        # Live-only detectors (unused in recording path)
        self.blink_tracker = StreamingBlinkTracker()
        self.processor = FrameProcessor(region_size=region_size)

        # Recording-path flutter transition state
        self._prev_in_flutter: bool = False
        self._last_flutter_event: FlutterEvent | None = None

        self.signals = SignalBus()
        self.last_color_reading: ColorReading | None = None

    # ------------------------------------------------------------------
    # Live path
    # ------------------------------------------------------------------

    def process_live(
        self,
        message: Message,
        now: float,
        frame_data: NDArray[np.uint8] | None = None,
    ) -> SignalBus:
        """Process one ZMQ message from Pupil Capture.

        Args:
            message:    Message yielded by PupilCaptureClient.stream_realtime().
            now:        Current time from time.monotonic().
            frame_data: Gamma-corrected frame array. If None, message.frame.data is used as-is.

        Returns:
            The updated SignalBus (same object each call).
        """
        s = self.signals
        s.clear_events()
        s.timestamp = now

        if message.gaze is not None:
            if self.processor.update_gaze(message.gaze):
                s.eye.confidence = message.gaze.confidence
                s.eye.norm_pos = message.gaze.norm_pos

        if message.fixation is not None:
            s.eye.fixation_id = message.fixation.id

        if message.blink is not None:
            b = message.blink
            blink_event, _ = self.blink_tracker.update(b.blink_type, b.timestamp, b.confidence)
            if blink_event is not None:
                s.eye.blink = blink_event

        flutter_event = self.blink_tracker.tick(now)
        if flutter_event is not None:
            s.eye.flutter = flutter_event

        s.eye.is_eyes_closed = self.blink_tracker.is_eyes_closed
        s.eye.is_flutter_active = self.blink_tracker.is_flutter_active
        s.eye.flutter_blink_count = self.blink_tracker.active_flutter_blink_count
        s.eye.total_blinks = self.blink_tracker.blink_count
        s.eye.total_flutters = self.blink_tracker.flutter_count

        if message.frame is not None:
            actual_data = frame_data if frame_data is not None else message.frame.data

            if frame_data is not None:
                corrected = FrameData(
                    timestamp=message.frame.timestamp,
                    width=message.frame.width,
                    height=message.frame.height,
                    data=frame_data,
                    topic=message.frame.topic,
                )
                self.processor.update_frame(corrected)
            else:
                self.processor.update_frame(message.frame)

            s.frame_width = message.frame.width
            s.frame_height = message.frame.height
            s.env.scene_change = self.scene_detector.update(actual_data)

            gaze_region = self.processor.extract_region()
            if gaze_region is not None:
                color_reading = self.analyzer.analyze(gaze_region)
                self.last_color_reading = color_reading
                self._populate_env(color_reading)
                s.eye.px_pos = (gaze_region.center_x, gaze_region.center_y)
                if self.processor.last_gaze is not None:
                    self.gaze_vel.update(
                        self.processor.last_gaze.norm_pos,
                        self.processor.last_gaze.timestamp,
                        message.frame.width,
                        message.frame.height,
                    )
                    s.eye.velocity_px_s = self.gaze_vel.velocity

        return s

    # ------------------------------------------------------------------
    # Recording path
    # ------------------------------------------------------------------

    def process_recording_frame(
        self,
        frame: NDArray[np.uint8],
        gaze: GazeSample | None,
        blink: BlinkSample | None,
        flutter: FlutterEvent | None,
        fixation: FixationSample | None,
        timestamp: float,
        gaze_px: tuple[int, int] | None,
        min_confidence: float = 0.0,
    ) -> SignalBus:
        """Process one frame from a recording.

        Args:
            frame:          BGR frame array (already gamma-corrected if needed).
            gaze:           Gaze sample for this frame, or None.
            blink:          Blink event at this timestamp, or None.
            flutter:        Flutter event at this timestamp, or None.
            fixation:       Fixation event for this frame, or None.
            timestamp:      World timestamp for this frame.
            gaze_px:        Pre-computed pixel position of gaze (all confidence levels).
                            Pass None if gaze is absent.
            min_confidence: Minimum gaze confidence required for color analysis.
                            Position and velocity are tracked regardless.

        Returns:
            The updated SignalBus (same object each call).
        """
        s = self.signals
        s.clear_events()
        s.timestamp = timestamp
        height, width = frame.shape[:2]
        s.frame_width = width
        s.frame_height = height

        if gaze is not None:
            s.eye.confidence = gaze.confidence
            s.eye.norm_pos = gaze.norm_pos

        if gaze_px is not None:
            s.eye.px_pos = gaze_px
            if gaze is not None:
                self.gaze_vel.update(gaze.norm_pos, gaze.timestamp, width, height)
            s.eye.velocity_px_s = self.gaze_vel.velocity

        if blink is not None:
            s.eye.blink = blink
        s.eye.is_eyes_closed = blink is not None

        if fixation is not None:
            s.eye.fixation_id = fixation.id

        in_flutter = flutter is not None
        if self._prev_in_flutter and not in_flutter and self._last_flutter_event is not None:
            s.eye.flutter = self._last_flutter_event
        s.eye.is_flutter_active = in_flutter
        s.eye.flutter_blink_count = flutter.blink_count if flutter else 0
        if in_flutter and flutter is not None:
            self._last_flutter_event = flutter
        self._prev_in_flutter = in_flutter

        s.env.scene_change = self.scene_detector.update(frame)

        confidence_ok = gaze is None or gaze.confidence >= min_confidence
        if gaze_px is not None and confidence_ok:
            region_array = self._crop_region(frame, gaze_px)
            if region_array is not None:
                gaze_region = GazeRegion(
                    center_x=gaze_px[0],
                    center_y=gaze_px[1],
                    region=region_array,
                    frame_width=width,
                    frame_height=height,
                    timestamp=timestamp,
                    confidence=gaze.confidence if gaze is not None else 0.0,
                )
                color_reading = self.analyzer.analyze(gaze_region)
                self.last_color_reading = color_reading
                self._populate_env(color_reading)

        return s

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all stateful detectors. Call on seek in the player."""
        self.scene_detector.reset()
        self.gaze_vel.reset()
        self.analyzer.reset()
        self.blink_tracker = StreamingBlinkTracker()
        self.processor = FrameProcessor(region_size=self.region_size)
        self._prev_in_flutter = False
        self._last_flutter_event = None
        self.last_color_reading = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _populate_env(self, cr: ColorReading) -> None:
        s = self.signals
        s.env.hue = cr.smoothed_hue
        s.env.raw_hue = cr.hue
        s.env.hue_normalized = cr.smoothed_hue / 179.0
        s.env.saturation = cr.saturation
        s.env.brightness = cr.smoothed_brightness
        s.env.brightness_normalized = cr.smoothed_brightness / 255.0
        s.has_env_reading = True

    def _crop_region(
        self, frame: NDArray[np.uint8], gaze_px: tuple[int, int]
    ) -> NDArray[np.uint8] | None:
        height, width = frame.shape[:2]
        gx, gy = gaze_px
        half = self.region_size // 2
        x1 = max(0, gx - half)
        y1 = max(0, gy - half)
        x2 = min(width, gx + half)
        y2 = min(height, gy + half)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2]
