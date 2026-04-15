"""Colour and brightness detection at the gaze point.

Two classes work in sequence:

    FrameProcessor  — extracts the pixel region around the gaze point
    ColorAnalyzer   — analyses that region for hue, saturation, brightness
                      and maps them to musical note and octave

NoteGate sits on top of ColorAnalyzer and fires when the detected note
stabilises after a real gaze transition, suppressing jitter.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from enum import IntEnum

import cv2
import numpy as np
from numpy.typing import NDArray

from pupil_tracker.input.live import FrameData, GazeData


# ---------------------------------------------------------------------------
# Note mapping
# ---------------------------------------------------------------------------

class Note(IntEnum):
    """Musical notes mapped from colour wavelengths (longer → lower)."""
    C = 0  # Red   ~700nm
    D = 1  # Orange ~620nm
    E = 2  # Yellow ~580nm
    F = 3  # Green  ~530nm
    G = 4  # Cyan   ~500nm
    A = 5  # Blue   ~470nm
    B = 6  # Violet ~400nm


NOTE_SEMITONES = {
    Note.C: 0, Note.D: 2, Note.E: 4, Note.F: 5,
    Note.G: 7, Note.A: 9, Note.B: 11,
}


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GazeRegion:
    """A pixel region extracted from the frame around the gaze point."""
    center_x: int
    center_y: int
    region: NDArray[np.uint8]   # BGR image patch
    frame_width: int
    frame_height: int
    timestamp: float
    confidence: float


@dataclass(frozen=True)
class ColorReading:
    """Full colour/brightness reading for one frame."""
    timestamp: float
    note: Note
    octave: int
    midi_note: int
    raw_midi_note: int          # Pre-stability (for transition detection)
    hue: float | None           # Raw hue 0–179, None if saturation too low
    smoothed_hue: float
    saturation: float
    brightness: float
    smoothed_brightness: float
    center_x: int
    center_y: int
    confidence: float


@dataclass(frozen=True)
class NoteEvent:
    """A discrete note trigger produced by NoteGate."""
    timestamp: float
    note: Note
    octave: int
    midi_note: int
    brightness: float           # 0–1 normalised
    center_x: int
    center_y: int


# ---------------------------------------------------------------------------
# FrameProcessor
# ---------------------------------------------------------------------------

class FrameProcessor:
    """Extracts a pixel region around the current gaze point.

    Keeps the last frame and gaze sample independently so they can arrive
    at different rates (gaze ~120Hz, frames ~30Hz).
    """

    def __init__(self, region_size: int = 50) -> None:
        self._region_size = region_size
        self._last_frame: FrameData | None = None
        self._last_gaze: GazeData | None = None

    @property
    def region_size(self) -> int:
        return self._region_size

    @property
    def last_frame(self) -> FrameData | None:
        return self._last_frame

    @property
    def last_gaze(self) -> GazeData | None:
        return self._last_gaze

    def update_frame(self, frame: FrameData) -> None:
        self._last_frame = frame

    def update_gaze(self, gaze: GazeData, min_confidence: float = 0.5) -> bool:
        """Accept a gaze sample if it passes confidence and bounds checks."""
        if gaze.confidence < min_confidence:
            return False
        x, y = gaze.norm_pos
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            return False
        self._last_gaze = gaze
        return True

    def norm_to_pixel(
        self, norm_x: float, norm_y: float, width: int, height: int
    ) -> tuple[int, int]:
        """Convert normalised Pupil coordinates to OpenCV pixel coordinates.

        Pupil uses (0,0) at bottom-left; OpenCV uses top-left — flip Y.
        """
        return int(norm_x * width), int((1.0 - norm_y) * height)

    def extract_region(self) -> GazeRegion | None:
        """Extract the region around the current gaze point."""
        if self._last_frame is None or self._last_gaze is None:
            return None

        frame = self._last_frame
        gaze = self._last_gaze
        pixel_x, pixel_y = self.norm_to_pixel(
            gaze.norm_pos[0], gaze.norm_pos[1], frame.width, frame.height
        )

        half = self._region_size // 2
        x1 = max(0, pixel_x - half)
        x2 = min(frame.width, pixel_x + half)
        y1 = max(0, pixel_y - half)
        y2 = min(frame.height, pixel_y + half)

        return GazeRegion(
            center_x=pixel_x,
            center_y=pixel_y,
            region=frame.data[y1:y2, x1:x2].copy(),
            frame_width=frame.width,
            frame_height=frame.height,
            timestamp=gaze.timestamp,
            confidence=gaze.confidence,
        )


# ---------------------------------------------------------------------------
# ColorAnalyzer
# ---------------------------------------------------------------------------

class ColorAnalyzer:
    """Analyses colour and brightness of a GazeRegion.

    Maps hue → Note (C–B, wavelength order) and brightness → octave (3–6).
    Applies Gaussian-weighted spatial averaging and temporal smoothing with
    stability hysteresis to suppress jitter.
    """

    HUE_RANGES: list[tuple[tuple[int, int], Note]] = [
        ((0, 8), Note.C),
        ((8, 25), Note.D),
        ((25, 38), Note.E),
        ((38, 75), Note.F),
        ((75, 95), Note.G),
        ((95, 125), Note.A),
        ((125, 165), Note.B),
        ((165, 180), Note.C),
    ]

    MIN_OCTAVE = 2
    MAX_OCTAVE = 6
    DEFAULT_MIN_SATURATION = 20

    def __init__(
        self,
        smoothing_window: int = 3,
        note_stability_frames: int = 2,
        octave_stability_frames: int = 3,
        note_stability_threshold: float = 0.5,
        octave_stability_threshold: float = 0.5,
        min_saturation: int = DEFAULT_MIN_SATURATION,
    ) -> None:
        self._smoothing_window = smoothing_window
        self._note_stability_frames = note_stability_frames
        self._octave_stability_frames = octave_stability_frames
        self._note_stability_threshold = note_stability_threshold
        self._octave_stability_threshold = octave_stability_threshold
        self._min_saturation = min_saturation

        self._hue_history: deque[float] = deque(maxlen=smoothing_window)
        self._brightness_history: deque[float] = deque(maxlen=smoothing_window)
        self._saturation_history: deque[float] = deque(maxlen=smoothing_window)
        self._note_history: deque[Note] = deque(maxlen=note_stability_frames)
        self._octave_history: deque[int] = deque(maxlen=octave_stability_frames)
        self._current_note: Note = Note.C
        self._current_octave: int = self.MIN_OCTAVE
        self._gaussian_kernel: NDArray[np.float64] | None = None
        self._kernel_size: tuple[int, int] = (0, 0)

    def _get_gaussian_kernel(self, height: int, width: int) -> NDArray[np.float64]:
        if self._gaussian_kernel is not None and self._kernel_size == (height, width):
            return self._gaussian_kernel
        y = np.arange(height) - height / 2
        x = np.arange(width) - width / 2
        gauss_y = np.exp(-0.5 * (y / (height / 4)) ** 2)
        gauss_x = np.exp(-0.5 * (x / (width / 4)) ** 2)
        kernel = np.outer(gauss_y, gauss_x)
        kernel = kernel / kernel.sum()
        self._gaussian_kernel = kernel
        self._kernel_size = (height, width)
        return kernel

    def _weighted_mean_hsv(
        self, hsv: NDArray[np.uint8]
    ) -> tuple[float | None, float, float]:
        height, width = hsv.shape[:2]
        kernel = self._get_gaussian_kernel(height, width)

        saturation = hsv[:, :, 1].astype(np.float64)
        value = hsv[:, :, 2].astype(np.float64)
        mean_saturation = float(np.sum(saturation * kernel))
        mean_value = float(np.sum(value * kernel))

        sat_mask = saturation >= self._min_saturation
        if not np.any(sat_mask):
            return None, mean_saturation, mean_value

        masked_kernel = kernel * sat_mask
        kernel_sum = masked_kernel.sum()
        if kernel_sum < 1e-10:
            return None, mean_saturation, mean_value
        masked_kernel = masked_kernel / kernel_sum

        hue = hsv[:, :, 0].astype(np.float64)
        hue_rad = hue * (2 * np.pi / 180)
        cos_sum = np.sum(np.cos(hue_rad) * masked_kernel)
        sin_sum = np.sum(np.sin(hue_rad) * masked_kernel)
        mean_hue = (np.arctan2(sin_sum, cos_sum) * 180 / (2 * np.pi)) % 180

        return float(mean_hue), mean_saturation, mean_value

    def _hue_to_note(self, hue: float) -> Note:
        for (low, high), note in self.HUE_RANGES:
            if low <= hue < high:
                return note
        return Note.C

    def _get_stable_note(self, raw_note: Note) -> Note:
        self._note_history.append(raw_note)
        if len(self._note_history) >= self._note_stability_frames:
            counts = Counter(self._note_history)
            most_common, count = counts.most_common(1)[0]
            if count >= self._note_stability_frames * self._note_stability_threshold:
                self._current_note = most_common
        return self._current_note

    def _brightness_to_octave(self, brightness: float) -> int:
        octave_range = self.MAX_OCTAVE - self.MIN_OCTAVE
        octave = self.MIN_OCTAVE + int(brightness / 255 * octave_range)
        octave = min(max(octave, self.MIN_OCTAVE), self.MAX_OCTAVE)
        return 3 if octave == 2 else octave

    def _get_stable_octave(self, raw_octave: int) -> int:
        self._octave_history.append(raw_octave)
        if len(self._octave_history) >= self._octave_stability_frames:
            counts = Counter(self._octave_history)
            most_common, count = counts.most_common(1)[0]
            if count >= self._octave_stability_frames * self._octave_stability_threshold:
                self._current_octave = most_common
        return self._current_octave

    def _calculate_midi_note(self, note: Note, octave: int) -> int:
        return min(max((octave + 1) * 12 + NOTE_SEMITONES[note], 0), 127)

    def analyze(self, gaze_region: GazeRegion) -> ColorReading:
        """Analyse a gaze region and return a ColorReading."""
        region = gaze_region.region

        if region.size == 0:
            current_midi = self._calculate_midi_note(self._current_note, self._current_octave)
            return ColorReading(
                timestamp=gaze_region.timestamp,
                note=self._current_note,
                octave=self._current_octave,
                midi_note=current_midi,
                raw_midi_note=current_midi,
                hue=0.0, smoothed_hue=0.0, saturation=0.0,
                brightness=0.0, smoothed_brightness=0.0,
                center_x=gaze_region.center_x,
                center_y=gaze_region.center_y,
                confidence=gaze_region.confidence,
            )

        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        raw_hue, raw_saturation, raw_value = self._weighted_mean_hsv(hsv)

        self._saturation_history.append(raw_saturation)
        self._brightness_history.append(raw_value)
        if raw_hue is not None:
            self._hue_history.append(raw_hue)

        smoothed_brightness = (
            sum(self._brightness_history) / len(self._brightness_history)
        )

        if self._hue_history:
            hue_rad = [h * (2 * np.pi / 180) for h in self._hue_history]
            cos_mean = sum(np.cos(h) for h in hue_rad) / len(hue_rad)
            sin_mean = sum(np.sin(h) for h in hue_rad) / len(hue_rad)
            smoothed_hue = (np.arctan2(sin_mean, cos_mean) * 180 / (2 * np.pi)) % 180
            stable_note = self._get_stable_note(self._hue_to_note(smoothed_hue))
        else:
            smoothed_hue = 0.0
            stable_note = self._current_note

        raw_octave = self._brightness_to_octave(smoothed_brightness)
        stable_octave = self._get_stable_octave(raw_octave)

        midi_note = self._calculate_midi_note(stable_note, stable_octave)
        raw_note_val = self._hue_to_note(raw_hue) if raw_hue is not None else stable_note
        raw_midi_note = self._calculate_midi_note(raw_note_val, stable_octave)

        return ColorReading(
            timestamp=gaze_region.timestamp,
            note=stable_note,
            octave=stable_octave,
            midi_note=midi_note,
            raw_midi_note=raw_midi_note,
            hue=raw_hue,
            smoothed_hue=smoothed_hue,
            saturation=sum(self._saturation_history) / len(self._saturation_history),
            brightness=raw_value,
            smoothed_brightness=smoothed_brightness,
            center_x=gaze_region.center_x,
            center_y=gaze_region.center_y,
            confidence=gaze_region.confidence,
        )

    def reset(self) -> None:
        self._hue_history.clear()
        self._brightness_history.clear()
        self._saturation_history.clear()
        self._note_history.clear()
        self._octave_history.clear()
        self._current_note = Note.C
        self._current_octave = self.MIN_OCTAVE


# ---------------------------------------------------------------------------
# NoteGate
# ---------------------------------------------------------------------------

class NoteGate:
    """Fires when the detected MIDI note stabilises after a real gaze transition.

    Suppresses jitter and duplicate triggers while resting on the same content.
    Responds to fixation ID changes as well as raw note transitions.
    """

    def __init__(self, stable_frames: int = 4, min_transition_frames: int = 3):
        self.stable_frames = stable_frames
        self.min_transition_frames = min_transition_frames
        self._recent: deque[int] = deque(maxlen=stable_frames)
        self._last_triggered: int | None = None
        self._diff_streak: int = 0
        self._transition_detected: bool = False
        self._last_fixation_id: int | None = None

    def new_fixation(self, fixation_id: int) -> None:
        if self._last_fixation_id is not None and fixation_id != self._last_fixation_id:
            self._transition_detected = True
        self._last_fixation_id = fixation_id

    def update(self, midi_note: int, raw_midi_note: int | None = None) -> bool:
        self._recent.append(midi_note)

        detect_note = raw_midi_note if raw_midi_note is not None else midi_note
        if self._last_triggered is not None and detect_note != self._last_triggered:
            self._diff_streak += 1
            if self._diff_streak >= self.min_transition_frames:
                self._transition_detected = True
        else:
            self._diff_streak = 0

        if len(self._recent) < self.stable_frames:
            return False
        if len(set(self._recent)) != 1:
            return False

        current = self._recent[0]
        if current != self._last_triggered or self._transition_detected:
            self._last_triggered = current
            self._transition_detected = False
            self._diff_streak = 0
            return True

        return False
