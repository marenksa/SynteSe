"""Brightness and color analyzer for gaze regions."""

from collections import deque
from dataclasses import dataclass
from enum import IntEnum

import cv2
import numpy as np
from numpy.typing import NDArray

from pupil_tracker.processor import GazeRegion


class Note(IntEnum):
    """Musical notes mapped from color wavelengths.

    Longer wavelength colors map to lower notes:
    - Red (longest wavelength ~700nm) → C (lowest)
    - Violet (shortest wavelength ~400nm) → B (highest)
    """

    C = 0  # Red
    D = 1  # Orange
    E = 2  # Yellow
    F = 3  # Green
    G = 4  # Cyan
    A = 5  # Blue
    B = 6  # Violet


# Semitone offsets for each note in the major scale
NOTE_SEMITONES = {
    Note.C: 0,
    Note.D: 2,
    Note.E: 4,
    Note.F: 5,
    Note.G: 7,
    Note.A: 9,
    Note.B: 11,
}


@dataclass(frozen=True)
class BrightnessReading:
    """A brightness reading at a specific point in time."""

    timestamp: float
    brightness: float  # 0-255 scale
    smoothed_brightness: float  # Smoothed value
    center_x: int
    center_y: int
    confidence: float


@dataclass(frozen=True)
class ColorReading:
    """A color and brightness reading for musical synthesis.

    Maps color to musical notes and brightness to octave:
    - Hue → Note (C through B based on wavelength)
    - Brightness → Octave (darker = lower octave)
    """

    timestamp: float
    note: Note  # Musical note from color
    octave: int  # Octave from brightness (2-6)
    midi_note: int  # Combined MIDI note number
    hue: float | None  # Raw hue value (0-179), None if saturation too low
    smoothed_hue: float  # Smoothed hue for display
    saturation: float  # Color saturation (0-255)
    brightness: float  # Value/brightness (0-255)
    smoothed_brightness: float  # Smoothed brightness
    center_x: int
    center_y: int
    confidence: float


class BrightnessAnalyzer:
    """Analyzes brightness in gaze regions."""

    def __init__(self, smoothing_window: int = 5) -> None:
        """Initialize the brightness analyzer.

        Args:
            smoothing_window: Number of readings to average for smoothing.
        """
        self._smoothing_window = smoothing_window
        self._brightness_history: deque[float] = deque(maxlen=smoothing_window)

    @property
    def smoothing_window(self) -> int:
        """Get the smoothing window size."""
        return self._smoothing_window

    def calculate_brightness(self, region: NDArray[np.uint8]) -> float:
        """Calculate the mean brightness of a BGR image region.

        Uses the luminance formula: Y = 0.299*R + 0.587*G + 0.114*B

        Args:
            region: BGR image array.

        Returns:
            Mean brightness value (0-255).
        """
        if region.size == 0:
            return 0.0

        # Convert to grayscale using luminance formula
        # OpenCV uses BGR order, so we need to account for that
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray))

    def analyze(self, gaze_region: GazeRegion) -> BrightnessReading:
        """Analyze the brightness of a gaze region.

        Args:
            gaze_region: The extracted region around the gaze point.

        Returns:
            BrightnessReading with raw and smoothed brightness values.
        """
        brightness = self.calculate_brightness(gaze_region.region)

        # Add to history for smoothing
        self._brightness_history.append(brightness)

        # Calculate smoothed value
        smoothed = sum(self._brightness_history) / len(self._brightness_history)

        return BrightnessReading(
            timestamp=gaze_region.timestamp,
            brightness=brightness,
            smoothed_brightness=smoothed,
            center_x=gaze_region.center_x,
            center_y=gaze_region.center_y,
            confidence=gaze_region.confidence,
        )

    def reset(self) -> None:
        """Reset the brightness history."""
        self._brightness_history.clear()


class ColorAnalyzer:
    """Analyzes color and brightness for musical note generation.

    Maps visible light spectrum to musical notes based on wavelength:
    - Longer wavelength (red) → lower notes
    - Shorter wavelength (violet) → higher notes

    Brightness determines the octave:
    - Darker → lower octave (2)
    - Brighter → higher octave (6)

    Uses Gaussian-weighted averaging (center pixels weigh more) and
    temporal smoothing for stable output.
    """

    # Hue ranges in OpenCV (0-179) mapped to notes
    # Adjusted ranges for better color separation (especially orange)
    HUE_RANGES: list[tuple[tuple[int, int], Note]] = [
        ((0, 8), Note.C),  # Red (low hue)
        ((8, 25), Note.D),  # Orange (expanded range)
        ((25, 38), Note.E),  # Yellow
        ((38, 75), Note.F),  # Green
        ((75, 95), Note.G),  # Cyan
        ((95, 125), Note.A),  # Blue
        ((125, 165), Note.B),  # Violet/Purple
        ((165, 180), Note.C),  # Red (high hue, wraps around)
    ]

    # Octave range for brightness mapping
    MIN_OCTAVE = 2
    MAX_OCTAVE = 6

    # Default minimum saturation for reliable hue detection (0-255 scale)
    # Pixels below this threshold have unreliable hue values
    DEFAULT_MIN_SATURATION = 50

    def __init__(
        self,
        smoothing_window: int = 15,
        note_stability_frames: int = 8,
        octave_stability_frames: int = 15,
        note_stability_threshold: float = 0.7,
        octave_stability_threshold: float = 0.8,
        min_saturation: int = DEFAULT_MIN_SATURATION,
    ) -> None:
        """Initialize the color analyzer.

        Args:
            smoothing_window: Number of readings to average for smoothing.
            note_stability_frames: Frames a note must be stable before changing.
            octave_stability_frames: Frames an octave must be stable before changing.
            note_stability_threshold: Fraction of frames that must agree for note change (0-1).
            octave_stability_threshold: Fraction of frames that must agree for octave change (0-1).
            min_saturation: Minimum saturation (0-255) for hue to be considered
                reliable. Pixels below this are excluded from note detection.
        """
        self._smoothing_window = smoothing_window
        self._note_stability_frames = note_stability_frames
        self._octave_stability_frames = octave_stability_frames
        self._note_stability_threshold = note_stability_threshold
        self._octave_stability_threshold = octave_stability_threshold
        self._min_saturation = min_saturation

        # Smoothing histories
        self._hue_history: deque[float] = deque(maxlen=smoothing_window)
        self._brightness_history: deque[float] = deque(maxlen=smoothing_window)
        self._saturation_history: deque[float] = deque(maxlen=smoothing_window)

        # Stability tracking for note changes
        self._note_history: deque[Note] = deque(maxlen=note_stability_frames)
        self._current_note: Note = Note.C

        # Stability tracking for octave changes
        self._octave_history: deque[int] = deque(maxlen=octave_stability_frames)
        self._current_octave: int = self.MIN_OCTAVE

        # Cached Gaussian kernel (will be created on first use)
        self._gaussian_kernel: NDArray[np.float64] | None = None
        self._kernel_size: tuple[int, int] = (0, 0)

    def _get_gaussian_kernel(self, height: int, width: int) -> NDArray[np.float64]:
        """Get or create a 2D Gaussian kernel for weighted averaging.

        Args:
            height: Height of the region.
            width: Width of the region.

        Returns:
            2D Gaussian kernel normalized to sum to 1.
        """
        if self._gaussian_kernel is not None and self._kernel_size == (height, width):
            return self._gaussian_kernel

        # Create 1D Gaussian kernels
        sigma_y = height / 4  # Standard deviation
        sigma_x = width / 4
        y = np.arange(height) - height / 2
        x = np.arange(width) - width / 2

        # Create 2D Gaussian
        gauss_y = np.exp(-0.5 * (y / sigma_y) ** 2)
        gauss_x = np.exp(-0.5 * (x / sigma_x) ** 2)
        kernel = np.outer(gauss_y, gauss_x)

        # Normalize to sum to 1
        kernel = kernel / kernel.sum()

        self._gaussian_kernel = kernel
        self._kernel_size = (height, width)
        return kernel

    def _weighted_mean_hsv(
        self, hsv: NDArray[np.uint8]
    ) -> tuple[float | None, float, float]:
        """Calculate Gaussian-weighted mean of HSV channels.

        For hue calculation, only pixels with saturation >= min_saturation are
        considered, since low-saturation pixels have unreliable hue values.

        Args:
            hsv: HSV image array (H, W, 3).

        Returns:
            Tuple of (weighted_hue, weighted_saturation, weighted_value).
            Hue may be None if no pixels meet the saturation threshold.
        """
        height, width = hsv.shape[:2]
        kernel = self._get_gaussian_kernel(height, width)

        # Saturation and Value are calculated from all pixels
        saturation = hsv[:, :, 1].astype(np.float64)
        value = hsv[:, :, 2].astype(np.float64)
        mean_saturation = np.sum(saturation * kernel)
        mean_value = np.sum(value * kernel)

        # For hue, only consider pixels with sufficient saturation
        # Low saturation = gray-ish pixels with unreliable hue
        sat_mask = saturation >= self._min_saturation

        if not np.any(sat_mask):
            # No pixels meet saturation threshold - hue is unreliable
            return None, float(mean_saturation), float(mean_value)

        # Create masked kernel (only high-saturation pixels contribute to hue)
        masked_kernel = kernel * sat_mask
        kernel_sum = masked_kernel.sum()

        if kernel_sum < 1e-10:
            # Edge case: masked kernel has negligible weight
            return None, float(mean_saturation), float(mean_value)

        # Normalize masked kernel
        masked_kernel = masked_kernel / kernel_sum

        # Hue requires special handling due to circular nature (0-179 wraps)
        # Convert to complex representation for circular mean
        hue = hsv[:, :, 0].astype(np.float64)
        hue_rad = hue * (2 * np.pi / 180)  # Convert to radians
        cos_sum = np.sum(np.cos(hue_rad) * masked_kernel)
        sin_sum = np.sum(np.sin(hue_rad) * masked_kernel)
        mean_hue_rad = np.arctan2(sin_sum, cos_sum)
        mean_hue = (mean_hue_rad * 180 / (2 * np.pi)) % 180

        return float(mean_hue), float(mean_saturation), float(mean_value)

    def _hue_to_note(self, hue: float) -> Note:
        """Convert OpenCV hue (0-179) to a musical note.

        Args:
            hue: Hue value from OpenCV HSV (0-179).

        Returns:
            The corresponding musical note.
        """
        for (low, high), note in self.HUE_RANGES:
            if low <= hue < high:
                return note
        # Fallback to C for any edge cases
        return Note.C

    def _get_stable_note(self, raw_note: Note) -> Note:
        """Get stable note with hysteresis to prevent rapid changes.

        Args:
            raw_note: The raw detected note.

        Returns:
            The stable note (may lag behind raw).
        """
        self._note_history.append(raw_note)

        # Check if enough recent frames agree on a new note
        if len(self._note_history) >= self._note_stability_frames:
            # Count occurrences of each note in history
            from collections import Counter

            counts = Counter(self._note_history)
            most_common_note, count = counts.most_common(1)[0]

            # Only change if threshold agrees
            if count >= self._note_stability_frames * self._note_stability_threshold:
                self._current_note = most_common_note

        return self._current_note

    def _brightness_to_octave(self, brightness: float) -> int:
        """Convert brightness (0-255) to an octave number.

        Args:
            brightness: Brightness value (0-255).

        Returns:
            Octave number (MIN_OCTAVE to MAX_OCTAVE).
        """
        # Map 0-255 to MIN_OCTAVE-MAX_OCTAVE
        octave_range = self.MAX_OCTAVE - self.MIN_OCTAVE
        octave = self.MIN_OCTAVE + int(brightness / 255 * octave_range)
        return min(max(octave, self.MIN_OCTAVE), self.MAX_OCTAVE)

    def _get_stable_octave(self, raw_octave: int) -> int:
        """Get stable octave with hysteresis to prevent rapid changes.

        Args:
            raw_octave: The raw detected octave.

        Returns:
            The stable octave (may lag behind raw).
        """
        self._octave_history.append(raw_octave)

        # Check if enough recent frames agree on a new octave
        if len(self._octave_history) >= self._octave_stability_frames:
            from collections import Counter

            counts = Counter(self._octave_history)
            most_common_octave, count = counts.most_common(1)[0]

            # Only change if threshold agrees (stricter than note by default)
            if count >= self._octave_stability_frames * self._octave_stability_threshold:
                self._current_octave = most_common_octave

        return self._current_octave

    def _calculate_midi_note(self, note: Note, octave: int) -> int:
        """Calculate MIDI note number from note and octave.

        MIDI note formula: (octave + 1) * 12 + semitone_offset
        Middle C (C4) = 60

        Args:
            note: The musical note.
            octave: The octave number.

        Returns:
            MIDI note number (0-127).
        """
        semitone = NOTE_SEMITONES[note]
        midi = (octave + 1) * 12 + semitone
        return min(max(midi, 0), 127)

    def analyze(self, gaze_region: GazeRegion) -> ColorReading:
        """Analyze color and brightness of a gaze region.

        Uses Gaussian-weighted averaging and temporal smoothing for stability.

        Args:
            gaze_region: The extracted region around the gaze point.

        Returns:
            ColorReading with note, octave, and MIDI note information.
        """
        region = gaze_region.region

        if region.size == 0:
            # Return default values for empty region
            return ColorReading(
                timestamp=gaze_region.timestamp,
                note=self._current_note,
                octave=self._current_octave,
                midi_note=self._calculate_midi_note(
                    self._current_note, self._current_octave
                ),
                hue=0.0,
                smoothed_hue=0.0,
                saturation=0.0,
                brightness=0.0,
                smoothed_brightness=0.0,
                center_x=gaze_region.center_x,
                center_y=gaze_region.center_y,
                confidence=gaze_region.confidence,
            )

        # Convert to HSV color space
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

        # Calculate Gaussian-weighted mean values
        # Note: raw_hue may be None if saturation is too low for reliable hue
        raw_hue, raw_saturation, raw_value = self._weighted_mean_hsv(hsv)

        # Add saturation and brightness to histories (always)
        self._saturation_history.append(raw_saturation)
        self._brightness_history.append(raw_value)

        # Only add hue to history if it's reliable (saturation was high enough)
        if raw_hue is not None:
            self._hue_history.append(raw_hue)

        # Calculate smoothed values
        smoothed_saturation = (
            sum(self._saturation_history) / len(self._saturation_history)
            if self._saturation_history
            else raw_saturation
        )
        smoothed_brightness = (
            sum(self._brightness_history) / len(self._brightness_history)
            if self._brightness_history
            else raw_value
        )

        # Calculate smoothed hue (circular mean) - only if we have hue history
        if self._hue_history:
            hue_rad = [h * (2 * np.pi / 180) for h in self._hue_history]
            cos_mean = sum(np.cos(h) for h in hue_rad) / len(hue_rad)
            sin_mean = sum(np.sin(h) for h in hue_rad) / len(hue_rad)
            smoothed_hue = (np.arctan2(sin_mean, cos_mean) * 180 / (2 * np.pi)) % 180

            # Map smoothed hue to note with stability
            raw_note = self._hue_to_note(smoothed_hue)
            stable_note = self._get_stable_note(raw_note)
        else:
            # No reliable hue readings yet - use current note
            smoothed_hue = 0.0
            stable_note = self._current_note

        # Map smoothed brightness to octave with stability
        raw_octave = self._brightness_to_octave(smoothed_brightness)
        stable_octave = self._get_stable_octave(raw_octave)

        # Calculate MIDI note
        midi_note = self._calculate_midi_note(stable_note, stable_octave)

        return ColorReading(
            timestamp=gaze_region.timestamp,
            note=stable_note,
            octave=stable_octave,
            midi_note=midi_note,
            hue=raw_hue,
            smoothed_hue=smoothed_hue,
            saturation=smoothed_saturation,
            brightness=raw_value,
            smoothed_brightness=smoothed_brightness,
            center_x=gaze_region.center_x,
            center_y=gaze_region.center_y,
            confidence=gaze_region.confidence,
        )

    def reset(self) -> None:
        """Reset all histories."""
        self._hue_history.clear()
        self._brightness_history.clear()
        self._saturation_history.clear()
        self._note_history.clear()
        self._octave_history.clear()
        self._current_note = Note.C
        self._current_octave = self.MIN_OCTAVE

