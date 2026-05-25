"""Colour and brightness detection at the gaze point.

Two classes work in sequence:

    FrameProcessor  — extracts the pixel region around the gaze point
    ColorAnalyzer   — analyses that region for hue, saturation, brightness
                      with Gaussian-weighted spatial averaging and temporal
                      smoothing

Note/octave mapping is prototype-specific and lives in patches/TNC_v1/.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from base.input.live import FrameData, GazeData


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
    """Raw colour/brightness reading for one frame."""
    timestamp: float
    hue: float | None           # Instantaneous hue 0–179; None if saturation too low
    smoothed_hue: float         # Temporally smoothed hue
    saturation: float           # Smoothed saturation 0–255
    brightness: float           # Instantaneous brightness 0–255
    smoothed_brightness: float  # Temporally smoothed brightness
    center_x: int
    center_y: int
    confidence: float


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
        self._last_raw_confidence: float = 0.0

    @property
    def region_size(self) -> int:
        return self._region_size

    @property
    def last_frame(self) -> FrameData | None:
        return self._last_frame

    @property
    def last_gaze(self) -> GazeData | None:
        return self._last_gaze

    @property
    def last_raw_confidence(self) -> float:
        """Most recent gaze confidence regardless of filtering, for display."""
        return self._last_raw_confidence

    def update_frame(self, frame: FrameData) -> None:
        self._last_frame = frame

    def update_gaze(self, gaze: GazeData, min_confidence: float = 0.5) -> bool:
        """Accept a gaze sample if confidence passes; position may exceed [0,1] at frame edges."""
        self._last_raw_confidence = gaze.confidence
        if gaze.confidence < min_confidence:
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

    Returns hue, saturation and brightness — smoothed over a temporal window.
    Note/octave mapping is not done here; see patches/color_music/ for that.
    """

    DEFAULT_MIN_SATURATION = 20

    def __init__(
        self,
        smoothing_window: int = 3,
        min_saturation: int = DEFAULT_MIN_SATURATION,
    ) -> None:
        self._smoothing_window = smoothing_window
        self._min_saturation = min_saturation

        self._hue_history: deque[float] = deque(maxlen=smoothing_window)
        self._brightness_history: deque[float] = deque(maxlen=smoothing_window)
        self._saturation_history: deque[float] = deque(maxlen=smoothing_window)
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

    def analyze(self, gaze_region: GazeRegion) -> ColorReading:
        """Analyse a gaze region and return a ColorReading."""
        region = gaze_region.region

        if region.size == 0:
            return ColorReading(
                timestamp=gaze_region.timestamp,
                hue=None,
                smoothed_hue=0.0,
                saturation=0.0,
                brightness=0.0,
                smoothed_brightness=0.0,
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
        else:
            smoothed_hue = 0.0

        return ColorReading(
            timestamp=gaze_region.timestamp,
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
