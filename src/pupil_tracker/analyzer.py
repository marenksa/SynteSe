"""Brightness analyzer for gaze regions."""

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from pupil_tracker.processor import GazeRegion


@dataclass(frozen=True)
class BrightnessReading:
    """A brightness reading at a specific point in time."""

    timestamp: float
    brightness: float  # 0-255 scale
    smoothed_brightness: float  # Smoothed value
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

