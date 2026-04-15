"""Full-frame scene change detector.

Detects changes in the entire video frame, independent of gaze position.
Useful for detecting head movements, scene cuts, or large environmental changes.
"""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np
from numpy.typing import NDArray


class SceneChangeDetector:
    """Measures how much the full video frame changed from the previous frame.

    Returns a value 0.0–1.0 (mean absolute pixel difference, normalized).
    Smoothed over a short window to reduce noise.

    A value near 0.0 means the scene is stable.
    A value near 1.0 means a dramatic cut or fast head movement.
    """

    def __init__(self, smoothing: int = 3) -> None:
        self._prev_gray: NDArray[np.float32] | None = None
        self._history: deque[float] = deque(maxlen=smoothing)

    def update(self, frame: NDArray[np.uint8]) -> float:
        """Feed a new BGR frame. Returns smoothed change magnitude 0.0–1.0."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

        if self._prev_gray is None or self._prev_gray.shape != gray.shape:
            self._prev_gray = gray
            return 0.0

        diff = float(np.mean(np.abs(gray - self._prev_gray)) / 255.0)
        self._prev_gray = gray
        self._history.append(diff)
        return float(np.mean(self._history))

    def reset(self) -> None:
        self._prev_gray = None
        self._history.clear()
