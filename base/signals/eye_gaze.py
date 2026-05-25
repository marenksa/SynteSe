"""Gaze position and velocity tracking.

Gaze position and confidence come directly from Pupil Capture and are
forwarded to the signal bus without transformation (see signals/bus.py).

This module adds velocity — how fast the gaze is moving — which is not
provided by Pupil Capture directly and must be computed from consecutive
position samples. Velocity is useful for distinguishing saccades (high
velocity spikes) from fixations (near zero) and smooth pursuit (low–medium).
"""

from __future__ import annotations

import math
from collections import deque


class GazeVelocityTracker:
    """Computes gaze movement speed in pixels/second from consecutive positions."""

    def __init__(self, smoothing: int = 3) -> None:
        self._prev_norm: tuple[float, float] | None = None
        self._prev_ts: float | None = None
        self._history: deque[float] = deque(maxlen=smoothing)
        self._velocity: float = 0.0

    def update(
        self,
        norm_pos: tuple[float, float],
        timestamp: float,
        frame_width: int,
        frame_height: int,
    ) -> float:
        """Feed new normalised gaze position. Returns smoothed speed in px/s."""
        if self._prev_norm is None or self._prev_ts is None:
            self._prev_norm = norm_pos
            self._prev_ts = timestamp
            return 0.0

        dt = timestamp - self._prev_ts
        if dt <= 0.0:
            return self._velocity

        dx = (norm_pos[0] - self._prev_norm[0]) * frame_width
        dy = (norm_pos[1] - self._prev_norm[1]) * frame_height
        speed = math.sqrt(dx * dx + dy * dy) / dt

        self._prev_norm = norm_pos
        self._prev_ts = timestamp
        self._history.append(speed)
        self._velocity = sum(self._history) / len(self._history)
        return self._velocity

    def reset(self) -> None:
        self._prev_norm = None
        self._prev_ts = None
        self._history.clear()
        self._velocity = 0.0

    @property
    def velocity(self) -> float:
        return self._velocity
