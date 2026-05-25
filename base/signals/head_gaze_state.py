"""Head-gaze state classification.

Classifies each frame into one of four states from two signals:
  - norm_pos stability: is gaze moving in the camera frame?
  - scene_change magnitude: is the camera/scene moving?

                scene_change low    scene_change high
  pos stable    Rest                SmoothPan
  pos changing  Scanning            RagLock
"""

from __future__ import annotations

from collections import deque
from enum import Enum, auto


class HeadGazeState(Enum):
    Rest      = auto()  # gaze still, head still
    SmoothPan = auto()  # gaze rides camera (stable in frame), head moves
    Scanning  = auto()  # gaze moves, head still
    RagLock   = auto()  # gaze moves, head moves (ragged pan or lock-on)


_SCENE_CHANGE_THRESHOLD = 0.05
_POS_STABILITY_THRESHOLD = 0.05


class HeadGazeClassifier:
    """Classifies head-gaze state from norm_pos history and scene_change.

    Parameters:
        window:         Frames over which to assess norm_pos stability.
        min_confidence: Gaze samples below this are skipped (not added to history).
    """

    def __init__(self, window: int = 4, min_confidence: float = 0.6) -> None:
        self._window = window
        self._min_confidence = min_confidence
        self._pos_history: deque[tuple[float, float]] = deque(maxlen=window)
        self._state = HeadGazeState.Rest

    @property
    def state(self) -> HeadGazeState:
        return self._state

    def update(
        self,
        norm_pos: tuple[float, float],
        scene_change: float,
        confidence: float,
    ) -> HeadGazeState:
        """Feed one frame and return the current state."""
        if confidence >= self._min_confidence:
            self._pos_history.append(norm_pos)

        head_moving = scene_change > _SCENE_CHANGE_THRESHOLD
        pos_stable = self._pos_is_stable()

        if not head_moving and pos_stable:
            self._state = HeadGazeState.Rest
        elif head_moving and pos_stable:
            self._state = HeadGazeState.SmoothPan
        elif not head_moving:
            self._state = HeadGazeState.Scanning
        else:
            self._state = HeadGazeState.RagLock

        return self._state

    def reset(self) -> None:
        self._pos_history.clear()
        self._state = HeadGazeState.Rest

    def _pos_is_stable(self) -> bool:
        if len(self._pos_history) < self._window:
            return True  # not enough data yet — default to stable
        xs = [p[0] for p in self._pos_history]
        ys = [p[1] for p in self._pos_history]
        return (
            max(xs) - min(xs) < _POS_STABILITY_THRESHOLD
            and max(ys) - min(ys) < _POS_STABILITY_THRESHOLD
        )
