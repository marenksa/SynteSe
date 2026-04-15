"""Hue-to-color-ID mapping for the color_toggle patch.

Maps OpenCV hue (0–179) to a stable color ID in 1–7.
No octave or brightness dimension.
"""

from __future__ import annotations

from collections import Counter, deque

# Maps hue ranges to color IDs 1–7 (wavelength order: red→violet)
# ID  Color    Hue range
# 1   Red      0–8, 165–180
# 2   Orange   8–25
# 3   Yellow   25–38
# 4   Green    38–75
# 5   Cyan     75–95
# 6   Blue     95–125
# 7   Violet   125–165
HUE_RANGES: list[tuple[tuple[int, int], int]] = [
    ((0, 8),     1),
    ((8, 25),    2),
    ((25, 38),   3),
    ((38, 75),   4),
    ((75, 95),   5),
    ((95, 125),  6),
    ((125, 165), 7),
    ((165, 180), 1),
]

COLOR_NAMES: dict[int, str] = {
    1: "Red",
    2: "Orange",
    3: "Yellow",
    4: "Green",
    5: "Cyan",
    6: "Blue",
    7: "Violet",
}

_STABILITY_THRESHOLD = 0.7


def hue_to_color_id(hue: float) -> int:
    for (low, high), color_id in HUE_RANGES:
        if low <= hue < high:
            return color_id
    return 1


class ColorIdMapper:
    """Maps smoothed hue to a stable color ID (1–7).

    Uses a mode-based stability window to suppress per-frame jitter.
    Returns None when the color is achromatic (low saturation) or the window
    has no stable consensus.
    """

    _MIN_SATURATION = 20  # below this: treat as achromatic

    def __init__(self, stability_frames: int = 30) -> None:
        self._stability_frames = stability_frames
        self._history: deque[int | None] = deque(maxlen=stability_frames)
        self._current_id: int | None = None

    def update(self, smoothed_hue: float, saturation: float) -> int | None:
        """Push a new reading and return the stable color ID, or None."""
        raw_id = hue_to_color_id(smoothed_hue) if saturation >= self._MIN_SATURATION else None
        self._history.append(raw_id)

        if len(self._history) < self._stability_frames:
            return self._current_id

        non_none = [v for v in self._history if v is not None]
        if not non_none:
            self._current_id = None
            return None

        counts = Counter(non_none)
        most_common_id, count = counts.most_common(1)[0]
        if count >= self._stability_frames * _STABILITY_THRESHOLD:
            self._current_id = most_common_id
        else:
            self._current_id = None

        return self._current_id

    def reset(self) -> None:
        self._history.clear()
        self._current_id = None
