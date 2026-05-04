"""NoteGate: fires when a MIDI note stabilises after a real gaze transition."""

from __future__ import annotations

from collections import deque


class NoteGate:
    """Fires when the detected MIDI note stabilises after a real gaze transition.

    Checks run in priority order and exit at the first hit:

      1. Note changed   — stable note differs from last triggered; covers all
                          colour-change transitions regardless of movement type.
      2. Diff streak    — raw note passed through different values for
                          min_transition_frames consecutive frames; covers
                          same-colour saccades via a chromatic path.

    Parameters:
        stable_frames:          Consecutive identical frames required to confirm note.
        min_transition_frames:  Frames of different raw note to arm diff-streak (check 2).
    """

    def __init__(
        self,
        stable_frames: int = 3,
        min_transition_frames: int = 3,
    ) -> None:
        self.stable_frames = stable_frames
        self.min_transition_frames = min_transition_frames

        self._recent: deque[int] = deque(maxlen=stable_frames)
        self._last_triggered: int | None = None

        self._diff_streak: int = 0
        self._transition_detected: bool = False

    def update(
        self,
        midi_note: int,
        raw_midi_note: int | None = None,
    ) -> bool:
        """Return True when a note trigger should fire."""
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

        if current != self._last_triggered:
            return self._fire(current)

        if self._transition_detected:
            return self._fire(current)

        return False

    def reset(self) -> None:
        self._recent.clear()
        self._last_triggered = None
        self._diff_streak = 0
        self._transition_detected = False

    def _fire(self, note: int) -> bool:
        self._last_triggered = note
        self._transition_detected = False
        self._diff_streak = 0
        return True
