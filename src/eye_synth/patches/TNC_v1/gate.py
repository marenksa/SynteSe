"""NoteGate: fires when a MIDI note stabilises after a real gaze transition."""

from __future__ import annotations

from collections import deque


class NoteGate:
    """Fires when the detected MIDI note stabilises after a real gaze transition.

    Suppresses jitter and duplicate triggers while resting on the same content.
    Responds to fixation ID changes as well as raw note transitions.

    Parameters:
        stable_frames:          Consecutive identical frames required to trigger.
        min_transition_frames:  Frames of different content before re-triggering same note.
    """

    def __init__(self, stable_frames: int = 4, min_transition_frames: int = 3) -> None:
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
        if current != self._last_triggered or self._transition_detected:
            self._last_triggered = current
            self._transition_detected = False
            self._diff_streak = 0
            return True

        return False
