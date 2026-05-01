"""NoteGate: fires when a MIDI note stabilises after a real gaze transition."""

from __future__ import annotations

from collections import deque

# Minimum scene_change magnitude (0–1) that indicates the head has moved.
_SCENE_CHANGE_THRESHOLD = 0.05

# Maximum norm_pos range per axis (0–1) over the stable window to count as stable.
# ~0.05 ≈ 64 px on a 1280-wide frame.
_POS_STABILITY_THRESHOLD = 0.05


class NoteGate:
    """Fires when the detected MIDI note stabilises after a real gaze transition.

    Checks run in priority order and exit at the first hit:

      1. Note changed      — stable note differs from last triggered; covers all
                             colour-change transitions regardless of head movement.
      2. Fixation changed  — Pupil detected a new fixation target; covers same-colour
                             saccades when the fixation detector fires.
      3. Smooth pan        — scene_change above threshold (head moved) AND norm_pos
                             stable over the window (gaze rode with the camera);
                             covers smooth same-colour pans missed by checks 1–2.
      4. Diff streak       — raw note passed through different values for
                             min_transition_frames consecutive frames; covers same-colour
                             saccades via a chromatic path when fixation was missed.

    Parameters:
        stable_frames:          Consecutive identical frames required to confirm note.
        min_transition_frames:  Frames of different raw note to arm diff-streak (check 4).
    """

    def __init__(
        self,
        stable_frames: int = 4,
        min_transition_frames: int = 3,
    ) -> None:
        self.stable_frames = stable_frames
        self.min_transition_frames = min_transition_frames

        self._recent: deque[int] = deque(maxlen=stable_frames)
        self._pos_history: deque[tuple[float, float]] = deque(maxlen=stable_frames)

        self._last_triggered: int | None = None

        # Check 2 state
        self._fixation_changed: bool = False
        self._last_fixation_id: int | None = None

        # Check 4 state
        self._diff_streak: int = 0
        self._transition_detected: bool = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def new_fixation(self, fixation_id: int) -> None:
        """Call whenever a new fixation event arrives from the tracker."""
        if self._last_fixation_id is not None and fixation_id != self._last_fixation_id:
            self._fixation_changed = True
        self._last_fixation_id = fixation_id

    def update(
        self,
        midi_note: int,
        raw_midi_note: int | None = None,
        norm_pos: tuple[float, float] = (0.5, 0.5),
        scene_change: float = 0.0,
    ) -> bool:
        """Return True when a note trigger should fire."""
        self._recent.append(midi_note)
        self._pos_history.append(norm_pos)

        # Accumulate diff streak for check 4
        detect_note = raw_midi_note if raw_midi_note is not None else midi_note
        if self._last_triggered is not None and detect_note != self._last_triggered:
            self._diff_streak += 1
            if self._diff_streak >= self.min_transition_frames:
                self._transition_detected = True
        else:
            self._diff_streak = 0

        # Require a full, uniform stable window before any trigger
        if len(self._recent) < self.stable_frames:
            return False
        if len(set(self._recent)) != 1:
            return False

        current = self._recent[0]

        # Check 1: note changed (colour transition, any movement type)
        if current != self._last_triggered:
            return self._fire(current)

        # Check 2: new fixation on same-colour object
        if self._fixation_changed:
            return self._fire(current)

        # Check 3: smooth same-colour pan
        # (scene moved + gaze rode with camera → norm_pos stayed stable)
        if scene_change > _SCENE_CHANGE_THRESHOLD and self._pos_is_stable():
            return self._fire(current)

        # Check 4: diff streak — saccade via chromatic path, fixation missed
        if self._transition_detected:
            return self._fire(current)

        return False

    def reset(self) -> None:
        self._recent.clear()
        self._pos_history.clear()
        self._last_triggered = None
        self._fixation_changed = False
        self._last_fixation_id = None
        self._diff_streak = 0
        self._transition_detected = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fire(self, note: int) -> bool:
        self._last_triggered = note
        self._fixation_changed = False
        self._transition_detected = False
        self._diff_streak = 0
        return True

    def _pos_is_stable(self) -> bool:
        """True if norm_pos range over the stable window is below threshold."""
        if len(self._pos_history) < self.stable_frames:
            return False
        xs = [p[0] for p in self._pos_history]
        ys = [p[1] for p in self._pos_history]
        return (
            max(xs) - min(xs) < _POS_STABILITY_THRESHOLD
            and max(ys) - min(ys) < _POS_STABILITY_THRESHOLD
        )
