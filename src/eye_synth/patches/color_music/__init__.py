"""Color-to-music patch: the original prototype behavior.

Maps:
- Hue at gaze point  →  MIDI note  (C–B, wavelength order)
- Brightness         →  octave
- Flutter burst end  →  am_lfo frequency (1–50 Hz, scaled by blink count)
- Intentional blink  →  am_lfo 0  (clears the effect)
- Gaze confidence during noise events  →  confidence message

To make a new prototype, copy this file and change what's in update().
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from enum import IntEnum

from eye_synth.signals.bus import OutputBus, SignalBus
from eye_synth.signals.eye_blinks import FLUTTER_MIN_BLINKS, BlinkType


# ---------------------------------------------------------------------------
# Note mapping constants
# ---------------------------------------------------------------------------

class Note(IntEnum):
    """Musical notes mapped from colour wavelengths (longer → lower)."""
    C = 0  # Red    ~700nm
    D = 1  # Orange ~620nm
    E = 2  # Yellow ~580nm
    F = 3  # Green  ~530nm
    G = 4  # Cyan   ~500nm
    A = 5  # Blue   ~470nm
    B = 6  # Violet ~400nm


NOTE_SEMITONES: dict[Note, int] = {
    Note.C: 0, Note.D: 2, Note.E: 4, Note.F: 5,
    Note.G: 7, Note.A: 9, Note.B: 11,
}

HUE_RANGES: list[tuple[tuple[int, int], Note]] = [
    ((0, 8),    Note.C),
    ((8, 25),   Note.D),
    ((25, 38),  Note.E),
    ((38, 75),  Note.F),
    ((75, 95),  Note.G),
    ((95, 125), Note.A),
    ((125, 165), Note.B),
    ((165, 180), Note.C),
]

NOTE_BGR_COLORS: dict[Note, tuple[int, int, int]] = {
    Note.C: (60, 60, 220),
    Note.D: (60, 140, 255),
    Note.E: (60, 220, 255),
    Note.F: (60, 180, 60),
    Note.G: (180, 180, 60),
    Note.A: (220, 120, 60),
    Note.B: (180, 60, 180),
}

NOTE_COLOR_NAMES: dict[Note, str] = {
    Note.C: "Red",
    Note.D: "Orange",
    Note.E: "Yellow",
    Note.F: "Green",
    Note.G: "Cyan",
    Note.A: "Blue",
    Note.B: "Violet",
}


# ---------------------------------------------------------------------------
# NoteReading
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NoteReading:
    """Note/octave/MIDI for one frame, produced by NoteMapper."""
    note: Note
    octave: int
    midi_note: int
    raw_midi_note: int   # pre-stability note, same octave — for NoteGate transition detection


# ---------------------------------------------------------------------------
# NoteMapper
# ---------------------------------------------------------------------------

class NoteMapper:
    """Maps smoothed hue and brightness to stable note/octave/MIDI.

    Owns the stability windows that suppress per-frame jitter.
    """

    _MIN_OCTAVE = 2
    _MAX_OCTAVE = 6
    _NOTE_STABILITY_THRESHOLD = 0.5

    def __init__(
        self,
        note_stability_frames: int = 2,
        octave_stability_frames: int = 3,
        octave_stability_threshold: float = 0.5,
    ) -> None:
        self._note_stability_frames = note_stability_frames
        self._octave_stability_frames = octave_stability_frames
        self._octave_stability_threshold = octave_stability_threshold

        self._note_history: deque[Note] = deque(maxlen=note_stability_frames)
        self._octave_history: deque[int] = deque(maxlen=octave_stability_frames)
        self._current_note: Note = Note.C
        self._current_octave: int = self._MIN_OCTAVE

    def update(
        self,
        smoothed_hue: float,
        raw_hue: float | None,
        smoothed_brightness: float,
    ) -> NoteReading:
        """Compute stable note/octave and return a NoteReading."""
        raw_note = self._hue_to_note(smoothed_hue)
        stable_note = self._get_stable_note(raw_note)

        raw_octave = self._brightness_to_octave(smoothed_brightness)
        stable_octave = self._get_stable_octave(raw_octave)

        midi_note = self._calculate_midi_note(stable_note, stable_octave)

        # raw_midi_note: uses raw (instantaneous) hue for transition detection
        raw_note_val = self._hue_to_note(raw_hue) if raw_hue is not None else stable_note
        raw_midi_note = self._calculate_midi_note(raw_note_val, stable_octave)

        return NoteReading(
            note=stable_note,
            octave=stable_octave,
            midi_note=midi_note,
            raw_midi_note=raw_midi_note,
        )

    def reset(self) -> None:
        self._note_history.clear()
        self._octave_history.clear()
        self._current_note = Note.C
        self._current_octave = self._MIN_OCTAVE

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _hue_to_note(self, hue: float) -> Note:
        for (low, high), note in HUE_RANGES:
            if low <= hue < high:
                return note
        return Note.C

    def _get_stable_note(self, raw_note: Note) -> Note:
        self._note_history.append(raw_note)
        if len(self._note_history) >= self._note_stability_frames:
            counts = Counter(self._note_history)
            most_common, count = counts.most_common(1)[0]
            if count >= self._note_stability_frames * self._NOTE_STABILITY_THRESHOLD:
                self._current_note = most_common
        return self._current_note

    def _brightness_to_octave(self, brightness: float) -> int:
        octave_range = self._MAX_OCTAVE - self._MIN_OCTAVE
        octave = self._MIN_OCTAVE + int(brightness / 255 * octave_range)
        octave = min(max(octave, self._MIN_OCTAVE), self._MAX_OCTAVE)
        return 3 if octave == self._MIN_OCTAVE else octave

    def _get_stable_octave(self, raw_octave: int) -> int:
        self._octave_history.append(raw_octave)
        if len(self._octave_history) >= self._octave_stability_frames:
            counts = Counter(self._octave_history)
            most_common, count = counts.most_common(1)[0]
            if count >= self._octave_stability_frames * self._octave_stability_threshold:
                self._current_octave = most_common
        return self._current_octave

    def _calculate_midi_note(self, note: Note, octave: int) -> int:
        return min(max((octave + 1) * 12 + NOTE_SEMITONES[note], 0), 127)


# ---------------------------------------------------------------------------
# NoteGate
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def flutter_to_lfo(blink_count: int, min_blinks: int, max_hz: float = 50.0) -> float:
    """Map flutter blink count to an LFO frequency in Hz.

    Scales linearly from 1 Hz (min_blinks) to max_hz (2 * min_blinks).
    """
    ratio = min(1.0, (blink_count - min_blinks) / max(1, min_blinks))
    return 1.0 + ratio * (max_hz - 1.0)


# ---------------------------------------------------------------------------
# Patch
# ---------------------------------------------------------------------------

class ColorMusicPatch:
    """Self-contained patch implementing the original color-to-music behavior."""

    def __init__(self) -> None:
        self._note_mapper = NoteMapper()
        self._note_gate = NoteGate()
        self._noise_was_active = False

    def reset(self) -> None:
        """Reset state — call on seek or restart."""
        self._note_mapper = NoteMapper()
        self._note_gate = NoteGate()
        self._noise_was_active = False

    def update(self, signals: SignalBus, outputs: OutputBus) -> None:
        # --- Eye events ---

        # Flutter ended → map blink count to LFO frequency
        if signals.eye.flutter is not None:
            hz = flutter_to_lfo(signals.eye.flutter.blink_count, FLUTTER_MIN_BLINKS)
            outputs.send("am_lfo", hz)

        # Intentional blink → clear LFO
        if (
            signals.eye.blink is not None
            and signals.eye.blink.blink_type == BlinkType.INTENTIONAL
        ):
            outputs.send("am_lfo", 0)

        # Confidence gating: pass raw confidence during noise, reset on exit
        noise_active = signals.eye.is_eyes_closed or signals.eye.is_flutter_active
        if noise_active:
            outputs.send("confidence", signals.eye.confidence)
        elif self._noise_was_active:
            outputs.send("confidence", 1.0)
        self._noise_was_active = noise_active

        # --- Environment events ---

        if not signals.has_env_reading:
            return

        note = self._note_mapper.update(
            signals.env.hue,
            signals.env.raw_hue,
            signals.env.brightness,
        )

        # Feed fixation events to the note gate
        if signals.eye.fixation_id is not None:
            self._note_gate.new_fixation(signals.eye.fixation_id)

        # Content-based note trigger (suppressed during flutter)
        if not signals.eye.is_flutter_active and self._note_gate.update(
            note.midi_note, note.raw_midi_note
        ):
            outputs.send("note_on", note.midi_note, signals.env.brightness_normalized)
