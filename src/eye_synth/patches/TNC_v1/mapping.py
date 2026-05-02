"""Colour-to-note mapping for the TNC_v1 patch.

Defines the Note enum, hue/brightness → note/octave/MIDI mapping,
and stability-windowed NoteMapper.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from enum import IntEnum


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
    ((0, 8),     Note.C),
    ((8, 25),    Note.D),
    ((25, 38),   Note.E),
    ((38, 75),   Note.F),
    ((75, 95),   Note.G),
    ((95, 125),  Note.A),
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


@dataclass(frozen=True)
class NoteReading:
    """Note/octave/MIDI for one frame, produced by NoteMapper."""
    note: Note
    octave: int
    midi_note: int
    raw_midi_note: int  # pre-stability note, same octave — for NoteGate transition detection


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
        # _MIN_OCTAVE=2 gives a 4-step range so octave 3 covers half the
        # brightness (0–127); octave 2 is clamped out — too low to use.
        octave_range = self._MAX_OCTAVE - self._MIN_OCTAVE
        octave = self._MIN_OCTAVE + int(brightness / 255 * octave_range)
        return max(3, min(octave, self._MAX_OCTAVE))

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
