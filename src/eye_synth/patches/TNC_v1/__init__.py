"""TNC_v1 patch: hue → MIDI note, brightness → octave, flutter → AM-LFO."""

from eye_synth.patches.TNC_v1.gate import NoteGate
from eye_synth.patches.TNC_v1.mapping import (
    HUE_RANGES,
    NOTE_BGR_COLORS,
    NOTE_COLOR_NAMES,
    NOTE_SEMITONES,
    Note,
    NoteMapper,
    NoteReading,
)
from eye_synth.patches.TNC_v1.patch import ColorMusicPatch, flutter_to_lfo

__all__ = [
    "ColorMusicPatch",
    "NoteGate",
    "NoteMapper",
    "NoteReading",
    "Note",
    "NOTE_SEMITONES",
    "HUE_RANGES",
    "NOTE_BGR_COLORS",
    "NOTE_COLOR_NAMES",
    "flutter_to_lfo",
]
