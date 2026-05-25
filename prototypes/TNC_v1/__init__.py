"""TNC_v1 patch: hue → MIDI note, brightness → octave, flutter → effect."""

from base.patches import register_patch
from .gate import NoteGate
from .mapping import (
    HUE_RANGES,
    NOTE_BGR_COLORS,
    NOTE_COLOR_NAMES,
    NOTE_SEMITONES,
    Note,
    NoteMapper,
    NoteReading,
)
from .patch import ColorMusicPatch

register_patch("TNC_v1", ColorMusicPatch)

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
]
