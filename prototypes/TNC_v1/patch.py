"""ColorMusicPatch: maps eye/environment signals to MIDI notes and effect."""

from __future__ import annotations

from base.output.overlay import OverlayConfig
from .gate import NoteGate
from .mapping import NoteMapper
from base.signals.bus import OutputBus, SignalBus
from base.signals.eye_blinks import FLUTTER_MIN_BLINKS, BlinkType


class ColorMusicPatch:
    """Maps gaze signals to color-music output.

    - Hue at gaze point  →  MIDI note  (C–B, wavelength order)
    - Brightness         →  octave
    - Flutter burst end  →  effect value (0–1, scaled by blink count)
    - Intentional blink  →  effect 0  (clears the effect)
    """

    overlay = OverlayConfig(
        show_brightness_bar=True,
        show_color_info=True,
        show_eye_panel=True,
        show_blink_flutter=True,
        brightness_octave_markers=((0, "3"), (128, "4"), (192, "5")),
    )

    def __init__(self) -> None:
        self._note_mapper = NoteMapper()
        self._note_gate = NoteGate()
        self.current_midi_note: int | None = None

    def reset(self) -> None:
        """Reset state — call on seek or restart."""
        self._note_mapper = NoteMapper()
        self._note_gate = NoteGate()
        self.current_midi_note = None

    def shutdown(self, outputs: OutputBus) -> None:
        outputs.send("effect", 0)

    def update(self, signals: SignalBus, outputs: OutputBus) -> None:
        # --- Eye events ---

        # Flutter ended → map blink count to effect intensity (0–1)
        if signals.eye.flutter is not None:
            value = min(1.0, (signals.eye.flutter.blink_count - FLUTTER_MIN_BLINKS) / max(1, FLUTTER_MIN_BLINKS))
            outputs.send("effect", value)

        # Intentional blink → clear effect
        if (
            signals.eye.blink is not None
            and signals.eye.blink.blink_type == BlinkType.INTENTIONAL
        ):
            outputs.send("effect", 0)

        # --- Environment events ---

        if not signals.has_env_reading:
            return

        note = self._note_mapper.update(
            signals.env.hue,
            signals.env.raw_hue,
            signals.env.brightness,
        )
        self.current_midi_note = note.midi_note

        # Content-based note trigger (suppressed during flutter)
        if not signals.eye.is_flutter_active and self._note_gate.update(
            note.midi_note,
            note.raw_midi_note,
        ):
            outputs.send("note_on", note.midi_note, signals.env.brightness / 255.0)
