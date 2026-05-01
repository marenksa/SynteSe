"""ColorMusicPatch: maps eye/environment signals to MIDI notes and effect."""

from __future__ import annotations

from eye_synth.output.overlay import OverlayConfig
from eye_synth.patches.TNC_v1.gate import NoteGate
from eye_synth.patches.TNC_v1.mapping import NoteMapper
from eye_synth.signals.bus import OutputBus, SignalBus
from eye_synth.signals.eye_blinks import FLUTTER_MIN_BLINKS, BlinkType


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
    )

    def __init__(self) -> None:
        self._note_mapper = NoteMapper()
        self._note_gate = NoteGate()

    def reset(self) -> None:
        """Reset state — call on seek or restart."""
        self._note_mapper = NoteMapper()
        self._note_gate = NoteGate()

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

        # Feed fixation events to the note gate
        if signals.eye.fixation_id is not None:
            self._note_gate.new_fixation(signals.eye.fixation_id)

        # Content-based note trigger (suppressed during flutter)
        if not signals.eye.is_flutter_active and self._note_gate.update(
            note.midi_note,
            note.raw_midi_note,
            norm_pos=signals.eye.norm_pos,
            scene_change=signals.env.scene_change,
        ):
            outputs.send("note_on", note.midi_note, signals.env.brightness / 255.0)
