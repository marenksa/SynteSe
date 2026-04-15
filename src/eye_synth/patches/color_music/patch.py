"""ColorMusicPatch: maps eye/environment signals to MIDI notes and LFO."""

from __future__ import annotations

from eye_synth.patches.color_music.gate import NoteGate
from eye_synth.patches.color_music.mapping import NoteMapper
from eye_synth.signals.bus import OutputBus, SignalBus
from eye_synth.signals.eye_blinks import FLUTTER_MIN_BLINKS, BlinkType


def flutter_to_lfo(blink_count: int, min_blinks: int, max_hz: float = 50.0) -> float:
    """Map flutter blink count to an LFO frequency in Hz.

    Scales linearly from 1 Hz (min_blinks) to max_hz (2 * min_blinks).
    """
    ratio = min(1.0, (blink_count - min_blinks) / max(1, min_blinks))
    return 1.0 + ratio * (max_hz - 1.0)


class ColorMusicPatch:
    """Maps gaze signals to color-music output.

    - Hue at gaze point  →  MIDI note  (C–B, wavelength order)
    - Brightness         →  octave
    - Flutter burst end  →  am_lfo frequency (1–50 Hz, scaled by blink count)
    - Intentional blink  →  am_lfo 0  (clears the effect)
    - Eye closure/flutter  →  confidence message (noise gating in PD)
    """

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
