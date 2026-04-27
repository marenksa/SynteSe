"""Patch protocol and factory."""

from __future__ import annotations

from typing import Protocol

from eye_synth.signals.bus import OutputBus, SignalBus


class Patch(Protocol):
    """Protocol that all patches must implement."""

    def update(self, signals: SignalBus, outputs: OutputBus) -> None:
        """Called every loop iteration with current signals."""
        ...

    def reset(self) -> None:
        """Reset internal state (e.g. on seek or restart)."""
        ...

    def shutdown(self, outputs: OutputBus) -> None:
        """Send any necessary cleanup messages to PD before exit."""
        ...


def load_patch(name: str) -> Patch:
    """Load a patch by name.

    Available patches:
        TNC_v1        —  hue→note, brightness→octave, flutter→effect
        TgSqC_v1      —  hue→color ID (1–7), stability→PD toggle
        SCfBF_v1      —  eye confidence stream → PD confidence signal (PD v1)
        SCfBF_v2      —  eye confidence stream → PD confidence signal (PD v2)
        RAVE_v1       —  gaze/colour/velocity → RAVE latent dims z0–z4 for nn~
        SPXV_v1       —  gaze coords + velocity → pitch/loudness (PD v1)
        SPXV_v2       —  gaze coords + velocity → pitch/loudness inverted (PD v2)
        SPXV_v3       —  gaze coords + velocity → pitch/loudness (PD v3)
    """
    if name == "TNC_v1":
        from eye_synth.patches.TNC_v1 import ColorMusicPatch
        return ColorMusicPatch()
    if name == "TgSqC_v1":
        from eye_synth.patches.TgSqC_v1 import ColorTogglePatch
        return ColorTogglePatch()
    if name in ("SCfBF_v1", "SCfBF_v2"):
        from eye_synth.patches.SCfBF import ConfidenceStreamPatch
        return ConfidenceStreamPatch()
    if name == "RAVE_v1":
        from eye_synth.patches.RAVE_v1 import RAVEPatch
        return RAVEPatch()
    if name in ("SPXV_v1", "SPXV_v2", "SPXV_v3"):
        from eye_synth.patches.SPXV import GazeStreamPatch
        return GazeStreamPatch()
    raise ValueError(
        f"Unknown patch: {name!r}. Available patches: TNC_v1, TgSqC_v1, SCfBF_v1, SCfBF_v2, RAVE_v1, SPXV_v1, SPXV_v2, SPXV_v3"
    )
