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
        TNC_v1     —  hue→note, brightness→octave, flutter→effect
        TgSqC_v1   —  hue→color ID (1–7), stability→PD toggle
        SCf_v1     —  eye confidence stream → PD confidence signal
    """
    if name == "TNC_v1":
        from eye_synth.patches.TNC_v1 import ColorMusicPatch
        return ColorMusicPatch()
    if name == "TgSqC_v1":
        from eye_synth.patches.TgSqC_v1 import ColorTogglePatch
        return ColorTogglePatch()
    if name == "SCf_v1":
        from eye_synth.patches.SCf_v1 import ConfidenceStreamPatch
        return ConfidenceStreamPatch()
    raise ValueError(
        f"Unknown patch: {name!r}. Available patches: TNC_v1, TgSqC_v1, SCf_v1"
    )
