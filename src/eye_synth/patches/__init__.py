"""Patches: creative mappings from signals to outputs.

A patch is any object that implements:
    update(signals: SignalBus, outputs: OutputBus) -> None
    reset() -> None  (called on seek/restart)

Each patch is a self-contained prototype. It reads from the SignalBus
and writes to the OutputBus — it never touches detectors or sinks directly.
"""

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


def load_patch(name: str) -> Patch:
    """Load a patch by name.

    Available patches:
        color_music  —  hue→note, brightness→octave, flutter→AM-LFO
    """
    if name == "color_music":
        from eye_synth.patches.color_music import ColorMusicPatch
        return ColorMusicPatch()
    raise ValueError(
        f"Unknown patch: {name!r}. Available patches: color_music"
    )
