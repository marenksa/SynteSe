"""Patches: creative mappings from signals to outputs.

A patch is any object that implements:
    update(signals: SignalBus, outputs: OutputBus) -> None
    reset() -> None  (called on seek/restart)

Each patch is a self-contained prototype. It reads from the SignalBus
and writes to the OutputBus — it never touches detectors or sinks directly.
"""

from eye_synth.patches.base import Patch, load_patch

__all__ = ["Patch", "load_patch"]
