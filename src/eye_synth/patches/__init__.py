"""Patches: creative mappings from signals to outputs.

A patch is any object that implements:
    update(signals: SignalBus, outputs: OutputBus) -> None
    reset() -> None  (called on seek/restart)

Each patch is a self-contained prototype. It reads from the SignalBus
and writes to the OutputBus — it never touches detectors or sinks directly.
"""

from eye_synth.patches.base import Patch, load_patch, register_patch

# Import all patches to trigger self-registration
import eye_synth.patches.RAVE_v1  # noqa: F401
import eye_synth.patches.SCfBF  # noqa: F401
import eye_synth.patches.SPX  # noqa: F401
import eye_synth.patches.TgSqC_v1  # noqa: F401
import eye_synth.patches.TNC_v1  # noqa: F401

__all__ = ["Patch", "load_patch", "register_patch"]
