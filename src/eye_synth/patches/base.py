"""Patch protocol, registry, and factory."""

from __future__ import annotations

from typing import Protocol

from eye_synth.signals.bus import OutputBus, SignalBus

_REGISTRY: dict[str, type] = {}


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


def register_patch(name: str, cls: type) -> None:
    """Register a patch class under a given name."""
    _REGISTRY[name] = cls


def load_patch(name: str) -> Patch:
    """Load a patch by name. Patches self-register via register_patch()."""
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"Unknown patch: {name!r}. Available: {available}")
    return _REGISTRY[name]()
