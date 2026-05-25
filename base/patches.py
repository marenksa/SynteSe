"""Patch protocol, registry, and auto-discovery."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Protocol

from base.signals.bus import OutputBus, SignalBus

_REGISTRY: dict[str, type] = {}
_PROTO_DIR = Path(__file__).parent.parent.parent / "prototypes"


class Patch(Protocol):
    """Protocol that all patches must implement."""

    def update(self, signals: SignalBus, outputs: OutputBus) -> None: ...
    def reset(self) -> None: ...
    def shutdown(self, outputs: OutputBus) -> None: ...


def register_patch(name: str, cls: type) -> None:
    _REGISTRY[name] = cls


def load_patch(name: str) -> Patch:
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"Unknown patch: {name!r}. Available: {available}")
    return _REGISTRY[name]()


def _discover() -> None:
    for init in sorted(_PROTO_DIR.glob("*/__init__.py")):
        mod = f"prototypes.{init.parent.name}"
        if mod not in sys.modules:
            importlib.import_module(mod)


_discover()

__all__ = ["Patch", "load_patch", "register_patch"]
