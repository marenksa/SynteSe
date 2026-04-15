"""ConfidenceStreamPatch: streams raw eye confidence and eye-state bools to PD."""

from __future__ import annotations

import time

from eye_synth.output.overlay import OverlayConfig
from eye_synth.signals.bus import OutputBus, SignalBus

INTENTIONAL_MS = 400  # ms eyes must be closed to count as intentional


class ConfidenceStreamPatch:
    """Continuously streams eye confidence and eye-state signals to Pure Data.

    Messages sent every frame:
        confidence <0–1>   —  raw tracker confidence

    Messages sent on change only:
        flutter <0|1>      —  1 while a flutter burst is active
        intentional <0|1>  —  1 while eyes have been closed >= INTENTIONAL_MS
        blink <0|1>        —  1 while eyes are closed
    """

    overlay = OverlayConfig(
        show_confidence=True,
        show_eye_panel=True,
        show_blink_flutter=True,
    )

    def __init__(self) -> None:
        self._eyes_closed_since: float | None = None
        self._prev_flutter: int = 0
        self._prev_intentional: int = -1  # sentinel: force send on first update
        self._prev_blink: int = 0

    def reset(self) -> None:
        self._eyes_closed_since = None
        self._prev_flutter = 0
        self._prev_intentional = -1  # sentinel: force send on first update after reset
        self._prev_blink = 0

    def shutdown(self, outputs: OutputBus) -> None:
        outputs.send("confidence", 1.0)
        outputs.send("flutter", 0)
        outputs.send("intentional", 1)
        outputs.send("blink", 0)

    def update(self, signals: SignalBus, outputs: OutputBus) -> None:
        if signals.eye.is_eyes_closed:
            if self._eyes_closed_since is None:
                self._eyes_closed_since = time.monotonic()
            elapsed_ms = (time.monotonic() - self._eyes_closed_since) * 1000
            is_intentional = int(elapsed_ms >= INTENTIONAL_MS)
        else:
            self._eyes_closed_since = None
            is_intentional = 0

        flutter = int(signals.eye.is_flutter_active)
        blink = int(signals.eye.is_eyes_closed)

        outputs.send("confidence", signals.eye.confidence)

        if flutter != self._prev_flutter:
            outputs.send("flutter", flutter)
            self._prev_flutter = flutter
        if is_intentional != self._prev_intentional:
            outputs.send("intentional", is_intentional)
            self._prev_intentional = is_intentional
        if blink != self._prev_blink:
            outputs.send("blink", blink)
            self._prev_blink = blink
