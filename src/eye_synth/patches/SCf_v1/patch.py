"""ConfidenceStreamPatch: streams raw eye confidence and eye-state bools to PD."""

from __future__ import annotations

from eye_synth.signals.bus import OutputBus, SignalBus


class ConfidenceStreamPatch:
    """Continuously streams eye confidence and eye-state signals to Pure Data.

    Messages sent every frame:
        confidence <0–1>   —  raw tracker confidence
        blink <0|1>        —  1 while eyes are closed
        flutter <0|1>      —  1 while a flutter burst is active
    """

    def reset(self) -> None:
        pass

    def shutdown(self, outputs: OutputBus) -> None:
        outputs.send("confidence", 1.0)
        outputs.send("blink", 0)
        outputs.send("flutter", 0)

    def update(self, signals: SignalBus, outputs: OutputBus) -> None:
        outputs.send("confidence", signals.eye.confidence)
        outputs.send("blink", int(signals.eye.is_eyes_closed))
        outputs.send("flutter", int(signals.eye.is_flutter_active))
