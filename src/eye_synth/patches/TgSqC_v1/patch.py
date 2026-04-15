"""ColorTogglePatch: maps gaze color to a PD toggle with color ID routing."""

from __future__ import annotations

from eye_synth.patches.TgSqC_v1.mapping import ColorIdMapper
from eye_synth.signals.bus import OutputBus, SignalBus


class ColorTogglePatch:
    """Sends color_id and toggle messages to Pure Data based on gaze color stability.

    Messages sent:
        toggle 0       — color became unstable (turn sound off)
        color_id <N>   — new stable color ID 1–7 (which sound to use)
        toggle 1       — color is stable (turn sound on)

    On color change (X → Y):
        toggle 0  →  color_id Y  →  toggle 1

    On stability loss:
        toggle 0

    On first stable detection:
        color_id N  →  toggle 1
    """

    def __init__(self) -> None:
        self._mapper = ColorIdMapper()
        self._active_id: int | None = None   # color ID currently toggled on
        self._toggle_is_on: bool = False

    def reset(self) -> None:
        self._mapper.reset()
        self._active_id = None
        self._toggle_is_on = False

    def shutdown(self, outputs: OutputBus) -> None:
        outputs.send("toggle", 0)

    def update(self, signals: SignalBus, outputs: OutputBus) -> None:
        if not signals.has_env_reading:
            return

        stable_id = self._mapper.update(signals.env.hue, signals.env.saturation)

        if stable_id is None:
            # Color is unstable — turn off if currently on
            if self._toggle_is_on:
                outputs.send("toggle", 0)
                self._toggle_is_on = False
                self._active_id = None

        elif stable_id != self._active_id:
            # New stable color — switch over
            if self._toggle_is_on:
                outputs.send("toggle", 0)
            outputs.send("color_id", stable_id)
            outputs.send("toggle", 1)
            self._active_id = stable_id
            self._toggle_is_on = True

        # stable_id == self._active_id: no change, do nothing
