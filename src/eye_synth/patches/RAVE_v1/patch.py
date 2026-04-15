"""RAVE_v1: maps eye/env signals to RAVE latent dimensions.

Pd messages sent every frame:
    z0 <-3..3>   gaze X (left=−3, right=+3)
    z1 <-3..3>   gaze Y (bottom=−3, top=+3)  [Pupil: 0=bottom, 1=top]
    z2 <-3..3>   colour hue (0=−3, 1=+3)
    z3 <-3..3>   brightness (dark=−3, bright=+3)
    z4 <-3..3>   gaze velocity (still=−3, fast=+3, clamped at VEL_MAX px/s)

Pd messages sent on change:
    blink <0|1>      eyes closed
    flutter <0|1>    rapid-blink burst active
    intentional <0|1> eyes closed ≥ INTENTIONAL_MS
"""

from __future__ import annotations

import time

from eye_synth.output.overlay import OverlayConfig
from eye_synth.signals.bus import OutputBus, SignalBus

LATENT_SCALE = 3.0      # output range ±LATENT_SCALE
VEL_MAX = 800.0         # px/s clamp for velocity → latent
INTENTIONAL_MS = 400


def _norm_to_latent(v: float) -> float:
    """Map 0.0–1.0 to −LATENT_SCALE..+LATENT_SCALE."""
    return (v * 2.0 - 1.0) * LATENT_SCALE


def _vel_to_latent(vel_px_s: float) -> float:
    """Map velocity 0..VEL_MAX to −LATENT_SCALE..+LATENT_SCALE."""
    clamped = min(vel_px_s, VEL_MAX) / VEL_MAX
    return _norm_to_latent(clamped)


class RAVEPatch:
    """Streams eye signals as RAVE latent values to Pure Data / nn~."""

    overlay = OverlayConfig(
        show_confidence=True,
        show_color_info=True,
        show_gaze_crosshair=True,
        show_blink_flutter=True,
    )

    def __init__(self) -> None:
        self._eyes_closed_since: float | None = None
        self._prev_blink = 0
        self._prev_flutter = 0
        self._prev_intentional = -1  # sentinel: force send on first update

    def reset(self) -> None:
        self._eyes_closed_since = None
        self._prev_blink = 0
        self._prev_flutter = 0
        self._prev_intentional = -1

    def shutdown(self, outputs: OutputBus) -> None:
        for i in range(5):
            outputs.send(f"z{i}", 0.0)
        outputs.send("blink", 0)
        outputs.send("flutter", 0)
        outputs.send("intentional", 1)

    def update(self, signals: SignalBus, outputs: OutputBus) -> None:
        eye = signals.eye
        env = signals.env

        # Latent dims
        z0 = _norm_to_latent(eye.norm_pos[0])
        z1 = _norm_to_latent(eye.norm_pos[1])  # Pupil: 0=bottom, 1=top
        z2 = _norm_to_latent(env.hue_normalized)
        z3 = _norm_to_latent(env.brightness_normalized)
        z4 = _vel_to_latent(eye.velocity_px_s)

        outputs.send("z0", z0)
        outputs.send("z1", z1)
        outputs.send("z2", z2)
        outputs.send("z3", z3)
        outputs.send("z4", z4)

        # Event booleans
        blink = int(eye.is_eyes_closed)
        flutter = int(eye.is_flutter_active)

        if eye.is_eyes_closed:
            if self._eyes_closed_since is None:
                self._eyes_closed_since = time.monotonic()
            elapsed_ms = (time.monotonic() - self._eyes_closed_since) * 1000
            intentional = int(elapsed_ms >= INTENTIONAL_MS)
        else:
            self._eyes_closed_since = None
            intentional = 0

        if blink != self._prev_blink:
            outputs.send("blink", blink)
            self._prev_blink = blink
        if flutter != self._prev_flutter:
            outputs.send("flutter", flutter)
            self._prev_flutter = flutter
        if intentional != self._prev_intentional:
            outputs.send("intentional", intentional)
            self._prev_intentional = intentional
