"""RAVE_v1: maps eye/env signals to RAVE latent dimensions.

Pd messages sent every frame:
    x      <-3..3>   gaze X (left=−3, right=+3)
    y      <-3..3>   gaze Y (bottom=−3, top=+3)  [Pupil: 0=bottom, 1=top]
    hue    <-3..3>   colour hue (0=−3, 1=+3)
    bright <-3..3>   brightness (dark=−3, bright=+3)
    vel    <-3..3>   gaze velocity (still=−3, fast=+3, clamped at VEL_MAX px/s)
    conf   <-3..3>   tracker confidence (0=−3, 1=+3)
    intent <-3..3>   intentional closure ramp (open=−3, closed=+3)

Pd messages sent on change:
    blink <0|1>   eyes closed
    flut  <0|1>   rapid-blink burst active
"""

from __future__ import annotations

from eye_synth.output.overlay import OverlayConfig
from eye_synth.signals.bus import OutputBus, SignalBus

LATENT_SCALE = 3.0      # output range ±LATENT_SCALE
VEL_MAX = 800.0         # px/s clamp for velocity → latent
INTENTIONAL_MS = 600
RAMP_DURATION = 3.0     # seconds to ramp intentional -3↔+3


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
        self._prev_blink = 0
        self._prev_flutter = 0
        self._prev_intentional = -1          # sentinel: force ramp init on first update
        self._int_value = -LATENT_SCALE      # current ramped value
        self._int_target = -LATENT_SCALE
        self._int_ramp_start: float | None = None
        self._int_ramp_from = -LATENT_SCALE

    def reset(self) -> None:
        self._prev_blink = 0
        self._prev_flutter = 0
        self._prev_intentional = -1
        self._int_value = -LATENT_SCALE
        self._int_target = -LATENT_SCALE
        self._int_ramp_start = None
        self._int_ramp_from = -LATENT_SCALE

    def shutdown(self, outputs: OutputBus) -> None:
        for name in ("x", "y", "hue", "bright", "vel", "conf", "intent"):
            outputs.send(name, 0.0)
        outputs.send("blink", 0)
        outputs.send("flut", 0)

    def update(self, signals: SignalBus, outputs: OutputBus) -> None:
        eye = signals.eye
        env = signals.env

        outputs.send("x",      _norm_to_latent(eye.norm_pos[0]))
        outputs.send("y",      _norm_to_latent(eye.norm_pos[1]))  # Pupil: 0=bottom, 1=top
        outputs.send("hue",    _norm_to_latent(env.hue / 179.0))
        outputs.send("bright", _norm_to_latent(env.brightness / 255.0))
        outputs.send("vel",    _vel_to_latent(eye.velocity))
        outputs.send("conf",   _norm_to_latent(eye.confidence))

        # Event booleans
        blink = int(eye.is_eyes_closed)
        flutter = int(eye.is_flutter_active)

        if eye.is_eyes_closed:
            intentional = int(eye.eyes_closed_elapsed_ms >= INTENTIONAL_MS)
        else:
            intentional = 0

        if blink != self._prev_blink:
            outputs.send("blink", blink)
            self._prev_blink = blink
        if flutter != self._prev_flutter:
            outputs.send("flut", flutter)
            self._prev_flutter = flutter

        # Intent: ramp -3↔+3 over RAMP_DURATION seconds
        if intentional != self._prev_intentional:
            self._int_target = LATENT_SCALE if intentional else -LATENT_SCALE
            self._int_ramp_from = self._int_value
            self._int_ramp_start = time.monotonic()
            self._prev_intentional = intentional

        if self._int_ramp_start is not None:
            t = min((time.monotonic() - self._int_ramp_start) / RAMP_DURATION, 1.0)
            self._int_value = self._int_ramp_from + t * (self._int_target - self._int_ramp_from)

        outputs.send("intent", self._int_value)
