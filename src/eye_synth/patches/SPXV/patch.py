"""GazeStreamPatch: streams raw gaze coordinates and velocity to PD.

Messages sent every frame:
    gaze_x <0–1>      —  normalised horizontal gaze position (0=left, 1=right)
    gaze_y <0–1>      —  normalised vertical gaze position (Pupil: 0=bottom, 1=top)
    velocity <float>  —  gaze speed in pixels/second (held at last good value during low confidence)

PD patches (SPXV_v1, SPXV_v2, SPXV_v3) handle all pitch/loudness mapping from these values.
"""

from __future__ import annotations

from eye_synth.output.overlay import OverlayConfig
from eye_synth.signals.bus import OutputBus, SignalBus

CONF_THRESHOLD = 0.5  # below this, velocity is held at its last trusted value


class GazeStreamPatch:
    """Streams raw gaze position and velocity to Pure Data."""

    overlay = OverlayConfig(
        show_gaze_crosshair=True,
    )

    def __init__(self) -> None:
        self._last_velocity: float = 0.0

    def reset(self) -> None:
        self._last_velocity = 0.0

    def shutdown(self, outputs: OutputBus) -> None:
        outputs.send("gaze_x", 0.0)
        outputs.send("gaze_y", 0.0)
        outputs.send("velocity", 0.0)

    def update(self, signals: SignalBus, outputs: OutputBus) -> None:
        eye = signals.eye
        if eye.confidence >= CONF_THRESHOLD:
            self._last_velocity = eye.velocity
        outputs.send("gaze_x", eye.norm_pos[0])
        outputs.send("gaze_y", eye.norm_pos[1])
        outputs.send("velocity", self._last_velocity)
