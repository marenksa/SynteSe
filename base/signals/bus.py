"""Signal bus: shared data structures passed between detectors and patches."""

from __future__ import annotations

from dataclasses import dataclass, field

from base.signals.eye_blinks import BlinkSample, FlutterEvent
from base.signals.head_gaze_state import HeadGazeState


@dataclass
class EyeSignals:
    """All eye-derived signals, updated every loop iteration."""

    # Gaze
    confidence: float = 0.0
    norm_pos: tuple[float, float] = (0.5, 0.5)   # 0.0–1.0, Pupil convention
    velocity: float = 0.0                    # gaze speed in pixels/second

    # Blink state
    is_eyes_closed: bool = False                  # between onset and offset
    eyes_closed_elapsed_ms: float = 0.0           # ms since most recent onset (resets per blink)
    blink: BlinkSample | None = None              # non-None for 1 iteration when a blink completes

    # Flutter state
    is_flutter_active: bool = False
    flutter_blink_count: int = 0                  # blinks accumulated during active burst
    flutter: FlutterEvent | None = None           # non-None for 1 iteration when flutter ends

    # Fixation
    fixation_id: int | None = None                # non-None for 1 iteration on new fixation


@dataclass
class EnvSignals:
    """All environment-derived signals, updated when a gaze region is analyzed."""

    # Color at gaze point
    hue: float = 0.0               # OpenCV 0–179 (temporally smoothed)
    raw_hue: float | None = None   # instantaneous hue (pre-smoothing); None if saturation too low
    saturation: float = 0.0       # 0–255
    brightness: float = 0.0       # 0–255

    # Change signals
    scene_change: float = 0.0     # full-frame change magnitude 0.0–1.0


@dataclass
class SignalBus:
    """All signals for the current iteration, passed to patches."""

    eye: EyeSignals = field(default_factory=EyeSignals)
    env: EnvSignals = field(default_factory=EnvSignals)
    head_gaze_state: HeadGazeState = HeadGazeState.Rest
    timestamp: float = 0.0
    has_env_reading: bool = False  # True when env signals were freshly updated this iteration

    def clear_events(self) -> None:
        """Clear one-shot event fields. Call at the start of each loop iteration."""
        self.eye.blink = None
        self.eye.flutter = None
        self.eye.fixation_id = None
        self.has_env_reading = False


class OutputBus:
    """Thin wrapper around output sinks with a generic send API.

    Patches call outputs.send("key", value) without knowing about the
    underlying protocol (FUDI, OSC, etc.).
    """

    def __init__(self, pd_sink=None) -> None:
        self._pd = pd_sink
        self.last_note_on: int | None = None  # set each time note_on is sent; caller clears

    def send(self, key: str, *values) -> None:
        """Send a named message to Pure Data via FUDI.

        Examples:
            outputs.send("note_on", 60, 0.8)
            outputs.send("effect", 0.8)
            outputs.send("confidence", 0.95)
        """
        if key == "note_on" and values:
            self.last_note_on = int(values[0])
        if self._pd is not None:
            self._pd.send(key, *values)

    def close(self) -> None:
        if self._pd is not None:
            self._pd.close()
