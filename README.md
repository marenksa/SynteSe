# Eye Tracking Music System

Real-time eye tracking system for building audiovisual prototypes with Pupil Core. Detects signals from the environment (colour, brightness, scene changes) and from the eyes (gaze position, confidence, blinks, flutter, velocity) and routes them to sound synthesis through Pure Data.

## Design Philosophy

The system is split into two layers:

- **Detectors** — extract signals from the camera and eye tracker. They run the same way regardless of what you're building.
- **Patches** — map those signals to sound. Each prototype is its own patch file.

This means experimenting with a new musical idea means writing a new patch, not touching any detection code. See [Writing a New Patch](#writing-a-new-patch).

## Prerequisites

1. **Pupil Core** hardware connected via USB
2. **Pupil Capture** running with **Frame Publisher plugin** enabled
   - Plugin Manager → Enable "Frame Publisher"
3. **Pure Data** for sound synthesis

## Installation

```bash
uv sync
```

## Quick Start

```bash
# 1. Open Pure Data and load a patch from puredata/
# 2. Enable DSP in Pure Data (Media → DSP On)
# 3. Run the tracker:
uv run pupil-tracker --pd
```

The default patch is `color_music`. See [`patches/color_music/README.md`](src/eye_synth/patches/color_music/README.md) for its specific mappings.

## Available Signals

Every loop iteration, detectors populate a `SignalBus` that patches can read from.

### Eye signals

| Signal | Type | Description |
|--------|------|-------------|
| `eye.confidence` | `float` 0–1 | Gaze tracking confidence |
| `eye.norm_pos` | `(float, float)` | Normalised gaze position (0–1) |
| `eye.px_pos` | `(int, int)` | Gaze position in pixels |
| `eye.velocity_px_s` | `float` | Gaze speed in pixels/second |
| `eye.is_eyes_closed` | `bool` | Between blink onset and offset |
| `eye.blink` | `BlinkSample \| None` | Non-None for one iteration when a blink completes |
| `eye.is_flutter_active` | `bool` | During a rapid blink burst |
| `eye.flutter_blink_count` | `int` | Blinks accumulated in current burst |
| `eye.flutter` | `FlutterEvent \| None` | Non-None for one iteration when a flutter ends |
| `eye.fixation_id` | `int \| None` | Non-None for one iteration on new fixation |

Blink types: `BLINK` (<400ms), `INTENTIONAL` (≥500ms), `AMBIGUOUS` (400–500ms).
Flutter threshold: 4+ blinks within 1.5s.

### Environment signals

| Signal | Type | Description |
|--------|------|-------------|
| `env.hue` | `float` 0–179 | OpenCV hue at gaze point |
| `env.hue_normalized` | `float` 0–1 | Normalised hue |
| `env.saturation` | `float` 0–255 | Colour saturation at gaze point |
| `env.brightness` | `float` 0–255 | Brightness at gaze point |
| `env.brightness_normalized` | `float` 0–1 | Normalised brightness |
| `env.note` | `Note` | Derived musical note (C–B) |
| `env.octave` | `int` 3–6 | Derived octave from brightness |
| `env.midi_note` | `int` | Derived MIDI note number |
| `env.raw_midi_note` | `int` | Pre-stability MIDI note (for transition detection) |
| `env.region_changed` | `bool` | True for one iteration when gaze content changes |
| `env.scene_change` | `float` 0–1 | Full-frame change magnitude (head movement, cuts) |

`signals.has_env_reading` is True only when a gaze region was successfully analysed on this iteration.

## Writing a New Patch

A patch is a class with two methods:

```python
class MyPatch:
    def update(self, signals: SignalBus, outputs: OutputBus) -> None:
        """Called every loop iteration."""
        ...

    def reset(self) -> None:
        """Called on seek or restart."""
        ...
```

`outputs.send("message_name", value1, value2, ...)` sends a FUDI message to Pure Data.

**Example — gaze speed controls reverb, scene cuts trigger a hit:**

```python
class SceneMotionPatch:
    def update(self, signals, outputs):
        outputs.send("reverb", signals.eye.velocity_px_s / 2000.0)

        if signals.env.scene_change > 0.4:
            outputs.send("hit", signals.env.scene_change)

        if signals.eye.blink is not None:
            from pupil_tracker.signals.eye_blinks import BlinkType
            if signals.eye.blink.blink_type == BlinkType.INTENTIONAL:
                outputs.send("freeze", 1)


    def reset(self):
        pass
```

Then register it in `patches/__init__.py`:

```python
def load_patch(name: str) -> Patch:
    ...
    if name == "scene_motion":
        from eye_synth.patches.scene_motion import SceneMotionPatch
        return SceneMotionPatch()
```

And run it:

```bash
uv run pupil-tracker --patch scene_motion --pd
uv run pupil-player recordings/000 --patch scene_motion --pd
```

Each patch should have its own `README.md` documenting its specific mappings.

## Playback

Test and develop against recorded sessions without needing hardware:

```bash
# Play a recording with gaze overlay
uv run pupil-player recordings/000

# Send patch output to Pure Data while playing
uv run pupil-player recordings/000 --pd

# Use a specific patch
uv run pupil-player recordings/000 --patch color_music --pd

# Brighten dark footage
uv run pupil-player recordings/000 --gamma 0.5
```

Controls: Space (play/pause), ←/→ (frame step), `[`/`]` (speed), 0–9 (jump), H (help), Q (quit).

## Project Structure

```
├── puredata/                       # Pure Data synthesis patches
│   └── color_music.pd
├── recordings/                     # Pupil Capture recordings
├── scripts/
│   ├── debug_connection.py
│   ├── test_puredata.py
│   └── test_color_grid.py
└── src/eye_synth/
    ├── tracker.py                  # Live tracker entry point (pupil-tracker)
    ├── player.py                   # Recording player entry point (pupil-player)
    ├── input/
    │   ├── live.py                 # ZMQ connection to Pupil Capture
    │   └── recording.py            # Recording loader + playback support
    ├── signals/
    │   ├── bus.py                  # SignalBus, OutputBus
    │   ├── pipeline.py             # Pipeline: owns detectors, populates SignalBus
    │   ├── env_color.py            # Colour analysis, NoteGate (hue→note, brightness→octave)
    │   ├── env_scene_change.py     # Full-frame change detection
    │   ├── eye_blinks.py           # Blink/flutter detection, types, constants
    │   └── eye_gaze.py             # Gaze velocity
    ├── output/
    │   ├── sinks.py                # PureDataSink, ColorConsoleSink, MultiSink
    │   └── overlay.py              # Stateless drawing functions
    └── patches/
        ├── __init__.py             # Patch protocol + load_patch()
        └── color_music/            # Prototype: colour→MIDI note
            ├── __init__.py
            └── README.md
```

## CLI Reference

### `pupil-tracker`

```
--host HOST              Pupil Capture host (default: 127.0.0.1)
--port PORT              Pupil Capture port (default: 50020)
--region-size N          Gaze region size in pixels (default: 50)
--smoothing N            Frames to average for smoothing (default: 3)
--no-video               Disable video display
--verbose, -v            Verbose console output
--gamma FLOAT            Gamma correction (< 1.0 brightens, default: 1.0)
--patch NAME             Patch to use (default: color_music)

Stability tuning:
  --note-stability N       Frames for note stability (default: 2)
  --octave-stability N     Frames for octave stability (default: 3)
  --octave-threshold T     Agreement threshold 0–1 (default: 0.5)

Pure Data output:
  --pd                     Send to Pure Data via FUDI/TCP
  --pd-host HOST           Pure Data host (default: 127.0.0.1)
  --pd-port PORT           Pure Data port (default: 9001)
```

### `pupil-player`

```
recording_path           Path to Pupil Capture recording directory
--start-frame N          Starting frame (default: 0)
--autoplay               Start playing immediately
--overlay                Show colour/note/brightness overlay
--patch NAME             Patch to use (default: color_music)
--pd                     Send to Pure Data
--pd-host HOST           Pure Data host
--pd-port PORT           Pure Data port
--region-size N          Gaze region size
--gamma FLOAT            Gamma correction
```

## Troubleshooting

**No data received** — Enable "Frame Publisher" plugin in Pupil Capture; check gaze mapping is active.

**High latency** — The system drains ZMQ buffers automatically. Restart if lag persists; reduce camera resolution if needed.

**Connection timeout** — Verify Pupil Capture is on port 50020; check firewall if connecting remotely.

## License

MIT
