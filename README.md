# Eye Tracking Music System

Eye tracking system for building audiovisual prototypes with Pupil Core. Run live with the tracker or from recorded sessions with the player — both are full performance modes. Detects signals from the environment (colour, brightness, scene changes) and from the eyes (gaze position, confidence, blinks, flutter, velocity) and routes them to sound synthesis through Pure Data.

## Installation

### Prerequisites

1. **Pupil Core** hardware connected via USB
2. **Pupil Capture** running with administrator privileges and **Frame Publisher plugin** enabled
   - Run Pupil Capture as admin (required for USB camera access on some systems)
   - Plugin Manager → Enable "Frame Publisher"
3. **Pure Data** for sound synthesis

```bash
uv sync
```

## Quick Start

```bash
# 1. Open puredata/TNC_v1.pd in Pure Data
# 2. Enable DSP in Pure Data (Media → DSP On)

# Run live with Pupil Core hardware:
uv run pupil-tracker

# Or run from a recording:
uv run pupil-player recordings/000
```

Player controls: Space (play/pause), ←/→ (frame step), `[`/`]` (speed), 0–9 (jump), H (help), Q (quit).

The default patch is `TNC_v1`. See [`patches/TNC_v1/README.md`](src/eye_synth/patches/TNC_v1/README.md) for its specific mappings.

## CLI Reference

### Common flags

Both commands share these flags:

```
--patch NAME             Patch to use (default: TNC_v1)
--pd-host HOST           Pure Data host (default: 127.0.0.1)
--pd-port PORT           Pure Data port (default: 9001)
--verbose, -v            Verbose console output
--no-overlay             Disable colour/brightness overlay
```

### `pupil-tracker`

```
--host HOST              Pupil Capture host (default: 127.0.0.1)
--port PORT              Pupil Capture port (default: 50020)
```

### `pupil-player`

```
recording_path           Path to Pupil Capture recording directory
```

## Project Structure

```
├── puredata/                       # Pure Data synthesis patches
│   ├── TNC_v1.pd                   # Default patch (hue→note, brightness→octave)
│   ├── TgSqC_v1.pd                 # Toggle sequence by colour
│   ├── SCfBF_v1.pd                 # Confidence stream + eye-state booleans (v1)
│   ├── SCfBF_v2.pd                 # Confidence stream + eye-state booleans (v2)
│   ├── SPX_v1.pd                   # Gaze position + velocity → pitch/loudness (v1)
│   ├── SPX_v2.pd                   # Gaze position + velocity → pitch/loudness, inverted (v2)
│   ├── SPX_v3.pd                   # Gaze position + velocity → pitch/loudness (v3)
│   └── RAVE_v1.pd                  # RAVE latent dimensions for nn~
├── recordings/                     # Pupil Capture recordings
└── src/eye_synth/
    ├── tracker.py                  # Live tracker entry point (pupil-tracker)
    ├── player.py                   # Recording player entry point (pupil-player)
    ├── input/
    │   ├── live.py                 # ZMQ connection to Pupil Capture
    │   └── recording.py            # Recording loader + playback support
    ├── signals/
    │   ├── bus.py                  # SignalBus, OutputBus
    │   ├── pipeline.py             # Pipeline: owns detectors, populates SignalBus
    │   ├── head_gaze_state.py      # HeadGazeState + HeadGazeClassifier
    │   ├── env_color.py            # Colour analysis (raw HSV extraction)
    │   ├── env_scene_change.py     # Full-frame change detection
    │   ├── eye_blinks.py           # Blink/flutter detection, types, constants
    │   └── eye_gaze.py             # Gaze velocity
    ├── output/
    │   ├── sinks.py                # PureDataSink, ColorConsoleSink, MultiSink
    │   └── overlay.py              # Stateless drawing functions
    └── patches/
        ├── base.py                 # Patch protocol, registry + load_patch()
        ├── TNC_v1/                 # Trigger Note by Colour: hue→MIDI note
        │   ├── mapping.py          # Note, NoteMapper, hue/brightness constants
        │   ├── gate.py             # NoteGate
        │   ├── patch.py            # ColorMusicPatch
        │   └── README.md
        ├── TgSqC_v1/              # Toggle Sequence by Colour: hue→PD toggle
        │   ├── mapping.py          # ColorIdMapper
        │   └── patch.py            # ColorTogglePatch
        ├── SCfBF/                  # Stream Confidence from Blinks/Flutter
        │   └── patch.py            # ConfidenceStreamPatch
        ├── SPX/                    # Stream gaze position + velocity
        │   └── patch.py            # GazeStreamPatch
        └── RAVE_v1/               # Gaze/colour/velocity → RAVE latents
            └── patch.py            # RAVEPatch
```

## Writing a New Patch

The system is split into two layers:

- **Detectors** — extract signals from the camera and eye tracker. They run the same way regardless of what you're building.
- **Patches** — map those signals to sound. Each prototype is its own patch file.

This means experimenting with a new musical idea means writing a new patch, not touching any detection code.

There is an AGENTS.md file to facilitate the use of coding agents to keep working on this project, including creating prototypes.

### Available Signals

Every loop iteration, detectors populate a `SignalBus` that patches can read from.

#### Head-gaze state

`signals.head_gaze_state` classifies the relationship between eye movement and head movement each frame:

| State | Gaze in frame | Scene change |
|-------|---------------|--------------|
| `Rest` | stable | low |
| `SmoothPan` | stable | high — head moves, gaze rides the camera |
| `Scanning` | moving | low |
| `RagLock` | moving | high |

Import: `from eye_synth.signals.head_gaze_state import HeadGazeState`

#### Eye signals (`signals.eye.*`)

| Signal | Type | Description |
|--------|------|-------------|
| `confidence` | `float` 0–1 | Gaze tracking confidence |
| `norm_pos` | `(float, float)` | Normalised gaze position (0–1), Pupil convention |
| `velocity` | `float` | Gaze speed in pixels/second |
| `is_eyes_closed` | `bool` | Between blink onset and offset |
| `eyes_closed_elapsed_ms` | `float` | ms since most recent onset; resets each blink |
| `blink` | `BlinkSample \| None` | Non-None for one iteration when a blink completes |
| `is_flutter_active` | `bool` | During a rapid blink burst |
| `flutter_blink_count` | `int` | Blinks accumulated in current burst |
| `flutter` | `FlutterEvent \| None` | Non-None for one iteration when a flutter ends |
| `fixation_id` | `int \| None` | Non-None for one iteration on new fixation |

Blink types: `BLINK` (≤400ms), `INTENTIONAL` (≥500ms), `AMBIGUOUS` (>400ms and <500ms).
Flutter threshold: 3+ blinks within 1.5s.

#### Environment signals (`signals.env.*`)

Only populated when `signals.has_env_reading` is True (a gaze region was successfully analysed this iteration).

| Signal | Type | Description |
|--------|------|-------------|
| `hue` | `float` 0–179 | OpenCV hue at gaze point (temporally smoothed) |
| `raw_hue` | `float \| None` | Instantaneous hue (pre-smoothing); None if saturation too low |
| `saturation` | `float` 0–255 | Colour saturation at gaze point |
| `brightness` | `float` 0–255 | Brightness at gaze point (smoothed) |
| `scene_change` | `float` 0–1 | Full-frame change magnitude (head movement, scene cuts) |

### Patch structure

A patch is a class with three methods:

```python
class MyPatch:
    def update(self, signals: SignalBus, outputs: OutputBus) -> None:
        """Called every loop iteration."""
        ...

    def reset(self) -> None:
        """Called on seek or restart."""
        ...

    def shutdown(self, outputs: OutputBus) -> None:
        """Called on exit — send any cleanup messages (e.g. silence notes)."""
        ...
```

`outputs.send("message_name", value1, value2, ...)` sends a FUDI message to Pure Data.

**Example — gaze speed controls reverb, smooth pans trigger a hit:**

```python
from eye_synth.signals.head_gaze_state import HeadGazeState

class SceneMotionPatch:
    def update(self, signals, outputs):
        outputs.send("reverb", signals.eye.velocity / 2000.0)

        if signals.head_gaze_state is HeadGazeState.SmoothPan:
            outputs.send("hit", 1)

        if signals.eye.blink is not None:
            from eye_synth.signals.eye_blinks import BlinkType
            if signals.eye.blink.blink_type == BlinkType.INTENTIONAL:
                outputs.send("freeze", 1)

    def reset(self):
        pass

    def shutdown(self, outputs):
        outputs.send("reverb", 0)
```

Then register it in your patch's `__init__.py`:

```python
from eye_synth.patches.base import register_patch
from eye_synth.patches.scene_motion.patch import SceneMotionPatch

register_patch("scene_motion", SceneMotionPatch)
```

And import it in `patches/__init__.py` so it loads at startup:

```python
import eye_synth.patches.scene_motion  # noqa: F401
```

Then run it:

```bash
uv run pupil-tracker --patch scene_motion
uv run pupil-player recordings/000 --patch scene_motion
```

Each patch should have its own `README.md` documenting its specific mappings.

### Patch naming convention

Patches are named `{Control}{Target}{Source}_v{N}`, where each token describes what the patch does: "trigger note by color" → `TNC_v1`.

| Control | | Target | | Source | |
|:---|:---|:---|:---|:---|:---|
| Trigger | `T` | Note | `N` | Color | `C` |
| Toggle | `Tg` | Pitch | `P` | Brightness | `Br` |
| Stream | `S` | Sequence | `Sq` | Coordinates | `X` |
| | | Effect | `E` | Velocity | `V` |
| | | | | Confidence | `Cf` |
| | | | | Blink | `B` |
| | | | | Flutter | `F` |


## Troubleshooting

**No data received** — Enable "Frame Publisher" plugin in Pupil Capture; check gaze mapping is active.

**High latency** — The system drains ZMQ buffers automatically. Restart if lag persists; reduce camera resolution if needed.

**Connection timeout** — Verify Pupil Capture is on port 50020; check firewall if connecting remotely.

