# The SynteSe Toolkit

SynteSe is a toolkit for building audiovisual prototypes with Pupil Core. Run live with the tracker or from recorded sessions with the player — both are full performance modes. Detects signals from the environment (colour, brightness, scene changes) and from the eyes (gaze position, confidence, blinks, flutter, velocity) and routes them to sound synthesis through Pure Data.

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
# 1. Open prototypes/TNC_v1/TNC_v1.pd in Pure Data
# 2. Enable DSP in Pure Data (Media → DSP On)

# Run live with Pupil Core hardware:
uv run pupil-tracker

# Or run from a recording:
uv run pupil-player recordings/000
```

Player controls: Space (play/pause), ←/→ (frame step), `[`/`]` (speed), 0–9 (jump), H (help), Q (quit).

The default patch is `TNC_v1`. See [`prototypes/TNC_v1/README.md`](prototypes/TNC_v1/README.md) for its specific mappings.

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
├── prototypes/                     # Instrument prototypes — Python mapping + PD synthesis
│   ├── PROTOTYPING.md              # Guide for building new prototypes
│   ├── TNC_v1/                     # Trigger Note by Colour: hue→MIDI note
│   │   ├── patch.py                # ColorMusicPatch
│   │   ├── mapping.py              # Note, NoteMapper, hue/brightness constants
│   │   ├── gate.py                 # NoteGate
│   │   ├── TNC_v1.pd               # Pure Data patch
│   │   └── README.md
│   ├── TgSqC_v1/                   # Toggle Sequence by Colour: hue→PD toggle
│   │   ├── patch.py                # ColorTogglePatch
│   │   ├── mapping.py              # ColorIdMapper
│   │   └── TgSqC_v1.pd
│   ├── SCfBF/                      # Stream Confidence from Blinks/Flutter
│   │   ├── patch.py                # ConfidenceStreamPatch
│   │   ├── SCfBF_v1.pd
│   │   └── SCfBF_v2.pd
│   ├── SPX/                        # Stream gaze position + velocity
│   │   ├── patch.py                # GazeStreamPatch
│   │   ├── SPX_v1.pd
│   │   ├── SPX_v2.pd
│   │   └── SPX_v3.pd
│   └── RAVE_v1/                    # Gaze/colour/velocity → RAVE latents
│       ├── patch.py                # RAVEPatch
│       └── RAVE_v1.pd
├── recordings/                     # Pupil Capture recordings
└── base/                           # Detection and routing engine
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
        └── base.py                 # Patch protocol, registry + load_patch()
```

## Creating a New Prototype

```bash
uv run new-prototype MyPatch_v1
```

Scaffolds a Python patch and a pre-wired Pure Data template. See [prototypes/PROTOTYPING.md](prototypes/PROTOTYPING.md) for a full walkthrough.


## Troubleshooting

**No data received** — Enable "Frame Publisher" plugin in Pupil Capture; check gaze mapping is active.

**High latency** — The system drains ZMQ buffers automatically. Restart if lag persists; reduce camera resolution if needed.

**Connection timeout** — Verify Pupil Capture is on port 50020; check firewall if connecting remotely.

