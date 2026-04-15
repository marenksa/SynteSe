# Pupil Color-to-Music Tracker

Real-time eye tracking color-to-music system using Pupil Core. Streams gaze and video data from Pupil Capture, analyzes color at the gaze point, and outputs MIDI notes for sound synthesis through Pure Data.

## Features

- Map colors to musical notes based on wavelength physics
  - Longer wavelength (red) → lower notes (C)
  - Shorter wavelength (violet) → higher notes (B)
- Brightness determines octave (darker = lower octave 2-6)
- Gaussian-weighted spatial averaging (center pixels weigh more)
- Temporal smoothing with configurable stability

## Prerequisites

1. **Pupil Core** eye tracking hardware connected via USB
2. **Pupil Capture** software running on your machine
3. **Frame Publisher plugin** enabled in Pupil Capture:
   - Open Pupil Capture → Plugin Manager → Enable "Frame Publisher"

## Installation

This project uses [UV](https://docs.astral.sh/uv/) for dependency management (not pip or poetry).

```bash
uv sync
```

## Quick Start

```bash
# 1. Open Pure Data and load puredata/color_music.pd
# 2. Turn on DSP in Pure Data (Media → DSP On)
# 3. Run the tracker:
uv run pupil-tracker --pd
```

The color-to-music mapping:

| Color | Wavelength | Note |
|-------|-----------|------|
| Red | ~700nm | C |
| Orange | ~620nm | D |
| Yellow | ~580nm | E |
| Green | ~530nm | F |
| Cyan | ~500nm | G |
| Blue | ~470nm | A |
| Violet | ~400nm | B |

Brightness maps to octave (2-6): darker = lower octave, brighter = higher octave.

## Testing & Calibration

### Playback Recordings

Test with Pupil Capture recordings without needing the hardware:

```bash
# Play a recording with gaze overlay
uv run gaze-player recordings/000

# Send color-to-music to Pure Data while playing
uv run gaze-player recordings/000 --pd

# Brighten dark footage with gamma correction
uv run gaze-player recordings/000 --gamma 0.5
```

Controls: Space (play/pause), arrow keys (frame step), H (help), Q (quit).

### Test Images

Generate calibration images to verify color detection:

```bash
uv run python scripts/generate_test_image.py
```

This creates two images in `test_images/`:
- **color_test_grid.png**: 7x5 grid (colors × octaves) for full calibration
- **rainbow_strip.png**: Simple horizontal color strip

Open the test grid on your screen and look at each square with the glasses to verify detection.

### Test Pure Data Connection

Test the Pure Data connection without Pupil hardware:

```bash
uv run python scripts/test_puredata.py
```

### Test Color Grid Audio

Play through the entire color-brightness grid systematically:

```bash
# Run with Pure Data's color_music.pd open and DSP on
uv run python scripts/test_color_grid.py --auto

# Interactive mode (press Enter between tests)
uv run python scripts/test_color_grid.py

# Adjust note duration (default 0.8 seconds)
uv run python scripts/test_color_grid.py --duration 0.5 --auto
```

This plays two test sequences:
1. **By rows**: Each color from darkest to brightest (C2→C6, D2→D6, etc.)
2. **By columns**: Each octave through all colors (C2→B2, C3→B3, etc.)

## CLI Options

```
usage: pupil-tracker [-h] [--host HOST] [--port PORT] [--region-size SIZE]
                     [--smoothing FRAMES] [--no-video] [--verbose]
                     [--note-stability N] [--octave-stability N] [--octave-threshold T]
                     [--pd] [--pd-host HOST] [--pd-port PORT]

Options:
  --host HOST              Pupil Capture host address (default: 127.0.0.1)
  --port PORT              Pupil Capture control port (default: 50020)
  --region-size SIZE       Size of gaze region to analyze in pixels (default: 50)
  --smoothing FRAMES       Number of frames for smoothing (default: 5)
  --no-video               Disable video display
  --verbose, -v            Enable verbose console output

Stability tuning:
  --note-stability N       Frames for note stability (default: 8, lower = faster)
  --octave-stability N     Frames for octave stability (default: 15, higher = more stable)
  --octave-threshold T     Agreement threshold for octave 0-1 (default: 0.8)

Pure Data output:
  --pd                     Send to Pure Data via FUDI protocol (TCP)
  --pd-host HOST           Pure Data host address (default: 127.0.0.1)
  --pd-port PORT           Pure Data port (default: 9001)
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Pupil Capture                          │
│                    (localhost:50020)                        │
└─────────────────┬───────────────────┬───────────────────────┘
                  │                   │
            gaze.3d.*           frame.world
                  │                   │
                  ▼                   ▼
┌─────────────────────────────────────────────────────────────┐
│                   ZMQ Subscriber                            │
│    • Zero-buffer real-time streaming (RCVHWM=1)             │
│    • Drains to latest message only                          │
│                      client.py                              │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   Frame Processor                           │
│    • Maps normalized gaze (0-1) to pixel coordinates        │
│    • Extracts region around gaze point                      │
│    • Filters low-confidence gaze data                       │
│                     processor.py                            │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   Color Analyzer                            │
│    • HSV color space with Gaussian-weighted averaging       │
│    • Maps hue to note (C-B) based on wavelength             │
│    • Maps brightness to octave (2-6)                        │
│    • Temporal smoothing with stability tracking             │
│                      analyzer.py                            │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    Output Sinks                             │
│    • ColorConsoleSink: Real-time note/color display         │
│    • PureDataFUDISink: FUDI/TCP to Pure Data                │
│                       output.py                             │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
pupil-color-tracker/
├── pyproject.toml              # UV project configuration
├── README.md                   # This file
├── AGENTS.md                   # Instructions for AI agents
├── puredata/
│   └── color_music.pd          # Pure Data patch for synthesis
├── scripts/
│   ├── debug_connection.py     # Debug Pupil Capture connection
│   ├── test_puredata.py        # Test Pure Data communication
│   ├── test_color_grid.py      # Play through color-brightness grid
│   └── generate_test_image.py  # Generate calibration images
├── test_images/                # Generated calibration images
│   ├── color_test_grid.png     # 7x5 color-brightness grid
│   └── rainbow_strip.png       # Simple color strip
├── recordings/                 # Pupil Capture recordings for playback
│   └── 000/                    # Example recording directory
└── src/pupil_tracker/
    ├── __init__.py
    ├── client.py               # ZMQ client for Pupil Capture
    ├── processor.py            # Frame/gaze processing
    ├── analyzer.py             # Color analysis
    ├── output.py               # Output sink interfaces (FUDI)
    ├── recording.py            # Recording playback support
    ├── player.py               # Video player with gaze overlay
    └── main.py                 # CLI entry point
```

## Key Technical Details

### Real-Time Performance

The system is optimized for zero-latency streaming:
- `ZMQ_RCVHWM = 1`: Minimal receive buffer
- `ZMQ_LINGER = 0`: No waiting on socket close
- Buffer draining: Always processes the LATEST message, discards old ones

### Data Flow

1. **Gaze data** arrives at ~120Hz (high frequency)
2. **Frame data** arrives at ~30Hz (world camera rate)
3. Processing only occurs on new frames to prevent lag
4. Gaze is filtered by confidence (>0.5) and bounds (0-1)

### Frame Format

Pupil Capture sends frames as JPEG-encoded images. The client automatically decodes them using OpenCV's `imdecode`.

## Troubleshooting

### No data received
- Ensure Pupil Capture is running
- Enable "Frame Publisher" plugin in Pupil Capture
- Check that gaze mapping is active (eyes detected)

### High latency / lag
- The system automatically drains buffers - restart if issues persist
- Reduce camera resolution in Pupil Capture if needed

### Connection timeout
- Verify Pupil Capture is on port 50020 (default)
- Check firewall settings if connecting remotely

## License

MIT
