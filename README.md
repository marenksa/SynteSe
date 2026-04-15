# Pupil Brightness Tracker

Real-time eye tracking brightness and color analyzer using Pupil Core. Streams gaze and video data from Pupil Capture, analyzes brightness and color at the gaze point, and outputs signals for sound synthesis through Pure Data.

## Features

- **Brightness Mode**: Analyze brightness at gaze point, map to pitch (original mode)
- **Color-to-Music Mode**: Map colors to musical notes based on wavelength physics
  - Longer wavelength (red) → lower notes (C)
  - Shorter wavelength (violet) → higher notes (B)
  - Brightness determines octave (darker = lower octave)

## Prerequisites

1. **Pupil Core** eye tracking hardware connected via USB
2. **Pupil Capture** software running on your machine
3. **Frame Publisher plugin** enabled in Pupil Capture:
   - Open Pupil Capture → Plugin Manager → Enable "Frame Publisher"

## Installation

This project uses [UV](https://docs.astral.sh/uv/) for dependency management (not pip or poetry).

```bash
cd pupil-brightness-tracker
uv sync
```

## Quick Start

```bash
# Run with video display (brightness mode)
uv run pupil-tracker

# Run without video (console only)
uv run pupil-tracker --no-video

# Run with verbose output
uv run pupil-tracker --verbose

# Log data to file
uv run pupil-tracker -o brightness_data.csv
```

### Color-to-Music Mode

Stream color data to Pure Data for musical note synthesis:

```bash
# 1. Open Pure Data and load puredata/color_music.pd
# 2. Turn on DSP in Pure Data (Media → DSP On)
# 3. Run the tracker with color mode:
uv run pupil-tracker --pd-color-fudi

# Or use OSC protocol (requires mrpeach external in Pd):
uv run pupil-tracker --pd-color-osc
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

Test the Pure Data patches without Pupil hardware:

```bash
# Test brightness mode (sine wave pattern)
uv run python scripts/test_puredata.py --fudi

# Test color mode (cycles through colors)
uv run python scripts/test_puredata.py --color-fudi
```

### Test Color Grid Audio

Play through the entire color-brightness grid systematically:

```bash
# Run with Pure Data's color_music.pd open and DSP on
uv run python scripts/test_color_grid.py --fudi --auto

# Interactive mode (press Enter between tests)
uv run python scripts/test_color_grid.py --fudi

# Adjust note duration (default 0.8 seconds)
uv run python scripts/test_color_grid.py --fudi --duration 0.5 --auto
```

This plays two test sequences:
1. **By rows**: Each color from darkest to brightest (C2→C6, D2→D6, etc.)
2. **By columns**: Each octave through all colors (C2→B2, C3→B3, etc.)

## CLI Options

```
usage: pupil-tracker [-h] [--host HOST] [--port PORT] [--region-size REGION_SIZE]
                     [--smoothing SMOOTHING] [--output OUTPUT] [--no-video] [--verbose]
                     [--note-stability N] [--octave-stability N] [--octave-threshold T]
                     [--pd-osc] [--pd-fudi] [--pd-color-osc] [--pd-color-fudi]
                     [--pd-host HOST] [--pd-port PORT]

Options:
  --host HOST              Pupil Capture host address (default: 127.0.0.1)
  --port PORT              Pupil Capture control port (default: 50020)
  --region-size SIZE       Size of gaze region to analyze in pixels (default: 50)
  --smoothing FRAMES       Number of frames for brightness smoothing (default: 5)
  --output, -o FILE        Output file for data (JSONL or CSV based on extension)
  --no-video               Disable video display
  --verbose, -v            Enable verbose console output

Color mode stability:
  --note-stability N       Frames for note stability (default: 8, lower = faster)
  --octave-stability N     Frames for octave stability (default: 15, higher = more stable)
  --octave-threshold T     Agreement threshold for octave 0-1 (default: 0.8)

Pure Data output:
  --pd-osc                 Stream brightness via OSC (use with brightness_receiver.pd)
  --pd-fudi                Stream brightness via FUDI (use with brightness_simple.pd)
  --pd-color-osc           Stream color/note via OSC (use with color_music.pd)
  --pd-color-fudi          Stream color/note via FUDI (use with color_music.pd)
  --pd-host HOST           Pure Data host address (default: 127.0.0.1)
  --pd-port PORT           Pure Data port (default: 9000 for OSC, 9001 for FUDI)
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
│                  Brightness Analyzer                        │
│    • Calculates mean luminance of gaze region               │
│    • Smooths values over configurable window                │
│                      analyzer.py                            │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    Output Sinks                             │
│    • ConsoleSink: Real-time brightness bar                  │
│    • FileSink: JSONL or CSV logging                         │
│    • ThresholdSink: Triggers on brightness changes          │
│    • (Future: USBSink for external device control)          │
│                       output.py                             │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
pupil-brightness-tracker/
├── pyproject.toml              # UV project configuration
├── README.md                   # This file
├── AGENTS.md                   # Instructions for AI agents
├── puredata/
│   ├── brightness_receiver.pd  # OSC-based brightness patch
│   ├── brightness_simple.pd    # FUDI-based brightness patch
│   └── color_music.pd          # Color-to-music patch (FUDI)
├── scripts/
│   ├── debug_connection.py     # Debug Pupil Capture connection
│   ├── test_puredata.py        # Test Pure Data communication
│   ├── test_color_grid.py      # Play through color-brightness grid
│   └── generate_test_image.py  # Generate calibration images
├── test_images/                # Generated calibration images
│   ├── color_test_grid.png     # 7x5 color-brightness grid
│   └── rainbow_strip.png       # Simple color strip
└── src/pupil_tracker/
    ├── __init__.py
    ├── client.py               # ZMQ client for Pupil Capture
    ├── processor.py            # Frame/gaze processing
    ├── analyzer.py             # Brightness and color analysis
    ├── output.py               # Output sink interfaces
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
