# Pupil Brightness Tracker

Real-time eye tracking brightness analyzer using Pupil Core. Streams gaze and video data from Pupil Capture, analyzes brightness at the gaze point, and outputs signals through an extensible interface.

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
# Run with video display
uv run pupil-tracker

# Run without video (console only)
uv run pupil-tracker --no-video

# Run with verbose output
uv run pupil-tracker --verbose

# Log data to file
uv run pupil-tracker -o brightness_data.csv
```

## CLI Options

```
usage: pupil-tracker [-h] [--host HOST] [--port PORT] [--region-size REGION_SIZE]
                     [--smoothing SMOOTHING] [--output OUTPUT] [--no-video] [--verbose]

Options:
  --host HOST              Pupil Capture host address (default: 127.0.0.1)
  --port PORT              Pupil Capture control port (default: 50020)
  --region-size SIZE       Size of gaze region to analyze in pixels (default: 50)
  --smoothing FRAMES       Number of frames for brightness smoothing (default: 5)
  --output, -o FILE        Output file for data (JSONL or CSV based on extension)
  --no-video               Disable video display
  --verbose, -v            Enable verbose console output
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
├── scripts/
│   └── debug_connection.py     # Debug utility for testing connection
└── src/pupil_tracker/
    ├── __init__.py
    ├── client.py               # ZMQ client for Pupil Capture
    ├── processor.py            # Frame/gaze processing
    ├── analyzer.py             # Brightness analysis
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
