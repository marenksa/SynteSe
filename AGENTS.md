# AI Agent Instructions

This document provides guidance for AI agents working in this repository.

## Project Overview

**Goal**: Build a real-time eye tracking system that:
1. Streams gaze and video data from Pupil Core hardware
2. Analyzes what the user is looking at (brightness and color detection)
3. Outputs signals to Pure Data for sound synthesis

**Current State**:
- Brightness tracker working (brightness → pitch)
- Color-to-music mode working (color → note, brightness → octave)

**Future Direction**: Add ML-based object/material classification for more sophisticated gaze-based audio feedback.

## Architecture

```
Pupil Capture (external app) 
    → ZMQ/MessagePack protocol 
    → Our Python client 
    → Processing pipeline 
    → Output sinks (console/file/USB)
```

### Key Components

| File | Purpose |
|------|---------|
| `client.py` | ZMQ connection to Pupil Capture, message parsing |
| `processor.py` | Gaze-to-pixel mapping, region extraction |
| `analyzer.py` | Brightness and color analysis (BrightnessAnalyzer, ColorAnalyzer) |
| `output.py` | Output sink protocol and implementations (console, file, Pure Data) |
| `main.py` | CLI entry point, main loop |

### Pure Data Patches

| File | Purpose |
|------|---------|
| `brightness_receiver.pd` | OSC-based brightness → pitch synthesis |
| `brightness_simple.pd` | FUDI-based brightness → pitch synthesis |
| `color_music.pd` | FUDI-based color → MIDI note synthesis |

## Critical Technical Knowledge

### Color-to-Music Mapping

The `ColorAnalyzer` maps visible light wavelength to musical notes:

```
Color     Wavelength    OpenCV Hue    Note    Semitone
─────────────────────────────────────────────────────
Red       ~700nm        0-10, 160+    C       0
Orange    ~620nm        10-25         D       2
Yellow    ~580nm        25-40         E       4
Green     ~530nm        40-80         F       5
Cyan      ~500nm        80-100        G       7
Blue      ~470nm        100-130       A       9
Violet    ~400nm        130-160       B       11
```

Brightness (HSV Value channel) maps to octave 2-6:
- 0-51 → Octave 2
- 51-102 → Octave 3
- 102-153 → Octave 4
- 153-204 → Octave 5
- 204-255 → Octave 6

MIDI note calculation: `midi = (octave + 1) * 12 + semitone`

### Note/Octave Stability

To prevent jarring jumps, note and octave detection use separate stability tracking:

| Parameter | Note | Octave |
|-----------|------|--------|
| Frames | 8 | 15 |
| Threshold | 70% | 80% |
| Min agreement | 6/8 | 12/15 |

- **Note stability**: Faster response (8 frames, 70% threshold)
- **Octave stability**: Slower response (15 frames, 80% threshold) to avoid jarring octave jumps
- **Saturation threshold**: Pixels with saturation < 50 are excluded from hue calculation (gray pixels have unreliable hue)

Configurable via CLI: `--note-stability`, `--octave-stability`, `--octave-threshold`

### ZMQ Real-Time Streaming

**IMPORTANT**: ZMQ SUB sockets buffer messages by default, causing massive lag. We MUST:

```python
# Set minimal buffer
subscriber.setsockopt(zmq.RCVHWM, 1)
subscriber.setsockopt(zmq.LINGER, 0)

# Drain to get LATEST message, not oldest
while True:
    try:
        parts = subscriber.recv_multipart(flags=zmq.NOBLOCK)
        latest = parts  # Keep draining
    except zmq.Again:
        break  # No more messages, use 'latest'
```

### Pupil Capture Network API

- **Control port**: TCP 50020 (REQ/REP pattern)
- **Data port**: Dynamic, obtained via `SUB_PORT` command
- **Protocol**: ZeroMQ + MessagePack
- **Topics**: `gaze.3d.*`, `frame.world`, `pupil.*`

### Gaze Data Format

```python
{
    "topic": "gaze.3d.0.",
    "norm_pos": [0.5, 0.5],  # Normalized (0-1), origin at BOTTOM-LEFT
    "confidence": 0.85,       # 0-1, filter < 0.5
    "timestamp": 1234567.89,
}
```

**Coordinate flip**: Pupil uses bottom-left origin, OpenCV uses top-left. Flip Y:
```python
pixel_y = int((1.0 - norm_y) * height)
```

### Frame Data Format

```python
{
    "topic": "frame.world",
    "width": 320,
    "height": 240,
    "format": "jpeg",  # Usually JPEG, not raw BGR
    # Raw bytes in third message part
}
```

**JPEG decoding required**:
```python
image_data = np.frombuffer(frame_bytes, dtype=np.uint8)
frame = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
```

## Development Guidelines

### Python Standards

- **Package manager**: UV only (not pip, poetry, conda)
- **Python version**: 3.12+
- **Type hints**: Required on all functions
- **Dataclasses**: Use `@dataclass(frozen=True)` for data objects
- **Protocols**: Use `typing.Protocol` for interfaces (duck typing)

### Code Style

```python
# Good: Type hints, docstrings, frozen dataclasses
@dataclass(frozen=True)
class GazeData:
    """Gaze data from Pupil Capture."""
    timestamp: float
    norm_pos: tuple[float, float]
    confidence: float

def process(data: GazeData) -> Result:
    """Process gaze data and return result."""
    ...
```

### Testing Commands

```bash
# Run tracker with video
uv run pupil-tracker

# Run without video (headless)
uv run pupil-tracker --no-video

# Debug connection
uv run python scripts/debug_connection.py

# Use gtimeout to prevent hanging
gtimeout 30 uv run pupil-tracker --no-video
```

### Debugging Tips

1. **Always use timeouts** on ZMQ operations
2. **Use `gtimeout`** when running commands that might hang
3. **Flush print output**: `print(..., flush=True)`
4. **Check Pupil Capture**:
   - Is it running?
   - Is Frame Publisher enabled?
   - Are eyes being detected? (gaze confidence > 0)

## External Documentation

### Pupil Core (NOT Neon!)

- **Network API**: https://docs.pupil-labs.com/core/developer/network-api/
- **Data Format**: https://docs.pupil-labs.com/core/terminology/
- **Frame Publisher**: https://docs.pupil-labs.com/core/software/pupil-capture/

### Libraries

- **ZeroMQ**: https://zeromq.org/
- **MessagePack**: https://msgpack.org/
- **OpenCV Python**: https://docs.opencv.org/4.x/

## Future Work

### Phase 1: Sound Refinement
- Add more sophisticated synthesis in Pure Data (filters, ADSR envelopes)
- Explore different scales/modes beyond major scale
- Add saturation-based effects (more saturated = more intense sound)

### Phase 2: Object Detection
- Integrate YOLOv8 or similar for object detection
- Intersect detected objects with gaze position
- Map objects to sound characteristics

### Phase 3: Custom Classification
- Train custom model for specific materials/surfaces
- Collect labeled training data from gaze sessions
- Material-based sound textures

## Common Pitfalls

1. **ZMQ buffering** - Always set `RCVHWM=1` and drain buffers
2. **Coordinate systems** - Pupil uses bottom-left origin, flip Y for OpenCV
3. **JPEG frames** - World camera sends JPEG, not raw BGR
4. **High-frequency gaze** - Don't process every gaze message, sync to frame rate
5. **Confidence filtering** - Always filter gaze with confidence < 0.5
6. **Out-of-bounds gaze** - Some gaze values are outside 0-1 range, filter them

## Quick Reference

```bash
# Install
uv sync

# Run (brightness mode)
uv run pupil-tracker

# Run with brightness → Pure Data
uv run pupil-tracker --pd-fudi

# Run color-to-music mode
uv run pupil-tracker --pd-color-fudi

# Run with options
uv run pupil-tracker --region-size 100 --smoothing 10 -o data.csv

# Test Pure Data connection (without Pupil hardware)
uv run python scripts/test_puredata.py --fudi
uv run python scripts/test_puredata.py --color-fudi

# Test color grid (plays all notes systematically)
uv run python scripts/test_color_grid.py --fudi --auto

# Generate calibration test images
uv run python scripts/generate_test_image.py

# Debug Pupil connection
uv run python scripts/debug_connection.py
```

## Test Scripts

| Script | Purpose |
|--------|---------|
| `test_puredata.py` | Test Pure Data connection with sine wave patterns |
| `test_color_grid.py` | Play through entire color-brightness grid (rows then columns) |
| `generate_test_image.py` | Generate calibration images (7×5 grid + rainbow strip) |
| `debug_connection.py` | Debug Pupil Capture ZMQ connection |

