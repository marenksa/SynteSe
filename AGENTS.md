# AI Agent Instructions

This document provides guidance for AI agents working in this repository.

## Project Overview

**Goal**: Build a real-time eye tracking system that:
1. Streams gaze and video data from Pupil Core hardware
2. Analyzes what the user is looking at (currently: brightness detection)
3. Outputs signals to external devices based on gaze analysis

**Current State**: MVP brightness tracker is working. The system detects brightness at the user's gaze point in real-time.

**Future Direction**: Replace brightness analysis with ML-based object/material classification, then stream classification signals to external devices (e.g., via USB).

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
| `analyzer.py` | Brightness calculation (replace with ML later) |
| `output.py` | Output sink protocol and implementations |
| `main.py` | CLI entry point, main loop |

## Critical Technical Knowledge

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

### Phase 1: Object Detection (Next)
- Integrate YOLOv8 or similar for object detection
- Intersect detected objects with gaze position
- Classify what user is looking at

### Phase 2: Custom Classification
- Train custom model for specific materials/surfaces
- Collect labeled training data from gaze sessions

### Phase 3: USB Output
- Implement `USBSink` in `output.py`
- Serial protocol for external device communication
- Real-time signal streaming based on classification

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

# Run
uv run pupil-tracker

# Run with options
uv run pupil-tracker --region-size 100 --smoothing 10 -o data.csv

# Debug
uv run python scripts/debug_connection.py
```

