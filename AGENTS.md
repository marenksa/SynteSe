# AI Agent Instructions

This document provides guidance for AI agents working in this repository.

## Project Overview

**Goal**: Build a real-time eye tracking color-to-music system that:
1. Streams gaze and video data from Pupil Core hardware
2. Analyzes color at the gaze point (hue → note, brightness → octave)
3. Triggers notes on fixation events (when gaze dwells on a point)
4. Outputs to Pure Data for sound synthesis

**Current State**: Content-based note triggering via `NoteGate`. Notes trigger when the detected MIDI note stabilizes after a real transition — no dependency on gaze velocity or Pupil fixation events.

**Future Direction**: Explore spatial mapping, pentatonic scales, or other approaches for more predictable/playable melodies.

## Architecture

```
Pupil Capture (external app)
    → ZMQ/MessagePack protocol
    → Our Python client (gaze, frames)
    → Color analysis at gaze point
    → NoteGate: trigger when MIDI note stabilizes after transition
    → PureDataSink → Pure Data (ADSR envelope synthesis)
```

### Key Components

| File | Purpose |
|------|---------|
| `client.py` | ZMQ connection to Pupil Capture, parses gaze/frame/fixation data |
| `processor.py` | Gaze-to-pixel mapping, region extraction |
| `analyzer.py` | Color analysis (ColorAnalyzer), NoteGate, NoteEvent |
| `output.py` | Output sinks (ColorConsoleSink, PureDataSink) |
| `recording.py` | Loads Pupil recordings (gaze, fixations, video) |
| `player.py` | Plays back recordings with gaze overlay and note triggering |
| `main.py` | CLI entry point for live tracking |

### Data Flow for Note Triggering

```
Each frame:
    → Extract gaze region from frame
    → ColorAnalyzer.analyze() → ColorReading (midi_note, note, octave)
    → NoteGate.update(midi_note) → True if note just stabilized
    → If triggered: create NoteEvent, send to PureDataSink
    → PureDataSink sends "note_on <midi> <brightness>" to Pd
```

### Pure Data Patch

| File | Purpose |
|------|---------|
| `color_music.pd` | Receives `note_on` messages, ADSR envelope synthesis |

Message format: `note_on <midi_note> <brightness>`
- `midi_note`: 36-84 (C2-B6)
- `brightness`: 0.0-1.0 (can be used for velocity)

## Critical Technical Knowledge

### Content-Based Note Triggering (NoteGate)

`NoteGate` in `analyzer.py` replaces earlier velocity-gated and fixation-based approaches. It tracks the **MIDI note stream** directly instead of gaze position or velocity.

**Why**: With a head-mounted eye tracker, gaze pixel position doesn't represent world position — the scene moves, not the gaze. Velocity-based gating fails because eyes compensate for head movement. NoteGate works by detecting when the *content* changes and stabilizes.

**Parameters:**
- `stable_frames` (default 4) — consecutive identical frames needed to trigger
- `min_transition_frames` (default 3) — frames of different content needed before re-triggering the same note

**Behavior:**
- Fires on first stable note
- Fires when note changes and stabilizes (saccade to new color)
- Fires when same note returns after a real transition (head turn away and back)
- Does NOT fire on brief 1-2 frame jitter (text on book covers)
- Does NOT re-fire while resting on the same content

### Color-to-Music Mapping (7-Note Major Scale)

```
Color     Wavelength    OpenCV Hue    Note    Semitone
─────────────────────────────────────────────────────
Red       ~700nm        0-8, 165+     C       0
Orange    ~620nm        8-25          D       2
Yellow    ~580nm        25-38         E       4
Green     ~530nm        38-75         F       5
Cyan      ~500nm        75-95         G       7
Blue      ~470nm        95-125        A       9
Violet    ~400nm        125-165       B       11
```

Brightness (HSV Value channel) maps to octave 2-6.

MIDI note calculation: `midi = (octave + 1) * 12 + semitone`

### Note/Octave Stability

Parameters are tuned for responsive detection (reduced from earlier sluggish values):

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `smoothing_window` | 3 | Frames to average for color smoothing |
| `note_stability_frames` | 2 | Frames before note can change |
| `octave_stability_frames` | 3 | Frames before octave can change |
| `note_stability_threshold` | 0.5 | Agreement ratio for note change |
| `octave_stability_threshold` | 0.5 | Agreement ratio for octave change |
| `min_saturation` | 20 | Minimum saturation for hue detection |

**Saturation threshold**: Lowered to 20 (from 50) to detect pastel colors (light pink, pale blue). Below 20 is typically gray/white where hue is meaningless.

### Binocular Gaze Handling

For recordings with binocular data (separate gaze.3d.0 and gaze.3d.1 topics):
- Prefers combined gaze (gaze.3d.01.) if available
- Otherwise selects the eye with **highest average confidence**
- Prevents flickering from interleaved left/right eye data

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
- **Topics**: `gaze.3d.*`, `frame.world`, `fixations`

### Gaze Data Format

```python
{
    "topic": "gaze.3d.01.",  # Combined binocular preferred
    "norm_pos": [0.5, 0.5],  # Normalized (0-1), origin at BOTTOM-LEFT
    "confidence": 0.85,       # 0-1, filter < 0.5
    "timestamp": 1234567.89,
}
```

**Coordinate flip**: Pupil uses bottom-left origin, OpenCV uses top-left. Flip Y:
```python
pixel_y = int((1.0 - norm_y) * height)
```

### Fixation Data Format

```python
{
    "topic": "fixations",
    "id": 42,
    "timestamp": 1234567.89,
    "duration": 150.0,        # ms
    "norm_pos": [0.5, 0.5],
    "dispersion": 1.2,        # degrees
    "confidence": 0.9,
}
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

## Session Transcripts

This is a research project. All agent sessions should be logged for later review.

**Transcript location:** `ai-transcripts/` in the repo root.

**Do not read old transcripts** unless the user asks you to or you need the format as a reference. They are for human review, not agent context.

When the user says **"make a transcript"**, write a summary of the current session to `ai-transcripts/YYYY-MM-DD-<topic>.md` following the format of existing transcripts in that folder.

**Format:**
- Header: date, model, tool, topic
- Each user prompt reproduced verbatim under `### User`
- Each assistant response condensed under `### Assistant`: summarize actions taken, key findings, and decisions — don't reproduce full tool output
- End with a `## Summary of Changes` section listing files modified, key design decisions, and commit hashes

## Critical Rule: Tracker and Player Parity

**`main.py` (live tracker) and `player.py` (recording player) must always have identical functionality.**

Any feature, message, or behaviour added to one must be added to the other in the same session. This includes:
- New Pd messages sent (e.g. `confidence`, `am_lfo`, `am_amp`)
- Gating logic (e.g. only send during blink/flutter)
- Cleanup/reset on exit
- State variables related to Pd output

Do not close a task or end a session with one file updated but not the other.

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
class NoteEvent:
    """A discrete note trigger event from content-based gaze settling."""
    timestamp: float
    note: Note
    octave: int
    midi_note: int
    brightness: float
    center_x: int
    center_y: int

def process(data: GazeData) -> Result:
    """Process gaze data and return result."""
    ...
```

### Testing Commands

```bash
# Run live tracker with video
uv run pupil-tracker

# Run with Pure Data output
uv run pupil-tracker --pd

# Run without video (headless)
uv run pupil-tracker --no-video --pd

# Play back a recording
uv run gaze-player /path/to/recording

# Play recording with Pure Data output
uv run gaze-player /path/to/recording --pd

# Play with gamma correction (for dark footage)
uv run gaze-player /path/to/recording --gamma 0.5

# Debug connection
uv run python scripts/debug_connection.py
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

### Note Mapping Alternatives

The current 7-note color mapping has practical challenges (lighting variations, mixed colors). Consider:

1. **Spatial mapping** - Screen regions = notes (like a visual piano)
2. **Pentatonic scale** - 5 notes (C,D,E,G,A) with bold colors, always sounds good
3. **Color markers** - Dedicated colored targets for reliable detection
4. **Hybrid** - Spatial for pitch, color for expression/timbre

### Sound Refinement
- Explore different scales/modes
- Add saturation-based effects (more saturated = more intense sound)
- Map fixation duration to note length or dynamics

### Object Detection
- Integrate YOLOv8 or similar for object detection
- Intersect detected objects with gaze position
- Map objects to sound characteristics

## Common Pitfalls

1. **ZMQ buffering** - Always set `RCVHWM=1` and drain buffers
2. **Coordinate systems** - Pupil uses bottom-left origin, flip Y for OpenCV
3. **JPEG frames** - World camera sends JPEG, not raw BGR
4. **Binocular flickering** - Filter to combined gaze or best-confidence eye
5. **Pastel colors** - Use low saturation threshold (20) to detect them
7. **Confidence filtering** - Always filter gaze with confidence < 0.5
8. **Out-of-bounds gaze** - Some gaze values are outside 0-1 range, filter them

## Quick Reference

```bash
# Install
uv sync

# Run live tracker
uv run pupil-tracker

# Run with Pure Data output
uv run pupil-tracker --pd

# Run with custom stability settings
uv run pupil-tracker --pd --note-stability 2 --octave-stability 3

# Play recording
uv run gaze-player /path/to/recording --pd

# Test Pure Data connection
uv run python scripts/test_puredata.py

# Debug Pupil connection
uv run python scripts/debug_connection.py
```

## Key Files for Note Triggering

If working on note triggering logic:

| File | What's There |
|------|--------------|
| `analyzer.py` | NoteGate class, NoteEvent dataclass, ColorAnalyzer |
| `main.py` | Live tracking loop with NoteGate integration |
| `player.py` | Recording playback with NoteGate integration |
| `output.py` | PureDataSink class, sends `note_on` to Pure Data |
