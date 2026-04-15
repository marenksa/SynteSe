# AI Agent Instructions

This document provides guidance for AI agents working in this repository.

## Project Overview

**Goal**: Build a modular eye tracking system for creating audiovisual prototypes with Pupil Core. The system detects signals from the gaze and environment, and routes them to sound synthesis in Pure Data via swappable patches.

**Design philosophy**: Detectors and creative mappings are strictly separated. Detectors run the same way regardless of the prototype. Each prototype is a self-contained patch file. See `patches/__init__.py` for the `Patch` protocol.

## Architecture

```
Pupil Capture (external app)
    → ZMQ/MessagePack
    → input/live.py        — raw data (gaze, frames, blinks, fixations)
    → tracker.py           — live entry point
    → player.py            — recording entry point
    → signals/pipeline.py  — owns detectors, populates SignalBus each iteration

SignalBus (signals/bus.py)
    eye.*   — confidence, position, velocity, blinks, flutter, fixation_id
    env.*   — hue, brightness, note, octave, region_changed, scene_change
    → patch.update(signals, outputs)

OutputBus (signals/bus.py)
    → outputs.send("key", value)  — generic FUDI to PureDataSink
    → Pure Data (audio synthesis)
```

### Key Files

| File | Purpose |
|------|---------|
| `input/live.py` | ZMQ connection to Pupil Capture, parses gaze/frame/blink/fixation |
| `input/recording.py` | Loads Pupil recordings (gaze, fixations, blinks, flutter, video) |
| `signals/pipeline.py` | Pipeline: owns all detectors, populates SignalBus each iteration |
| `signals/env_color.py` | Colour analysis (ColorAnalyzer, FrameProcessor), NoteGate, NoteEvent |
| `signals/env_scene_change.py` | Full-frame change magnitude |
| `signals/eye_blinks.py` | Streaming blink/flutter detection, BlinkType, constants |
| `signals/eye_gaze.py` | Gaze speed in px/s |
| `signals/bus.py` | SignalBus, EyeSignals, EnvSignals, OutputBus |
| `output/sinks.py` | PureDataSink (FUDI/TCP), ColorConsoleSink, MultiSink |
| `output/overlay.py` | Stateless drawing functions (shared by tracker.py and player.py) |
| `tracker.py` | Live entry point: gamma → pipeline.process_live() → overlay → patch |
| `player.py` | Recording entry point: pipeline.process_recording_frame() → patch |
| `patches/__init__.py` | Patch protocol + load_patch() factory |
| `patches/color_music/` | Prototype: hue→note, brightness→octave, flutter→AM-LFO |

### Signal Bus Fields

**EyeSignals** (updated every iteration):
- `confidence: float` — gaze tracking confidence 0–1
- `norm_pos: (float, float)` — normalised gaze (0–1), Pupil convention
- `px_pos: (int, int)` — pixel coordinates in world frame
- `velocity_px_s: float` — gaze speed in pixels/second
- `is_eyes_closed: bool` — between blink onset and offset
- `blink: BlinkSample | None` — non-None for 1 iteration when blink completes
- `is_flutter_active: bool` — during rapid blink burst
- `flutter_blink_count: int` — blinks accumulated in active burst
- `flutter: FlutterEvent | None` — non-None for 1 iteration when flutter ends
- `fixation_id: int | None` — non-None for 1 iteration on new fixation

**EnvSignals** (updated when gaze region is analysed, i.e. `signals.has_env_reading == True`):
- `hue: float` — OpenCV hue 0–179 (temporally smoothed)
- `raw_hue: float | None` — instantaneous hue (pre-smoothing); None if saturation too low
- `hue_normalized: float` — 0–1
- `saturation: float` — 0–255
- `brightness: float` — 0–255 (smoothed)
- `brightness_normalized: float` — 0–1
- `scene_change: float` — full-frame change magnitude 0–1

### Data Flow Per Loop Iteration

Both entry points call `Pipeline.process_live()` or `Pipeline.process_recording_frame()`, which does:

```
signals.clear_events()

if gaze:            → signals.eye.confidence, norm_pos
if fixation:        → signals.eye.fixation_id
if blink:           → signals.eye.blink, is_eyes_closed, is_flutter_active
blink_tracker.tick: → signals.eye.flutter (on burst end)

if frame:
    scene_detector  → signals.env.scene_change
    gaze region     → signals.env.hue, raw_hue, saturation, brightness
    gaze velocity   → signals.eye.velocity_px_s, px_pos
    signals.has_env_reading = True

patch.update(signals, outputs)   ← called by tracker.py / player.py after pipeline
```

### Writing a Patch

```python
class MyPatch:
    def update(self, signals: SignalBus, outputs: OutputBus) -> None:
        if not signals.has_env_reading:
            return  # skip if no fresh frame data
        outputs.send("key", value)   # sends "key value;" to Pure Data

    def reset(self) -> None:
        pass  # called on seek/restart
```

Register in `patches/__init__.py` under `load_patch()`. Create `patches/<name>/` as a package with `__init__.py` containing the patch class and a `README.md` documenting the mappings.

## Critical Technical Knowledge

### NoteMapper and NoteGate (color_music patch)

These live entirely in `patches/color_music/` — they are prototype-specific.

**NoteMapper** converts `signals.env.hue`, `signals.env.raw_hue`, and `signals.env.brightness` into a `NoteReading(note, octave, midi_note, raw_midi_note)`. Owns stability windows for note and octave to suppress per-frame jitter.

**NoteGate** fires when the MIDI note stabilises after a real gaze transition. Uses `raw_midi_note` (pre-stability) for transition streak counting.

**Why not gaze velocity or fixation events**: With a head-mounted tracker, gaze pixel position doesn't represent world position — the scene moves, not the gaze. NoteGate detects when *content* changes and stabilises instead.

**NoteGate parameters:**
- `stable_frames` (default 4) — consecutive identical frames to trigger
- `min_transition_frames` (default 3) — frames of different content before re-triggering same note

**Fires when**: new stable note, same note after real transition, new fixation ID
**Does not fire**: brief 1–2 frame jitter, resting on same content, during flutter

### Blink Detection

Uses Pupil Capture's built-in blink detector (onset/offset events), not confidence-based gating.

- `BlinkType.BLINK` — duration < 400ms
- `BlinkType.INTENTIONAL` — duration ≥ 500ms
- `BlinkType.AMBIGUOUS` — 400–500ms

**Flutter**: 4+ blinks in a 1.5s sliding window. Ends after 0.5s with no new blink.

Real-time uses `StreamingBlinkTracker` (`signals/eye_blinks.py`). Recording playback reads pre-classified blink/flutter data from the recording file; `Pipeline.process_recording_frame()` handles flutter transition detection internally.

### Colour-to-Note Mapping (color_music patch)

Defined in `patches/color_music/` (`HUE_RANGES`, `NOTE_SEMITONES`).

```
Red     ~700nm  OpenCV 0–8, 165+   C   semitone 0
Orange  ~620nm  8–25               D   semitone 2
Yellow  ~580nm  25–38              E   semitone 4
Green   ~530nm  38–75              F   semitone 5
Cyan    ~500nm  75–95              G   semitone 7
Blue    ~470nm  95–125             A   semitone 9
Violet  ~400nm  125–165            B   semitone 11
```

MIDI note: `(octave + 1) * 12 + semitone`
Saturation threshold: 20/255 (below this, `raw_hue` is None — treat as grey).

### ZMQ Real-Time Streaming

ZMQ SUB sockets buffer by default. Always:
```python
subscriber.setsockopt(zmq.RCVHWM, 1)
subscriber.setsockopt(zmq.LINGER, 0)
```
Drain to latest message before processing. See `input/live.py`.

### Pupil Capture API

- Control port: TCP 50020 (REQ/REP)
- Data port: dynamic, obtained via `SUB_PORT` command
- Protocol: ZeroMQ + MessagePack
- Topics: `gaze.*`, `frame.world`, `fixations`, `blinks`, `frame.eye.0`

### Coordinate System

Pupil uses (0,0) at bottom-left; OpenCV uses top-left. Flip Y:
```python
pixel_y = int((1.0 - norm_y) * height)
```

## Critical Rule: Tracker and Player Parity

**`tracker.py` and `player.py` must produce identical signal behaviour for the same input.**

Parity is now largely enforced by `Pipeline` — both entry points call the same detectors through the same class. Adding a new signal means adding it to `Pipeline`, not separately to both files.

The remaining parity obligation:
- Any new detector added to `Pipeline` must be reset in `Pipeline.reset()`
- `player.py`'s `_reset_on_seek()` calls `pipeline.reset()` — keep that call in place

## Python Standards

- **Package manager**: UV only (not pip, poetry, conda)
- **Python version**: 3.12+
- **Type hints**: Required on all functions
- **Dataclasses**: Use `@dataclass(frozen=True)` for data objects
- **Protocols**: Use `typing.Protocol` for interfaces

## Session Transcripts

All agent sessions should be logged for later review.

**Location:** `ai-transcripts/` in the repo root.

**Do not read old transcripts** unless the user asks or you need the format as reference.

When the user says **"make a transcript"**, write a summary to `ai-transcripts/YYYY-MM-DD-<topic>.md` following existing transcript format:
- Header: date, model, tool, topic
- Each user prompt verbatim under `### User`
- Each assistant response condensed under `### Assistant`
- End with `## Summary of Changes`: files modified, key decisions, commit hashes

## Testing Commands

```bash
# Live tracker
uv run pupil-tracker
uv run pupil-tracker --patch color_music

# Recording playback
uv run pupil-player recordings/000
uv run pupil-player recordings/000 --patch color_music --gamma 0.5

# Utility scripts
uv run python scripts/debug_connection.py
uv run python scripts/test_puredata.py
uv run python scripts/test_color_grid.py --auto
```

## Common Pitfalls

1. **ZMQ buffering** — Always set `RCVHWM=1` and drain buffers
2. **Coordinate systems** — Pupil uses bottom-left origin, flip Y for OpenCV
3. **JPEG frames** — World camera sends JPEG, not raw BGR
4. **Binocular flickering** — Filter to combined gaze or best-confidence eye
5. **Pastel colours** — Use low saturation threshold (20) to detect them
6. **Confidence filtering** — Filter gaze with confidence < 0.5
7. **Out-of-bounds gaze** — Some gaze values fall outside 0–1, filter them
8. **Patch reset on seek** — player.py's `_reset_on_seek()` must call `patch.reset()`

## External Documentation

### Pupil Core (NOT Neon)

- Network API: https://docs.pupil-labs.com/core/developer/network-api/
- Data format: https://docs.pupil-labs.com/core/terminology/

### Libraries

- ZeroMQ: https://zeromq.org/
- MessagePack: https://msgpack.org/
- OpenCV Python: https://docs.opencv.org/4.x/
