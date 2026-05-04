# AI Agent Instructions

This document provides guidance for AI agents working in this repository.

## Project Overview

**Goal**: Build a modular eye tracking system for creating audiovisual prototypes with Pupil Core. The system detects signals from the gaze and environment, and routes them to sound synthesis in Pure Data via swappable patches.

**Design philosophy**: Detectors and creative mappings are strictly separated. Detectors run the same way regardless of the prototype. Each prototype is a self-contained patch file. See `patches/base.py` for the `Patch` protocol.

## Architecture

```
Pupil Capture (external app)
    → ZMQ/MessagePack
    → input/live.py        — raw data (gaze, frames, blinks, fixations)
    → tracker.py           — live entry point
    → player.py            — recording entry point
    → signals/pipeline.py  — owns detectors, populates SignalBus each iteration

SignalBus (signals/bus.py)
    head_gaze_state   — Rest | SmoothPan | Scanning | RagLock
    eye.*             — confidence, norm_pos, velocity, blinks, flutter, fixation_id
    env.*             — hue, raw_hue, saturation, brightness, scene_change
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
| `signals/bus.py` | SignalBus, EyeSignals, EnvSignals, OutputBus |
| `signals/head_gaze_state.py` | HeadGazeState enum + HeadGazeClassifier |
| `signals/env_color.py` | Colour analysis: ColorAnalyzer, FrameProcessor |
| `signals/env_scene_change.py` | Full-frame change magnitude |
| `signals/eye_blinks.py` | Streaming blink/flutter detection, BlinkType, constants |
| `signals/eye_gaze.py` | Gaze speed in px/s |
| `output/sinks.py` | PureDataSink (FUDI/TCP), ColorConsoleSink, MultiSink |
| `output/overlay.py` | Stateless drawing functions (shared by tracker.py and player.py) |
| `tracker.py` | Live entry point: pipeline.process_live() → overlay → patch |
| `player.py` | Recording entry point: pipeline.process_recording_frame() → patch |
| `patches/base.py` | Patch protocol + load_patch() factory |
| `patches/TNC_v1/` | Trigger Note by Colour: hue→note, brightness→octave, flutter→effect |
| `patches/TgSqC_v1/` | Toggle Sequence by Colour: hue→color ID, stability→PD toggle |
| `patches/SCfBF/` | Stream Confidence from Blinks/Flutter: eye state → PD booleans |
| `patches/SPX/` | Stream Pitch/X-Y/Velocity: gaze coords + velocity → PD |
| `patches/RAVE_v1/` | Gaze/colour/velocity → RAVE latent dims for nn~ |

### Signal Bus Fields

**SignalBus** (top-level):
- `head_gaze_state: HeadGazeState` — `Rest | SmoothPan | Scanning | RagLock` (see below)
- `has_env_reading: bool` — True when env signals were freshly updated this iteration
- `timestamp: float` — current frame timestamp

**EyeSignals** (`signals.eye.*`, updated every iteration):
- `confidence: float` — gaze tracking confidence 0–1
- `norm_pos: (float, float)` — normalised gaze (0–1), Pupil convention
- `velocity: float` — gaze speed in pixels/second
- `is_eyes_closed: bool` — between blink onset and offset
- `eyes_closed_elapsed_ms: float` — ms since most recent onset; resets each blink
- `blink: BlinkSample | None` — non-None for 1 iteration when a blink completes
- `is_flutter_active: bool` — during rapid blink burst
- `flutter_blink_count: int` — blinks accumulated in active burst
- `flutter: FlutterEvent | None` — non-None for 1 iteration when flutter ends
- `fixation_id: int | None` — non-None for 1 iteration on new fixation

**EnvSignals** (`signals.env.*`, updated when `has_env_reading` is True):
- `hue: float` — OpenCV hue 0–179 (temporally smoothed)
- `raw_hue: float | None` — instantaneous hue (pre-smoothing); None if saturation too low
- `saturation: float` — 0–255
- `brightness: float` — 0–255 (smoothed)
- `scene_change: float` — full-frame change magnitude 0.0–1.0

**HeadGazeState** classifies eye vs. head movement each frame:

|  | scene_change low | scene_change high |
|--|--|--|
| **pos stable** | `Rest` — both still | `SmoothPan` — head moves, gaze rides camera |
| **pos changing** | `Scanning` — gaze moves, head still | `RagLock` — both moving |

### Data Flow Per Loop Iteration

Both entry points call `Pipeline.process_live()` or `Pipeline.process_recording_frame()`, which does:

```
signals.clear_events()

if gaze:            → signals.eye.confidence, norm_pos
if fixation:        → signals.eye.fixation_id
if blink:           → signals.eye.blink, is_eyes_closed, eyes_closed_elapsed_ms, is_flutter_active
blink_tracker.tick: → signals.eye.flutter (on burst end)

if frame:
    scene_detector  → signals.env.scene_change
    head_gaze       → signals.head_gaze_state
    gaze region     → signals.env.hue, raw_hue, saturation, brightness
                       signals.has_env_reading = True
    gaze velocity   → signals.eye.velocity

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

    def shutdown(self, outputs: OutputBus) -> None:
        pass  # send cleanup messages to PD before exit (e.g. silence notes)
```

Register the patch in `patches/base.py` under `load_patch()`. Create `patches/<name>/` as a package with `__init__.py` exporting the patch class, and a `README.md` documenting the mappings.

## Critical Technical Knowledge

### NoteMapper and NoteGate (TNC_v1 patch)

These live entirely in `patches/TNC_v1/` — they are prototype-specific.

**NoteMapper** converts `signals.env.hue`, `signals.env.raw_hue`, and `signals.env.brightness` into a `NoteReading(note, octave, midi_note, raw_midi_note)`. Owns the octave stability window only; note stability is handled downstream by NoteGate.

**NoteGate** fires when the MIDI note stabilises after a real gaze transition. Uses `raw_midi_note` (pre-smoothing) for transition streak counting.

**Why not gaze velocity or fixation events**: With a head-mounted tracker, gaze pixel position doesn't represent world position — the scene moves, not the gaze. NoteGate detects when *content* changes and stabilises instead.

**NoteGate parameters:**
- `stable_frames` (default 3) — consecutive identical frames to confirm note
- `min_transition_frames` (default 3) — frames of different raw note to arm a diff-streak re-trigger

**Fires when** (in priority order):
1. Stable note differs from last triggered (all colour transitions)
2. Raw note passed through different values for `min_transition_frames` frames (same-colour saccades via a chromatic path)

**Does not fire**: brief 1–2 frame jitter, resting on same content, during flutter, same-colour saccades with no chromatic path

### Blink Detection

Uses Pupil Capture's built-in blink detector (onset/offset events), not confidence-based gating.

- `BlinkType.BLINK` — duration < 400ms
- `BlinkType.INTENTIONAL` — duration ≥ 500ms
- `BlinkType.AMBIGUOUS` — 400–500ms

**Flutter**: 3+ blinks in a 1.5s sliding window. Ends after 0.3s with no new blink.

Real-time uses `StreamingBlinkTracker` (`signals/eye_blinks.py`). Recording playback reads pre-classified blink/flutter data from the recording file; `Pipeline.process_recording_frame()` handles flutter transition detection internally.

### Colour-to-Note Mapping (TNC_v1 patch)

Defined in `patches/TNC_v1/mapping.py` (`HUE_RANGES`, `NOTE_SEMITONES`).

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

**`ai-transcripts/` is gitignored. Never commit transcript files** — not directly, not force-added, not in any form. They are local records only.

**Do not read old transcripts** unless the user asks or you need the format as reference.

When the user says **"make a transcript"**, write a summary to `ai-transcripts/YYYY-MM-DD-<topic>.md` following the format in `ai-transcripts/TEMPLATE.md`:
- Header: date, model, tool, topic, commit(s)
- Context paragraph explaining the why
- Each user prompt condensed under `### User`
- Each assistant response condensed under `### Assistant`
- End with `## Summary of Changes`: design rationale, files table, key decisions, commit hashes

## Testing Commands

```bash
# Live tracker
uv run pupil-tracker
uv run pupil-tracker --patch TNC_v1

# Recording playback
uv run pupil-player recordings/000
uv run pupil-player recordings/000 --patch TNC_v1

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
