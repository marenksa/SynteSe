# Building an Instrument Prototype

This project is built around two separate layers:

- **Detectors** — extract signals from the eye tracker and camera. They run the same way for every prototype.
- **Patches** — map those signals to sound. Each prototype is its own patch: a small Python file that decides what to send to Pure Data, and a PD patch that does the synthesis.

Making a new instrument means writing a new patch. You don't touch the detection code.

## Create a prototype

```bash
uv run new-prototype MyPatch_v1
```

This creates:
- `prototypes/MyPatch_v1/patch.py` — your Python signal mapping
- `prototypes/MyPatch_v1/MyPatch_v1.pd` — your Pure Data synthesis patch (pre-wired with `netreceive`)

And registers it so `--patch MyPatch_v1` works immediately.

## The Python side: mapping signals to sound

Edit `patch.py`. The `update` method is called every loop iteration. Read from `signals`, send messages to Pure Data via `outputs.send`:

```python
def update(self, signals: SignalBus, outputs: OutputBus) -> None:
    outputs.send("reverb", signals.eye.velocity / 2000.0)
```

`outputs.send("reverb", 0.7)` sends the FUDI message `reverb 0.7;` to Pure Data.

### Available signals

**Gaze position** — where you're looking:

| Signal | Type | Range | Description |
|--------|------|-------|-------------|
| `signals.eye.norm_pos` | `(float, float)` | 0–1 | Normalised gaze (x, y). Pupil convention: 0,0 = bottom-left |
| `signals.eye.velocity` | `float` | px/s | How fast the gaze is moving |
| `signals.eye.confidence` | `float` | 0–1 | Tracker confidence — filter below 0.5 if you need clean data |

**Blinks and gestures:**

| Signal | Type | Description |
|--------|------|-------------|
| `signals.eye.is_eyes_closed` | `bool` | True while eyes are closed |
| `signals.eye.eyes_closed_elapsed_ms` | `float` | ms since eyes closed; resets each blink |
| `signals.eye.blink` | `BlinkSample \| None` | Non-None for one iteration when a blink completes |
| `signals.eye.is_flutter_active` | `bool` | True during a rapid blink burst (3+ blinks in 1.5s) |
| `signals.eye.flutter_blink_count` | `int` | Blinks accumulated in current burst |
| `signals.eye.flutter` | `FlutterEvent \| None` | Non-None for one iteration when a flutter burst ends |

**Colour and environment** — only populated when `signals.has_env_reading` is True:

| Signal | Type | Range | Description |
|--------|------|-------|-------------|
| `signals.env.hue` | `float` | 0–179 | OpenCV hue at gaze point (smoothed) |
| `signals.env.raw_hue` | `float \| None` | 0–179 | Instantaneous hue; None if colour is too grey |
| `signals.env.saturation` | `float` | 0–255 | Colour saturation |
| `signals.env.brightness` | `float` | 0–255 | Brightness (smoothed) |
| `signals.env.scene_change` | `float` | 0–1 | Full-frame change magnitude — spikes on head movement |

**Head-gaze state** (`signals.head_gaze_state`) — the relationship between eye movement and head movement:

| State | Gaze | Scene change |
|-------|------|-------------|
| `Rest` | stable | low — nothing moving |
| `SmoothPan` | stable | high — head panning, gaze rides the camera |
| `Scanning` | moving | low — eyes moving, head still |
| `RagLock` | moving | high — both moving |

**Optional imports** — most signals are plain numbers and booleans, but two require an import to be useful:

```python
from base.signals.eye_blinks import BlinkType        # to compare blink.blink_type
from base.signals.head_gaze_state import HeadGazeState  # to compare signals.head_gaze_state
```

`blink.blink_type` values: `BlinkType.BLINK` (≤400ms), `BlinkType.INTENTIONAL` (≥500ms), `BlinkType.AMBIGUOUS`.

### Example

```python
from base.signals.bus import OutputBus, SignalBus
from base.signals.eye_blinks import BlinkType
from base.signals.head_gaze_state import HeadGazeState


class SceneMotionPatch:
    def update(self, signals: SignalBus, outputs: OutputBus) -> None:
        # Gaze speed drives reverb amount
        outputs.send("reverb", signals.eye.velocity / 2000.0)

        # Smooth camera pan triggers a percussion hit
        if signals.head_gaze_state is HeadGazeState.SmoothPan:
            outputs.send("hit", 1)

        # Intentional blink (held ≥500ms) freezes playback
        if signals.eye.blink is not None:
            if signals.eye.blink.blink_type == BlinkType.INTENTIONAL:
                outputs.send("freeze", 1)

    def reset(self) -> None:
        pass

    def shutdown(self, outputs: OutputBus) -> None:
        outputs.send("reverb", 0)
```

### Overlay

Add a class-level `overlay` attribute to control what the live preview shows:

```python
from base.output.overlay import OverlayConfig

class MyPatch:
    overlay = OverlayConfig(
        show_gaze_crosshair=True,   # on by default
        show_color_info=True,       # hue/saturation readout
        show_brightness_bar=True,   # brightness meter
        show_blink_flutter=True,    # blink/flutter indicators
    )
```

## The Pure Data side

Open `prototypes/MyPatch_v1/MyPatch_v1.pd`. It starts with:

- `[netreceive 9001]` — receives FUDI messages from Python over TCP
- `[route my_message]` — routes messages by name to outlets

Edit the `[route]` arguments to match the message names you send from Python. Each outlet corresponds to one message name, in order.

`outputs.send("reverb", 0.7)` arrives as `reverb 0.7;` → the `reverb` outlet of your `[route reverb hit freeze]` object.

Build your synthesis below and connect audio to `[dac~]`.

## Running your prototype

```bash
# Live (Pupil Core hardware required):
uv run pupil-tracker --patch MyPatch_v1

# From a recording:
uv run pupil-player recordings/000 --patch MyPatch_v1
```

## Naming convention

Patches are named `{Control}{Target}{Source}_v{N}`, where each token describes what the patch does. For example, "trigger note by color" → `TNC_v1`.

| Control | | Target | | Source | |
|:---|:---|:---|:---|:---|:---|
| Trigger | `T` | Note | `N` | Color | `C` |
| Toggle | `Tg` | Pitch | `P` | Brightness | `Br` |
| Stream | `S` | Sequence | `Sq` | Coordinates | `X` |
| | | Effect | `E` | Velocity | `V` |
| | | | | Confidence | `Cf` |
| | | | | Blink | `B` |
| | | | | Flutter | `F` |

## Existing patches as reference

| Patch | What it does | Example recording |
|-------|-------------|-------------------|
| `TNC_v1` | Hue at gaze point → MIDI note, brightness → octave, flutter → effect | `recordings/001` |
| `SPX_v3` | Raw gaze coordinates + velocity → Pure Data (simplest patch to read) | `recordings/002` |
| `TgSqC_v1` | Hue → colour ID, stability → PD sequence toggle | `recordings/003` |
| `SCfBF_v2` | Streams confidence + blink/flutter state booleans | `recordings/004` |
| `RAVE_v1` | Maps everything to RAVE latent dimensions for `nn~` | — |
