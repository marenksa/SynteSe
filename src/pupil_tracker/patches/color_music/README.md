# Patch: color_music

Maps visible light wavelength to musical pitch and gaze dynamics to sound modulation. The core idea: what you look at determines what note plays; how you blink determines how it sounds.

## Signal Mappings

### Note triggering

The note at your gaze point is determined by colour (hue) and brightness:

| Colour | Wavelength | Note | Hue (OpenCV 0–179) |
|--------|-----------|------|--------------------|
| Red | ~700nm | C | 0–8, 165–179 |
| Orange | ~620nm | D | 8–25 |
| Yellow | ~580nm | E | 25–38 |
| Green | ~530nm | F | 38–75 |
| Cyan | ~500nm | G | 75–95 |
| Blue | ~470nm | A | 95–125 |
| Violet | ~400nm | B | 125–165 |

Brightness maps to octave (3–6): darker → lower, brighter → higher.

A note triggers when the gaze settles on new content. This is handled by `NoteGate` — it fires once when the detected MIDI note stabilises after a real transition, and suppresses jitter from text and fine details.

**Pd message:** `note_on <midi_note> <brightness>`
- `midi_note`: 36–83 (C3–B6)
- `brightness`: 0.0–1.0 (can be used for velocity/dynamics)

### Blink effects

| Eye event | Effect | Pd message |
|-----------|--------|------------|
| Intentional blink (≥500ms) | Clears AM-LFO | `am_lfo 0` |
| Flutter burst ends | Sets AM-LFO frequency | `am_lfo <hz>` |

**Flutter → AM-LFO:** blink count within the burst maps linearly to 1–50 Hz. 4 blinks = 1 Hz, 15 blinks = 50 Hz. The tremolo frequency rises with the intensity of the flutter.

**Note triggering is suppressed during active flutter** — use flutter as a distinct performance gesture.

### Confidence gating

During blinks and flutter, the raw gaze confidence (0–1) is forwarded to Pure Data. When the event ends, a single `confidence 1.0` reset message is sent.

**Pd message:** `confidence <value>`

This can be used to fade, mute, or modulate the sound while the gaze signal is unreliable.

## Pure Data Patch

`puredata/color_music.pd` — receives all messages above and synthesises sound using ADSR envelopes.

```
note_on <midi> <brightness>   →  triggers a note
am_lfo <hz>                   →  sets amplitude modulation frequency (0 = off)
confidence <0–1>              →  modulation/gate signal during noise
```

To use:
1. Open `puredata/color_music.pd` in Pure Data
2. Enable DSP (Media → DSP On)
3. Run the tracker with `--pd`

## Running

```bash
# Live tracking
uv run pupil-tracker --pd

# Playback
uv run gaze-player recordings/000 --pd

# This patch is the default, but can be explicit:
uv run pupil-tracker --patch color_music --pd
```

## Stability Tuning

The colour analyser applies temporal smoothing and stability hysteresis before emitting note values. These are tunable:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `--smoothing` | 3 | Frames averaged for colour smoothing. Higher = more stable but slower. |
| `--note-stability` | 2 | Consecutive frames needed before a note change is accepted. |
| `--octave-stability` | 3 | Stricter stability for octave changes (prevents brightness flicker). |
| `--octave-threshold` | 0.5 | Fraction of the stability window that must agree. |
| `--gamma` | 1.0 | Brighten dark footage (< 1.0) to spread brightness across more octaves. |

For a more reactive response (fast saccades, dense scenes):
```bash
uv run pupil-tracker --pd --note-stability 1 --octave-stability 2
```

For a more stable response (slow contemplative looking):
```bash
uv run pupil-tracker --pd --smoothing 5 --note-stability 3 --octave-stability 5
```

## Known Limitations

- **Same-colour transitions**: Moving gaze between two objects of identical hue and brightness produces no note re-trigger, because neither the raw MIDI note nor the fixation ID changes visibly during a smooth head turn. Pupil's fixation detector is sparse and doesn't always catch this.
- **Low-saturation colours**: Greys, whites, and near-white pastels have unreliable hue. The patch uses brightness and defaults the note to the current stable note. Minimum saturation threshold is 20/255.
- **Mixed colour regions**: The gaze region is 50×50px by default. If the region straddles two colours, the Gaussian-weighted average determines the note, which may not correspond to either colour cleanly.
