# Patch: TNC_v1

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
| Intentional blink (≥500ms) | Clears effect | `effect 0` |
| Flutter burst ends | Sets effect value | `effect <0–1>` |

**Flutter → effect:** blink count within the burst maps linearly to 0–1. 0.0 at the minimum flutter threshold, 1.0 at twice that. The value rises with the intensity of the flutter. PD scales it as needed.

**Note triggering is suppressed during active flutter** — use flutter as a distinct performance gesture.

## Pure Data Patch

`puredata/TNC_v1.pd` — receives all messages above and synthesises sound using ADSR envelopes.

```
note_on <midi> <brightness>   →  triggers a note
effect <0–1>                  →  sets effect intensity (0 = off)
```

To use:
1. Open `puredata/TNC_v1.pd` in Pure Data
2. Enable DSP (Media → DSP On)
3. Run the tracker (it connects to PD automatically)

## Running

```bash
# Live tracking (TNC_v1 is the default patch)
uv run pupil-tracker

# Playback
uv run pupil-player recordings/000

# Explicit patch selection:
uv run pupil-tracker --patch TNC_v1
```

## Known Limitations

- **Same-colour smooth pans**: Moving gaze across a uniform surface while the head pans produces no note re-trigger if hue and brightness are identical throughout. NoteGate's SmoothPan detection handles most cases (fires once on pan entry), but a continuous, slow pan over a perfectly uniform region will only trigger at the start.
- **Low-saturation colours**: Greys, whites, and near-white pastels have unreliable hue. The patch uses brightness and defaults the note to the current stable note. Minimum saturation threshold is 20/255.
- **Mixed colour regions**: The gaze region is 50×50px by default. If the region straddles two colours, the average determines the note, which may not correspond to either colour cleanly.
