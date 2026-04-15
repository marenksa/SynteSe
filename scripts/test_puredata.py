#!/usr/bin/env python3
"""Test script to verify Pure Data communication.

This sends fake brightness/color values to Pure Data so you can test the
connection and sound generation without the Pupil hardware.

Usage:
    # Brightness mode - for OSC (requires mrpeach in Pd):
    uv run python scripts/test_puredata.py --osc

    # Brightness mode - for FUDI (no externals needed):
    uv run python scripts/test_puredata.py --fudi

    # Color-to-music mode - cycles through colors:
    uv run python scripts/test_puredata.py --color-fudi

Make sure Pure Data is running with the appropriate patch open first!
"""

import argparse
import math
import time

from pythonosc import udp_client


# Note names for display
NOTE_NAMES = ["C", "D", "E", "F", "G", "A", "B"]
COLOR_NAMES = ["Red", "Orange", "Yellow", "Green", "Cyan", "Blue", "Violet"]

# Semitone offsets for major scale
NOTE_SEMITONES = [0, 2, 4, 5, 7, 9, 11]


def calculate_midi_note(note: int, octave: int) -> int:
    """Calculate MIDI note from note index (0-6) and octave."""
    return (octave + 1) * 12 + NOTE_SEMITONES[note]


def test_osc(host: str, port: int, duration: float) -> None:
    """Send test OSC messages to Pure Data."""
    print(f"[OSC] Sending to {host}:{port}")
    print("[OSC] Open 'brightness_receiver.pd' in Pure Data")
    print("[OSC] Make sure DSP is ON")
    print()

    client = udp_client.SimpleUDPClient(host, port)

    start = time.time()
    while time.time() - start < duration:
        # Generate a sine wave brightness pattern
        t = time.time() - start
        brightness = (math.sin(t * 0.5) + 1) / 2  # 0-1 range, slow oscillation

        client.send_message("/brightness", brightness)
        print(f"\r  Brightness: {brightness:.3f}", end="", flush=True)
        time.sleep(0.05)  # 20 Hz update rate

    print("\n[OSC] Done!")


def test_fudi(host: str, port: int, duration: float) -> None:
    """Send test FUDI messages to Pure Data."""
    import socket

    print(f"[FUDI] Connecting to {host}:{port}")
    print("[FUDI] Open 'brightness_simple.pd' in Pure Data")
    print("[FUDI] Make sure DSP is ON")
    print()

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((host, port))
        print("[FUDI] Connected!")
    except ConnectionRefusedError:
        print("[FUDI] ERROR: Could not connect. Is Pure Data running with the patch open?")
        return

    start = time.time()
    try:
        while time.time() - start < duration:
            # Generate a sine wave brightness pattern
            t = time.time() - start
            brightness = (math.sin(t * 0.5) + 1) / 2  # 0-1 range

            message = f"brightness {brightness:.4f};\n"
            sock.send(message.encode("utf-8"))
            print(f"\r  Brightness: {brightness:.3f}", end="", flush=True)
            time.sleep(0.05)
    finally:
        sock.close()

    print("\n[FUDI] Done!")


def test_color_osc(host: str, port: int, duration: float) -> None:
    """Send test color/MIDI OSC messages to Pure Data."""
    print(f"[COLOR-OSC] Sending to {host}:{port}")
    print("[COLOR-OSC] Open 'color_music.pd' in Pure Data")
    print("[COLOR-OSC] Make sure DSP is ON")
    print()
    print("Cycling through colors (wavelength order)...")
    print()

    client = udp_client.SimpleUDPClient(host, port)

    start = time.time()
    while time.time() - start < duration:
        t = time.time() - start

        # Cycle through notes (0-6) slowly
        note = int(t / 2) % 7

        # Vary brightness/octave with sine wave (octaves 2-6)
        brightness = (math.sin(t * 0.3) + 1) / 2  # 0-1 range
        octave = 2 + int(brightness * 4)  # 2-6

        midi_note = calculate_midi_note(note, octave)

        client.send_message("/midinote", midi_note)
        client.send_message("/note", note)
        client.send_message("/octave", octave)
        client.send_message("/brightness", brightness)

        note_name = NOTE_NAMES[note]
        color_name = COLOR_NAMES[note]
        print(
            f"\r  {color_name:7} -> {note_name}{octave} (MIDI {midi_note:3d}) "
            f"Brightness: {brightness:.2f}",
            end="",
            flush=True,
        )
        time.sleep(0.05)

    print("\n[COLOR-OSC] Done!")


def test_color_fudi(host: str, port: int, duration: float) -> None:
    """Send test color/MIDI FUDI messages to Pure Data."""
    import socket

    print(f"[COLOR-FUDI] Connecting to {host}:{port}")
    print("[COLOR-FUDI] Open 'color_music.pd' in Pure Data")
    print("[COLOR-FUDI] Make sure DSP is ON")
    print()
    print("Cycling through colors (wavelength order)...")
    print()

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((host, port))
        print("[COLOR-FUDI] Connected!")
    except ConnectionRefusedError:
        print("[COLOR-FUDI] ERROR: Could not connect. Is Pure Data running with the patch open?")
        return

    start = time.time()
    try:
        while time.time() - start < duration:
            t = time.time() - start

            # Cycle through notes (0-6) slowly
            note = int(t / 2) % 7

            # Vary brightness/octave with sine wave (octaves 2-6)
            brightness = (math.sin(t * 0.3) + 1) / 2  # 0-1 range
            octave = 2 + int(brightness * 4)  # 2-6

            midi_note = calculate_midi_note(note, octave)

            # Send all values
            message = (
                f"midinote {midi_note};\n"
                f"note {note};\n"
                f"octave {octave};\n"
                f"brightness {brightness:.4f};\n"
            )
            sock.send(message.encode("utf-8"))

            note_name = NOTE_NAMES[note]
            color_name = COLOR_NAMES[note]
            print(
                f"\r  {color_name:7} -> {note_name}{octave} (MIDI {midi_note:3d}) "
                f"Brightness: {brightness:.2f}",
                end="",
                flush=True,
            )
            time.sleep(0.05)
    finally:
        sock.close()

    print("\n[COLOR-FUDI] Done!")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test Pure Data communication",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Brightness mode options
    parser.add_argument(
        "--osc",
        action="store_true",
        help="Brightness via OSC (use with brightness_receiver.pd)",
    )
    parser.add_argument(
        "--fudi",
        action="store_true",
        help="Brightness via FUDI (use with brightness_simple.pd)",
    )

    # Color mode options
    parser.add_argument(
        "--color-osc",
        action="store_true",
        help="Color-to-music via OSC (use with color_music.pd)",
    )
    parser.add_argument(
        "--color-fudi",
        action="store_true",
        help="Color-to-music via FUDI (use with color_music.pd)",
    )

    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Pure Data host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port (default: 9000 for OSC, 9001 for FUDI)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Test duration in seconds (default: 30)",
    )

    args = parser.parse_args()

    if not any([args.osc, args.fudi, args.color_osc, args.color_fudi]):
        print("Please specify a mode:")
        print()
        print("Brightness mode:")
        print("  --osc        : Use with 'brightness_receiver.pd' (needs mrpeach)")
        print("  --fudi       : Use with 'brightness_simple.pd' (no externals)")
        print()
        print("Color-to-music mode:")
        print("  --color-osc  : Use with 'color_music.pd' (needs mrpeach)")
        print("  --color-fudi : Use with 'color_music.pd' (no externals)")
        return

    if args.osc:
        port = args.port or 9000
        test_osc(args.host, port, args.duration)
    elif args.fudi:
        port = args.port or 9001
        test_fudi(args.host, port, args.duration)
    elif args.color_osc:
        port = args.port or 9000
        test_color_osc(args.host, port, args.duration)
    elif args.color_fudi:
        port = args.port or 9001
        test_color_fudi(args.host, port, args.duration)


if __name__ == "__main__":
    main()

