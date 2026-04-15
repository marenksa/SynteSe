#!/usr/bin/env python3
"""Test script to verify Pure Data communication.

This sends fake color/note values to Pure Data so you can test the
connection and sound generation without the Pupil hardware.

Usage:
    # OSC (requires mrpeach in Pd):
    uv run python scripts/test_puredata.py --osc

    # FUDI (no externals needed):
    uv run python scripts/test_puredata.py --fudi

Make sure Pure Data is running with color_music.pd open first!
"""

import argparse
import math
import socket
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
    print("[OSC] Open 'color_music.pd' in Pure Data")
    print("[OSC] Make sure DSP is ON")
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

    print("\n[OSC] Done!")


def test_fudi(host: str, port: int, duration: float) -> None:
    """Send test FUDI messages to Pure Data."""
    print(f"[FUDI] Connecting to {host}:{port}")
    print("[FUDI] Open 'color_music.pd' in Pure Data")
    print("[FUDI] Make sure DSP is ON")
    print()
    print("Cycling through colors (wavelength order)...")
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

    print("\n[FUDI] Done!")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test Pure Data communication",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--osc",
        action="store_true",
        help="Send via OSC (requires mrpeach external in Pd)",
    )
    parser.add_argument(
        "--fudi",
        action="store_true",
        help="Send via FUDI/TCP (no externals needed)",
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

    if not args.osc and not args.fudi:
        print("Please specify --osc or --fudi")
        print()
        print("  --osc  : Requires mrpeach external in Pure Data")
        print("  --fudi : No externals needed (recommended)")
        return

    if args.osc:
        port = args.port or 9000
        test_osc(args.host, port, args.duration)
    else:
        port = args.port or 9001
        test_fudi(args.host, port, args.duration)


if __name__ == "__main__":
    main()
