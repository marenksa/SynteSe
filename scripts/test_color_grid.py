#!/usr/bin/env python3
"""Test script that plays through the color-brightness grid systematically.

Sends MIDI notes to Pure Data matching the test_images/color_test_grid.png

Two test sequences:
1. BY ROWS: Each color from darkest to brightest (Oct 2 → 6)
   Red: C2, C3, C4, C5, C6
   Orange: D2, D3, D4, D5, D6
   ... through Violet

2. BY COLUMNS: Each octave cycling through all colors
   Octave 2: C2, D2, E2, F2, G2, A2, B2
   Octave 3: C3, D3, E3, F3, G3, A3, B3
   ... through Octave 6

Usage:
    uv run python scripts/test_color_grid.py
"""

import argparse
import socket
import time

# Color/Note definitions (matching analyzer.py)
COLORS = [
    ("Red", "C", 0),      # Note index 0, semitone 0
    ("Orange", "D", 2),   # Note index 1, semitone 2
    ("Yellow", "E", 4),   # Note index 2, semitone 4
    ("Green", "F", 5),    # Note index 3, semitone 5
    ("Cyan", "G", 7),     # Note index 4, semitone 7
    ("Blue", "A", 9),     # Note index 5, semitone 9
    ("Violet", "B", 11),  # Note index 6, semitone 11
]

OCTAVES = [2, 3, 4, 5, 6]

# Brightness values for each octave (matching generate_test_image.py)
OCTAVE_BRIGHTNESS = {
    2: 32 / 255,   # Very dark
    3: 96 / 255,   # Dark
    4: 160 / 255,  # Medium
    5: 210 / 255,  # Bright
    6: 250 / 255,  # Very bright
}


def midi_note(semitone: int, octave: int) -> int:
    """Calculate MIDI note number."""
    return (octave + 1) * 12 + semitone


def test_by_rows(sock: socket.socket, note_duration: float) -> None:
    """Play each color from darkest to brightest."""
    print("\n" + "=" * 60)
    print("TEST 1: BY ROWS (each color across octaves)")
    print("=" * 60)

    for color_name, note_name, semitone in COLORS:
        print(f"\n--- {color_name} ({note_name}) ---")

        for octave in OCTAVES:
            midi = midi_note(semitone, octave)
            brightness = OCTAVE_BRIGHTNESS[octave]
            note_idx = COLORS.index((color_name, note_name, semitone))

            # Send to Pure Data
            message = (
                f"midinote {midi};\n"
                f"note {note_idx};\n"
                f"octave {octave};\n"
                f"brightness {brightness:.4f};\n"
            )
            sock.send(message.encode("utf-8"))

            print(f"  {note_name}{octave} (MIDI {midi:3d}) - Brightness: {brightness:.2f}")
            time.sleep(note_duration)

    print("\n[Rows complete]")


def test_by_columns(sock: socket.socket, note_duration: float) -> None:
    """Play each octave cycling through all colors."""
    print("\n" + "=" * 60)
    print("TEST 2: BY COLUMNS (each octave across colors)")
    print("=" * 60)

    for octave in OCTAVES:
        brightness = OCTAVE_BRIGHTNESS[octave]
        print(f"\n--- Octave {octave} (Brightness: {brightness:.2f}) ---")

        for note_idx, (color_name, note_name, semitone) in enumerate(COLORS):
            midi = midi_note(semitone, octave)

            # Send to Pure Data
            message = (
                f"midinote {midi};\n"
                f"note {note_idx};\n"
                f"octave {octave};\n"
                f"brightness {brightness:.4f};\n"
            )
            sock.send(message.encode("utf-8"))

            print(f"  {note_name}{octave} ({color_name:7}) MIDI {midi:3d}")
            time.sleep(note_duration)

    print("\n[Columns complete]")


def run_test(host: str, port: int, note_duration: float, auto: bool = False) -> None:
    """Run tests using FUDI protocol."""
    print(f"Connecting to {host}:{port}")
    print("Open 'color_music.pd' in Pure Data")
    print("Make sure DSP is ON")

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((host, port))
        print("Connected!")
    except ConnectionRefusedError:
        print("ERROR: Could not connect. Is Pure Data running with the patch open?")
        return

    try:
        if auto:
            print("\n[Auto mode] Starting TEST 1 in 2 seconds...")
            time.sleep(2)
        else:
            input("\nPress Enter to start TEST 1 (by rows)...")
        test_by_rows(sock, note_duration)

        if auto:
            print("\n[Auto mode] Starting TEST 2 in 3 seconds...")
            time.sleep(3)
        else:
            input("\nPress Enter to start TEST 2 (by columns)...")
        test_by_columns(sock, note_duration)

        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETE")
        print("=" * 60)
    finally:
        sock.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test color grid through Pure Data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Pure Data host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9001,
        help="Pure Data FUDI port (default: 9001)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.8,
        help="Duration of each note in seconds (default: 0.8)",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Run automatically without waiting for Enter key",
    )

    args = parser.parse_args()

    print()
    print("Color Grid Test")
    print("===============")
    print()
    print("This test will play through the color-brightness grid:")
    print()
    print("  TEST 1 (Rows): Each color from darkest to brightest")
    print("    Red:    C2 → C3 → C4 → C5 → C6")
    print("    Orange: D2 → D3 → D4 → D5 → D6")
    print("    ...etc")
    print()
    print("  TEST 2 (Columns): Each octave through all colors")
    print("    Oct 2: C2 → D2 → E2 → F2 → G2 → A2 → B2")
    print("    Oct 3: C3 → D3 → E3 → F3 → G3 → A3 → B3")
    print("    ...etc")
    print()

    run_test(args.host, args.port, args.duration, args.auto)


if __name__ == "__main__":
    main()
