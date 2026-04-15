#!/usr/bin/env python3
"""Test script to verify Pure Data communication.

This sends fake brightness values to Pure Data so you can test the
connection and sound generation without the Pupil hardware.

Usage:
    # For OSC (requires mrpeach in Pd):
    uv run python scripts/test_puredata.py --osc

    # For FUDI (no externals needed):
    uv run python scripts/test_puredata.py --fudi

Make sure Pure Data is running with the appropriate patch open first!
"""

import argparse
import math
import time

from pythonosc import udp_client


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
        print(f"[FUDI] Connected!")
    except ConnectionRefusedError:
        print(f"[FUDI] ERROR: Could not connect. Is Pure Data running with the patch open?")
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test Pure Data communication",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--osc",
        action="store_true",
        help="Use OSC protocol (requires mrpeach external in Pd)",
    )
    parser.add_argument(
        "--fudi",
        action="store_true",
        help="Use FUDI protocol (no externals needed)",
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
        print("  --osc  : Use with 'brightness_receiver.pd' (needs mrpeach external)")
        print("  --fudi : Use with 'brightness_simple.pd' (no externals needed)")
        return

    if args.osc:
        port = args.port or 9000
        test_osc(args.host, port, args.duration)
    elif args.fudi:
        port = args.port or 9001
        test_fudi(args.host, port, args.duration)


if __name__ == "__main__":
    main()

